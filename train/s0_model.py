import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm

""" Complex Neural Network Layers
Adopted from complexPyTorch https://github.com/wavefrontshaping/complexPyTorch
"""
class ComplexReLU(nn.Module):
     def forward(self,input):
        return torch.relu(input.real).type(
            torch.complex64
        ) + 1j * torch.relu(input.imag).type(
            torch.complex64
        )  
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features, bias)
        self.fc_i = nn.Linear(in_features, out_features, bias)

    def forward(self, input):
        return (self.fc_r(input.real)-self.fc_i(input.imag)).type(torch.complex64) \
            + 1j*(self.fc_r(input.imag)+self.fc_i(input.real)).type(torch.complex64)

def get_adj_matrix(pts, r):
    """ Compute the sparse adjacency matrix of the given point cloud.
    Args:
        pts: (n, ?) tensor, the input point cloud.
        r: float, the radius of the ball.
    Returns:
        adj_matrix: sparse (n, n) matrix, the adjacency matrix. 
                    adj_matrix[i,j] equals 1 if ||pts[i] - pts[j]|| < r, else 0.
    """
    
    # This is the batch size when computing the adjacency matrix.
    # It can adjusted based on your GPU memory. 8192 ** 2 is for 12GB GPU.
    MAX_SIZE = 8192 ** 2

    N = pts.shape[0]
    if N > MAX_SIZE ** 0.5:
        step_size = MAX_SIZE // N
        slice_grid = torch.arange(0, N, step_size)
        slice_grid = torch.cat([slice_grid, torch.tensor([N])])
        non_zero_indices = []
        for j in range(1, len(slice_grid)):
            dist = torch.cdist(pts[slice_grid[j-1]:slice_grid[j]], pts)
            indices = torch.nonzero(dist < r, as_tuple=False)
            indices[:,0] += slice_grid[j-1]
            non_zero_indices.append(indices)
        non_zero_indices = torch.cat(non_zero_indices).T
        adj_matrix = torch.sparse_coo_tensor(
            non_zero_indices, 
            torch.ones_like(non_zero_indices[0], dtype=torch.float32), 
            size=(N, N)
        )
        return adj_matrix
    else:
        dist = torch.cdist(pts, pts)
        adj_matrix = torch.where(dist < r, torch.ones_like(dist), torch.zeros_like(dist))
        return adj_matrix

def strict_standard_normal(d):
    """ Adopted from VecKM.
    this function generate very similar outcomes as torch.randn(d)
    but the numbers are strictly standard normal, no randomness.
    """
    y = np.linspace(0, 1, d+2)
    x = norm.ppf(y)[1:-1]
    np.random.shuffle(x)
    x = torch.tensor(x).float()
    return x

class VecKM(nn.Module):
    def __init__(self, d=384, alpha=5, radius=1.):
        super().__init__()
        self.sqrt_d = d ** 0.5
        self.alpha, self.d, self.radius = alpha, d, radius

        self.A = torch.stack(
            [strict_standard_normal(d) for _ in range(3)], 
            dim=0
        ) * alpha
        self.A = nn.Parameter(self.A, False)                                    # (3, d)

    def forward(self, pts):
        """ Compute the dense local geometry encodings of the given point cloud.
        Args:
            pts: (n, 3) tensor, the input point cloud.

        Returns:
            G: (n, d) tensor
               the dense local geometry encodings. 
               note: it is complex valued. 
        """
        J = get_adj_matrix(pts[:,1:], self.radius)                              # SparseReal(n, n)
                                                                                # only use x, y for adjacency matrix, ignore t.
        pA = pts @ self.A                                                       # Real(n, d
        epA = torch.cat([torch.cos(pA), torch.sin(pA)], dim=1)                  # Real(n, 2d)
        G = J @ epA                                                             # Real(n, 2d)
        G = torch.complex(
            G[:, :self.d], G[:, self.d:]
        ) / torch.complex(
            epA[:, :self.d], epA[:, self.d:]
        )                                                                       # Complex(n, d)
        G = G / torch.norm(G, dim=-1, keepdim=True) * self.sqrt_d               # Complex(n, d)
        return G

    def __repr__(self):
        return f'VecKM(d={self.d}, alpha={self.alpha}, radius={self.radius}.)'

class ResidualBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            ComplexLinear(d, d, bias=False),
            ComplexReLU(),
            ComplexLinear(d, d, bias=False),
            ComplexReLU(),
            ComplexLinear(d, d, bias=False),
        )

    def forward(self, x):
        return x + self.net(x)
    
class FeatureTransform(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.res1 = ResidualBlock(d)
        self.down = ComplexLinear(d, d // 2, bias=False)
        self.res2 = ResidualBlock(d // 2)
        self.out = ComplexLinear(d // 2, 1, bias=False)
        self.relu = ComplexReLU()
        
    def forward(self, G):
        G = self.relu(self.res1(G))
        G = self.relu(self.down(G))
        G = self.relu(self.res2(G))
        G = self.out(G)
        return G
    
class NormalEstimator(nn.Module):
    def __init__(self, d, alpha):
        super().__init__()
        self.vkm = VecKM(d, alpha)
        self.feat_trans = FeatureTransform(d)

    def forward(self, events):
        # events has shape (n, 3), txy.
        G = self.vkm(events)
        pred = self.feat_trans(G)
        return torch.concatenate((pred.real, pred.imag), dim=-1)

    def inference(self, events, ensemble=3):
        all_preds = []
        alpha_list = np.linspace(0, 2*np.pi, ensemble, endpoint=False)
        for e in range(ensemble):
            alpha = alpha_list[e]

            R = torch.tensor([
                [1, 0, 0],
                [0, np.cos(alpha), -np.sin(alpha)],
                [0, np.sin(alpha), np.cos(alpha)]   
            ]).float().to(events.device)
            R2d = torch.tensor([
                [np.cos(alpha), -np.sin(alpha)],
                [np.sin(alpha), np.cos(alpha)]
            ]).float().to(events.device)

            pred = self(events @ R.T) @ torch.inverse(R2d).T
            all_preds.append(pred.detach().cpu())

        all_preds = torch.stack(all_preds, dim=1)
        return self.vote(all_preds)

    def vote(self, all_preds):
        # all_preds has shape (n, ensemble, 2)
        radius = torch.norm(all_preds, dim=-1)                                  # (n, ensemble)

        angles = torch.atan2(all_preds[...,1], all_preds[...,0])
        sin_angles = torch.sin(angles)                                          # (n, ensemble)
        cos_angles = torch.cos(angles)                                          # (n, ensemble)
        sum_sin = sin_angles.sum(dim=1, keepdim=True)                           # (n, 1)
        sum_cos = cos_angles.sum(dim=1, keepdim=True)                           # (n, 1)
        mean_angle = torch.atan2(sum_sin, sum_cos)                              # (n, 1)

        circular_variance = torch.sqrt(
            torch.sum(
                torch.cos(angles - mean_angle), dim=1
            ) ** 2 + torch.sum(
                torch.sin(angles - mean_angle), dim=1
            ) ** 2
        ) / angles.size(1)                                                      # (n,)
        circular_std = torch.sqrt(-2 * torch.log(circular_variance))            # (n,)

        mean_radius = radius.mean(1)                                            # (n,)
        vote_flows = torch.stack([
            mean_radius * torch.cos(mean_angle[:,0]),
            mean_radius * torch.sin(mean_angle[:,0])
        ], dim=-1)
        
        return vote_flows, circular_std


def MotionFieldLoss(pred_flows, focused_flows, lambd):

    angular_loss = - torch.sum(
            (pred_flows-focused_flows/2) * focused_flows, dim=-1
        ) / (
            torch.linalg.norm(
                pred_flows-focused_flows/2, dim=-1
            ) * torch.linalg.norm(
                focused_flows, dim=-1
            ) + 1e-7
        )

    y = torch.norm(focused_flows/2, dim=-1)
    z = torch.norm(pred_flows-focused_flows/2, dim=-1)

    norm_loss = (torch.log(lambd+y) - torch.log(lambd+z)).square()

    return angular_loss.mean() + norm_loss.mean()

def EndPointLoss(pred_flows, focused_flow):
    return torch.norm(pred_flows - focused_flow, dim=-1).mean()