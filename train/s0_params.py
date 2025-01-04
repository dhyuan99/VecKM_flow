from dataclasses import dataclass

dataset_path = './'

@dataclass
class MVSECParams:
    # VecKM parameters
    d: int = 384
    alpha: float = 5
    # local scaling parameters
    pxl_radius: float = 0.0225
    t_radius: float = 0.025
    train_norm_range: tuple = (0.05, 3)
    val_norm_range: tuple = (0.05, 3)
    # training parameters
    dataset_path: str = '/fs/nexus-projects/VecKM_flow/data'
    loss_fn: str = 'mfl' # motion field loss
    mfl_loss_lambda: float = 0.1
    # inference parameters
    width: int = 346
    height: int = 260
    K: tuple = (223.9940010790056, 223.61783486959376, 170.7684322973841, 128.18711828338436)
   
@dataclass
class EVIMOParams:
    # VecKM parameters
    d: int = 384
    alpha: float = 5
    # local scaling parameters
    pxl_radius: float = 0.0225
    t_radius: float = 0.01
    train_norm_range: tuple = (0.03, 3) # only flows with magnitude > 0.03, < 3 are used in training.
    val_norm_range: tuple = (0.03, 3)   # only flows with magnitude > 0.03, < 3 are used in validation.
    # training parameters
    dataset_path: str = '/fs/nexus-projects/VecKM_flow/data'
    loss_fn: str = 'mfl' # motion field loss
    mfl_loss_lambda: float = 0.1
    # inference parameters
    width: int = 640
    height: int = 480
    K: tuple = (520.257996, 520.297974, 321.259003, 242.490005)
    
@dataclass
class DSECParams:
    # VecKM parameters
    d: int = 384
    alpha: float = 5
    # local scaling parameters
    pxl_radius: float = 0.0225
    t_radius: float = 0.01
    train_norm_range: tuple = (0.01, 3)
    val_norm_range: tuple = (0.01, 3)
    # training parameters
    dataset_path: str = '/fs/nexus-projects/VecKM_flow/data'
    loss_fn: str = 'mfl' # motion field loss
    mfl_loss_lambda: float = 0.1
    # inference parameters
    width: int = 640
    height: int = 480
    K: tuple = (569.7632987676102, 569.7632987676102, 335.0999870300293, 221.23667526245117)

@dataclass
class UNIONParams:
    # VecKM parameters
    d: int = 384
    alpha: float = 5
    # local scaling parameters
    pxl_radius: float = 0.0225
    t_radius: float = 0.01
    train_norm_range: tuple = (0.01, 3)
    val_norm_range: tuple = (0.01, 3)
    # training parameters
    dataset_path: str = '/fs/nexus-projects/VecKM_flow/data'
    loss_fn: str = 'mfl' # motion field loss
    mfl_loss_lambda: float = 0.1
    # inference parameters
    width: int = 640
    height: int = 480
    K: tuple = (569.7632987676102, 569.7632987676102, 335.0999870300293, 221.23667526245117)