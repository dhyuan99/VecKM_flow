from dataclasses import dataclass

@dataclass
class MVSECParams:
    # VecKM parameters
    d: int = 384
    alpha: float = 5
    # local scaling parameters
    pxl_radius: float = 0.0225
    t_radius: float = 0.025
   
@dataclass
class EVIMOParams:
    # VecKM parameters
    d: int = 384
    alpha: float = 5
    # local scaling parameters
    pxl_radius: float = 0.0225
    t_radius: float = 0.01
    
@dataclass
class DSECParams:
    # VecKM parameters
    d: int = 384
    alpha: float = 5
    # local scaling parameters
    pxl_radius: float = 0.0225
    t_radius: float = 0.01

@dataclass
class UNIONParams:
    # VecKM parameters
    d: int = 384
    alpha: float = 5
    # local scaling parameters
    pxl_radius: float = 0.0225
    t_radius: float = 0.01
