from dataclasses import dataclass

import numpy as np

__all__ = ["model_params"]

_alpha_viscosity = np.array([0.1])
_dimless_accrate = np.array([0.1, 1, 10])
_dimless_bhmass = np.array([1e6, 1e7])
_gas_index = np.array([3.0])
_wind_index = np.array([0.0])
_dimless_radius_in = np.array([1.0])
_dimless_radius_out = np.array([10000.0])


@dataclass(frozen=True)
class ModelParams:
    alpha_viscosity: np.ndarray = _alpha_viscosity
    dimless_accrate: np.ndarray = _dimless_accrate
    dimless_bhmass: np.ndarray = _dimless_bhmass
    gas_index: np.ndarray = _gas_index
    wind_index: np.ndarray = _wind_index
    dimless_radius_in: np.ndarray = _dimless_radius_in
    dimless_radius_out: np.ndarray = _dimless_radius_out


model_params: ModelParams = ModelParams()
