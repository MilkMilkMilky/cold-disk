from dataclasses import dataclass

import numpy as np

_alpha_viscosity = np.array([0.1])
_dimless_accrate = np.array([1000])
_dimless_bhmass = np.array([1e7])
_gas_index = np.array([3.0])
_wind_index = np.array([0.0])
_dimless_radius_in = np.array([1.0])
_dimless_radius_out = np.array([10000.0])


@dataclass(frozen=True)
class ModelParams:
    alpha_viscosity: np.ndarray
    dimless_accrate: np.ndarray
    dimless_bhmass: np.ndarray
    gas_index: np.ndarray
    wind_index: np.ndarray
    dimless_radius_in: np.ndarray
    dimless_radius_out: np.ndarray


model_params: ModelParams = ModelParams(
    alpha_viscosity=_alpha_viscosity,
    dimless_accrate=_dimless_accrate,
    dimless_bhmass=_dimless_bhmass,
    gas_index=_gas_index,
    wind_index=_wind_index,
    dimless_radius_in=_dimless_radius_in,
    dimless_radius_out=_dimless_radius_out,
)
