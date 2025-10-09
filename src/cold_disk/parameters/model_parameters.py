"""Module `cold_disk.parameters.model_parameters`.

Provides default adjustable parameters for accretion disk models.
This module defines a frozen dataclass `ModelParams` containing arrays of
possible values for various disk parameters. A singleton instance
`model_params` is provided with pre-filled example values suitable for
parameter sweeps or testing.

Notes:
-----
- `ModelParams` is a frozen dataclass containing arrays of parameter values.
- `model_params` is a module-level singleton with default example values.
- All parameter arrays are NumPy arrays for efficient computation.

Example:
-------
>>> from cold_disk import model_params
>>> model_params.dimless_accrate
array([0.1, 1.0, 10.0])
>>> model_params.alpha_viscosity
array([0.1])

"""
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
    """Frozen dataclass storing arrays of adjustable parameters for accretion disk models.

    Attributes
    ----------
    alpha_viscosity : np.ndarray
        Possible values for the dimensionless alpha viscosity parameter.
    dimless_accrate : np.ndarray
        Possible values for the dimensionless accretion rate.
    dimless_bhmass : np.ndarray
        Possible values for black hole masses (dimensionless units).
    gas_index : np.ndarray
        Possible values for the gas index parameter.
    wind_index : np.ndarray
        Possible values for the wind index parameter.
    dimless_radius_in : np.ndarray
        Possible values for the inner disk radius (dimensionless units).
    dimless_radius_out : np.ndarray
        Possible values for the outer disk radius (dimensionless units).

    """

    alpha_viscosity: np.ndarray = _alpha_viscosity
    dimless_accrate: np.ndarray = _dimless_accrate
    dimless_bhmass: np.ndarray = _dimless_bhmass
    gas_index: np.ndarray = _gas_index
    wind_index: np.ndarray = _wind_index
    dimless_radius_in: np.ndarray = _dimless_radius_in
    dimless_radius_out: np.ndarray = _dimless_radius_out


model_params: ModelParams = ModelParams()
"""Module-level singleton instance of `ModelParams` containing default example values."""
