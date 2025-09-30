import functools
from collections.abc import Callable

import numpy as np
import scipy as sp

from disk_solver.parameter_init import cgs_consts


def vectorize_functions(func) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> np.ndarray:
        # Turn both args and kwargs into arrays
        arrays = [np.asarray(a) for a in args] + [np.asarray(v) for v in kwargs.values()]
        bcast = np.broadcast_arrays(*arrays)
        out_shape = bcast[0].shape

        # Split back args and kwargs after broadcasting
        arg_arrays = bcast[: len(args)]
        kwarg_arrays = dict(zip(kwargs.keys(), bcast[len(args) :], strict=False))

        results = []
        for idx in np.ndindex(out_shape):
            scalars = [arr[idx] for arr in arg_arrays]
            kw_scalars = {k: arr[idx] for k, arr in kwarg_arrays.items()}
            results.append(func(*scalars, **kw_scalars))

        sample_output = np.asarray(results[0])
        return np.array(results).reshape(out_shape + sample_output.shape)

    return wrapper


def get_bhmass(*, dimless_bhmass) -> float | np.ndarray:
    dimless_bhmass = np.asarray(dimless_bhmass)
    bhmass = dimless_bhmass * cgs_consts.cgs_msun
    return bhmass


def get_accrate_edd(*, dimless_bhmass) -> float | np.ndarray:
    dimless_bhmass = np.asarray(dimless_bhmass)
    accrate_edd = 1.44e18 * dimless_bhmass
    return accrate_edd


def get_accrate_init(*, dimless_accrate, dimless_bhmass) -> float | np.ndarray:
    dimless_accrate, dimless_bhmass = np.asarray(dimless_accrate), np.asarray(dimless_bhmass)
    accrate_edd = get_accrate_edd(dimless_bhmass=dimless_bhmass)
    accrate_init = accrate_edd * dimless_accrate
    return accrate_init


def get_radius_sch(*, dimless_bhmass, dimless_radius) -> float | np.ndarray:
    dimless_bhmass, dimless_radius = np.asarray(dimless_bhmass), np.asarray(dimless_radius)
    bhmass = get_bhmass(dimless_bhmass=dimless_bhmass)
    radius_sch = 2 * cgs_consts.cgs_gra * bhmass / cgs_consts.cgs_c / cgs_consts.cgs_c
    return radius_sch


def get_coeff_in(*, gas_index) -> float | np.ndarray:
    gas_index = np.asarray(gas_index)
    numerator = (2**gas_index * sp.special.gamma(gas_index + 1)) ** 2
    denominator = sp.special.gamma(2 * gas_index + 2)
    coeff_in = numerator / denominator
    return coeff_in


def get_radius_fromdimless(*, dimless_bhmass, dimless_radius) -> float | np.ndarray:
    dimless_bhmass, dimless_radius = np.asarray(dimless_bhmass), np.asarray(dimless_radius)
    radius_sch = get_radius_sch(dimless_bhmass=dimless_bhmass, dimless_radius=dimless_radius)
    radius = dimless_radius * radius_sch
    return radius


def get_accrate_fromdimless(*, dimless_bhmass, dimless_accrate) -> float | np.ndarray:
    dimless_bhmass, dimless_accrate = np.asarray(dimless_bhmass), np.asarray(dimless_accrate)
    accrate_edd = get_accrate_edd(dimless_bhmass=dimless_bhmass)
    accrate = accrate_edd * dimless_accrate
    return accrate
