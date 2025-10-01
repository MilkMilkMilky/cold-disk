import numpy as np
import scipy as sp

from disk_solver.parameter_init import DiskParams, cgs_consts


def get_bhmass(*, par: DiskParams) -> float:
    bhmass = par.dimless_bhmass * cgs_consts.cgs_msun
    return bhmass


def get_accrate_edd(*, par: DiskParams) -> float:
    accrate_edd = 1.44e18 * par.dimless_bhmass
    return accrate_edd


def get_accrate_init(*, par: DiskParams) -> float:
    accrate_edd = get_accrate_edd(par=par)
    accrate_init = accrate_edd * par.dimless_accrate
    return accrate_init


def get_radius_sch(*, par: DiskParams) -> float:
    bhmass = get_bhmass(par=par)
    radius_sch = 2 * cgs_consts.cgs_gra * bhmass / cgs_consts.cgs_c / cgs_consts.cgs_c
    return radius_sch


def get_coeff_in(*, par: DiskParams) -> float:
    numerator = (2**par.gas_index * sp.special.gamma(par.gas_index + 1)) ** 2
    denominator = sp.special.gamma(2 * par.gas_index + 2)
    coeff_in = numerator / denominator
    return coeff_in


def get_radius_fromdimless(*, par: DiskParams, dimless_radius) -> float | np.ndarray:
    dimless_radius = np.asarray(dimless_radius)
    radius_sch = get_radius_sch(par=par)
    radius = dimless_radius * radius_sch
    return radius


def get_accrate_fromdimless(*, par: DiskParams) -> float:
    accrate_edd = get_accrate_edd(par=par)
    accrate = accrate_edd * par.dimless_accrate
    return accrate
