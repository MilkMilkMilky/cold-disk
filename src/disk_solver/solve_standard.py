import math

import numpy as np
import scipy as sp

from disk_solver.parameter_init import cgs_consts
from parameters import model_params

alpha_viscosity = model_params.alpha_viscosity[0]
dimless_accrate = model_params.dimless_accrate[0]
dimless_bhmass = model_params.dimless_bhmass[0]
dimless_radius_out = model_params.dimless_radius_out[0]
gas_index = model_params.gas_index[0]


def get_bhmass(*, dimless_bhmass):
    dimless_bhmass = np.asarray(dimless_bhmass)
    bhmass = dimless_bhmass * cgs_consts.cgs_msun
    return bhmass


def get_accrate_edd(*, dimless_bhmass):
    dimless_bhmass = np.asarray(dimless_bhmass)
    accrate_edd = 1.44e18 * dimless_bhmass
    return accrate_edd


def get_accrate_init(*, dimless_accrate, dimless_bhmass):
    dimless_accrate, dimless_bhmass = np.asarray(dimless_accrate), np.asarray(dimless_bhmass)
    accrate_edd = get_accrate_edd(dimless_bhmass=dimless_bhmass)
    accrate_init = accrate_edd * dimless_accrate
    return accrate_init


def get_radius_sch(*, dimless_bhmass, dimless_radius):
    dimless_bhmass, dimless_radius = np.asarray(dimless_bhmass), np.asarray(dimless_radius)
    bhmass = get_bhmass(dimless_bhmass=dimless_bhmass)
    radius_sch = 2 * cgs_consts.cgs_gra * bhmass / cgs_consts.cgs_c / cgs_consts.cgs_c
    return radius_sch


def get_coeff_in(*, gas_index):
    gas_index = np.asarray(gas_index)
    numerator = (2**gas_index * sp.special.gamma(gas_index + 1)) ** 2
    denominator = sp.special.gamma(2 * gas_index + 2)
    coeff_in = numerator / denominator
    return coeff_in
