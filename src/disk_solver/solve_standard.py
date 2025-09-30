import math

import numpy as np
import scipy as sp

from disk_solver import solve_tools
from disk_solver.parameter_init import cgs_consts
from parameters import model_params


def get_standard_angvel(*, dimless_bhmass, dimless_radius):
    dimless_bhmass, dimless_radius = np.asarray(dimless_bhmass), np.asarray(dimless_radius)
    bhmass = solve_tools.get_bhmass(dimless_bhmass=dimless_bhmass)
    radius = solve_tools.get_radius_fromdimless(dimless_bhmass=dimless_bhmass, dimless_radius=dimless_radius)
    standard_angvel = np.sqrt(cgs_consts.cgs_gra * bhmass / radius / radius / radius)
    return standard_angvel


def get_standard_pressure(*, standard_density, standard_temperature):
    standard_density, standard_temperature = np.asarray(standard_density), np.asarray(standard_temperature)
    part_1 = 2 * standard_density * cgs_consts.cgs_kb * standard_temperature / cgs_consts.cgs_mh
    part_2 = cgs_consts.cgs_a * standard_temperature**4 / 3
    standard_pressure = part_1 + part_2
    return standard_pressure

def get_standard_soundvel(*, standard_pressure, standard_density):
    standard_pressure, standard_density = np.asarray(standard_pressure), np.asarray(standard_density)
    standard_soundvel = np.sqrt(standard_pressure / standard_density)
    return standard_soundvel

def get_standard_averopacity(*, standard_density, standard_temperature):
    standard_density, standard_temperature = np.asarray(standard_density), np.asarray(standard_temperature)
    standard_averopacity = cgs_consts.cgs_kes + 6.4e22 * standard_density * standard_temperature ** (-3.5)
    return standard_averopacity

def get_standard_kineviscocity(*, standard_arealdensity, dimless_bhmass, dimless_accrate, dimless_radius):
    standard_arealdensity = np.asanyarray(standard_arealdensity)
    dimless_bhmass = np.asarray(dimless_bhmass)
    dimless_accrate, dimless_radius = np.asarray(dimless_accrate), np.asarray(dimless_radius)
    accrate = solve_tools.get_accrate_fromdimless(dimless_bhmass=dimless_bhmass, dimless_accrate=dimless_accrate)
    part_1 = accrate / 3 / math.pi / standard_arealdensity
    part_2 = 1 - np.sqrt(3 / dimless_radius)
    standard_kineviscocity = part_1 * part_2
    return standard_kineviscocity



if __name__ == "__main__":
    alpha_viscosity = model_params.alpha_viscosity[0]
    dimless_accrate = model_params.dimless_accrate[0]
    dimless_bhmass = model_params.dimless_bhmass[0]
    gas_index = model_params.gas_index[0]
    wind_index = model_params.wind_index[0]
    dimless_radius_in = model_params.dimless_radius_in[0]
    dimless_radius_out = model_params.dimless_radius_out[0]
