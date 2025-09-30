import math

import numpy as np
import scipy as sp

from disk_solver import solve_tools
from disk_solver.parameter_init import cgs_consts
from parameters import model_params


def get_standard_angvel(*, dimless_bhmass, dimless_radius) -> float | np.ndarray:
    dimless_bhmass, dimless_radius = np.asarray(dimless_bhmass), np.asarray(dimless_radius)
    bhmass = solve_tools.get_bhmass(dimless_bhmass=dimless_bhmass)
    radius = solve_tools.get_radius_fromdimless(dimless_bhmass=dimless_bhmass, dimless_radius=dimless_radius)
    standard_angvel = np.sqrt(cgs_consts.cgs_gra * bhmass / radius / radius / radius)
    return standard_angvel


def get_standard_pressure(*, standard_density, standard_temperature) -> float | np.ndarray:
    standard_density, standard_temperature = np.asarray(standard_density), np.asarray(standard_temperature)
    part_1 = 2 * standard_density * cgs_consts.cgs_kb * standard_temperature / cgs_consts.cgs_mh
    part_2 = cgs_consts.cgs_a * standard_temperature**4 / 3
    standard_pressure = part_1 + part_2
    return standard_pressure

def get_standard_soundvel(*, standard_pressure, standard_density) -> float | np.ndarray:
    standard_pressure, standard_density = np.asarray(standard_pressure), np.asarray(standard_density)
    standard_soundvel = np.sqrt(standard_pressure / standard_density)
    return standard_soundvel

def get_standard_averopacity(*, standard_density, standard_temperature) -> float | np.ndarray:
    standard_density, standard_temperature = np.asarray(standard_density), np.asarray(standard_temperature)
    standard_averopacity = cgs_consts.cgs_kes + 6.4e22 * standard_density * standard_temperature ** (-3.5)
    return standard_averopacity

def get_standard_kineviscocity(
    *,
    standard_arealdensity,
    dimless_bhmass,
    dimless_accrate,
    dimless_radius,
) -> float | np.ndarray:
    standard_arealdensity = np.asanyarray(standard_arealdensity)
    dimless_bhmass = np.asarray(dimless_bhmass)
    dimless_accrate, dimless_radius = np.asarray(dimless_accrate), np.asarray(dimless_radius)
    accrate = solve_tools.get_accrate_fromdimless(dimless_bhmass=dimless_bhmass, dimless_accrate=dimless_accrate)
    part_1 = accrate / 3 / math.pi / standard_arealdensity
    part_2 = 1 - np.sqrt(3 / dimless_radius)
    standard_kineviscocity = part_1 * part_2
    return standard_kineviscocity

def get_standard_halfheight_test(
    *,
    alpha_viscosity,
    dimless_bhmass,
    dimless_accrate,
    dimless_radius,
) -> float | np.ndarray:
    alpha_viscosity, dimless_bhmass = np.asarray(alpha_viscosity), np.asarray(dimless_bhmass)
    dimless_accrate, dimless_radius = np.asarray(dimless_accrate), np.asarray(dimless_radius)
    standard_halfheight_test = (
        1.5e3 * alpha_viscosity ** (-1 / 10) * dimless_bhmass ** (9 / 10) * (dimless_accrate / 0.1) ** (3 / 20)
    )
    standard_halfheight_test *= dimless_radius ** (9 / 8) * (1 - np.sqrt(3 / dimless_radius)) ** (3 / 20)
    return standard_halfheight_test


def get_standard_arealdensity_test(
    *,
    alpha_viscosity,
    dimless_bhmass,
    dimless_accrate,
    dimless_radius,
) -> float | np.ndarray:
    alpha_viscosity, dimless_bhmass = np.asarray(alpha_viscosity), np.asarray(dimless_bhmass)
    dimless_accrate, dimless_radius = np.asarray(dimless_accrate), np.asarray(dimless_radius)
    standard_arealdensity_test = (
        1.4e5 * alpha_viscosity ** (-4 / 5) * dimless_bhmass ** (1 / 5) * (dimless_accrate / 0.1) ** (7 / 10)
    )
    standard_arealdensity_test *= dimless_radius ** (-3 / 4) * (1 - np.sqrt(3 / dimless_radius)) ** (7 / 10)
    return standard_arealdensity_test


def get_standard_density_test(
    *,
    alpha_viscosity,
    dimless_bhmass,
    dimless_accrate,
    dimless_radius,
) -> float | np.ndarray:
    alpha_viscosity, dimless_bhmass = np.asarray(alpha_viscosity), np.asarray(dimless_bhmass)
    dimless_accrate, dimless_radius = np.asarray(dimless_accrate), np.asarray(dimless_radius)
    standard_density_test = (
        4.7 * alpha_viscosity ** (-7 / 10) * dimless_bhmass ** (-7 / 10) * (dimless_accrate / 0.1) ** (11 / 20)
    )
    standard_density_test *= dimless_radius ** (-15 / 8) * (1 - np.sqrt(3 / dimless_radius)) ** (11 / 20)
    return standard_density_test


def get_standard_radvel_test(
    *,
    alpha_viscosity,
    dimless_bhmass,
    dimless_accrate,
    dimless_radius,
) -> float | np.ndarray:
    alpha_viscosity, dimless_bhmass = np.asarray(alpha_viscosity), np.asarray(dimless_bhmass)
    dimless_accrate, dimless_radius = np.asarray(dimless_accrate), np.asarray(dimless_radius)
    standard_radvel_test = (
        -5.4e5 * alpha_viscosity ** (4 / 5) * dimless_bhmass ** (-1 / 5) * (dimless_accrate / 0.1) ** (3 / 10)
    )
    standard_radvel_test *= dimless_radius ** (-1 / 4) * (1 - np.sqrt(3 / dimless_radius)) ** (-7 / 10)
    return standard_radvel_test


def get_standard_temperature_test(
    *,
    alpha_viscosity,
    dimless_bhmass,
    dimless_accrate,
    dimless_radius,
) -> float | np.ndarray:
    alpha_viscosity, dimless_bhmass = np.asarray(alpha_viscosity), np.asarray(dimless_bhmass)
    dimless_accrate, dimless_radius = np.asarray(dimless_accrate), np.asarray(dimless_radius)
    standard_temperature_test = (
        6.9e7 * alpha_viscosity ** (-1 / 5) * dimless_bhmass ** (-1 / 5) * (dimless_accrate / 0.1) ** (3 / 10)
    )
    standard_temperature_test *= dimless_radius ** (-3 / 4) * (1 - np.sqrt(3 / dimless_radius)) ** (3 / 10)
    return standard_temperature_test


def get_standard_opticaldepth_test(
    *,
    alpha_viscosity,
    dimless_bhmass,
    dimless_accrate,
    dimless_radius,
) -> float | np.ndarray:
    alpha_viscosity, dimless_bhmass = np.asarray(alpha_viscosity), np.asarray(dimless_bhmass)
    dimless_accrate, dimless_radius = np.asarray(dimless_accrate), np.asarray(dimless_radius)
    standard_opticaldepth_test = (
        79 * alpha_viscosity ** (-4 / 5) * dimless_bhmass ** (1 / 5) * (dimless_accrate / 0.1) ** (1 / 5)
    )
    standard_opticaldepth_test *= (1 - np.sqrt(3 / dimless_radius)) ** (1 / 5)
    return standard_opticaldepth_test


@solve_tools.vectorize_functions
def standard_disk_solver(*, alpha_viscosity, dimless_bhmass, dimless_accrate, dimless_radius) -> np.ndarray:
    halfheight_test = get_standard_halfheight_test(
        alpha_viscosity=alpha_viscosity,
        dimless_bhmass=dimless_bhmass,
        dimless_accrate=dimless_accrate,
        dimless_radius=dimless_radius,
    )
    arealdensity_test = get_standard_arealdensity_test(
        alpha_viscosity=alpha_viscosity,
        dimless_bhmass=dimless_bhmass,
        dimless_accrate=dimless_accrate,
        dimless_radius=dimless_radius,
    )

    density_test = get_standard_density_test(
        alpha_viscosity=alpha_viscosity,
        dimless_bhmass=dimless_bhmass,
        dimless_accrate=dimless_accrate,
        dimless_radius=dimless_radius,
    )

    radvel_test = get_standard_radvel_test(
        alpha_viscosity=alpha_viscosity,
        dimless_bhmass=dimless_bhmass,
        dimless_accrate=dimless_accrate,
        dimless_radius=dimless_radius,
    )

    temperature_test = get_standard_temperature_test(
        alpha_viscosity=alpha_viscosity,
        dimless_bhmass=dimless_bhmass,
        dimless_accrate=dimless_accrate,
        dimless_radius=dimless_radius,
    )

    opticaldepth_test = get_standard_opticaldepth_test(
        alpha_viscosity=alpha_viscosity,
        dimless_bhmass=dimless_bhmass,
        dimless_accrate=dimless_accrate,
        dimless_radius=dimless_radius,
    )

    guess_solution = np.array([
        halfheight_test,
        arealdensity_test,
        density_test,
        radvel_test,
        temperature_test,
        opticaldepth_test,
    ])

    def get_standard_equations(x) -> np.ndarray:
        halfheight, arealdensity, density, radvel, temperature, opticaldepth = x
        radius = solve_tools.get_radius_fromdimless(dimless_bhmass=dimless_bhmass, dimless_radius=dimless_radius)
        accrate = solve_tools.get_accrate_fromdimless(dimless_bhmass=dimless_bhmass, dimless_accrate=dimless_accrate)
        angvel = get_standard_angvel(dimless_bhmass=dimless_bhmass, dimless_radius=dimless_radius)
        pressure = get_standard_pressure(standard_density=density, standard_temperature=temperature)
        soundvel = get_standard_soundvel(standard_pressure=pressure, standard_density=density)
        averopacity = get_standard_averopacity(standard_density=density, standard_temperature=temperature)
        kineviscocity = get_standard_kineviscocity(
            standard_arealdensity=arealdensity,
            dimless_bhmass=dimless_bhmass,
            dimless_accrate=dimless_accrate,
            dimless_radius=dimless_radius,
        )
        eq_1 = accrate + 2 * math.pi * radius * radvel * arealdensity
        eq_2 = halfheight - soundvel / angvel
        eq_3 = (
            9 * kineviscocity * arealdensity * angvel**2 / 4
            - 8 * cgs_consts.cgs_a * cgs_consts.cgs_c * temperature**4 / 3 / opticaldepth
        )
        eq_4 = arealdensity - 2 * density * halfheight
        eq_5 = opticaldepth - 0.5 * averopacity * arealdensity
        eq_6 = 1.5 * density * kineviscocity * angvel - alpha_viscosity * pressure

        return np.array([eq_1, eq_2, eq_3, eq_4, eq_5, eq_6])

    standard_solve = sp.optimize.root(get_standard_equations, guess_solution)
    if standard_solve.success:
        return np.array(standard_solve.x)
    else:
        raise ValueError(f"Did not converge: {standard_solve.message}")


def get_standard_halfheight_result(
    *,
    alpha_viscosity,
    dimless_bhmass,
    dimless_accrate,
    dimless_radius,
) -> float | np.ndarray:
    alpha_viscosity, dimless_bhmass = np.asarray(alpha_viscosity), np.asarray(dimless_bhmass)
    dimless_accrate, dimless_radius = np.asarray(dimless_accrate), np.asarray(dimless_radius)
    halfheight_result = standard_disk_solver(
        alpha_viscosity=alpha_viscosity,
        dimless_bhmass=dimless_bhmass,
        dimless_accrate=dimless_accrate,
        dimless_radius=dimless_radius,
    )[0]
    return halfheight_result


def get_standard_arealdensity_result(
    *,
    alpha_viscosity,
    dimless_bhmass,
    dimless_accrate,
    dimless_radius,
) -> float | np.ndarray:
    alpha_viscosity, dimless_bhmass = np.asarray(alpha_viscosity), np.asarray(dimless_bhmass)
    dimless_accrate, dimless_radius = np.asarray(dimless_accrate), np.asarray(dimless_radius)
    arealdensity_result = standard_disk_solver(
        alpha_viscosity=alpha_viscosity,
        dimless_bhmass=dimless_bhmass,
        dimless_accrate=dimless_accrate,
        dimless_radius=dimless_radius,
    )[1]
    return arealdensity_result


def get_standard_density_result(
    *,
    alpha_viscosity,
    dimless_bhmass,
    dimless_accrate,
    dimless_radius,
) -> float | np.ndarray:
    alpha_viscosity, dimless_bhmass = np.asarray(alpha_viscosity), np.asarray(dimless_bhmass)
    dimless_accrate, dimless_radius = np.asarray(dimless_accrate), np.asarray(dimless_radius)
    density_result = standard_disk_solver(
        alpha_viscosity=alpha_viscosity,
        dimless_bhmass=dimless_bhmass,
        dimless_accrate=dimless_accrate,
        dimless_radius=dimless_radius,
    )[2]
    return density_result


def get_standard_radvel_result(
    *,
    alpha_viscosity,
    dimless_bhmass,
    dimless_accrate,
    dimless_radius,
) -> float | np.ndarray:
    alpha_viscosity, dimless_bhmass = np.asarray(alpha_viscosity), np.asarray(dimless_bhmass)
    dimless_accrate, dimless_radius = np.asarray(dimless_accrate), np.asarray(dimless_radius)
    radvel_result = standard_disk_solver(
        alpha_viscosity=alpha_viscosity,
        dimless_bhmass=dimless_bhmass,
        dimless_accrate=dimless_accrate,
        dimless_radius=dimless_radius,
    )[3]
    return radvel_result


def get_standard_temperature_result(
    *,
    alpha_viscosity,
    dimless_bhmass,
    dimless_accrate,
    dimless_radius,
) -> float | np.ndarray:
    alpha_viscosity, dimless_bhmass = np.asarray(alpha_viscosity), np.asarray(dimless_bhmass)
    dimless_accrate, dimless_radius = np.asarray(dimless_accrate), np.asarray(dimless_radius)
    temperature_result = standard_disk_solver(
        alpha_viscosity=alpha_viscosity,
        dimless_bhmass=dimless_bhmass,
        dimless_accrate=dimless_accrate,
        dimless_radius=dimless_radius,
    )[4]
    return temperature_result


def get_standard_opticaldepth_result(
    *,
    alpha_viscosity,
    dimless_bhmass,
    dimless_accrate,
    dimless_radius,
) -> float | np.ndarray:
    alpha_viscosity, dimless_bhmass = np.asarray(alpha_viscosity), np.asarray(dimless_bhmass)
    dimless_accrate, dimless_radius = np.asarray(dimless_accrate), np.asarray(dimless_radius)
    opticaldepth_result = standard_disk_solver(
        alpha_viscosity=alpha_viscosity,
        dimless_bhmass=dimless_bhmass,
        dimless_accrate=dimless_accrate,
        dimless_radius=dimless_radius,
    )[4]
    return opticaldepth_result


def get_standard_angmom_result(*, dimless_bhmass, dimless_radius) -> float | np.ndarray:
    dimless_bhmass = np.asarray(dimless_bhmass)
    dimless_radius = np.asarray(dimless_radius)
    radius = solve_tools.get_radius_fromdimless(dimless_bhmass=dimless_bhmass, dimless_radius=dimless_radius)
    angvel = get_standard_angvel(dimless_bhmass=dimless_bhmass, dimless_radius=dimless_radius)
    angmom_result = angvel * radius**2
    return angmom_result


def get_standard_coff_eta_result(
    *,
    alpha_viscosity,
    dimless_bhmass,
    dimless_accrate,
    dimless_radius,
) -> float | np.ndarray:
    alpha_viscosity, dimless_bhmass = np.asarray(alpha_viscosity), np.asarray(dimless_bhmass)
    dimless_accrate, dimless_radius = np.asarray(dimless_accrate), np.asarray(dimless_radius)
    angvel = get_standard_angvel(dimless_bhmass=dimless_bhmass, dimless_radius=dimless_radius)
    radvel = get_standard_radvel_result(
        alpha_viscosity=alpha_viscosity,
        dimless_bhmass=dimless_bhmass,
        dimless_accrate=dimless_accrate,
        dimless_radius=dimless_radius,
    )
    halfheight = get_standard_halfheight_result(
        alpha_viscosity=alpha_viscosity,
        dimless_bhmass=dimless_bhmass,
        dimless_accrate=dimless_accrate,
        dimless_radius=dimless_radius,
    )
    soundvel = halfheight * angvel
    coff_eta_result = (radvel / soundvel) ** 2
    return coff_eta_result


if __name__ == "__main__":
    alpha_viscosity = model_params.alpha_viscosity[0]
    dimless_accrate = model_params.dimless_accrate[0]
    dimless_bhmass = model_params.dimless_bhmass[0]
    gas_index = model_params.gas_index[0]
    wind_index = model_params.wind_index[0]
    dimless_radius_in = model_params.dimless_radius_in[0]
    dimless_radius_out = model_params.dimless_radius_out[0]
    dimless_radius = dimless_radius_out
