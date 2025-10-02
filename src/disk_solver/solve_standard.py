import math

import numpy as np
import scipy as sp

from disk_solver.parameter_init import DiskParams, cgs_consts
from disk_solver.solve_tools import DiskTools
from parameters import model_params


class StandardDisk:
    @staticmethod
    def get_standard_angvel(*, par: DiskParams, dimless_radius) -> float | np.ndarray:
        dimless_radius = np.asarray(dimless_radius)
        bhmass = DiskTools.get_bhmass(par=par)
        radius = DiskTools.get_radius_fromdimless(par=par, dimless_radius=dimless_radius)
        standard_angvel = np.sqrt(cgs_consts.cgs_gra * bhmass / radius / radius / radius)
        return standard_angvel

    @staticmethod
    def get_standard_pressure(*, standard_density, standard_temperature) -> float | np.ndarray:
        standard_density, standard_temperature = np.asarray(standard_density), np.asarray(standard_temperature)
        part_1 = 2 * standard_density * cgs_consts.cgs_kb * standard_temperature / cgs_consts.cgs_mh
        part_2 = cgs_consts.cgs_a * standard_temperature**4 / 3
        standard_pressure = part_1 + part_2
        return standard_pressure

    @staticmethod
    def get_standard_soundvel(*, standard_pressure, standard_density) -> float | np.ndarray:
        standard_pressure, standard_density = np.asarray(standard_pressure), np.asarray(standard_density)
        standard_soundvel = np.sqrt(standard_pressure / standard_density)
        return standard_soundvel

    @staticmethod
    def get_standard_averopacity(*, standard_density, standard_temperature) -> float | np.ndarray:
        standard_density, standard_temperature = np.asarray(standard_density), np.asarray(standard_temperature)
        standard_averopacity = cgs_consts.cgs_kes + 6.4e22 * standard_density * standard_temperature ** (-3.5)
        return standard_averopacity

    @staticmethod
    def get_standard_kineviscocity(*, par: DiskParams, standard_arealdensity, dimless_radius) -> float | np.ndarray:
        standard_arealdensity = np.asanyarray(standard_arealdensity)
        dimless_radius = np.asarray(dimless_radius)
        accrate = DiskTools.get_accrate_fromdimless(par=par)
        part_1 = accrate / 3 / math.pi / standard_arealdensity
        part_2 = 1 - np.sqrt(3 / dimless_radius)
        standard_kineviscocity = part_1 * part_2
        return standard_kineviscocity

    @staticmethod
    def get_standard_halfheight_test(*, par: DiskParams, dimless_radius) -> float | np.ndarray:
        dimless_radius = np.asarray(dimless_radius)
        criterion_1 = (par.alpha_viscosity * par.dimless_bhmass) ** (-1 / 8) / 170
        criterion_2 = (
            150 * (par.alpha_viscosity * par.dimless_bhmass) ** (2 / 21) * (par.dimless_accrate / 0.1) ** (16 / 21)
        )
        criterion_3 = 6.3e3 * (par.dimless_accrate / 0.1) ** (2 / 3)
        if (par.dimless_accrate / 0.1) >= criterion_1 and dimless_radius < criterion_2:
            standard_halfheight_test = (
                5.5e4 * par.dimless_bhmass * par.dimless_accrate * (1 - np.sqrt(3 / dimless_radius))
            )
        elif dimless_radius < criterion_3:
            standard_halfheight_test = (
                2.7e3
                * par.alpha_viscosity ** (-1 / 10)
                * par.dimless_bhmass ** (9 / 10)
                * (par.dimless_accrate / 0.1) ** (1 / 5)
            )
            standard_halfheight_test *= dimless_radius ** (21 / 20) * (1 - np.sqrt(3 / dimless_radius)) ** (1 / 5)
        else:
            standard_halfheight_test = (
                1.5e3
                * par.alpha_viscosity ** (-1 / 10)
                * par.dimless_bhmass ** (9 / 10)
                * (par.dimless_accrate / 0.1) ** (3 / 20)
            )
            standard_halfheight_test *= dimless_radius ** (9 / 8) * (1 - np.sqrt(3 / dimless_radius)) ** (3 / 20)
        return standard_halfheight_test

    @staticmethod
    def get_standard_arealdensity_test(*, par: DiskParams, dimless_radius) -> float | np.ndarray:
        dimless_radius = np.asarray(dimless_radius)
        criterion_1 = (par.alpha_viscosity * par.dimless_bhmass) ** (-1 / 8) / 170
        criterion_2 = (
            150 * (par.alpha_viscosity * par.dimless_bhmass) ** (2 / 21) * (par.dimless_accrate / 0.1) ** (16 / 21)
        )
        criterion_3 = 6.3e3 * (par.dimless_accrate / 0.1) ** (2 / 3)
        if (par.dimless_accrate / 0.1) >= criterion_1 and dimless_radius < criterion_2:
            standard_arealdensity_test = (
                100
                * (par.alpha_viscosity) ** (-1)
                * (par.dimless_accrate / 0.1) ** (-1)
                * dimless_radius ** (3 / 2)
                * (1 - np.sqrt(3 / dimless_radius)) ** (-1)
            )
        elif dimless_radius < criterion_3:
            standard_arealdensity_test = (
                4.3e4
                * par.alpha_viscosity ** (-4 / 5)
                * par.dimless_bhmass ** (1 / 5)
                * (par.dimless_accrate / 0.1) ** (3 / 5)
            )
            standard_arealdensity_test *= dimless_radius ** (-3 / 5) * (1 - np.sqrt(3 / dimless_radius)) ** (3 / 5)
        else:
            standard_arealdensity_test = (
                1.4e5
                * par.alpha_viscosity ** (-4 / 5)
                * par.dimless_bhmass ** (1 / 5)
                * (par.dimless_accrate / 0.1) ** (7 / 10)
            )
            standard_arealdensity_test *= dimless_radius ** (-3 / 4) * (1 - np.sqrt(3 / dimless_radius)) ** (7 / 10)
        return standard_arealdensity_test

    @staticmethod
    def get_standard_density_test(*, par: DiskParams, dimless_radius) -> float | np.ndarray:
        dimless_radius = np.asarray(dimless_radius)
        criterion_1 = (par.alpha_viscosity * par.dimless_bhmass) ** (-1 / 8) / 170
        criterion_2 = (
            150 * (par.alpha_viscosity * par.dimless_bhmass) ** (2 / 21) * (par.dimless_accrate / 0.1) ** (16 / 21)
        )
        criterion_3 = 6.3e3 * (par.dimless_accrate / 0.1) ** (2 / 3)
        if (par.dimless_accrate / 0.1) >= criterion_1 and dimless_radius < criterion_2:
            standard_density_test = (
                9e-4
                * (par.alpha_viscosity) ** (-1)
                * (par.dimless_bhmass) ** (-1)
                * (par.dimless_accrate / 0.1) ** (-2)
            )
            standard_density_test *= dimless_radius ** (3 / 2) * (1 - np.sqrt(3 / dimless_radius)) ** (-2)
        elif dimless_radius < criterion_3:
            standard_density_test = (
                8
                * par.alpha_viscosity ** (-7 / 10)
                * par.dimless_bhmass ** (-7 / 10)
                * (par.dimless_accrate / 0.1) ** (2 / 5)
            )
            standard_density_test *= dimless_radius ** (-33 / 20) * (1 - np.sqrt(3 / dimless_radius)) ** (2 / 5)
        else:
            standard_density_test = (
                4.7
                * par.alpha_viscosity ** (-7 / 10)
                * par.dimless_bhmass ** (-7 / 10)
                * (par.dimless_accrate / 0.1) ** (11 / 20)
            )
            standard_density_test *= dimless_radius ** (-15 / 8) * (1 - np.sqrt(3 / dimless_radius)) ** (11 / 20)
        return standard_density_test

    @staticmethod
    def get_standard_radvel_test(*, par: DiskParams, dimless_radius) -> float | np.ndarray:
        dimless_radius = np.asarray(dimless_radius)
        criterion_1 = (par.alpha_viscosity * par.dimless_bhmass) ** (-1 / 8) / 170
        criterion_2 = (
            150 * (par.alpha_viscosity * par.dimless_bhmass) ** (2 / 21) * (par.dimless_accrate / 0.1) ** (16 / 21)
        )
        criterion_3 = 6.3e3 * (par.dimless_accrate / 0.1) ** (2 / 3)
        if (par.dimless_accrate / 0.1) >= criterion_1 and dimless_radius < criterion_2:
            standard_radvel_test = (
                -7.6e8
                * par.alpha_viscosity
                * (par.dimless_accrate / 0.1) ** 2
                * (dimless_radius) ** (-5 / 2)
                * (1 - np.sqrt(3 / dimless_radius))
            )
        elif dimless_radius < criterion_3:
            standard_radvel_test = (
                -1.7e6
                * par.alpha_viscosity ** (4 / 5)
                * par.dimless_bhmass ** (-1 / 5)
                * (par.dimless_accrate / 0.1) ** (2 / 5)
            )
            standard_radvel_test *= dimless_radius ** (-2 / 5) * (1 - np.sqrt(3 / dimless_radius)) ** (-3 / 5)
        else:
            standard_radvel_test = (
                -5.4e5
                * par.alpha_viscosity ** (4 / 5)
                * par.dimless_bhmass ** (-1 / 5)
                * (par.dimless_accrate / 0.1) ** (3 / 10)
            )
            standard_radvel_test *= dimless_radius ** (-1 / 4) * (1 - np.sqrt(3 / dimless_radius)) ** (-7 / 10)
        return standard_radvel_test

    @staticmethod
    def get_standard_temperature_test(*, par: DiskParams, dimless_radius) -> float | np.ndarray:
        dimless_radius = np.asarray(dimless_radius)
        criterion_1 = (par.alpha_viscosity * par.dimless_bhmass) ** (-1 / 8) / 170
        criterion_2 = (
            150 * (par.alpha_viscosity * par.dimless_bhmass) ** (2 / 21) * (par.dimless_accrate / 0.1) ** (16 / 21)
        )
        criterion_3 = 6.3e3 * (par.dimless_accrate / 0.1) ** (2 / 3)
        if (par.dimless_accrate / 0.1) >= criterion_1 and dimless_radius < criterion_2:
            standard_temperature_test = (
                4.9e7 * (par.alpha_viscosity * par.dimless_bhmass) ** (-1 / 4) * dimless_radius ** (-3 / 8)
            )
        elif dimless_radius < criterion_3:
            standard_temperature_test = (
                2.2e8
                * par.alpha_viscosity ** (-1 / 5)
                * par.dimless_bhmass ** (-1 / 5)
                * (par.dimless_accrate / 0.1) ** (2 / 5)
            )
            standard_temperature_test *= dimless_radius ** (-9 / 10) * (1 - np.sqrt(3 / dimless_radius)) ** (2 / 5)
        else:
            standard_temperature_test = (
                6.9e7
                * par.alpha_viscosity ** (-1 / 5)
                * par.dimless_bhmass ** (-1 / 5)
                * (par.dimless_accrate / 0.1) ** (3 / 10)
            )
            standard_temperature_test *= dimless_radius ** (-3 / 4) * (1 - np.sqrt(3 / dimless_radius)) ** (3 / 10)
        return standard_temperature_test

    @staticmethod
    def get_standard_opticaldepth_test(*, par: DiskParams, dimless_radius) -> float | np.ndarray:
        dimless_radius = np.asarray(dimless_radius)
        criterion_1 = (par.alpha_viscosity * par.dimless_bhmass) ** (-1 / 8) / 170
        criterion_2 = (
            150 * (par.alpha_viscosity * par.dimless_bhmass) ** (2 / 21) * (par.dimless_accrate / 0.1) ** (16 / 21)
        )
        criterion_3 = 6.3e3 * (par.dimless_accrate / 0.1) ** (2 / 3)
        if (par.dimless_accrate / 0.1) >= criterion_1 and dimless_radius < criterion_2:
            standard_opticaldepth_test = (
                8.4e-3
                * par.alpha_viscosity ** (-17 / 16)
                * par.dimless_bhmass ** (-1 / 16)
                * (par.dimless_accrate / 0.1) ** (-2)
            )
            standard_opticaldepth_test *= dimless_radius ** (93 / 32) * (1 - np.sqrt(3 / dimless_radius)) ** (-2)
        elif dimless_radius < criterion_3:
            standard_opticaldepth_test = (
                24
                * par.alpha_viscosity ** (-4 / 5)
                * par.dimless_bhmass ** (1 / 5)
                * (par.dimless_accrate / 0.1) ** (1 / 10)
            )
            standard_opticaldepth_test *= dimless_radius ** (3 / 20) * (1 - np.sqrt(3 / dimless_radius)) ** (1 / 10)
        else:
            standard_opticaldepth_test = (
                79
                * par.alpha_viscosity ** (-4 / 5)
                * par.dimless_bhmass ** (1 / 5)
                * (par.dimless_accrate / 0.1) ** (1 / 5)
            )
            standard_opticaldepth_test *= (1 - np.sqrt(3 / dimless_radius)) ** (1 / 5)
        return standard_opticaldepth_test

    @staticmethod
    def standard_disk_solver(*, par: DiskParams, dimless_radius: float) -> np.ndarray:
        halfheight_test = StandardDisk.get_standard_halfheight_test(par=par, dimless_radius=dimless_radius)
        arealdensity_test = StandardDisk.get_standard_arealdensity_test(par=par, dimless_radius=dimless_radius)
        density_test = StandardDisk.get_standard_density_test(par=par, dimless_radius=dimless_radius)
        radvel_test = StandardDisk.get_standard_radvel_test(par=par, dimless_radius=dimless_radius)
        temperature_test = StandardDisk.get_standard_temperature_test(par=par, dimless_radius=dimless_radius)
        opticaldepth_test = StandardDisk.get_standard_opticaldepth_test(par=par, dimless_radius=dimless_radius)
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
            radius = DiskTools.get_radius_fromdimless(par=par, dimless_radius=dimless_radius)
            accrate = DiskTools.get_accrate_fromdimless(par=par)
            angvel = StandardDisk.get_standard_angvel(par=par, dimless_radius=dimless_radius)
            pressure = StandardDisk.get_standard_pressure(standard_density=density, standard_temperature=temperature)
            soundvel = StandardDisk.get_standard_soundvel(standard_pressure=pressure, standard_density=density)
            averopacity = StandardDisk.get_standard_averopacity(
                standard_density=density,
                standard_temperature=temperature,
            )
            kineviscocity = StandardDisk.get_standard_kineviscocity(
                par=par,
                standard_arealdensity=arealdensity,
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
            eq_6 = 1.5 * density * kineviscocity * angvel - par.alpha_viscosity * pressure

            return np.array([eq_1, eq_2, eq_3, eq_4, eq_5, eq_6])

        standard_solve = sp.optimize.root(get_standard_equations, guess_solution, method="lm")
        if standard_solve.success:
            return np.array(standard_solve.x)
        else:
            raise ValueError(f"Did not converge: {standard_solve.message}")

    @staticmethod
    def get_standard_solve_result(
        *,
        par: DiskParams,
        dimless_radius: float | np.ndarray,
    ) -> np.ndarray:
        dimless_radius = np.atleast_1d(dimless_radius)
        result_array = np.zeros(
            len(dimless_radius),
            dtype=[
                ("halfheight", float),
                ("arealdensity", float),
                ("density", float),
                ("radvel", float),
                ("temperature", float),
                ("opticaldepth", float),
                ("angmom", float),
                ("coff_eta", float),
            ],
        )
        for i, r in enumerate(dimless_radius):
            solve = StandardDisk.standard_disk_solver(par=par, dimless_radius=r)
            halfheight, arealdensity, density, radvel, temperature, opticaldepth = solve
            angvel = float(StandardDisk.get_standard_angvel(par=par, dimless_radius=r))
            radius = float(DiskTools.get_radius_fromdimless(par=par, dimless_radius=r))
            angmom = angvel * radius**2
            coff_eta = (radvel / (halfheight * angvel)) ** 2
            result_array[i] = (
                halfheight,
                arealdensity,
                density,
                radvel,
                temperature,
                opticaldepth,
                angmom,
                coff_eta,
            )
        if result_array.size == 1:
            return result_array[0]
        return result_array


__all__ = ["StandardDisk"]
