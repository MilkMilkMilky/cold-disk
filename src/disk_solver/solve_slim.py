import math
import warnings

import numpy as np
import scipy as sp
from numpy._typing._array_like import NDArray

from disk_solver.parameter_init import DiskParams, cgs_consts
from disk_solver.solve_standard import StandardDisk
from disk_solver.solve_tools import DiskTools


class SlimDisk:
    @staticmethod
    def get_slim_angvelk(*, par: DiskParams, radius) -> float | np.ndarray:
        radius = np.asarray(radius)
        bhmass = DiskTools.get_bhmass(par=par)
        radius_sch = DiskTools.get_radius_sch(par=par)
        slim_angvelk = np.sqrt(cgs_consts.cgs_gra * bhmass / radius) / (radius - radius_sch)
        return slim_angvelk

    @staticmethod
    def get_slim_angmomk(*, par: DiskParams, radius) -> float | np.ndarray:
        radius = np.asarray(radius)
        angvelk = SlimDisk.get_slim_angvelk(par=par, radius=radius)
        slim_angmomk = angvelk * radius * radius
        return slim_angmomk

    @staticmethod
    def get_slim_accrate(*, par: DiskParams, radius) -> float | np.ndarray:
        radius = np.asarray(radius)
        radius_sch = DiskTools.get_radius_sch(par=par)
        dimless_radius = radius / radius_sch
        accrate_out = DiskTools.get_accrate_fromdimless(par=par, dimless_accrate=par.dimless_accrate)
        slim_accrate = accrate_out * np.float_power((dimless_radius / par.dimless_radius_out), par.wind_index)
        return slim_accrate

    @staticmethod
    def get_slim_dlnaccrate_dradius(*, par: DiskParams, radius) -> float | np.ndarray:
        radius = np.asarray(radius)
        slim_dlnaccrate_dradius = par.wind_index / radius
        return slim_dlnaccrate_dradius

    @staticmethod
    def get_slim_dlnangvelk_dradius(*, par: DiskParams, radius) -> float | np.ndarray:
        radius = np.asarray(radius)
        radius_sch = DiskTools.get_radius_sch(par=par)
        slim_dlnangvelk_dradius = -1 / 2 / radius - 1 / (radius - radius_sch)
        return slim_dlnangvelk_dradius

    @staticmethod
    def get_slim_angmomin(*, par: DiskParams, dimless_angmomin: float) -> float:
        radius_sch = DiskTools.get_radius_sch(par=par)
        slim_angmomin = dimless_angmomin * radius_sch * cgs_consts.cgs_c
        return slim_angmomin

    @staticmethod
    def get_slim_arealpressure(*, par: DiskParams, radius, angmom, angmomin) -> float | np.ndarray:
        radius, angmom = np.asarray(radius), np.asarray(angmom)
        slim_accrate = SlimDisk.get_slim_accrate(par=par, radius=radius)
        slim_arealpressure = slim_accrate * (angmom - angmomin) / (2 * math.pi * par.alpha_viscosity * radius * radius)
        return slim_arealpressure

    @staticmethod
    def get_slim_arealdensity(*, par: DiskParams, arealpressure, coff_eta, radius) -> float | np.ndarray:
        arealpressure = np.asarray(arealpressure)
        coff_eta, radius = np.asarray(coff_eta), np.asarray(radius)
        slim_accrate = SlimDisk.get_slim_accrate(par=par, radius=radius)
        slim_arealdensity = slim_accrate * slim_accrate / (4 * math.pi**2 * radius * radius * arealpressure * coff_eta)
        return slim_arealdensity

    @staticmethod
    def get_slim_halfheight(*, par: DiskParams, radius, arealpressure, arealdensity) -> float | np.ndarray:
        radius = np.asarray(radius)
        arealpressure, arealdensity = np.asarray(arealpressure), np.asarray(arealdensity)
        angvelk = SlimDisk.get_slim_angvelk(par=par, radius=radius)
        coff_k = np.sqrt(arealpressure / arealdensity)
        slim_halfheight = np.sqrt(2 * par.gas_index + 3) * coff_k / angvelk
        return slim_halfheight

    @staticmethod
    def get_slim_pressure(*, par: DiskParams, arealpressure, halfheight) -> float | np.ndarray:
        arealpressure, halfheight = np.asarray(arealpressure), np.asarray(halfheight)
        coff_in1 = DiskTools.get_coeff_in(index=par.gas_index + 1)
        slim_pressure = arealpressure / 2 / halfheight / coff_in1
        return slim_pressure

    @staticmethod
    def get_slim_density(*, par: DiskParams, arealdensity, halfheight) -> float | np.ndarray:
        arealdensity, halfheight = np.asarray(arealdensity), np.asarray(halfheight)
        coff_in = DiskTools.get_coeff_in(index=par.gas_index)
        slim_density = arealdensity / 2 / halfheight / coff_in
        return slim_density

    @staticmethod
    def get_slim_temperature(*, pressure: float, density: float) -> float:
        coff_b = cgs_consts.cgs_rg * density / cgs_consts.cgs_amm
        coff_c = -pressure

        def slim_temperature_func(t):
            return (cgs_consts.cgs_a / 3) * t**4 + coff_b * t + coff_c

        try:
            temperature = sp.optimize.root_scalar(
                slim_temperature_func,
                bracket=[1e-10, 1e8],
                method="brentq",
            )
            if temperature.converged and temperature.root > 0:
                slim_temperature = temperature.root
                return slim_temperature
        except Exception as e:
            warnings.warn(
                f"root_scalar failed with exception: {e}. Falling back to fsolve.",
                stacklevel=2,
            )
        temperature_array = sp.optimize.fsolve(slim_temperature_func, 10000)
        for temp in temperature_array:
            if temp > 0:
                slim_temperature = temp
                return slim_temperature
        raise RuntimeError(f"Failed to solve slim temperature. pressure={pressure}, density={density}")

    @staticmethod
    def get_slim_opacity(*, par: DiskParams, density, temperature) -> float | np.ndarray:
        density, temperature = np.asarray(density), np.asarray(temperature)
        coff_in = DiskTools.get_coeff_in(index=par.gas_index)
        opacity_abs = cgs_consts.cgs_kra * (coff_in * density)
        opacity_abs *= np.float_power((2 * temperature / 3), (-3.5))
        slim_opacity = cgs_consts.cgs_kes + opacity_abs
        return slim_opacity

    @staticmethod
    def get_slim_pressure_ratio(*, density, temperature, pressure) -> float | np.ndarray:
        pressure, density = np.asarray(pressure), np.asarray(density)
        temperature = np.asarray(temperature)
        slim_pressure_ratio = cgs_consts.cgs_rg / cgs_consts.cgs_amm * density * temperature / pressure
        return slim_pressure_ratio

    @staticmethod
    def get_slim_chandindex_1(*, pressure_ratio) -> float | np.ndarray:
        pressure_ratio = np.asarray(pressure_ratio)
        slim_chandindex_1 = (32 - 24 * pressure_ratio - 3 * pressure_ratio**2) / (24 - 21 * pressure_ratio)
        return slim_chandindex_1

    @staticmethod
    def get_slim_chandindex_3(*, pressure_ratio) -> float | np.ndarray:
        pressure_ratio = np.asarray(pressure_ratio)
        slim_chandindex_3 = (32 - 27 * pressure_ratio) / (24 - 21 * pressure_ratio)
        return slim_chandindex_3

    @staticmethod
    def get_slim_fluxz(*, density, halfheight, opacity, temperature) -> float | np.ndarray:
        density, halfheight = np.asarray(density), np.asarray(halfheight)
        opacity, temperature = np.asarray(opacity), np.asarray(temperature)
        slim_fluxz = 4 * cgs_consts.cgs_a * cgs_consts.cgs_c * np.float_power(temperature, 4)
        slim_fluxz /= 3 * opacity * density * halfheight
        return slim_fluxz

    @staticmethod
    def get_slim_apresstoadens(arealpressure, arealdensity) -> float | np.ndarray:
        arealpressure, arealdensity = np.asarray(arealpressure), np.asarray(arealdensity)
        slim_apresstoadens = arealpressure / arealdensity
        return slim_apresstoadens

    @staticmethod
    def get_slim_radvel(*, par: DiskParams, radius, arealdensity) -> float | np.ndarray:
        radius, arealdensity = np.asarray(radius), np.asarray(arealdensity)
        accrate = SlimDisk.get_slim_accrate(par=par, radius=radius)
        slim_radvel = -accrate / 2 / math.pi / radius / arealdensity
        return slim_radvel

    @staticmethod
    def get_slim_soundvel(*, pressure, density) -> float | np.ndarray:
        pressure, density = np.asarray(pressure), np.asarray(density)
        slim_soundvel = np.sqrt(pressure / density)
        return slim_soundvel

    @staticmethod
    def get_slim_rveltosvel(*, radvel, soundvel) -> float | np.ndarray:
        radvel, soundvel = np.asarray(radvel), np.asarray(soundvel)
        slim_rveltosvel = np.abs(radvel / soundvel)
        return slim_rveltosvel

    @staticmethod
    def get_slim_rveltosvel_fromfirst(
        *,
        par: DiskParams,
        dimless_radius,
        angmom,
        coff_eta,
        angmomin,
    ) -> float | np.ndarray:
        dimless_radius = np.asarray(dimless_radius)
        radius = DiskTools.get_radius_fromdimless(par=par, dimless_radius=dimless_radius)
        arealpressure = SlimDisk.get_slim_arealpressure(par=par, radius=radius, angmom=angmom, angmomin=angmomin)
        arealdensity = SlimDisk.get_slim_arealdensity(
            par=par,
            arealpressure=arealpressure,
            coff_eta=coff_eta,
            radius=radius,
        )
        halfheight = SlimDisk.get_slim_halfheight(
            par=par,
            radius=radius,
            arealpressure=arealpressure,
            arealdensity=arealdensity,
        )
        pressure = SlimDisk.get_slim_pressure(par=par, arealpressure=arealpressure, halfheight=halfheight)
        density = SlimDisk.get_slim_density(par=par, arealdensity=arealdensity, halfheight=halfheight)
        radvel = SlimDisk.get_slim_radvel(par=par, radius=radius, arealdensity=arealdensity)
        soundvel = SlimDisk.get_slim_soundvel(pressure=pressure, density=density)
        rveltosvel = np.abs(radvel / soundvel)
        return rveltosvel

    @staticmethod
    def get_slim_temperature_eff(*, fluxz) -> float | np.ndarray:
        fluxz = np.asarray(fluxz)
        slim_temperature_eff = np.float_power((fluxz / cgs_consts.cgs_sb), 0.25)
        return slim_temperature_eff

    @staticmethod
    def get_slim_initvalue(*, par: DiskParams) -> np.ndarray:
        standard_result = StandardDisk.get_standard_solve_result(par=par, dimless_radius=par.dimless_radius_out)
        angmom_init = standard_result["angmom"]
        coff_eta_init = standard_result["coff_eta"]
        slim_init_arr = np.array([angmom_init, coff_eta_init])
        return slim_init_arr

    @staticmethod
    def slim_disk_model(indep_var: float, dep_var: np.ndarray, par: DiskParams, angmomin: float) -> np.ndarray:
        radius_sch = DiskTools.get_radius_sch(par=par)
        radius = indep_var * radius_sch
        angmom, coff_eta = dep_var[0], abs(dep_var[1])
        accrate = SlimDisk.get_slim_accrate(par=par, radius=radius)
        dlnaccrate_dradius = SlimDisk.get_slim_dlnaccrate_dradius(par=par, radius=radius)
        dlnangvelk_dradius = SlimDisk.get_slim_dlnangvelk_dradius(par=par, radius=radius)
        arealpressure = SlimDisk.get_slim_arealpressure(par=par, radius=radius, angmom=angmom, angmomin=angmomin)
        arealdensity = SlimDisk.get_slim_arealdensity(
            par=par,
            arealpressure=arealpressure,
            coff_eta=coff_eta,
            radius=radius,
        )
        halfheight = SlimDisk.get_slim_halfheight(
            par=par,
            radius=radius,
            arealpressure=arealpressure,
            arealdensity=arealdensity,
        )
        pressure = SlimDisk.get_slim_pressure(par=par, arealpressure=arealpressure, halfheight=halfheight)
        density = SlimDisk.get_slim_density(par=par, arealdensity=arealdensity, halfheight=halfheight)
        temperature = SlimDisk.get_slim_temperature(pressure=float(pressure), density=float(density))
        opacity = SlimDisk.get_slim_opacity(par=par, density=density, temperature=temperature)
        pressure_ratio = SlimDisk.get_slim_pressure_ratio(density=density, temperature=temperature, pressure=pressure)
        chandindex_1 = SlimDisk.get_slim_chandindex_1(pressure_ratio=pressure_ratio)
        chandindex_3 = SlimDisk.get_slim_chandindex_3(pressure_ratio=pressure_ratio)
        fluxz = SlimDisk.get_slim_fluxz(
            density=density,
            halfheight=halfheight,
            opacity=opacity,
            temperature=temperature,
        )
        apresstoadens = SlimDisk.get_slim_apresstoadens(arealpressure=arealpressure, arealdensity=arealdensity)
        angmomk = SlimDisk.get_slim_angmomk(par=par, radius=radius)
        energy_qrad = 4 * math.pi * radius * fluxz
        fun1 = (
            (angmom**2 - angmomk**2) / (apresstoadens * radius**3)
            - dlnangvelk_dradius
            + 2 * (1 + coff_eta) / radius
            - coff_eta / radius
            - dlnaccrate_dradius
        )
        u1 = (chandindex_1 + 1) / ((chandindex_3 - 1) * radius)
        u2 = (chandindex_1 - 1) * dlnaccrate_dradius / (chandindex_3 - 1)
        u3 = (chandindex_1 - 1) * dlnangvelk_dradius / (chandindex_3 - 1)
        u4 = energy_qrad / apresstoadens / accrate
        u5 = 4 * math.pi * par.alpha_viscosity * arealpressure * angmom / (apresstoadens * accrate * radius)
        u6 = (3 * chandindex_1 - 1) * fun1 / (2 * coff_eta * (chandindex_3 - 1))
        part1 = u1 + u2 + u3 + u4 - u5 - u6
        d1 = (
            accrate
            * chandindex_1
            / (radius * radius * math.pi * par.alpha_viscosity * arealpressure * (chandindex_3 - 1))
        )
        d2 = 2 * math.pi * par.alpha_viscosity * arealpressure / (accrate * apresstoadens)
        d3 = (
            (1 + coff_eta)
            * accrate
            * (3 * chandindex_1 - 1)
            / (2 * math.pi * par.alpha_viscosity * arealpressure * radius * radius * 2 * coff_eta * (chandindex_3 - 1))
        )
        part2 = d1 - d2 - d3
        dangmom_dradius = part1 / part2
        fun2 = -(1 + coff_eta) * accrate / (2 * math.pi * par.alpha_viscosity * arealpressure * radius * radius)
        dcoffeta_dradius = fun1 + fun2 * dangmom_dradius
        dangmom_ddimlessradius = dangmom_dradius * radius_sch
        dcoffeta_ddimlessradius = dcoffeta_dradius * radius_sch
        deri_arr = np.array([dangmom_ddimlessradius, dcoffeta_ddimlessradius])
        return deri_arr

    @staticmethod
    def get_slim_indep_array(*, par: DiskParams) -> np.ndarray:
        indep_array = np.arange(par.dimless_radius_out, par.dimless_radius_in, -0.01)
        return indep_array

    @staticmethod
    def slim_disk_integrator(*, par: DiskParams, angmomin):
        indep_array = SlimDisk.get_slim_indep_array(par=par)
        initvalue = SlimDisk.get_slim_initvalue(par=par)
        slimintresult, slimintinfo = sp.integrate.odeint(
            func=SlimDisk.slim_disk_model,
            y0=initvalue,
            t=indep_array,
            args=(par, angmomin),
            full_output=True,
            printmessg=True,
            tfirst=True,
            atol=1e-10,
            rtol=1e-10,
        )
        return slimintresult, slimintinfo

    @staticmethod
    def slim_disk_solver(*, par: DiskParams):
        dimless_angmomin_min = 1
        dimless_angmomin_max = 2
        dimless_angmomin = 1.5
        solve_counter = 0
        rveltosvel_max = 0
        while True:
            angmomin = SlimDisk.get_slim_angmomin(par=par, dimless_angmomin=dimless_angmomin)
            slimintresult, slimintinfo = SlimDisk.slim_disk_integrator(par=par, angmomin=angmomin)
            dimless_radius_solve_array = slimintinfo["tcur"]
            slimintresult = slimintresult[: dimless_radius_solve_array.shape[0]]
            angmom_solve_array, coffeta_solve_array = slimintresult.T
            dimless_radius_solve_min = np.min(dimless_radius_solve_array)
            rveltosvel_solve_array = SlimDisk.get_slim_rveltosvel_fromfirst(
                par=par,
                dimless_radius=dimless_radius_solve_array,
                angmom=angmom_solve_array,
                coff_eta=coffeta_solve_array,
                angmomin=angmomin,
            )
            rveltosvel_solve_array = np.asarray(rveltosvel_solve_array)
            rveltosvel_max = max(
                rveltosvel_max,
                np.nanmax(rveltosvel_solve_array[np.isfinite(rveltosvel_solve_array)]),
            )
            print("dimless_angmomin:", dimless_angmomin)
            print("dimless_radius_solve_min:", dimless_radius_solve_min)
            print("rveltosvel_max:", rveltosvel_max)
            if dimless_radius_solve_min > 3:
                dimless_angmomin_max = dimless_angmomin
            else:
                if rveltosvel_max < 1:
                    dimless_angmomin_min = dimless_angmomin
                else:
                    slim_solver_result = (dimless_radius_solve_array, angmom_solve_array, coffeta_solve_array)
                    shoot_success = True
                    break
            dimless_angmomin = (dimless_angmomin_max + dimless_angmomin_min) / 2
            solve_counter += 1
            if solve_counter > 30:
                slim_solver_result = (dimless_radius_solve_array, angmom_solve_array, coffeta_solve_array)
                shoot_success = False
                break
        slim_solver_info = {
            "dimless_angmomin": dimless_angmomin,
            "shoot_count": solve_counter,
            "shoot_succcess": shoot_success,
        }
        return slim_solver_result, slim_solver_info


if __name__ == "__main__":
    ...
