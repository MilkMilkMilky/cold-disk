import math
import warnings
from typing import Any, cast

import numpy as np
import scipy.integrate
import scipy.optimize

from cold_disk.disk_solver.parameter_init import DiskParams, cgs_consts
from cold_disk.disk_solver.solve_standard import StandardDisk
from cold_disk.disk_solver.solve_tools import DiskTools

__all__ = ["SlimDisk"]

class SlimDisk:
    @staticmethod
    def get_slim_angvelk(*, par: DiskParams, radius: float | np.ndarray) -> float | np.ndarray:
        """Compute the Keplerian angular velocity in the slim disk model.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        par : DiskParams
            An object of the `DiskParams` class, which contains the adjustable
            parameters of the slim disk model.
        radius : float or array_like
            Radial coordinate(s) of the disk. Can be a scalar or an array.

        Returns
        -------
        float or ndarray
            Keplerian angular velocity at the given radius/radii.
            Returns a scalar if `radius` is scalar, or a NumPy array otherwise.

        """
        radius = np.asarray(radius)
        bhmass = DiskTools.get_bhmass(par=par)
        radius_sch = DiskTools.get_radius_sch(par=par)
        slim_angvelk = np.sqrt(cgs_consts.cgs_gra * bhmass / radius) / (radius - radius_sch)
        return slim_angvelk

    @staticmethod
    def get_slim_angmomk(*, par: DiskParams, radius: float | np.ndarray) -> float | np.ndarray:
        """Compute the Keplerian angular momentum in the slim disk model.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        par : DiskParams
            An object of the `DiskParams` class, which contains the adjustable
            parameters of the slim disk model.
        radius : float or array_like
            Radial coordinate(s) of the disk. Can be a scalar or an array.

        Returns
        -------
        float or ndarray
            Keplerian angular momentum at the given radius/radii.
            Returns a scalar if `radius` is scalar, or a NumPy array otherwise.

        """
        radius = np.asarray(radius)
        angvelk = SlimDisk.get_slim_angvelk(par=par, radius=radius)
        slim_angmomk = angvelk * radius * radius
        return slim_angmomk

    @staticmethod
    def get_slim_accrate(*, par: DiskParams, radius: float | np.ndarray) -> float | np.ndarray:
        """Compute the mass accretion rate at a given radius in the slim disk model.

        All physical quantities are assumed to be expressed in the CGS unit system.
        This function only applies when the disk wind follows a power-law profile.

        Parameters
        ----------
        par : DiskParams
            An object of the `DiskParams` class, which contains the adjustable
            parameters of the slim disk model.
        radius : float or array_like
            Radial coordinate(s) of the disk. Can be a scalar or an array.

        Returns
        -------
        float or ndarray
            Mass accretion rate at the given radius/radii.
            Returns a scalar if `radius` is scalar, or a NumPy array otherwise.

        """
        radius = np.asarray(radius)
        radius_sch = DiskTools.get_radius_sch(par=par)
        dimless_radius = radius / radius_sch
        accrate_out = DiskTools.get_accrate_fromdimless(par=par, dimless_accrate=par.dimless_accrate)
        slim_accrate = accrate_out * np.float_power((dimless_radius / par.dimless_radius_out), par.wind_index)
        return slim_accrate

    @staticmethod
    def get_slim_dlnaccrate_dradius(*, par: DiskParams, radius: float | np.ndarray) -> float | np.ndarray:
        """Compute the derivative of the logarithm of the mass accretion rate.

        All physical quantities are assumed to be expressed in the CGS unit system.
        This function only applies when the disk wind follows a power-law profile.

        Parameters
        ----------
        par : DiskParams
            An object of the `DiskParams` class, which contains the adjustable
            parameters of the slim disk model.
        radius : float or array_like
            Radial coordinate(s) of the disk. Can be a scalar or an array.

        Returns
        -------
        float or ndarray
            Derivative of ln(accretion rate) with respect to radius at the given radius/radii.
            Returns a scalar if `radius` is scalar, or a NumPy array otherwise.

        """
        radius = np.asarray(radius)
        slim_dlnaccrate_dradius = par.wind_index / radius
        return slim_dlnaccrate_dradius

    @staticmethod
    def get_slim_dlnangvelk_dradius(*, par: DiskParams, radius: float | np.ndarray) -> float | np.ndarray:
        """Compute the derivative of the logarithm of the Keplerian angular velocity.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        par : DiskParams
            An object of the `DiskParams` class, which contains the adjustable
            parameters of the slim disk model.
        radius : float or array_like
            Radial coordinate(s) of the disk. Can be a scalar or an array.

        Returns
        -------
        float or ndarray
            Derivative of ln(Keplerian angular velocity) with respect to radius
            at the given radius/radii.
            Returns a scalar if `radius` is scalar, or a NumPy array otherwise.

        """
        radius = np.asarray(radius)
        radius_sch = DiskTools.get_radius_sch(par=par)
        slim_dlnangvelk_dradius = -1 / 2 / radius - 1 / (radius - radius_sch)
        return slim_dlnangvelk_dradius

    @staticmethod
    def get_slim_angmomin(*, par: DiskParams, dimless_angmomin: float) -> float:
        """Compute the dimensional angular momentum of matter entering the black hole.

        Converts the dimensionless angular momentum to physical units in the CGS system.

        Parameters
        ----------
        par : DiskParams
            An object of the `DiskParams` class, which contains the adjustable
            parameters of the slim disk model.
        dimless_angmomin : float
            Dimensionless angular momentum at the inner boundary of the disk.

        Returns
        -------
        float
            Dimensional angular momentum of matter entering the black hole.

        """
        radius_sch = DiskTools.get_radius_sch(par=par)
        slim_angmomin = dimless_angmomin * radius_sch * cgs_consts.cgs_c
        return slim_angmomin

    @staticmethod
    def get_slim_arealpressure(
        *,
        par: DiskParams,
        radius: float | np.ndarray,
        angmom: float | np.ndarray,
        angmomin: float,
    ) -> float | np.ndarray:
        """Compute the areal pressure at a given radius in the slim disk model.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        par : DiskParams
            An object of the `DiskParams` class, which contains the adjustable
            parameters of the slim disk model.
        radius : float or array_like
            Radial coordinate(s) of the disk. Can be a scalar or an array.
        angmom : float or array_like
            Angular momentum at the corresponding radius/radii. Must have the same shape as `radius`.
        angmomin : float
            Angular momentum of matter entering the black hole.

        Returns
        -------
        float or ndarray
            Areal pressure at the given radius/radii.
            Returns a scalar if `radius` is scalar, or a NumPy array otherwise.

        """
        radius, angmom = np.asarray(radius), np.asarray(angmom)
        slim_accrate = SlimDisk.get_slim_accrate(par=par, radius=radius)
        slim_arealpressure = slim_accrate * (angmom - angmomin) / (2 * math.pi * par.alpha_viscosity * radius * radius)
        return slim_arealpressure

    @staticmethod
    def get_slim_arealdensity(
        *,
        par: DiskParams,
        arealpressure: float | np.ndarray,
        coff_eta: float | np.ndarray,
        radius: float | np.ndarray,
    ) -> float | np.ndarray:
        """Compute the areal density at a given radius in the slim disk model.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        par : DiskParams
            An object of the `DiskParams` class, which contains the adjustable
            parameters of the slim disk model.
        arealpressure : float or array_like
            Areal pressure at the corresponding radius/radii. Must have the same shape as `radius`.
        coff_eta : float or array_like
            One of the dependent variables from solving the slim disk equations,
            representing (radial velocity / (areal pressure / areal density))^2.
            Must have the same shape as `radius`.
        radius : float or array_like
            Radial coordinate(s) of the disk. Can be a scalar or an array.

        Returns
        -------
        float or ndarray
            Areal density at the given radius/radii.
            Returns a scalar if `radius` is scalar, or a NumPy array otherwise.

        """
        arealpressure = np.asarray(arealpressure)
        coff_eta, radius = np.asarray(coff_eta), np.asarray(radius)
        slim_accrate = SlimDisk.get_slim_accrate(par=par, radius=radius)
        slim_arealdensity = slim_accrate * slim_accrate / (4 * math.pi**2 * radius * radius * arealpressure * coff_eta)
        return slim_arealdensity

    @staticmethod
    def get_slim_halfheight(*, par: DiskParams, radius, arealpressure, arealdensity) -> float | np.ndarray:
        """Compute the half-thickness of the slim disk at a given radius.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        par : DiskParams
            An object of the `DiskParams` class, which contains the adjustable
            parameters of the slim disk model.
        radius : float or array_like
            Radial coordinate(s) of the disk. Can be a scalar or an array.
        arealpressure : float or array_like
            Areal pressure at the corresponding radius/radii. Must have the same shape as `radius`.
        arealdensity : float or array_like
            Areal density at the corresponding radius/radii. Must have the same shape as `radius`.

        Returns
        -------
        float or ndarray
            Half-thickness of the disk at the given radius/radii.
            Returns a scalar if `radius` is scalar, or a NumPy array otherwise.

        """
        radius = np.asarray(radius)
        arealpressure, arealdensity = np.asarray(arealpressure), np.asarray(arealdensity)
        angvelk = SlimDisk.get_slim_angvelk(par=par, radius=radius)
        coff_k = np.sqrt(arealpressure / arealdensity)
        slim_halfheight = np.sqrt(2 * par.gas_index + 3) * coff_k / angvelk
        return slim_halfheight

    @staticmethod
    def get_slim_pressure(
        *,
        par: DiskParams,
        arealpressure: float | np.ndarray,
        halfheight: float | np.ndarray,
    ) -> float | np.ndarray:
        """Compute the central pressure of the slim disk at a given radius.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        par : DiskParams
            An object of the `DiskParams` class, which contains the adjustable
            parameters of the slim disk model.
        arealpressure : float or array_like
            Areal pressure at the corresponding radius/radii.
            Must have the same shape as `halfheight`.
        halfheight : float or array_like
            Half-thickness of the disk at the corresponding radius/radii.
            Must have the same shape as `arealpressure`.

        Returns
        -------
        float or ndarray
            Central pressure of the disk at the given radius/radii.
            Returns a scalar if inputs are scalars, or a NumPy array otherwise.

        """
        arealpressure, halfheight = np.asarray(arealpressure), np.asarray(halfheight)
        coff_in1 = DiskTools.get_coeff_in(index=par.gas_index + 1)
        slim_pressure = arealpressure / 2 / halfheight / coff_in1
        return slim_pressure

    @staticmethod
    def get_slim_density(
        *,
        par: DiskParams,
        arealdensity: float | np.ndarray,
        halfheight: float | np.ndarray,
    ) -> float | np.ndarray:
        """Compute the central density of the slim disk at a given radius.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        par : DiskParams
            An object of the `DiskParams` class, which contains the adjustable
            parameters of the slim disk model.
        arealdensity : float or array_like
            Areal density at the corresponding radius/radii.
            Must have the same shape as `halfheight`.
        halfheight : float or array_like
            Half-thickness of the disk at the corresponding radius/radii.
            Must have the same shape as `arealdensity`.

        Returns
        -------
        float or ndarray
            Central density of the disk at the given radius/radii.
            Returns a scalar if inputs are scalars, or a NumPy array otherwise.

        """
        arealdensity, halfheight = np.asarray(arealdensity), np.asarray(halfheight)
        coff_in = DiskTools.get_coeff_in(index=par.gas_index)
        slim_density = arealdensity / 2 / halfheight / coff_in
        return slim_density

    @staticmethod
    def get_slim_temperature(*, pressure: float, density: float) -> float:
        """Compute the central temperature of the slim disk from pressure and density.

        Solves the equation for temperature:

            (a / 3) * T^4 + (R_g * density / μ) * T - pressure = 0

        where:
            - a is the radiation constant,
            - R_g is the molar gas constant,
            - μ is the average molar mass,
            - T is the temperature.

        The solution is searched in the positive range. The method proceeds as follows:
            1. Attempt to solve using `root_scalar` with the Brent's method (`brentq`).
            2. If `root_scalar` fails, fall back to `fsolve` with default settings.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        pressure : float
            Central pressure of the disk at the given radius.
        density : float
            Central density of the disk at the given radius.

        Returns
        -------
        float
            Central temperature of the disk at the given radius.

        Raises
        ------
        RuntimeError
            If the temperature could not be solved successfully.

        """
        coff_b = cgs_consts.cgs_rg * density / cgs_consts.cgs_amm
        coff_c = -pressure

        def slim_temperature_func(t):
            return (cgs_consts.cgs_a / 3) * t**4 + coff_b * t + coff_c

        try:
            temperature = scipy.optimize.root_scalar(
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
        temperature_array = scipy.optimize.fsolve(slim_temperature_func, 10000)
        temperature_array = cast("np.ndarray", temperature_array)
        for temp in temperature_array:
            if temp > 0:
                slim_temperature = temp
                return slim_temperature
        raise RuntimeError(f"Failed to solve slim temperature. pressure={pressure}, density={density}")

    @staticmethod
    def get_slim_opacity(
        *,
        par: DiskParams,
        density: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        """Compute the opacity of the slim disk at a given radius.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        par : DiskParams
            An object of the `DiskParams` class, which contains the adjustable
            parameters of the slim disk model.
        density : float or array_like
            Central density at the corresponding radius/radii.
            Must have the same shape as `temperature`.
        temperature : float or array_like
            Central temperature at the corresponding radius/radii.
            Must have the same shape as `density`.

        Returns
        -------
        float or ndarray
            Opacity at the given radius/radii.
            Returns a scalar if inputs are scalars, or a NumPy array otherwise.

        """
        density, temperature = np.asarray(density), np.asarray(temperature)
        coff_in = DiskTools.get_coeff_in(index=par.gas_index)
        opacity_abs = cgs_consts.cgs_kra * (coff_in * density)
        opacity_abs *= np.float_power((2 * temperature / 3), (-3.5))
        slim_opacity = cgs_consts.cgs_kes + opacity_abs
        return slim_opacity

    @staticmethod
    def get_slim_pressure_ratio(
        *,
        density: float | np.ndarray,
        temperature: float | np.ndarray,
        pressure: float | np.ndarray,
    ) -> float | np.ndarray:
        """Compute the ratio of gas pressure to total pressure at a given radius.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        density : float or array_like
            Central density at the corresponding radius/radii.
            Must have the same shape as `temperature` and `pressure`.
        temperature : float or array_like
            Central temperature at the corresponding radius/radii.
            Must have the same shape as `density` and `pressure`.
        pressure : float or array_like
            Central total pressure at the corresponding radius/radii.
            Must have the same shape as `density` and `temperature`.

        Returns
        -------
        float or ndarray
            Ratio of gas pressure to total pressure at the given radius/radii.
            Returns a scalar if inputs are scalars, or a NumPy array otherwise.

        """
        pressure, density = np.asarray(pressure), np.asarray(density)
        temperature = np.asarray(temperature)
        slim_pressure_ratio = cgs_consts.cgs_rg / cgs_consts.cgs_amm * density * temperature / pressure
        return slim_pressure_ratio

    @staticmethod
    def get_slim_chandindex_1(*, pressure_ratio: float | np.ndarray) -> float | np.ndarray:
        """Compute the first Chandrasekhar generalized adiabatic index at a given radius.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        pressure_ratio : float or array_like
            Ratio of gas pressure to total pressure at the corresponding radius/radii.

        Returns
        -------
        float or ndarray
            First Chandrasekhar generalized adiabatic index at the given radius/radii.
            Returns a scalar if `pressure_ratio` is scalar, or a NumPy array otherwise.

        """
        pressure_ratio = np.asarray(pressure_ratio)
        slim_chandindex_1 = (32 - 24 * pressure_ratio - 3 * pressure_ratio**2) / (24 - 21 * pressure_ratio)
        return slim_chandindex_1

    @staticmethod
    def get_slim_chandindex_3(*, pressure_ratio: float | np.ndarray) -> float | np.ndarray:
        """Compute the third Chandrasekhar generalized adiabatic index at a given radius.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        pressure_ratio : float or array_like
            Ratio of gas pressure to total pressure at the corresponding radius/radii.

        Returns
        -------
        float or ndarray
            Third Chandrasekhar generalized adiabatic index at the given radius/radii.
            Returns a scalar if `pressure_ratio` is scalar, or a NumPy array otherwise.

        """
        pressure_ratio = np.asarray(pressure_ratio)
        slim_chandindex_3 = (32 - 27 * pressure_ratio) / (24 - 21 * pressure_ratio)
        return slim_chandindex_3

    @staticmethod
    def get_slim_fluxz(
        *,
        density: float | np.ndarray,
        halfheight: float | np.ndarray,
        opacity: float | np.ndarray,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        """Compute the vertical radiative flux of the slim disk at a given radius.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        density : float or array_like
            Central density at the corresponding radius/radii.
        halfheight : float or array_like
            Half-thickness of the disk at the corresponding radius/radii.
        opacity : float or array_like
            Opacity at the corresponding radius/radii.
        temperature : float or array_like
            Central temperature at the corresponding radius/radii.

            Note
            ----
            All input parameters must have the same shape.

        Returns
        -------
        float or ndarray
            Vertical radiative flux at the given radius/radii.
            Returns a scalar if inputs are scalars, or a NumPy array otherwise.

        """
        density, halfheight = np.asarray(density), np.asarray(halfheight)
        opacity, temperature = np.asarray(opacity), np.asarray(temperature)
        slim_fluxz = 4 * cgs_consts.cgs_a * cgs_consts.cgs_c * np.float_power(temperature, 4)
        slim_fluxz /= 3 * opacity * density * halfheight
        return slim_fluxz

    @staticmethod
    def get_slim_apresstoadens(
        arealpressure: float | np.ndarray,
        arealdensity: float | np.ndarray,
    ) -> float | np.ndarray:
        """Compute the ratio of areal pressure to areal density at a given radius.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        arealpressure : float or array_like
            Areal pressure at the corresponding radius/radii.
            Must have the same shape as `arealdensity`.
        arealdensity : float or array_like
            Areal density at the corresponding radius/radii.
            Must have the same shape as `arealpressure`.

        Returns
        -------
        float or ndarray
            Ratio of areal pressure to areal density at the given radius/radii.
            Returns a scalar if inputs are scalars, or a NumPy array otherwise.

        """
        arealpressure, arealdensity = np.asarray(arealpressure), np.asarray(arealdensity)
        slim_apresstoadens = arealpressure / arealdensity
        return slim_apresstoadens

    @staticmethod
    def get_slim_radvel(
        *,
        par: DiskParams,
        radius: float | np.ndarray,
        arealdensity: float | np.ndarray,
    ) -> float | np.ndarray:
        """Compute the radial velocity of the slim disk at a given radius.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        par : DiskParams
            An object of the `DiskParams` class, which contains the adjustable
            parameters of the slim disk model.
        radius : float or array_like
            Radial coordinate(s) of the disk.
        arealdensity : float or array_like
            Areal density at the corresponding radius/radii.
            Must have the same shape as `radius`.

        Returns
        -------
        float or ndarray
            Radial velocity at the given radius/radii.
            Returns a scalar if inputs are scalars, or a NumPy array otherwise.

        """
        radius, arealdensity = np.asarray(radius), np.asarray(arealdensity)
        accrate = SlimDisk.get_slim_accrate(par=par, radius=radius)
        slim_radvel = -accrate / 2 / math.pi / radius / arealdensity
        return slim_radvel

    @staticmethod
    def get_slim_soundvel(*, pressure: float | np.ndarray, density: float | np.ndarray) -> float | np.ndarray:
        """Compute the sound speed of the slim disk at a given radius.

        The sound speed is defined as the square root of the ratio of
        central pressure to central density at the given radius.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        pressure : float or array_like
            Central pressure at the corresponding radius/radii.
            Must have the same shape as `density`.
        density : float or array_like
            Central density at the corresponding radius/radii.
            Must have the same shape as `pressure`.

        Returns
        -------
        float or ndarray
            Sound speed at the given radius/radii.
            Returns a scalar if inputs are scalars, or a NumPy array otherwise.

        """
        pressure, density = np.asarray(pressure), np.asarray(density)
        slim_soundvel = np.sqrt(pressure / density)
        return slim_soundvel

    @staticmethod
    def get_slim_rveltosvel(*, radvel: float | np.ndarray, soundvel: float | np.ndarray) -> float | np.ndarray:
        """Compute the ratio of radial velocity to sound speed at a given radius.

        The result is defined as the absolute value of radial velocity divided by
        the sound speed: |v_r / c_s|.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        radvel : float or array_like
            Radial velocity at the corresponding radius/radii.
            Must have the same shape as `soundvel`.
        soundvel : float or array_like
            Sound speed at the corresponding radius/radii.
            Must have the same shape as `radvel`.

        Returns
        -------
        float or ndarray
            Absolute value of the ratio of radial velocity to sound speed at the given radius/radii.
            Returns a scalar if inputs are scalars, or a NumPy array otherwise.

        """
        radvel, soundvel = np.asarray(radvel), np.asarray(soundvel)
        slim_rveltosvel = np.abs(radvel / soundvel)
        return slim_rveltosvel

    @staticmethod
    def get_slim_temperature_eff(*, fluxz: float | np.ndarray) -> float | np.ndarray:
        """Compute the effective temperature of the slim disk at a given radius.

        The effective temperature is derived from the vertical radiative flux
        (fluxz) and is generally different from the central temperature.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        fluxz : float or array_like
            Vertical radiative flux at the corresponding radius/radii.

        Returns
        -------
        float or ndarray
            Effective temperature at the given radius/radii.
            Returns a scalar if `fluxz` is scalar, or a NumPy array otherwise.

        """
        fluxz = np.asarray(fluxz)
        slim_temperature_eff = np.float_power((fluxz / cgs_consts.cgs_sb), 0.25)
        return slim_temperature_eff

    @staticmethod
    def get_slim_initvalue(*, par: DiskParams) -> np.ndarray:
        """Compute the initial values at the outer boundary of the slim disk.

        The initial values are taken from the solution of the standard disk
        model at the outer boundary radius. These values serve as the starting
        point for solving the slim disk equations.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        par : DiskParams
            An object of the `DiskParams` class, which contains the adjustable
            parameters of the slim disk model.

        Returns
        -------
        ndarray
            Array containing the initial values at the outer boundary of the slim disk.
            Typically includes angular momentum and the dependent variable coff_eta.

        """
        standard_result = StandardDisk.get_standard_solve_result(par=par, dimless_radius=par.dimless_radius_out)
        angmom_init = standard_result["angmom"]
        coff_eta_init = standard_result["coff_eta"]
        slim_init_arr = np.array([angmom_init, coff_eta_init])
        return slim_init_arr

    @staticmethod
    def slim_disk_model(indep_var: float, dep_var: np.ndarray, par: DiskParams, angmomin: float) -> np.ndarray:
        """Compute the derivatives of angular momentum and coff_eta for the slim disk model.

        This function defines the system of ordinary differential equations (ODEs)
        used for solving the slim disk structure. The independent variable is the
        dimensionless radius, and the dependent variables are:

            - angmom: angular momentum
            - coff_eta: squared ratio of radial velocity to (areal pressure / areal density)

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        indep_var : float
            Dimensionless radius at the current step.
        dep_var : ndarray
            Array containing the current values of the dependent variables:
                dep_var[0] = angular momentum (angmom)
                dep_var[1] = coff_eta
        par : DiskParams
            An object of the `DiskParams` class containing the adjustable parameters
            of the slim disk model.
        angmomin : float
            Angular momentum of matter entering the black hole (inner boundary).

        Returns
        -------
        ndarray
            Array of derivatives with respect to dimensionless radius:
                [d(angmom)/d(dimless_radius), d(coff_eta)/d(dimless_radius)]

        """
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
        dangmom_numerator = u1 + u2 + u3 + u4 - u5 - u6
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
        dangmom_denominator = d1 - d2 - d3
        dangmom_dradius = dangmom_numerator / dangmom_denominator
        fun2 = -(1 + coff_eta) * accrate / (2 * math.pi * par.alpha_viscosity * arealpressure * radius * radius)
        dcoffeta_dradius = fun1 + fun2 * dangmom_dradius
        dangmom_ddimlessradius = dangmom_dradius * radius_sch
        dcoffeta_ddimlessradius = dcoffeta_dradius * radius_sch
        deri_arr = np.array([dangmom_ddimlessradius, dcoffeta_ddimlessradius])
        return deri_arr

    @staticmethod
    def slim_disk_model_output(
        *,
        indep_var: np.ndarray,
        dep_var_0: np.ndarray,
        dep_var_1: np.ndarray,
        par: DiskParams,
        angmomin: float,
        output_mode: str,
    ) -> np.ndarray:
        """Compute the main physical quantities of the slim disk as functions of radius.

        This function evaluates various disk properties at the given array of
        dimensionless radii based on the current values of angular momentum and
        coff_eta. It can be used to output either selected quantities for
        ODE integration diagnostics or the full set of quantities after the
        ODE solution is complete.

        The `output_mode` parameter controls which quantities are returned:

            - "rveltosvel": only returns the ratio of radial velocity to sound speed.
            - "dangmom": returns rveltosvel along with components of the angular momentum derivative.
            - "fulloutput": returns the complete set of slim disk quantities, including
              radius, accretion rate, angular momentum, coff_eta, velocities, pressures,
              densities, temperature, opacity, Chandrasekhar indices, flux, and etc.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        indep_var : ndarray
            Array of dimensionless radii at which to evaluate the disk quantities.
        dep_var_0 : ndarray
            Array of angular momentum values corresponding to `indep_var`.
        dep_var_1 : ndarray
            Array of coff_eta values corresponding to `indep_var`.
        par : DiskParams
            An object of the `DiskParams` class containing the adjustable parameters
            of the slim disk model.
        angmomin : float
            Angular momentum of matter entering the black hole (inner boundary).
        output_mode : str
            Determines which quantities are returned. Must be one of "rveltosvel",
            "dangmom", or "fulloutput".

        Returns
        -------
        ndarray
            Structured array containing the requested slim disk quantities, with
            fields determined by `output_mode`. The shape matches that of `indep_var`.

        """
        radius_sch = DiskTools.get_radius_sch(par=par)
        radius = indep_var * radius_sch
        angmom, coff_eta = dep_var_0, np.abs(dep_var_1)
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
        pressure = np.asarray(SlimDisk.get_slim_pressure(par=par, arealpressure=arealpressure, halfheight=halfheight))
        density = np.asarray(SlimDisk.get_slim_density(par=par, arealdensity=arealdensity, halfheight=halfheight))
        radvel = SlimDisk.get_slim_radvel(par=par, radius=radius, arealdensity=arealdensity)
        soundvel = SlimDisk.get_slim_soundvel(pressure=pressure, density=density)
        rveltosvel = SlimDisk.get_slim_rveltosvel(radvel=radvel, soundvel=soundvel)
        match output_mode:
            case "rveltosvel":
                slim_output_dtype = [
                    (name, "f8")
                    for name in [
                        "rveltosvel",
                    ]
                ]
                slim_output = np.zeros_like(rveltosvel, dtype=slim_output_dtype)
                slim_output["rveltosvel"] = rveltosvel
                return slim_output

        temperature = np.array([
            SlimDisk.get_slim_temperature(pressure=p, density=d) if p > 0 and d > 0 else 0
            for p, d in zip(pressure, density, strict=True)
        ])
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
        dangmom_numerator = u1 + u2 + u3 + u4 - u5 - u6
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
        dangmom_denominator = d1 - d2 - d3
        match output_mode:
            case "dangmom":
                slim_output_dtype = [
                    (name, "f8") for name in ["rveltosvel", "dangmom_numerator", "dangmom_denominator"]
                ]
                slim_output = np.zeros_like(rveltosvel, dtype=slim_output_dtype)
                slim_output["rveltosvel"] = rveltosvel
                slim_output["dangmom_numerator"] = dangmom_numerator
                slim_output["dangmom_denominator"] = dangmom_denominator
                return slim_output
        dangmom_dradius = dangmom_numerator / dangmom_denominator
        fun2 = -(1 + coff_eta) * accrate / (2 * math.pi * par.alpha_viscosity * arealpressure * radius * radius)
        dcoffeta_dradius = fun1 + fun2 * dangmom_dradius
        dangmom_ddimlessradius = dangmom_dradius * radius_sch
        dcoffeta_ddimlessradius = dcoffeta_dradius * radius_sch
        match output_mode:
            case "fulloutput":
                slim_output_dtype = [
                    (name, "f8")
                    for name in [
                        "dimless_radius",
                        "radius",
                        "accrate",
                        "angmom",
                        "coff_eta",
                        "angvelk",
                        "arealpressure",
                        "arealdensity",
                        "halfheight",
                        "pressure",
                        "density",
                        "radvel",
                        "soundvel",
                        "rveltosvel",
                        "temperature",
                        "opacity",
                        "pressure_ratio",
                        "chandindex_1",
                        "chandindex_3",
                        "fluxz",
                        "dangmom_numerator",
                        "dangmom_denominator",
                        "dangmom_ddimlessradius",
                        "dcoffeta_ddimlessradius",
                    ]
                ]
                slim_output = np.zeros_like(indep_var, dtype=slim_output_dtype)
                slim_output["dimless_radius"] = indep_var
                slim_output["radius"] = radius
                slim_output["accrate"] = accrate
                slim_output["angmom"] = angmom
                slim_output["coff_eta"] = coff_eta
                slim_output["angvelk"] = angmomk / radius / radius
                slim_output["arealpressure"] = arealpressure
                slim_output["arealdensity"] = arealdensity
                slim_output["halfheight"] = halfheight
                slim_output["pressure"] = pressure
                slim_output["density"] = density
                slim_output["radvel"] = radvel
                slim_output["soundvel"] = soundvel
                slim_output["rveltosvel"] = rveltosvel
                slim_output["temperature"] = temperature
                slim_output["opacity"] = opacity
                slim_output["pressure_ratio"] = pressure_ratio
                slim_output["chandindex_1"] = chandindex_1
                slim_output["chandindex_3"] = chandindex_3
                slim_output["fluxz"] = fluxz
                slim_output["dangmom_numerator"] = dangmom_numerator
                slim_output["dangmom_denominator"] = dangmom_denominator
                slim_output["dangmom_ddimlessradius"] = dangmom_ddimlessradius
                slim_output["dcoffeta_ddimlessradius"] = dcoffeta_ddimlessradius
                return slim_output
        raise ValueError("Invalid output_mode")

    @staticmethod
    def get_slim_indep_array(*, par: DiskParams) -> np.ndarray:
        """Generate the initial array of dimensionless radii for slim disk ODE solving.

        The array spans from the outer radius to the inner radius with a step
        of -0.01. **Do not change this step size under any circumstances**,
        as practical tests show that altering it can cause **fatal errors**
        in the global solution process of the slim disk model.

        Note that after ODE integration, the returned independent and dependent
        variable arrays from the integrator will most likely **have different lengths
        and step sizes** than this initial sequence.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        par : DiskParams
            An object of the `DiskParams` class containing the adjustable
            parameters of the slim disk model, including `dimless_radius_in`
            and `dimless_radius_out`.

        Returns
        -------
        ndarray
            Array of dimensionless radii for ODE integration, decreasing from
            outer to inner boundary with step -0.01.

        """
        indep_array = np.arange(par.dimless_radius_out, par.dimless_radius_in, -0.01)
        return indep_array

    @staticmethod
    def slim_disk_odeint(*, par: DiskParams, angmomin: float) -> tuple[np.ndarray, dict[str, Any]]:
        """Integrate the slim disk ODE system over the dimensionless radius array.

        This function uses `scipy.integrate.odeint` to solve the system of
        ODEs defined by `slim_disk_model`, starting from the outer boundary
        initial conditions. The returned arrays provide the dependent variables
        (angular momentum and coff_eta) along the integration path.

        **Important constraints:**
        - `tfirst=True`, `atol=1e-10`, and `rtol=1e-10` **must not be changed**.
          Modifying these values can cause **fatal errors** in the global
          solution of the slim disk model.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        par : DiskParams
            An object of the `DiskParams` class containing the adjustable
            parameters of the slim disk model.
        angmomin : float
            Angular momentum of matter entering the black hole (inner boundary).

        Returns
        -------
        tuple
            - slimintresult : ndarray
                Array of integrated dependent variables along the dimensionless
                radius array (angmom, coff_eta).
            - slimintinfo : dict
                Dictionary of integration information returned by `odeint`.

        """
        indep_array = SlimDisk.get_slim_indep_array(par=par)
        initvalue = SlimDisk.get_slim_initvalue(par=par)
        slimintresult, slimintinfo = scipy.integrate.odeint(
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
    def slim_disk_odeint_manager(
        *,
        par: DiskParams,
        angmomin: float,
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float]:
        """Process the result of a single ODE integration of the slim disk model.

        This function organizes the output from `slim_disk_integrator` into
        properly shaped arrays of independent and dependent variables, suitable
        for further analysis. The shapes of the returned arrays generally
        **do not match** the initial dimensionless radius sequence provided
        to the integrator.

        The function identifies the smallest non-zero value in the integrator's
        internal `tcur` array, which indicates where the integration effectively
        terminated. All trailing zeros in `tcur` are removed, and the
        corresponding dependent variables are truncated accordingly.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        par : DiskParams
            An object of the `DiskParams` class containing the adjustable
            parameters of the slim disk model.
        angmomin : float
            Angular momentum of matter entering the black hole (inner boundary).

        Returns
        -------
        tuple
            - manageresult : tuple of ndarrays
                (dimless_radius_solve_array, angmom_solve_array, coffeta_solve_array)
                Arrays of independent dimensionless radius and the corresponding
                angular momentum and coff_eta after removing trailing zeros.
            - manageinfo : float
                Minimum non-zero value of the integrator's internal `tcur`,
                indicating the effective termination radius of the integration.

        """
        indep_array = SlimDisk.get_slim_indep_array(par=par)
        slimintresult, slimintinfo = SlimDisk.slim_disk_odeint(par=par, angmomin=angmomin)
        dimless_radius_solve_cur = slimintinfo["tcur"]
        dimless_radius_solve_index = np.nonzero(dimless_radius_solve_cur)
        dimless_radius_solve_min = np.min(dimless_radius_solve_cur[dimless_radius_solve_index])
        dimless_radius_solve_array = indep_array[dimless_radius_solve_index]
        slimintresult = slimintresult[dimless_radius_solve_index]
        angmom_solve_array, coffeta_solve_array = slimintresult.T
        manageresult = (dimless_radius_solve_array, angmom_solve_array, coffeta_solve_array)
        manageinfo = dimless_radius_solve_min
        return manageresult, manageinfo

    @staticmethod
    def slim_disk_odeint_solver(*, par: DiskParams) -> tuple[np.ndarray, np.ndarray]:
        """Get the slim disk global solution using a shooting method.

        This routine performs a binary-search (shooting) loop over a dimensionless
        inner angular-momentum parameter (dimless_angmomin) to obtain a global
        solution of the slim-disk equations. For each trial value the function:

          1. Converts the trial dimensionless angmomin to a physical angmomin,
          2. Integrates the ODEs (via `slim_disk_integrate_manager`),
          3. Evaluates a diagnostic quantity `rveltosvel = |v_r / c_s|` over the
             successfully integrated radius range (using `slim_disk_model_output`).

        The search logic (per iteration) is:

          - If the integrator terminated *outside* the ISCO region (code tests
            `dimless_radius_solve_min > 3`), the current trial is considered too
            large and the upper search bound is reduced.
          - Otherwise (integration reached deep enough), if the solution never
            becomes transonic (i.e. `rveltosvel_max < 1`) the trial is considered too
            small and the lower search bound is increased.
          - If the integration reaches sufficiently small radius *and* a transonic
            point is found (`rveltosvel_max >= 1`), the solver accepts the current
            solution and returns the full output.

        A hard iteration cap of 50 is enforced; if exceeded the routine returns the
        most recent (closest) result and flags the shoot as unsuccessful.

        All physical quantities are assumed to be expressed in the CGS unit system.

        Parameters
        ----------
        par : DiskParams
            An object of the `DiskParams` class containing the adjustable parameters
            of the slim disk model.

        Returns
        -------
        tuple
            - slim_solver_result : ndarray
                Structured array produced by `slim_disk_model_output(..., output_mode='fulloutput')`
                corresponding to the final (accepted or last) trial. When the shooting
                succeeds this is the accepted full solution; when the shooting fails
                after the iteration cap it is the most recent attempt.
            - slim_solver_info : ndarray
                Numpy structured array of length 1 with fields:
                - "dimless_angmomin" : float
                Final dimensionless inner angular momentum (the last trial value).
                - "shoot_count" : int
                Number of shooting iterations performed.
                - "shoot_succcess" : bool
                True if the shooting terminated successfully, False if the
                iteration cap was reached without success.

        Notes
        -----
        - Initial search bounds and guess: dimless_angmomin_min = 1, dimless_angmomin_max = 2,
          initial dimless_angmomin = 1.5.
        - The diagnostic `rveltosvel_max` is the maximum of |v_r / c_s| over the
          integrator's valid radius interval; it is used to detect transonic crossing.
        - Multiple unsuccessful shooting attempts return the most recent computed
          solution (closest attempt), not a guaranteed physically valid final solution.

        """
        dimless_angmomin_min = 1
        dimless_angmomin_max = 2
        dimless_angmomin = 1.5
        solve_counter = 0
        rveltosvel_max = 0
        while True:
            angmomin = SlimDisk.get_slim_angmomin(par=par, dimless_angmomin=dimless_angmomin)
            manageresult, manageinfo = SlimDisk.slim_disk_odeint_manager(par=par, angmomin=angmomin)
            rveltosvel_solve_array = SlimDisk.slim_disk_model_output(
                indep_var=manageresult[0],
                dep_var_0=manageresult[1],
                dep_var_1=manageresult[2],
                par=par,
                angmomin=angmomin,
                output_mode="rveltosvel",
            )["rveltosvel"]
            dimless_radius_solve_min = manageinfo
            rveltosvel_max = np.nanmax(rveltosvel_solve_array[np.isfinite(rveltosvel_solve_array)])
            if dimless_radius_solve_min > 3:
                dimless_angmomin_max = dimless_angmomin
            else:
                if rveltosvel_max < 1:
                    dimless_angmomin_min = dimless_angmomin
                else:
                    slim_solver_result = SlimDisk.slim_disk_model_output(
                        indep_var=manageresult[0],
                        dep_var_0=manageresult[1],
                        dep_var_1=manageresult[2],
                        par=par,
                        angmomin=angmomin,
                        output_mode="fulloutput",
                    )
                    shoot_success = True
                    break
            dimless_angmomin = (dimless_angmomin_max + dimless_angmomin_min) / 2
            solve_counter += 1
            if solve_counter > 50:
                slim_solver_result = SlimDisk.slim_disk_model_output(
                    indep_var=manageresult[0],
                    dep_var_0=manageresult[1],
                    dep_var_1=manageresult[2],
                    par=par,
                    angmomin=angmomin,
                    output_mode="fulloutput",
                )
                shoot_success = False
                break
        slim_solver_info_dtype = [("shoot_count", "i4"), ("dimless_angmomin", "f8"), ("shoot_succcess", "bool")]
        slim_solver_info = np.zeros(1, dtype=slim_solver_info_dtype)
        slim_solver_info["shoot_count"] = solve_counter
        slim_solver_info["dimless_angmomin"] = dimless_angmomin
        slim_solver_info["shoot_succcess"] = shoot_success
        return slim_solver_result, slim_solver_info

    @staticmethod
    def slim_disk_sed_solver(
        *,
        par: DiskParams,
        dimless_radius: np.ndarray,
        fluxz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the the radiation characteristics of a slim accretion disk.

        This routine calculates the disk spectral energy distribution (SED)
        by integrating the local blackbody emission over the disk radius.
        It also computes the total bolometric luminosity and the effective
        radiative efficiency relative to the initial mass accretion rate.

        Important
        ----------
        - The integration is restricted to **dimensionless radii > 3**, corresponding
          to the ISCO of a Schwarzschild black hole. Radii below this limit are
          unstable to numerical evaluation, and including them can lead to
          serious errors in the SED, bolometric luminosity, and efficiency calculations.

        Parameters
        ----------
        par : DiskParams
            An object of the `DiskParams` class containing the adjustable
            parameters of the slim disk model.
        dimless_radius : np.ndarray
            Array of dimensionless radii corresponding to the slim-disk solution.
        fluxz : np.ndarray
            Vertical radiative flux at each radius, used to compute the effective temperature.

        Returns
        -------
        tuple
            - sed_output : np.ndarray
                Structured array with fields:
                    - "logfrequency" : float
                        Logarithm (base 10) of the frequency in Hz.
                    - "sed" : float
                        Spectral energy distribution at each frequency (erg/s/Hz).
            - lum_output : np.ndarray
                Structured array with fields:
                    - "lum_bol" : float
                        Total bolometric luminosity integrated over all frequencies (erg/s).
                    - "lum_eff" : float
                        Effective radiative efficiency, i.e. lum_bol / (accretion_rate_init * c^2).

        Notes
        -----
        - Frequencies are sampled logarithmically from 10^13 Hz to 10^17 Hz with step 0.01 dex.
        - The SED is integrated using the trapezoidal rule over the radius array.
        - Negative signs in the integration formula account for the inward radial flux convention.
        - Any NaNs or positive infinities arising from numerical issues are replaced with zero
          before computing bolometric luminosity.

        """
        log_frequencies = np.arange(13, 17, 0.01)
        frequencies = np.float_power(10, log_frequencies)
        sed_frequencies = np.empty(len(log_frequencies))
        mask = dimless_radius > 3
        dimless_radius = dimless_radius[mask]
        temperature_eff = np.asarray(SlimDisk.get_slim_temperature_eff(fluxz=fluxz))[mask]
        radius_sch = DiskTools.get_radius_sch(par=par)
        accrate_init = DiskTools.get_accrate_init(par=par)
        for item, frequency in enumerate(frequencies):
            dsed_numerator = (
                4
                * math.pi
                * cgs_consts.cgs_h
                * np.float_power(frequency, 3)
                / (cgs_consts.cgs_c**2)
                * radius_sch**2
                * dimless_radius
            )
            dsed_denominator = np.expm1(cgs_consts.cgs_h * frequency / cgs_consts.cgs_kb / temperature_eff)
            dsed_frequency = 2 * math.pi * dsed_numerator / dsed_denominator
            sed_frequencies[item] = -scipy.integrate.trapezoid(dsed_frequency, dimless_radius)
        sed = sed_frequencies * frequencies
        sed_frequencies_nonenan = np.nan_to_num(sed_frequencies, posinf=0)
        lum_bol = scipy.integrate.trapezoid(sed_frequencies_nonenan, frequencies)
        lum_eff = lum_bol / (accrate_init * cgs_consts.cgs_c**2)
        sed_output_dtype = [("logfrequency", "f8"), ("sed", "f8")]
        sed_output = np.zeros_like(log_frequencies, dtype=sed_output_dtype)
        sed_output["logfrequency"] = log_frequencies
        sed_output["sed"] = sed
        lum_output_dtype = [("lum_bol", "f8"), ("lum_eff", "f8")]
        lum_output = np.zeros(1, dtype=lum_output_dtype)
        lum_output["lum_bol"] = lum_bol
        lum_output["lum_eff"] = lum_eff
        return sed_output, lum_output
