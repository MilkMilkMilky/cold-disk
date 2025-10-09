from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.special

from cold_disk.disk_solver.parameter_init import DiskParams, cgs_consts

__all__ = ["DiskTools"]

class DiskTools:
    @staticmethod
    def vectorize_for_radius(func: Callable, on_fail: Any = np.nan) -> Callable:
        """Wrap a function to allow array-like inputs for a single argument `dimless_radius`.

        - func: the original function to vectorize.
        - on_fail: value to use when a single call raises an exception. Use 'raise' to re-raise.

        Returns a new function that accepts a scalar or 1D array input for dimless_radius.
        """

        def wrapper(*args, dimless_radius=None, **kwargs) -> Any:
            if dimless_radius is None:
                raise ValueError("dimless_radius must be provided")
            arr = np.atleast_1d(dimless_radius)
            results = []

            for r in arr:
                try:
                    val = func(*args, dimless_radius=r, **kwargs)
                except Exception:
                    if on_fail == "raise":
                        raise
                    val = on_fail
                results.append(val)

            results = np.array(results)
            if np.isscalar(dimless_radius) or (hasattr(dimless_radius, "shape") and dimless_radius.shape == ()):
                return results.item()
            return results

        return wrapper

    @staticmethod
    def get_bhmass(*, par: DiskParams) -> float:
        """Compute the physical black hole mass from the dimensionless mass parameter.

        Parameters
        ----------
        par : DiskParams
            An object containing the adjustable parameters of the accretion disk.
            `par.dimless_bhmass` specifies the black hole mass in dimensionless units.

        Returns
        -------
        float
            Black hole mass in CGS units.

        """
        bhmass = par.dimless_bhmass * cgs_consts.cgs_msun
        return bhmass

    @staticmethod
    def get_accrate_edd(*, par: DiskParams) -> float:
        """Compute the physical Eddington accretion rate from the dimensionless accretion rate.

        Parameters
        ----------
        par : DiskParams
            An object containing the adjustable parameters of the accretion disk.
            `par.dimless_bhmass` is used to scale the Eddington accretion rate.

        Returns
        -------
        float
            Eddington accretion rate in CGS units.

        """
        accrate_edd = 1.44e18 * par.dimless_bhmass
        return accrate_edd

    @staticmethod
    def get_accrate_init(*, par: DiskParams) -> float:
        """Compute the initial (outer-boundary) accretion rate for the disk.

        Parameters
        ----------
        par : DiskParams
            An object containing the adjustable parameters of the accretion disk.
            `par.dimless_accrate` specifies the accretion rate relative to the
            Eddington rate.

        Returns
        -------
        float
            Accretion rate at the disk outer boundary in CGS units.

        """
        accrate_edd = DiskTools.get_accrate_edd(par=par)
        accrate_init = accrate_edd * par.dimless_accrate
        return accrate_init

    @staticmethod
    def get_radius_sch(*, par: DiskParams) -> float:
        """Compute the Schwarzschild radius of the black hole.

        Parameters
        ----------
        par : DiskParams
            An object containing the adjustable parameters of the accretion disk.
            The black hole mass is obtained from `par.dimless_bhmass`.

        Returns
        -------
        float
            Schwarzschild radius in CGS units.

        """
        bhmass = DiskTools.get_bhmass(par=par)
        radius_sch = 2 * cgs_consts.cgs_gra * bhmass / cgs_consts.cgs_c / cgs_consts.cgs_c
        return radius_sch

    @staticmethod
    def get_coeff_in(*, index: float) -> float:
        """Compute a coefficient related to the gas index for disk calculations.

        This coefficient is often used in converting between areal and volumetric
        quantities in the accretion disk equations, and depends on the polytropic
        (or gas) index.

        Parameters
        ----------
        index : float
            Gas or polytropic index used in the coefficient calculation.

        Returns
        -------
        float
            Computed coefficient value.

        """
        numerator = (2**index * scipy.special.gamma(index + 1)) ** 2
        denominator = scipy.special.gamma(2 * index + 2)
        coeff_in = numerator / denominator
        return coeff_in

    @staticmethod
    def get_radius_fromdimless(*, par: DiskParams, dimless_radius: float | np.ndarray) -> float | np.ndarray:
        """Convert a dimensionless radius to a physical radius.

        Parameters
        ----------
        par : DiskParams
            An object containing the adjustable parameters of the accretion disk.
        dimless_radius : float or np.ndarray
            Dimensionless radius (in units of Schwarzschild radius).

        Returns
        -------
        float or np.ndarray
            Physical radius in CGS units. Matches the shape of `dimless_radius`.

        """
        dimless_radius = np.asarray(dimless_radius)
        radius_sch = DiskTools.get_radius_sch(par=par)
        radius = dimless_radius * radius_sch
        return radius

    @staticmethod
    def get_accrate_fromdimless(*, par: DiskParams, dimless_accrate: float) -> float:
        """Convert a dimensionless accretion rate to a physical accretion rate.

        Parameters
        ----------
        par : DiskParams
            An object containing the adjustable parameters of the accretion disk.
        dimless_accrate : float
            Dimensionless accretion rate (relative to the Eddington rate).

        Returns
        -------
        float
            Physical accretion rate in CGS units.

        """
        accrate_edd = DiskTools.get_accrate_edd(par=par)
        accrate = accrate_edd * dimless_accrate
        return accrate
