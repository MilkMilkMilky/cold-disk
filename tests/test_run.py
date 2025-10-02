import numpy as np
import scipy as sp

from disk_solver import DiskParams, DiskTools, SlimDisk, StandardDisk, cgs_consts
from parameters import model_params

if __name__ == "__main__":
    para = DiskParams(
        alpha_viscosity=model_params.alpha_viscosity[0],
        dimless_accrate=model_params.dimless_accrate[0],
        dimless_bhmass=model_params.dimless_bhmass[0],
        gas_index=model_params.gas_index[0],
        wind_index=model_params.wind_index[0],
        dimless_radius_in=model_params.dimless_radius_in[0],
        dimless_radius_out=model_params.dimless_radius_out[0],
    )
    lin = SlimDisk.get_slim_angmomin(par=para, dimless_angmomin=1.7)
    solve, solveinfo = SlimDisk.slim_disk_integrator(par=para, angmomin=lin)
    print(solve)
    print(solveinfo)
