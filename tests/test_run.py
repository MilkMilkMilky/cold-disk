import matplotlib.pyplot as plt
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
    angmomin = SlimDisk.get_slim_angmomin(par=para, dimless_angmomin=1.839)
    # solve, solveinfo = SlimDisk.slim_disk_solver(par=para)
    # dimless_radius_solve_array = solve[0]
    # angmom_solve_array, coffeta_solve_array = solve[1], solve[2]
    # rveltosvel_solve_array = SlimDisk.get_slim_rveltosvel_fromfirst(
    #     par=para,
    #     dimless_radius=dimless_radius_solve_array,
    #     angmom=angmom_solve_array,
    #     coff_eta=coffeta_solve_array,
    #     angmomin=lin,
    # )
    # rveltosvel_solve_array = np.asarray(rveltosvel_solve_array)
    # print(solveinfo)
    # print(max(rveltosvel_solve_array))
    # print(max(coffeta_solve_array))
    solveresult, solveinfo = SlimDisk.slim_disk_solver(par=para)
    print(solveinfo)
    x_data = np.log10(solveresult["dimless_radius"])
    numerator = solveresult["dangmom_numerator"]
    denominator = solveresult["dangmom_denominator"]

    scale_factor = 1e10
    denominator_scaled = denominator * scale_factor

    plt.figure(figsize=(8, 5))
    plt.plot(x_data, numerator, lw=2, label="dangmom_numerator")
    plt.plot(x_data, denominator_scaled, lw=2, label=f"dangmom_denominator × {scale_factor}")
    # plt.xlim(0, 2)
    # plt.ylim(-1e-10, 1e-10)
    plt.yscale("symlog", linthresh=1e-8)
    plt.xlabel("log10(dimless_radius)")
    plt.ylabel("Value")
    plt.title("dangmom_numerator & dangmom_denominator vs dimless_radius")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
