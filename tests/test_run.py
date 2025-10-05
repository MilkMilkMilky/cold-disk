from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

from disk_solver import DiskParams, DiskTools, SlimDisk, StandardDisk, cgs_consts
from parameters import model_params


def slimtest_save_csv(par: DiskParams):
    solveresult, solveinfo = SlimDisk.slim_disk_solver(par=par)
    print(solveinfo)
    print(np.max(solveresult["rveltosvel"]))
    results_df = pd.DataFrame(solveresult)
    results_df = results_df.set_index("dimless_radius")
    save_path = Path("./slim_results.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(save_path, index=True)


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
    slimtest_save_csv(para)
