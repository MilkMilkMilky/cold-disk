from pathlib import Path
from typing import cast

import h5py
import numpy as np

from cold_disk.disk_solver import DiskParams, SlimDisk, StandardDisk


class ResultGeneratorTools:
    @staticmethod
    def slimdisk_normalresult_generator(*, hdf5_file_path: Path) -> None:
        if not hdf5_file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_file_path}")
        log_path = hdf5_file_path.parent / f"{hdf5_file_path.stem}_solver.log"
        failed_tasks = []

        with h5py.File(hdf5_file_path, "a") as h5file:
            if "adjparamspace" not in h5file:
                raise RuntimeError("HDF5 file missing 'adjparamspace' dataset")
            adjparamspace_ds = h5file["adjparamspace"]
            if not isinstance(adjparamspace_ds, h5py.Dataset):
                raise TypeError("'adjparamspace' must be an h5py.Dataset")
            task_ids = adjparamspace_ds.fields("id")[:]
            task_group_names = [f"task_{task_id}" for task_id in task_ids]
            existing_objects = list(h5file.keys())
            existing_groups = [name for name in existing_objects if name.startswith("task_")]
            if set(existing_groups) != set(task_group_names):
                raise RuntimeError("HDF5 file incomplete or has extra task groups: mismatch with paraspace")
            for task_id in task_ids:
                group_name = f"task_{task_id}"
                task_group = h5file[group_name]
                if not isinstance(task_group, h5py.Group):
                    raise TypeError(f"{group_name} is not an h5py.Group")
                taskstate_ds = h5file["taskstate"]
                if not isinstance(taskstate_ds, h5py.Dataset):
                    raise TypeError("'taskstate' must be an h5py.Dataset")
                if taskstate_ds[task_id]:
                    continue
                try:
                    param_values = adjparamspace_ds[task_id]
                    par = DiskParams(
                        alpha_viscosity=param_values["alpha_viscosity"],
                        dimless_accrate=param_values["dimless_accrate"],
                        dimless_bhmass=param_values["dimless_bhmass"],
                        gas_index=param_values["gas_index"],
                        wind_index=param_values["wind_index"],
                        dimless_radius_in=param_values["dimless_radius_in"],
                        dimless_radius_out=param_values["dimless_radius_out"],
                    )
                    slim_solver_result, slim_solver_info = SlimDisk.slim_disk_odeint_solver(par=par)
                    if "resultnormal" in task_group:
                        del task_group["resultnormal"]
                    result_group = task_group.create_group("resultnormal")
                    for name in slim_solver_result.dtype.names:
                        result_group.create_dataset(
                            name,
                            data=slim_solver_result[name],
                            compression="gzip",
                            compression_opts=1,
                        )
                    result_group.create_dataset(
                        "slimsolveinfo",
                        data=slim_solver_info,
                        compression="gzip",
                        compression_opts=1,
                    )
                    taskstate_ds[task_id] = True
                except Exception as e:
                    failed_tasks.append((task_id, str(e)))

        with log_path.open("w") as f:
            if failed_tasks:
                for task_id, err in failed_tasks:
                    f.write(f"Task {task_id} failed: {err}\n")
            else:
                f.write("All tasks computed successfully.\n")

        return None
