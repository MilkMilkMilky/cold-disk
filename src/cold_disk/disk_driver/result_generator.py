from pathlib import Path
from typing import cast

import h5py
import numpy as np

from cold_disk.disk_solver import DiskParams, SlimDisk, StandardDisk

__all__ = ["ResultGeneratorTools"]

class ResultGeneratorTools:
    @staticmethod
    def slimdisk_normalresult_generator(*, hdf5_file_path: Path) -> None:
        """Compute and store slim-disk solutions for all tasks in an HDF5 file.

        This routine iterates over all parameter sets defined in the 'adjparamspace'
        dataset of the given HDF5 file, solves the slim disk equations using
        `SlimDisk.slim_disk_odeint_solver`, and stores the results in the corresponding
        task groups. It also updates the task completion state and logs any failures.

        The function handles the following operations per task:
          1. Skip tasks already marked as completed in the 'taskstate' dataset.
          2. Convert the stored parameter values into a `DiskParams` instance.
          3. Solve the slim disk ODE system via `SlimDisk.slim_disk_odeint_solver`.
          4. Store the structured result array in a subgroup 'resultnormal' under the task group.
          5. Store the solver metadata (`slim_solver_info`) alongside the results.
          6. Mark the task as completed in 'taskstate'.
          7. Collect and log any exceptions that occur during solving.

        Parameters
        ----------
        hdf5_file_path : Path
            Path to the HDF5 file containing:
              - 'adjparamspace' dataset with all parameter combinations,
              - 'taskstate' dataset tracking completion of each task,
              - task groups ('task_0', 'task_1', ...) for storing results.

        Returns
        -------
        None
            The function updates the HDF5 file in place and writes a log file
            `<hdf5_file_stem>_solver.log` in the same directory. No value is returned.

        Raises
        ------
        FileNotFoundError
            If `hdf5_file_path` does not exist.
        RuntimeError
            If 'adjparamspace' is missing or task groups do not match expected names.
        TypeError
            If datasets or groups are not of expected h5py types.

        Notes
        -----
        - Solver metadata `slim_solver_info` is stored as a structured NumPy array
          containing iteration count, final dimensionless inner angular momentum,
          and success flag.
        - Any failed tasks are logged with task ID and exception message.

        """
        if not hdf5_file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_file_path}")

        if not hdf5_file_path.stem.startswith("slimdiskdata_"):
            raise ValueError(f"HDF5 file {hdf5_file_path.name} does not appear to be a slim disk data file")

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

    @staticmethod
    def slimdisk_radiationresult_generator(*, hdf5_file_path: Path) -> None:
        """Compute and store slim-disk radiation outputs for all previously solved tasks in an HDF5 file.

        This routine iterates over all tasks marked as completed in the 'taskstate' dataset
        of the given HDF5 file, retrieves the previously computed slim disk solutions,
        calculates spectral energy distributions (SED) and related radiation properties using
        `SlimDisk.slim_disk_sed_solver`, and stores the results under 'resultradiation'
        subgroups in the corresponding task groups. Failed tasks are logged.

        The function handles the following operations per task:
          1. Skip tasks not yet marked as completed in the 'taskstate' dataset.
          2. Retrieve the `DiskParams` from the 'adjparamspace' dataset.
          3. Extract previously computed slim disk solution arrays ('dimless_radius', 'fluxz').
          4. Compute the SED and bolometric luminosity (including disk radiative efficiency).
          5. Store SED ('logfrequency' and 'sed') and luminosity ('lbol_and_eff') in 'resultradiation'.
          6. Collect and log any exceptions that occur during the radiation calculation.

        Parameters
        ----------
        hdf5_file_path : Path
            Path to the HDF5 file containing:
              - 'adjparamspace' dataset with all parameter combinations,
              - 'taskstate' dataset indicating which tasks have completed,
              - task groups ('task_0', 'task_1', ...) containing 'resultnormal' data.

        Returns
        -------
        None
            The function updates the HDF5 file in place and writes a log file
            `<hdf5_file_stem>_radiation.log` in the same directory. No value is returned.

        Raises
        ------
        FileNotFoundError
            If `hdf5_file_path` does not exist.
        RuntimeError
            If 'adjparamspace' is missing or task groups do not match expected names.
        TypeError
            If datasets or groups are not of expected h5py types.

        Notes
        -----
        - Only tasks with `taskstate=True` are processed; others are skipped.
        - Existing 'resultradiation' subgroups are overwritten.
        - The SED calculation integrates only over dimensionless radii > 3 to avoid
          instability within the ISCO region; including smaller radii would produce
          erroneous SED, bolometric luminosity, and radiative efficiency values.
        - Compression is not explicitly applied in this function; datasets are written as-is.
        - Failed tasks are logged with task ID and exception message.

        """
        if not hdf5_file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_file_path}")

        if not hdf5_file_path.stem.startswith("slimdiskdata_"):
            raise ValueError(f"HDF5 file {hdf5_file_path.name} does not appear to be a slim disk data file")

        log_path = hdf5_file_path.parent / f"{hdf5_file_path.stem}_radiation.log"
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
            taskstate_ds = h5file["taskstate"]
            if not isinstance(taskstate_ds, h5py.Dataset):
                raise TypeError("'taskstate' must be an h5py.Dataset")
            for task_id in task_ids:
                if not taskstate_ds[task_id]:
                    continue
                group_name = f"task_{task_id}"
                task_group = h5file[group_name]
                if not isinstance(task_group, h5py.Group):
                    raise TypeError(f"{group_name} is not an h5py.Group")
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
                resultnormal = cast("h5py.Group", task_group["resultnormal"])
                dimless_radius_ds = cast("h5py.Dataset", resultnormal["dimless_radius"])
                dimless_radius = dimless_radius_ds[:]
                fluxz_ds = cast("h5py.Dataset", resultnormal["fluxz"])
                fluxz = fluxz_ds[:]
                try:
                    sed_output, lum_output = SlimDisk.slim_disk_sed_solver(
                        par=par,
                        dimless_radius=dimless_radius,
                        fluxz=fluxz,
                    )
                    if "resultradiation" in task_group:
                        del task_group["resultradiation"]
                    result_group = task_group.create_group("resultradiation")
                    result_group.create_dataset("logfrequency", data=sed_output["logfrequency"])
                    result_group.create_dataset("sed", data=sed_output["sed"])
                    result_group.create_dataset("lbol_and_eff", data=lum_output)
                except Exception as e:
                    failed_tasks.append((task_id, str(e)))
                    continue
        with log_path.open("w") as f:
            if failed_tasks:
                for task_id, err in failed_tasks:
                    f.write(f"Task {task_id} failed: {err}\n")
            else:
                f.write("All radiation calculation computed successfully.\n")
        return None
