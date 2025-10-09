"""Module `cold_disk.disk_driver.result_generator`.

Provides tools for computing and storing accretion disk solutions in HDF5 files.
This module handles batch computation of disk models, result storage, and
completion tracking for parameter space exploration.

The primary class is `ResultGeneratorTools`, which offers static methods for
computing disk solutions and organizing results in structured HDF5 datasets.

Notes:
-----
- HDF5 files must contain 'adjparamspace' and 'taskstate' datasets for operation.
- Task completion is tracked to enable resumable computations.
- Results are stored in compressed HDF5 datasets with proper error handling.
- All methods are static and stateless; no instance of `ResultGeneratorTools` is required.

Example:
-------
>>> from cold_disk import ResultGeneratorTools
>>> from pathlib import Path
>>> # Compute slim disk solutions
>>> h5path = Path("slimdiskdata_20251001.h5")
>>> ResultGeneratorTools.slimdisk_normalresult_generator(hdf5_file_path=h5path)
>>> # Compute radiation outputs
>>> ResultGeneratorTools.slimdisk_radiationresult_generator(hdf5_file_path=h5path)

"""
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from cold_disk.disk_solver import DiskParams, DiskTools, SlimDisk, StandardDisk

__all__ = ["ResultGeneratorTools"]

_ADJPARAMSPACE_MISSING = "HDF5 file missing 'adjparamspace' dataset"
_TASKSTATE_MUST_BE_DATASET = "'taskstate' must be an h5py.Dataset"
_ADJPARAMSPACE_MUST_BE_DATASET = "'adjparamspace' must be an h5py.Dataset"
_ERR_GROUP_MISMATCH = "HDF5 file incomplete or has extra task groups: mismatch with paraspace"


class ResultGeneratorTools:
    """Collection of static methods for computing and storing accretion disk solutions.

    This class provides tools for batch computation of disk models, result storage,
    and completion tracking. It works with HDF5 files containing parameter spaces
    and manages the computation workflow for both standard and slim disk models.

    Methods include:

    - `slimdisk_normalresult_generator`: Compute and store slim disk structure solutions.
    - `slimdisk_radiationresult_generator`: Compute and store slim disk radiation outputs.
    - `standarddisk_normalresult_generator`: Compute and store standard disk solutions.

    All methods operate on HDF5 files with pre-initialized parameter spaces and
    provide robust error handling and logging for batch computations.

    Notes
    -----
    - HDF5 files must contain 'adjparamspace' dataset with parameter combinations
      and 'taskstate' dataset for completion tracking.
    - Task groups ('task_0', 'task_1', etc.) are created automatically for result storage.
    - Failed computations are logged with task ID and error messages.
    - Results are stored as compressed HDF5 datasets for efficient storage.
    - All main methods skip already completed tasks to enable resumable computations.

    """

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
            The file name must start with ``slimdiskdata_`` to be considered valid.

        Returns
        -------
        None
            The function updates the HDF5 file in place and writes a log file
            `<hdf5_file_stem>_solver.log` in the same directory. No value is returned.

        Raises
        ------
        FileNotFoundError
            If `hdf5_file_path` does not exist.
        ValueError
            If `hdf5_file_path` does not appear to be a slim disk data file.
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
                raise RuntimeError(_ADJPARAMSPACE_MISSING)
            adjparamspace_ds = h5file["adjparamspace"]
            if not isinstance(adjparamspace_ds, h5py.Dataset):
                raise TypeError(_ADJPARAMSPACE_MUST_BE_DATASET)
            task_ids = adjparamspace_ds.fields("id")[:]
            task_group_names = [f"task_{task_id}" for task_id in task_ids]
            existing_objects = list(h5file.keys())
            existing_groups = [name for name in existing_objects if name.startswith("task_")]
            if set(existing_groups) != set(task_group_names):
                raise RuntimeError(_ERR_GROUP_MISMATCH)
            for task_id in task_ids:
                group_name = f"task_{task_id}"
                task_group = h5file[group_name]
                if not isinstance(task_group, h5py.Group):
                    raise TypeError(f"{group_name} is not an h5py.Group")
                taskstate_ds = h5file["taskstate"]
                if not isinstance(taskstate_ds, h5py.Dataset):
                    raise TypeError(_TASKSTATE_MUST_BE_DATASET)
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
                f.write("All slim disk tasks computed successfully.\n")

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
            The file name must start with ``slimdiskdata_`` to be considered valid.

        Returns
        -------
        None
            The function updates the HDF5 file in place and writes a log file
            `<hdf5_file_stem>_radiation.log` in the same directory. No value is returned.

        Raises
        ------
        FileNotFoundError
            If `hdf5_file_path` does not exist.
        ValueError
            If `hdf5_file_path` does not appear to be a slim disk data file.
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
                raise RuntimeError(_ADJPARAMSPACE_MISSING)
            adjparamspace_ds = h5file["adjparamspace"]
            if not isinstance(adjparamspace_ds, h5py.Dataset):
                raise TypeError(_ADJPARAMSPACE_MUST_BE_DATASET)
            task_ids = adjparamspace_ds.fields("id")[:]
            task_group_names = [f"task_{task_id}" for task_id in task_ids]
            existing_objects = list(h5file.keys())
            existing_groups = [name for name in existing_objects if name.startswith("task_")]
            if set(existing_groups) != set(task_group_names):
                raise RuntimeError(_ERR_GROUP_MISMATCH)
            taskstate_ds = h5file["taskstate"]
            if not isinstance(taskstate_ds, h5py.Dataset):
                raise TypeError(_TASKSTATE_MUST_BE_DATASET)
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

    @staticmethod
    def standarddisk_normalresult_generator(*, hdf5_file_path: Path) -> None:
        """Compute and store standard-disk (SSD) solutions for all tasks in an HDF5 file.

        This routine iterates over all parameter sets defined in the 'adjparamspace'
        dataset of the given HDF5 file, solves the standard accretion disk
        structure using `StandardDisk.get_standard_solve_result`, and stores
        the results under corresponding task groups. It updates the
        task completion state and logs any failures.

        The function handles the following operations per task:
          1. Skip tasks already marked as completed in the 'taskstate' dataset.
          2. Convert the stored parameter values into a `DiskParams` instance.
          3. Construct the dimensionless and physical radius arrays.
          4. Solve the standard-disk structure equations via
             `StandardDisk.get_standard_solve_result`.
          5. Store the computed quantities in the subgroup 'resultnormal'
             under the corresponding task group.
          6. Mark the task as completed in 'taskstate'.
          7. Collect and log any exceptions that occur during solving.

        Parameters
        ----------
        hdf5_file_path : Path
            Path to the HDF5 file containing:
              - 'adjparamspace' dataset with all parameter combinations,
              - 'taskstate' dataset tracking completion of each task,
              - task groups ('task_0', 'task_1', ...) for storing results.
            The file name must start with ``standarddiskdata_`` to be considered valid.

        Returns
        -------
        None
            The function updates the HDF5 file in place and writes a log file
            `<hdf5_file_stem>_solver.log` in the same directory. No value is returned.

        Raises
        ------
        FileNotFoundError
            If `hdf5_file_path` does not exist.
        ValueError
            If `hdf5_file_path` does not appear to be a standard disk data file.
        RuntimeError
            If 'adjparamspace' is missing or task groups do not match expected names.
        TypeError
            If datasets or groups are not of expected h5py types.

        Notes
        -----
        - Each result group 'resultnormal' includes:
            * `dimless_radius`: array of dimensionless radii.
            * `radius`: array of physical radii.
            * Additional physical quantities computed by
              `StandardDisk.get_standard_solve_result`, stored as compressed datasets.
        - The solver operates over radii from `dimless_radius_out` down to
          `dimless_radius_in` with a fixed step of 0.1.
        - Any failed tasks are logged with their task ID and exception message.

        """
        if not hdf5_file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_file_path}")

        if not hdf5_file_path.stem.startswith("standarddiskdata_"):
            raise ValueError(f"HDF5 file {hdf5_file_path.name} does not appear to be a standard disk data file")

        log_path = hdf5_file_path.parent / f"{hdf5_file_path.stem}_solver.log"
        failed_tasks = []

        with h5py.File(hdf5_file_path, "a") as h5file:
            if "adjparamspace" not in h5file:
                raise RuntimeError(_ADJPARAMSPACE_MISSING)
            adjparamspace_ds = h5file["adjparamspace"]
            if not isinstance(adjparamspace_ds, h5py.Dataset):
                raise TypeError(_ADJPARAMSPACE_MUST_BE_DATASET)
            task_ids = adjparamspace_ds.fields("id")[:]
            task_group_names = [f"task_{task_id}" for task_id in task_ids]
            existing_objects = list(h5file.keys())
            existing_groups = [name for name in existing_objects if name.startswith("task_")]
            if set(existing_groups) != set(task_group_names):
                raise RuntimeError(_ERR_GROUP_MISMATCH)
            taskstate_ds = h5file["taskstate"]
            if not isinstance(taskstate_ds, h5py.Dataset):
                raise TypeError(_TASKSTATE_MUST_BE_DATASET)
            for task_id in task_ids:
                if taskstate_ds[task_id]:
                    continue
                group_name = f"task_{task_id}"
                task_group = h5file[group_name]
                if not isinstance(task_group, h5py.Group):
                    raise TypeError(f"{group_name} is not an h5py.Group")
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
                    dimless_radius_array = np.arange(
                        par.dimless_radius_out,
                        par.dimless_radius_in - 1e-8,
                        -0.1,
                    )
                    radius_array = DiskTools.get_radius_fromdimless(par=par, dimless_radius=dimless_radius_array)
                    ssd_result_array = StandardDisk.get_standard_solve_result(
                        par=par,
                        dimless_radius=dimless_radius_array,
                    )
                    if "resultnormal" in task_group:
                        del task_group["resultnormal"]
                    result_group = task_group.create_group("resultnormal")
                    result_group.create_dataset(
                        name="dimless_radius",
                        data=dimless_radius_array,
                        compression="gzip",
                        compression_opts=1,
                    )
                    result_group.create_dataset(
                        name="radius",
                        data=radius_array,
                        compression="gzip",
                        compression_opts=1,
                    )
                    for name in ssd_result_array.dtype.names:
                        result_group.create_dataset(
                            name,
                            data=ssd_result_array[name],
                            compression="gzip",
                            compression_opts=1,
                        )
                    taskstate_ds[task_id] = True
                except Exception as e:
                    failed_tasks.append((task_id, str(e)))
                    continue

        with log_path.open("w") as f:
            if failed_tasks:
                for task_id, err in failed_tasks:
                    f.write(f"Task {task_id} failed: {err}\n")
            else:
                f.write("All standard disk tasks computed successfully.\n")

        return None
