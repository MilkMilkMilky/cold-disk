from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from parameters import model_params


@dataclass
class _AdjustableParams:
    alpha_viscosity: np.ndarray
    dimless_accrate: np.ndarray
    dimless_bhmass: np.ndarray
    gas_index: np.ndarray
    wind_index: np.ndarray
    dimless_radius_in: np.ndarray
    dimless_radius_out: np.ndarray


class ParaspaceGeneratorTools:
    @staticmethod
    def get_current_utcdate() -> str:
        utcdate_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
        return utcdate_str

    @staticmethod
    def load_slimdisk_datafiles(*, data_date: str | None = None) -> Path:
        if data_date is None:
            data_date = ParaspaceGeneratorTools.get_current_utcdate()

        if not isinstance(data_date, str) or len(data_date) != 8 or not data_date.isdigit():
            raise ValueError(f"data_date must be 'YYYYMMDD', got: {data_date}")

        current_file_dir = Path(__file__).resolve().parent
        project_root = (current_file_dir.parent.parent).resolve()
        data_dir = project_root / "data"
        slimdiskdata_dir = data_dir / "slimdiskdata"
        target_dir_name = f"slimdiskdata_{data_date}"
        target_dir = slimdiskdata_dir / target_dir_name
        hdf5_file_path = target_dir / f"{target_dir_name}.h5"
        target_dir.mkdir(parents=True, exist_ok=True)

        if not hdf5_file_path.exists():
            h5py.File(hdf5_file_path, "w").close()

        return hdf5_file_path

    @staticmethod
    def load_adjparams_default() -> _AdjustableParams:
        adjparams = _AdjustableParams(
            alpha_viscosity=model_params.alpha_viscosity,
            dimless_accrate=model_params.dimless_accrate,
            dimless_bhmass=model_params.dimless_bhmass,
            gas_index=model_params.gas_index,
            wind_index=model_params.wind_index,
            dimless_radius_in=model_params.dimless_radius_in,
            dimless_radius_out=model_params.dimless_radius_out,
        )
        return adjparams

    @staticmethod
    def adjparams_dispatcher(adjparams_obj: _AdjustableParams, dispatch_mode: str) -> np.ndarray:
        allowed_modes = ("parasweep", "pairscan", "fullfactorial")
        if dispatch_mode not in allowed_modes:
            raise ValueError(f"dispatch_mode must be one of {allowed_modes}, got '{dispatch_mode}'")

        adjparams_dict = {field.name: getattr(adjparams_obj, field.name) for field in fields(adjparams_obj)}
        adjparams_names = list(adjparams_dict.keys())

        if dispatch_mode == "parasweep":
            lengths = [len(value) for value in adjparams_dict.values()]
            num_multi_value_params = sum(length_value > 1 for length_value in lengths)
            if num_multi_value_params != 1:
                raise ValueError(
                    "In 'parasweep' mode, exactly one adjustable parameter must have length > 1; "
                    "all other parameters must have length 1.",
                )
            max_length = max(lengths)
            adjparams_arrays = [
                np.full(max_length, value[0]) if len(value) == 1 else value for value in adjparams_dict.values()
            ]
            adjparams_matrix = np.column_stack(adjparams_arrays)

        elif dispatch_mode == "pairscan":
            lengths = [len(value) for value in adjparams_dict.values()]
            if len(set(lengths)) != 1:
                raise ValueError("In 'pairscan' mode, all adjustable parameters must have the same length.")
            adjparams_matrix = np.column_stack(list(adjparams_dict.values()))

        elif dispatch_mode == "fullfactorial":
            grids = np.meshgrid(*adjparams_dict.values(), indexing="ij")
            adjparams_matrix = np.column_stack([grid.ravel() for grid in grids])

        dtype = [("id", "i4")] + [(name, "f8") for name in adjparams_names]
        num_tasks = adjparams_matrix.shape[0]
        adjparams_space = np.zeros(num_tasks, dtype=dtype)
        adjparams_space["id"] = np.arange(num_tasks)
        for index, name in enumerate(adjparams_names):
            adjparams_space[name] = adjparams_matrix[:, index]

        return adjparams_space


if __name__ == "__main__":
    ...
