import os
import glob

import numpy as np

from typing import List

from .dataloader import dataset_base
from easydict import EasyDict
from pathlib import Path
from util.misc import validate_plot_filename, filter_files_by_id_and_num
from util import Mode
from .build import DATASETS


@DATASETS.register_module()
class EvoMS(dataset_base):
    def __init__(self, config: EasyDict):
        super().__init__(config)
        self._filenames = self._get_filenames()
        self._is_labeled = self._init_is_labeled()

    def _init_is_labeled(self) -> np.ndarray:
        plot_ids = [validate_plot_filename(file)[1] for file in self.filenames]
        is_labeled = np.isin(plot_ids, self.config.LABELED_IDS)
        return is_labeled

    @property
    def is_labeled(self):
        return self._is_labeled

    @is_labeled.setter
    def is_labeled(self, new_indices: np.ndarray):
        self._is_labeled = new_indices

    def _get_filenames(self) -> np.ndarray:
        """Get a list of filenames that correspond to the given mode

        Args:
            mode (Mode): mode for which to get the filenames

        Returns:
            np.ndarray: list of filenames
        """
        path_list = sorted(
            glob.glob(
                os.path.join(self.input_path, "*" + self.config.FILENAME_EXTENSION)
            )
        )
        file_list = [
            Path(path).name[: -len(self.config.FILENAME_EXTENSION)]
            for path in path_list
        ]
        filtered_file_list = filter_files_by_id_and_num(
            file_list,
            self.config.IDS,
            self.config.TEST_SET,
            self.mode is not Mode.test,
            False,
        )

        return filtered_file_list

    @property
    def filenames(self):
        return self._filenames

    @filenames.setter
    def filenames(self, new_names: List[str]):
        self._filenames = new_names
