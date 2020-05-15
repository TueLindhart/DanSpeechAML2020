import os
import random
from pathlib import Path

import numpy as np
import pandas as pd


class DatasetFactory:

    def __init__(self, root_dir):
        self.root_dir = root_dir
        csv_files = [f for f in os.listdir(self.root_dir) if ".csv" in f]
        assert len(
            csv_files) == 1, "Multiple csv files present in specified data directory"
        csv_file = csv_files[0]

        self.meta = pd.read_csv(os.path.join(self.root_dir, csv_file), encoding="utf-8")

    def split_file(self, train_size):
        np.random.seed(42)
        mask = np.random.rand(len(self.meta)) < train_size
        train = self.meta[mask]
        validation = self.meta[~mask]

        Path(os.path.join(self.root_dir, 'sets')).mkdir(parents=True, exist_ok=True)

        train.to_csv(os.path.join(self.root_dir, 'sets', 'train.csv'), sep=',', index=False)
        validation.to_csv(os.path.join(self.root_dir, 'sets', 'validation.csv'), sep=',', index=False)



