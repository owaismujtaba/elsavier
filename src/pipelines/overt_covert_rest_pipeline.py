import os
import numpy as np
import pandas as pd
from pathlib import Path

import config as config
from src.decoding.overt_covert_rest import SpeechEEGDatasetLoader
from src.decoding.overt_covert_rest_model import OvertCoverRestClassifier


class OvertCovertRestPipeline:
    def __init__(self, subject_id='01', session_id='01'):
        self.subject_id = subject_id
        self.session_id = session_id
        self.output_dir = Path(config.CURR_DIR, 'DecodingResults')
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = None
        self.history = None

    def _get_condition_config(self, trial_mode, trial_type):
        return {
            "trial_mode": trial_mode,
            "trial_unit": 'Words',
            "experiment_mode": 'Experiment',
            "trial_boundary": 'Start',
            "trial_type": trial_type,
            "modality": '',
            "tmin": -0.2,
            "tmax": 1.0
        }

    def _load_condition_data(self, label, config):
        loader = SpeechEEGDatasetLoader(
            subject_id=self.subject_id,
            session_id=self.session_id,
            label=label,
            condition_config=config
        )
        return loader.get_data()

    def load_data(self):
        overt_cfg = self._get_condition_config('Real', 'Speech')
        covert_cfg = self._get_condition_config('Silent', 'Speech')
        rest_cfg = self._get_condition_config('', 'Fixation')

        overt, overt_labels = self._load_condition_data(0, overt_cfg)
        covert, covert_labels = self._load_condition_data(1, covert_cfg)
        rest, rest_labels = self._load_condition_data(2, rest_cfg)

        self.X = np.concatenate([overt, covert, rest], axis=0)
        self.y = np.concatenate([overt_labels, covert_labels, rest_labels], axis=0)

        print(f"Data loaded: {self.X.shape[0]} samples, {self.X.shape[1]} channels, {self.X.shape[2]} timepoints")

    def train(self, test_split=0.2):
        input_shape = (self.X.shape[1], self.X.shape[2])
        self.model = OvertCoverRestClassifier(inputShape=input_shape)
        self.model.compileModel()
        self.model.summary()
        self.history = self.model.trainWithSplit(self.X, self.y, validationSplit=test_split)
        print("Training completed.")

    def save_history(self):
        if self.history is None:
            print("No history to save.")
            return
        hist_df = pd.DataFrame(self.history.history)
        filename = f"sub-{self.subject_id}_ses-{self.session_id}_overt_covert_rest.csv"
        filepath = Path(self.output_dir, filename)
        hist_df.to_csv(filepath, index=False)
        print(f"Training history saved to {filepath}")

    def run(self):
        if not config.OVERT_COVERT_REST_CLASSIFICATION:
            print("Pipeline not enabled in config.")
            return
        self.load_data()
        self.train()
        self.save_history()
