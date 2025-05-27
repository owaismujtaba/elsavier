import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.dataset.data_reader import BIDSDatasetReader
from src.dataset.data_loader import DataLoader
from src.utils.graphics import styled_print

import config as config

import pdb

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from src.dataset.data_reader import BIDSDatasetReader
from src.dataset.data_loader import DataLoader
from src.utils.graphics import styled_print

import config as config


import numpy as np
from pathlib import Path
from src.dataset.data_reader import BIDSDatasetReader
from src.dataset.data_loader import DataLoader
from src.utils.graphics import styled_print
import config


class EEGEpochExtractor:
    """
    Handles EEG data loading and epoch creation for visual and fixation trials.
    """
    def __init__(self, tmin=-0.2, tmax=0.5):
        styled_print('', 'Initializing VisualEpochExtractor', color='red', panel=True)
        self.tmin = tmin
        self.tmax = tmax
        self.raw_data = {}
        self.subjects = []
        self.sessions = []

        self._load_data()

    def _load_data(self):
        styled_print('', 'Loading raw EEG data', color='green')
        for _, sub_id, ses_id in config.filepaths:
            try:
                styled_print('', f'Loading sub-{sub_id}, ses-{ses_id}', color='green')
                reader = BIDSDatasetReader(sub_id=sub_id, ses_id=ses_id)
                raw = reader.raw
                if raw is not None:
                    self.subjects.append(sub_id)
                    self.sessions.append(ses_id)
                    self.raw_data[(sub_id, ses_id)] = raw
                else:
                    styled_print('', f'No raw data for sub-{sub_id}, ses-{ses_id}', color='yellow')
            except Exception as e:
                styled_print('', f'Error loading sub-{sub_id}, ses-{ses_id}: {e}', color='red')

    def create_epochs(
            self, trial_mode, trial_unit, experiment_mode, 
            trial_boundary, trial_type, modality,  
            tmin=None, tmax=None,):
        """
        Creates epochs for all subjects/sessions for a given trial type.
        """
        tmin = self.tmin if tmin is None else tmin
        tmax = self.tmax if tmax is None else tmax
        epochs = {}
        
        for (sub_id, ses_id), raw in self.raw_data.items():
            styled_print('', f'Creating epochs: sub-{sub_id}, ses-{ses_id}', color='green')
            try:
                loader = DataLoader(
                    eeg_data=raw, trial_mode=trial_mode, trial_type=trial_type,
                    trial_unit=trial_unit, experiment_mode=experiment_mode,
                    trial_boundary=trial_boundary, modality=modality
                )
                epochs[(sub_id, ses_id)] = loader.create_epochs(tmin=tmin, tmax=tmax)
            except Exception as e:
                styled_print('', f'Skipping sub-{sub_id}, ses-{ses_id} due to: {e}', color='yellow')

        return epochs















class VisualRestExtractor:
    """
    Extracts and plots evoked EEG responses in occipital channels
    for pictorial vs. fixation events across multiple subjects/sessions.
    """
    def __init__(self, raw=None, tmin=-0.2, tmax=0.5):
        styled_print('', 'Initializing VisualRestExtractor', color='red', panel=True)

        self.raw = raw
        self.annotations = getattr(raw, 'annotations', [])
        self.tmin = tmin
        self.tmax = tmax

        self.subjects = []
        self.sessions = []
        self.raw_data = {}

        if self.raw is None:
            self.load_all_data()

    def get_visual_events(self, include=('Visual',), exclude=None):
        exclude = exclude or []
        return [evt for evt in self.annotations
                if all(inc in evt['description'] for inc in include)
                and all(exc not in evt['description'] for exc in exclude)]

    def load_all_data(self):
        styled_print('', 'Loading raw data for all subjects', color='green')
        for sub_id, ses_id in config.filepaths:
            styled_print('', f'Loading raw data for sub-{sub_id}, ses-{ses_id}', color='green')
            try:
                bids = BIDSDatasetReader(sub_id=sub_id, ses_id=ses_id)
                raw = bids.raw
                if raw is not None:
                    self.subjects.append(sub_id)
                    self.sessions.append(ses_id)
                    self.raw_data[(sub_id, ses_id)] = raw
                else:
                    styled_print('', f'No raw data for sub-{sub_id} ses-{ses_id}', color='yellow')
            except Exception as e:
                styled_print('', f'Error loading sub-{sub_id}, ses-{ses_id}: {e}', color='red')

    def create_epochs_for_all(self,
                               trial_mode, trial_unit, experiment_mode,
                               trial_boundary, trial_type, modality,
                               tmin=None, tmax=None):
        tmin = self.tmin if tmin is None else tmin
        tmax = self.tmax if tmax is None else tmax
        epochs = {}

        for (sub_id, ses_id), raw in self.raw_data.items():
            styled_print('', f'Creating epochs for sub-{sub_id}, ses-{ses_id}', color='green')
            try:
                loader = DataLoader(
                    eeg_data=raw,
                    trial_mode=trial_mode,
                    trial_unit=trial_unit,
                    experiment_mode=experiment_mode,
                    trial_boundary=trial_boundary,
                    trial_type=trial_type,
                    modality=modality
                )
                epochs[(sub_id, ses_id)] = loader.create_epochs(tmin=tmin, tmax=tmax)
            except Exception as e:
                styled_print('', f'Skipping sub-{sub_id}, ses-{ses_id} due to error: {e}', color='yellow')

        return epochs

    def plot_occipital_all_subjects(self,
                                    trial_mode='Silent', trial_unit='Words',
                                    experiment_mode='Experiment', trial_boundary='Start',
                                    trial_type='Stimulus', modality='Pictures'):
        pict_epochs = self.create_epochs_for_all(
            trial_mode, trial_unit, experiment_mode,
            trial_boundary, trial_type, modality,
            tmin=-0.2, tmax=0.5
        )
        fix_epochs = self.create_epochs_for_all(
            trial_mode, trial_unit, experiment_mode,
            trial_boundary, 'Fixation', modality,
            tmin=0.3, tmax=1.0
        )

        if not pict_epochs or not fix_epochs:
            styled_print('', 'No epochs to plot.', color='red')
            return

        occ_channels = ['PO3', 'POz', 'PO4']
        total_subjects = len(pict_epochs)
        cols = 4
        rows = int(np.ceil(total_subjects / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), sharex=True, sharey=True)
        axes = axes.flatten()

        for idx, ((sub_id, ses_id), pict_epo) in enumerate(pict_epochs.items()):
            fix_epo = fix_epochs.get((sub_id, ses_id))
            if fix_epo is None:
                continue

            try:
                ev_pict = pict_epo.average().pick(occ_channels)
                ev_fix = fix_epo.average().pick(occ_channels)
            except Exception as e:
                styled_print('', f'Skipping plotting for {sub_id}, {ses_id}: {e}', color='yellow')
                continue

            mean_pict = ev_pict.data.mean(axis=0)
            mean_fix = ev_fix.data.mean(axis=0)
            times = ev_pict.times

            ax = axes[idx]
            ax.plot(times, mean_pict, label='Pictorial', color='green')
            ax.plot(times, mean_fix, label='Fixation', color='blue')
            ax.axvline(0, linestyle='--', color='cyan', label='Onset')
            ax.axvline(0.1, linestyle='--', color='red', label='100ms')
            ax.axvline(0.3, linestyle='--', color='black', label='300ms')

            ax.set_title(f"sub-{sub_id}_ses-{ses_id}")
            if idx % cols == 0:
                ax.set_ylabel('Amplitude (µV)')
            if idx >= (rows - 1) * cols:
                ax.set_xlabel('Time (s)')

        for ax in axes[total_subjects:]:
            ax.axis('off')

        axes[0].legend(loc='upper right')
        plt.tight_layout()

        out_dir = Path(config.IMAGES_DIR) / f"{trial_mode}_{trial_unit}_{experiment_mode}_{trial_boundary}_{trial_type}"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / 'occipital_evoked.png'
        fig.savefig(save_path, dpi=600)
        styled_print('', f'Saved plot to {save_path}', color='green')



class VisualRestExtractor1:
    def __init__(self, raw, tmin=-0.2, tmax=0.5):
        styled_print('', 'Initializing Register Class', color='red', panel=True)
        self.raw = raw
        self.annotations = self.raw.annotations
        self.tmin = tmin
        self.tmax = tmax

        self.load_data()

    def get_visual_events_(self):
        pass

    def _filter_events(self, criteria, exclude=None):
        exclude = exclude or []
        return [event for event in self.annotations 
                if all(criterion in event['description'] for criterion in criteria) 
                and all(excl not in event['description'] for excl in exclude)]
    



    def load_data(self):
        styled_print('', 'Loading raw data for all subjects',color='green')
        files = config.filepaths

        for item in files:
            subject = item[1]
            session = item[2]
            styled_print('', f'Loading raw data for sub-{subject}, ses-{session}', color='green')
            bids_data = BIDSDatasetReader(
                sub_id=subject,
                ses_id=session
            )
            
            raw = bids_data.raw  

            self.subjects.append(subject)
            self.sessions.append(session)
            self.raw_data[(subject, session)] = raw  

        if not self.raw_data:
            print("No raw data found. Check dataset paths or processing steps.")

    def create_epochs_for_all(
            self, trial_mode, trial_unit, experiment_mode, 
            trial_boundary, trial_type, modality, 
            tmin, tmax
        ):
        styled_print('', 'Creating EPOCHS for all subjects',color='red')
        epochs = {}

        for (subject, session), raw in self.raw_data.items():
            styled_print('', f'Loading for sub-{subject}, ses-{session}', color='green')
            data_loader = DataLoader(
                eeg_data=raw, trial_mode=trial_mode, trial_unit=trial_unit,
                experiment_mode=experiment_mode, trial_boundary=trial_boundary,
                trial_type=trial_type, modality=modality
            )
            try:
                epochs[(subject, session)] = data_loader.create_epochs(tmin=tmin, tmax=tmax)
            except:
                continue

        return epochs

    def plot_occipital_all_subjects(
            self, trial_mode='Silent', trial_unit='Words', experiment_mode='Experiment',
            trial_boundary='Start', trial_type='Stimulus',  modality='Pictures'
        ):
        """Plots evoked responses in occipital channels across all subjects."""
        
        


        epochs_pictorial = self.create_epochs_for_all(
            trial_mode=trial_mode, trial_unit=trial_unit,
            experiment_mode=experiment_mode, trial_boundary=trial_boundary,
            trial_type=trial_type, modality=modality,
            tmin=-0.2, tmax=0.5
        )

        epochs_fixation = self.create_epochs_for_all(
            trial_mode=trial_mode, trial_unit=trial_unit,
            experiment_mode=experiment_mode, trial_boundary=trial_boundary,
            trial_type='Fixation', modality=modality,
            tmin=0.3, tmax=1.0
        )
        if not epochs_pictorial and not epochs_fixation:
            print("No epochs available for plotting.")
            return
        
        #return epochs_fixation, epochs_pictorial
        occipital_channels = ['PO3', 'POz', 'PO4']

        fig, axes = plt.subplots(5, 4, figsize=(15, 10), sharex=True, sharey=False)
        axes = axes.flatten()

        for i, ((subject, session), epoch) in enumerate(epochs_pictorial.items()):
            evoked_pictorial = epoch.average()
            occipital_data_pictorial = evoked_pictorial.copy().pick(occipital_channels).data
            mean_signal_pictorial = np.mean(occipital_data_pictorial, axis=0)

            fixation_epoch = epochs_fixation[(subject, session)]
            evoked_fixation = fixation_epoch.average()
            occipital_data_fixation = evoked_fixation.copy().pick(occipital_channels).data
            mean_signal_fixation = np.mean(occipital_data_fixation, axis=0)

            ax = axes[i]
            ax.plot(mean_signal_pictorial, label='Pictorial', color='green')
            ax.plot(mean_signal_fixation, label='Fixation', color='blue')
            ax.axvline(200, color='cyan', linestyle='--', label="onset")
            ax.axvline(300, color='r', linestyle='--', label="100ms)")
            ax.axvline(500, color='black', linestyle='--', label="300ms)")

            ax.set_title(f"sub-{subject}_ses_{session}")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #ax.set_xlim([0, 0.7])
            if i == 0:
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("µV")

        axes[-1].legend(loc='upper right')  
        plt.tight_layout()
        images_dir = config.IMAGES_DIR
        image_filepath = Path(images_dir, f'{trial_mode}_{trial_unit}_{experiment_mode}_{trial_boundary}_{trial_type}_Pictures/Fixation_occipital_p100/p300_plot.png')
        plt.savefig(image_filepath, dpi=600)
        print(f"Saved plot to {image_filepath}")
        
        
