import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.dataset.data_reader import BIDSDatasetReader
from src.dataset.data_loader import DataLoader
from src.utils.graphics import styled_print

import config as config

import pdb


class VisualRestExtractor:
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
        
        
