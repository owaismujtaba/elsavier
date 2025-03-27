
import mne
import numpy as np

import config as config

from src.dataset.data_reader import BIDSDatasetReader
from src.utils.graphics import styled_print, print_criteria




class DataLoader:
    def __init__(self, eeg_data, trial_mode='', trial_unit='', 
                 experiment_mode='', trial_boundary='', 
                 trial_type='', modality=''):
        styled_print('', 'Initializing DataLoader Class', 'red', panel=True)
        self.eeg_data = eeg_data
        self.annotations = eeg_data.annotations
        self.criteria = [
            trial_mode, trial_unit, experiment_mode,
            trial_boundary, trial_type, modality
        ]
        

    def _filter_events(self):
        """Filters EEG event annotations based on predefined criteria."""
        filtered_events = []
        for event in self.annotations:
            if all(criterion in event['description'] for criterion in self.criteria):
                filtered_events.append(event)
        return filtered_events

    def create_epochs(self, tmin, tmax):
        """Epochs the EEG data based on filtered events."""
        styled_print('', 'Creating EPOCHS', color='green')
        print_criteria(self.criteria+[tmin, tmax])
        filtered_events = self._filter_events()

        if not filtered_events:
            raise ValueError("No matching events found for epoching.")

        event_list = []
        event_id_map = {} 
        event_counter = 1

        for event in filtered_events:
            onset_sample = int(event['onset'] * self.eeg_data.info['sfreq'])  
            description = event['description']

            if description not in event_id_map:
                event_id_map[description] = event_counter
                event_counter += 1

            event_list.append([onset_sample, 0, event_id_map[description]])

        events = np.array(event_list)  
        epochs = mne.Epochs(self.eeg_data, events, event_id=event_id_map, 
                            tmin=tmin, tmax=tmax, baseline=(tmin, tmin+0.2), 
                            preload=True)

        return epochs
    
    