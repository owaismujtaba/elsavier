import numpy as np
import mne
from matplotlib import pyplot as plt

from src.dataset.data_reader import BIDSDatasetReader
import config as config

class SpeechEventExtractor:
    def __init__(self,raw, tmin=-0.2, tmax=0.8):
        self.raw = raw
        self.annotations = self.raw.annotations
        self.tmin = tmin
        self.tmax = tmax
    
    def get_silence_events(self):
        return self._filter_events(['Experiment','Words', 'Start', 'Speech','Audio', 'silence'])
    
    def get_overt_speaking_events(self):
        return self._filter_events(['Real', 'Words', 'Experiment', 'Start', 'Speech', 'Audio'], exclude=['silence'])
    
    def get_covert_speaking_events(self):
        return self._filter_events(['Silent', 'Words', 'Experiment', 'Start', 'Speech', 'Audio'], exclude=['silence'])
    
    def _filter_events(self, criteria, exclude=None):
        exclude = exclude or []
        return [event for event in self.annotations 
                if all(criterion in event['description'] for criterion in criteria) 
                and all(excl not in event['description'] for excl in exclude)]
    
    def get_events_info(self, filtered_events):
        event_list = []
        event_id_map = {}
        event_counter = 1

        for event in filtered_events:
            onset_sample = int(event['onset'] * self.raw.info['sfreq'])
            description = event['description']

            if description not in event_id_map:
                event_id_map[description] = event_counter
                event_counter += 1

            event_list.append([onset_sample, 0, event_id_map[description]])

        events = np.array(event_list)
        return events, event_id_map
    
    def create_epochs(self, event_type):
        if event_type == 'silence':
            filtered_events = self.get_silence_events()
        elif event_type == 'overt':
            filtered_events = self.get_overt_speaking_events()
        elif event_type == 'covert':
            filtered_events = self.get_covert_speaking_events()
        else:
            raise ValueError("Invalid event type. Choose from 'silence', 'overt', or 'covert'.")
        
        events, event_id_map = self.get_events_info(filtered_events)
        epochs = mne.Epochs(self.raw, events, event_id=event_id_map, tmin=self.tmin, tmax=self.tmax, 
                            baseline=(None, 0), detrend=1, preload=True)
        return epochs
    
    def plot_erp(self):
        event_types = ['silence', 'overt', 'covert']
        colors = {'silence': 'blue', 'overt': 'green', 'covert': 'red'}
        channels = ['F3', 'F7']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
        
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        for event_type in event_types:
            epochs = self.create_epochs(event_type)
            evoked = epochs.average()
            
            # Plot F3
            ch_idx_f3 = epochs.ch_names.index('F3')
            axes[0].plot(evoked.times, evoked.data[ch_idx_f3], label=f'{event_type} - F3', color=colors[event_type])
            
            # Plot F7
            ch_idx_f7 = epochs.ch_names.index('F7')
            axes[1].plot(evoked.times, evoked.data[ch_idx_f7], label=f'{event_type} - F7', color=colors[event_type])
            
            # Plot all channels (mean across all)
            axes[2].plot(evoked.times, evoked.data.mean(axis=0), label=f'{event_type} - All', color=colors[event_type])
        
        
        
        for ax in axes:
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.legend()
        
        plt.tight_layout()
        plt.show()




def load_subject_data(subject_ids, session_ids):
    subjects_data = {}
    for index in range(len(session_ids)):
        subject_id = subject_ids[index]
        session_id = session_ids[index]
        raw = BIDSDatasetReader(sub_id=subject_id, ses_id=session_id)
        extractor = SpeechEventExtractor(
            raw=raw.raw  # Corrected data assignment
        )
        subjects_data[(subject_id, session_id)] = extractor
    return subjects_data

def plot_erp_all_subjects_sessions(subjects_data):
    event_types = ['silence', 'overt', 'covert']
    colors = {'silence': 'blue', 'overt': 'green', 'covert': 'red'}
    channels = ['F3', 'F7']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    for (subject_id, session_id), extractor in subjects_data.items():
        for event_type in event_types:
            epochs = extractor.create_epochs(event_type)
            evoked = epochs.average()
            
            # Plot F3
            ch_idx_f3 = epochs.ch_names.index('F3')
            axes[0].plot(evoked.times, evoked.data[ch_idx_f3], label=f'S{subject_id} Sess{session_id} {event_type}', color=colors[event_type], alpha=0.5)
            
            # Plot F7
            ch_idx_f7 = epochs.ch_names.index('F7')
            axes[1].plot(evoked.times, evoked.data[ch_idx_f7], label=f'S{subject_id} Sess{session_id} {event_type}', color=colors[event_type], alpha=0.5)
            
            # Plot all channels (mean across all)
            axes[2].plot(evoked.times, evoked.data.mean(axis=0), label=f'S{subject_id} Sess{session_id} {event_type}', color=colors[event_type], alpha=0.5)
    
    axes[0].set_title('ERP Comparison for F3')
    axes[1].set_title('ERP Comparison for F7')
    axes[2].set_title('ERP Comparison across All Channels')
    
    for ax in axes:
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right', fontsize='small', ncol=2)
    
    plt.tight_layout()
    plt.show()
