import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import mne
import pdb
import config as config


import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.stats import permutation_t_test


class ERPComparator:
    def __init__(self, raw, events, event_ids, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), picks='eeg'):
        """
        Initialize the ERPComparator.

        Parameters:
        - raw: mne.io.Raw object
        - events: event array
        - event_ids: dict with keys 'fixation' and 'stimulus' pointing to their event codes
        - tmin, tmax: time window for epochs
        - baseline: baseline correction interval
        - picks: channels to include
        """
        self.raw = raw
        self.events = events
        self.event_ids = event_ids
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.picks = picks

        self.fixation_epochs = None
        self.stimulus_epochs = None
        self.fixation_evoked = None
        self.stimulus_evoked = None
        self.diff_evoked = None

    def create_epochs(self):
        self.fixation_epochs = mne.Epochs(self.raw, self.events, event_id={'fixation': self.event_ids['fixation']},
                                          tmin=self.tmin, tmax=self.tmax, baseline=self.baseline,
                                          picks=self.picks, preload=True)
        self.stimulus_epochs = mne.Epochs(self.raw, self.events, event_id={'stimulus': self.event_ids['stimulus']},
                                          tmin=self.tmin, tmax=self.tmax, baseline=self.baseline,
                                          picks=self.picks, preload=True)

    def compute_evokeds(self):
        self.fixation_evoked = self.fixation_epochs.average()
        self.stimulus_evoked = self.stimulus_epochs.average()

    def plot_erps(self, picks='Cz'):
        mne.viz.plot_compare_evokeds({'Fixation': self.fixation_evoked, 'Stimulus': self.stimulus_evoked}, picks=picks)

    def compute_difference(self):
        self.diff_evoked = mne.combine_evoked([self.stimulus_evoked, self.fixation_evoked], weights=[1, -1])
        self.diff_evoked.plot(title="Stimulus - Fixation ERP Difference")

    def run_t_test(self, channel_name='Cz', n_permutations=1000):
        ch_idx = self.fixation_epochs.ch_names.index(channel_name)
        stim_data = self.stimulus_epochs.get_data()[:, ch_idx, :]  # shape: (n_epochs, n_times)
        fix_data = self.fixation_epochs.get_data()[:, ch_idx, :]

        # Paired t-test
        t_vals, p_vals = permutation_t_test(stim_data - fix_data, n_permutations=n_permutations)

        # Plot
        times = self.fixation_epochs.times
        plt.figure(figsize=(10, 4))
        plt.plot(times, t_vals, label='t-values')
        plt.axhline(0, color='black', linestyle='--')
        plt.xlabel("Time (s)")
        plt.ylabel("t-value")
        plt.title(f"T-test: Stimulus vs Fixation at {channel_name}")
        plt.legend()
        plt.tight_layout()
        plt.show()



class P100Plotter1:
    """
    Handles plotting of the P100 ERP component for visual trials.
    """
    def __init__(self, channels=None, time_window=(0.08, 0.12), baseline=(-0.2, 0)):
        """
        Parameters:
        - channels (list of str): EEG channel names to average (e.g., occipital electrodes).
        - time_window (tuple): Time window in seconds around expected P100 peak (default: 80–120 ms).
        - baseline (tuple): Time window for baseline correction (default: -200 to 0 ms).
        """
        self.channels = channels 
        self.time_window = time_window
        self.baseline = baseline

    def plot_p100(self, epochs, name, subject_id, session_id):
        """
        Plots the average ERP focusing on the P100 component.

        Parameters:
        - epochs: MNE Epochs object.
        - title: Plot title.
        - show: Whether to show the plot immediately.
        """

        epochs = epochs.copy().apply_baseline(self.baseline)

        picks = mne.pick_channels(epochs.info['ch_names'], include=self.channels)
        if picks.shape[0] < 1:
            raise ValueError(f"No channels from {self.channels} found in data.")

        evoked = epochs.average(picks=picks)

        # Plot the evoked response
        fig = evoked.plot(spatial_colors=True, show=False)
        for ax in fig.axes:
            ax.axvline(x=0.1, color='red', linestyle='--')
            ax.axvspan(self.time_window[0], self.time_window[1], color='orange', alpha=0.3)

        save_dir = Path(config.IMAGES_DIR, f'sub-{subject_id}', f'ses-{session_id}')
        os.makedirs(save_dir, exist_ok=True)
        save_path = Path(save_dir, f'{name}.png')
        for ax in fig.axes:
            ax.set_title(f'') 
        if save_path:
            fig.savefig(save_path, dpi=600)
            print(f"[INFO] Saved plot to {save_path}")

    def get_p100_amplitude(self, epochs):
        """
        Extracts the peak P100 amplitude and latency from average ERP.

        Returns:
        - amplitude (float): Maximum positive deflection in the time window.
        - latency (float): Time (in seconds) of the peak.
        """
        evoked = epochs.copy().apply_baseline(self.baseline).average(picks=self.channels)

        # Time mask
        time_mask = (evoked.times >= self.time_window[0]) & (evoked.times <= self.time_window[1])
        data = evoked.data
        peak_vals = data[:, time_mask]
        
        peak_amplitude = peak_vals.max()
        peak_index = peak_vals.argmax()
        peak_latency = evoked.times[time_mask][peak_index % peak_vals.shape[1]]

        return peak_amplitude, peak_latency
