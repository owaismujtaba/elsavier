import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

import config as config

class P100Plotter:
    def __init__(self, 
            condition1: 'P100ComponentAnalyzer', condition2: 'P100ComponentAnalyzer', 
            name1, name2, sub_id, ses_id
            ):
        """
        Initialize the P100 plotter.

        Args:
            condition1 (P100ComponentAnalyzer): First condition analyzer.
            condition2 (P100ComponentAnalyzer): Second condition analyzer.
        """
        print('Initializing Plotter')
        self.condition1 = condition1
        self.condition2 = condition2
        self.name1 = name1
        self.name2 = name2
        self.sub_id = sub_id
        self.ses_id = ses_id

    def plot_evokeds(self):
        """
        Plot the evoked responses from each channel individually for condition1 and condition2.
        """
        evoked_1 = self.condition1.get_evoked()
        evoked_2 = self.condition2.get_evoked()

        ch_names_1 = self.condition1.channels
        ch_names_2 = self.condition2.channels

        ch_idx_1 = [evoked_1.ch_names.index(ch) for ch in ch_names_1]
        ch_idx_2 = [evoked_2.ch_names.index(ch) for ch in ch_names_2]

        data_1 = evoked_1.data[ch_idx_1, :]
        data_2 = evoked_2.data[ch_idx_2, :]

        min_len = min(data_1.shape[1], data_2.shape[1])
        times = evoked_1.times[:min_len]

        data_1 = data_1[:, :min_len]
        data_2 = data_2[:, :min_len]
        
        plt.figure(figsize=(12, 6))

        for i, ch in enumerate(ch_names_1):
            plt.plot(times, data_1[i], label=f'{self.name1} - {ch}', linestyle='-')

        for i, ch in enumerate(ch_names_2):
            plt.plot(times, data_2[i], label=f'{self.name2} - {ch}', linestyle='--')

        directory = Path(config.IMAGES_DIR, f'sub-{self.sub_id}', f'ses-{self.ses_id}')
        os.makedirs(directory, exist_ok=True)
        plot_name = f'p_100_component_{self.name1}_{self.name2}.png'

        filepath = Path(directory, plot_name)

        #plt.title('Evoked Responses by Channel (Time)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (µV)')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filepath)
