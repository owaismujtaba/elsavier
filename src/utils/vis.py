
from matplotlib import pyplot  as plt


def plot_p100_occupital(self, epochs, sub_id, ses_id):
        """Plots the P100 ERP component from the filtered epochs."""
        
        # Select occipital electrodes (O1, O2, Oz)
        occipital_channels = ["O1", "O2", "Oz"]
        evoked = epochs.average()
        

        # Check if the EEG data contains these channels
        available_channels = [ch for ch in occipital_channels if ch in evoked.info["ch_names"]]
        if not available_channels:
            print("No occipital channels found in data.")
            return

        # Plot ERP
        plt.figure(figsize=(8, 5))
        evoked.plot(picks=available_channels, show=True, titles="Occipital Electrodes")
        

        plt.savefig()


