import os
import pandas as pd
from mne.epochs import Epochs

from src.dataset.data_reader import BIDSDatasetReader
from src.dataset.eeg_epoch_builder import EEGEpochBuilder
from src.analysis.p_100_analyser import P100ComponentAnalyzer
from src.visualizations.p100_plotter import P100Plotter


class P100AnalysisPipeline:
    """
    Pipeline to analyze and compare P100 ERP components between two experimental conditions.
    """

    def __init__(
        self,
        subject_id: str,
        session_id: str,
        condition1_config: dict,
        condition2_config: dict,
        channels: list[str]
    ) -> None:
        """
        Initialize the pipeline with subject/session info and condition configurations.

        Args:
            subject_id (str): BIDS subject identifier.
            session_id (str): BIDS session identifier.
            condition1_config (dict): Dict with trial and epoch params for condition 1.
            condition2_config (dict): Same as condition1_config, with optional 'time_window'.
            channels (list[str], optional): EEG channels to analyze. Defaults to ['PO3', 'POz', 'PO4'].
        """
        self.subject_id = subject_id
        self.session_id = session_id
        self.condition1_config = condition1_config
        self.condition2_config = condition2_config
        self.channels = channels
        self.analyzer1: P100ComponentAnalyzer = None
        self.analyzer2: P100ComponentAnalyzer = None
        self.eeg = None
        self.epochs1: Epochs = None
        self.epochs2: Epochs = None

    def load_data(self) -> 'P100AnalysisPipeline':
        """
        Load EEG data using BIDSDatasetReader.

        Returns:
            P100AnalysisPipeline: self
        """
        self.bids_reader = BIDSDatasetReader(
            sub_id=self.subject_id,
            ses_id=self.session_id
        )
        self.eeg = self.bids_reader.processed_file
        return self

    def build_epochs(self) -> 'P100AnalysisPipeline':
        """
        Construct MNE Epochs objects for both conditions.

        Returns:
            P100AnalysisPipeline: self
        """
        self.epochs1 = self._create_epochs(self.condition1_config)
        self.epochs2 = self._create_epochs(self.condition2_config)
        return self

    def _create_epochs(self, config: dict) -> Epochs:
        """
        Helper to create epochs using EEGEpochBuilder.

        Args:
            config (dict): Configuration dictionary for the condition.

        Returns:
            Epochs: MNE Epochs object
        """
        return EEGEpochBuilder(
            eeg_data=self.eeg,
            trial_mode=config["trial_mode"],
            trial_unit=config["trial_unit"],
            experiment_mode=config["experiment_mode"],
            trial_boundary=config["trial_boundary"],
            trial_type=config["trial_type"],
            modality=config["modality"]
        ).create_epochs(
            tmin=config["tmin"],
            tmax=config["tmax"]
        )

    def analyze(self) -> 'P100AnalysisPipeline':
        """
        Compute P100 peak, latency, and mean amplitude for both conditions.

        Returns:
            P100AnalysisPipeline: self
        """
        self.analyzer1 = P100ComponentAnalyzer(
            self.epochs1, channels=self.channels
        )
        self.analyzer2 = P100ComponentAnalyzer(
            self.epochs2,
            channels=self.channels,
            time_window=self.condition2_config.get("time_window")
        )

        lat1, peak1, mean1 = self.analyzer1.get_p100_peak()
        lat2, peak2, mean2 = self.analyzer2.get_p100_peak()

        print(f"{self.condition1_config['label']} P100: "
              f"latency={lat1:.3f}s, peak={peak1:.2f}µV, mean={mean1:.2f}µV")
        print(f"{self.condition2_config['label']} P100: "
              f"latency={lat2:.3f}s, peak={peak2:.2f}µV, mean={mean2:.2f}µV")

        return self

    def plot(self) -> 'P100AnalysisPipeline':
        """
        Generate and display ERP plots for the two conditions.

        Returns:
            P100AnalysisPipeline: self
        """
        plotter = P100Plotter(
            condition1=self.analyzer1,
            condition2=self.analyzer2,
            name1=self.condition1_config['label'],
            name2=self.condition2_config['label'],
            sub_id=self.subject_id,
            ses_id=self.session_id
        )
        plotter.plot_evokeds()
        return self

    def save_results(self, output_dir: str = "p100_results") -> 'P100AnalysisPipeline':
        """
        Save P100 peak metrics as a CSV file.

        Args:
            output_dir (str): Directory to save results. Defaults to "p100_results".

        Returns:
            P100AnalysisPipeline: self
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "condition": [
                self.condition1_config["label"],
                self.condition2_config["label"]
            ],
            "latency": [
                self.analyzer1.latency,
                self.analyzer2.latency
            ],
            "peak": [
                self.analyzer1.peak_amplitude,
                self.analyzer2.peak_amplitude
            ],
            "mean": [
                self.analyzer1.mean_amplitude,
                self.analyzer2.mean_amplitude
            ]
        }
        name1=self.condition1_config['label']
        name2=self.condition2_config['label']
        df = pd.DataFrame(results)
        csv_path = os.path.join(
            output_dir,
            f"sub-{self.subject_id}_ses-{self.session_id}_p100_{name1}_{name2}.csv"
        )
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        return self

    def run(self, save_csv: bool = True) -> 'P100AnalysisPipeline':
        """
        Execute the entire pipeline from loading to analysis and plotting.

        Args:
            save_csv (bool): Whether to save the output as a CSV. Defaults to True.

        Returns:
            P100AnalysisPipeline: self
        """
        self.load_data().build_epochs().analyze().plot()
        if save_csv:
            self.save_results()
        return self
