from src.dataset.bids import create_bids_dataset
from src.pipelines.p100_pipeline import P100AnalysisPipeline
import config as config
from bids import BIDSLayout

import pdb


if __name__== '__main__':
    
    if config.CREATE_BIDS_DATASET:
        create_bids_dataset(dataset_details=config.filepaths[8:])

    if config.P_100_ANALYSIS:
        visual = {
            "label": "Visual",
            "trial_type": "Stimulus",
            "tmin": -0.2,
            "tmax": 0.5,
            "trial_mode": "",
            "trial_unit": "Words",
            "experiment_mode": "Experiment",
            "trial_boundary": "Start",
            "modality": "Pictures"
        }

        rest = {
            "label": "Rest",
            "trial_type": "Fixation",
            "tmin": -0.2,
            "tmax": 0.5,
            "trial_mode": "",
            "trial_unit": "Words",
            "experiment_mode": "Experiment",
            "trial_boundary": "Start",
            "modality": "Pictures",
            "time_window": (0.08, 0.12)  # Optional window for P100
        }

        layout = BIDSLayout(config.BIDS_DIR, validate=True)
        subject_ids = layout.get_subjects()

        for sub in subject_ids:
            session_ids = layout.get_sessions(subject=sub)  # or any other subject
            for ses in session_ids:
                pipeline = P100AnalysisPipeline(
                    subject_id=sub,
                    session_id=ses,
                    condition1_config=visual,
                    condition2_config=rest,
                    channels = ['PO3', 'POz', 'PO4']
                )

                pipeline.run(save_csv=True)