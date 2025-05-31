from src.dataset.bids import create_bids_dataset
from src.pipelines.p100_pipeline import P100AnalysisPipeline
from src.decoding.overt_covert_rest import SpeechEEGDatasetLoader
import config as config
from bids import BIDSLayout
import numpy as np
import pandas as pd
import os
from pathlib import Path
import pdb

from src.decoding.overt_covert_rest_model import OvertCoverRestClassifier
from src.anonymization.voice_snonymizer import VoiceAnonymizerPipeline

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


    if config.OVERT_COVERT_REST_CLASSIFICATION:
        
        from src.pipelines.overt_covert_rest_pipeline import OvertCovertRestPipeline

        layout = BIDSLayout(config.BIDS_DIR, validate=True)
        subject_ids = layout.get_subjects()

        for sub in subject_ids:
            session_ids = layout.get_sessions(subject=sub)  # or any other subject
            for ses in session_ids:
                pipeline = OvertCovertRestPipeline(
                    subject_id=sub, session_id=ses
                )
                pipeline.run()


    if config.ANONYMIZE_AUDIO:
        layout = BIDSLayout(config.BIDS_DIR, validate=True)
        subject_ids = layout.get_subjects()

        for sub in subject_ids:
            session_ids = layout.get_sessions(subject=sub)  # or any other subject
            for ses in session_ids:
                directory = Path(config.BIDS_DIR, f'sub-{sub}', f'ses-{ses}', 'audio')
                
                filepath = [os.path.join(directory, file) for file in os.listdir(directory) if file.lower().endswith(".wav")][0]
                print(filepath)
                pipeline = VoiceAnonymizerPipeline(pitch_steps=5, formant_ratio=1.3)
                anonymized_audio = pipeline.fit_transform(filepath)
                pipeline.save(anonymized_audio, pipeline.target_sr, Path(directory, "anonymized.wav"))
