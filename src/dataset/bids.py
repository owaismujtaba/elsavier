import os
import pandas as pd
import numpy as np
from pathlib import Path
import csv
from scipy.io.wavfile import write
from mne.annotations import Annotations
from mne_bids import BIDSPath, write_raw_bids

import config as config
from src.dataset.data_reader import XDFDataReader
from src.utils.graphics import styled_print

import pdb

class BIDSDataset:
    def __init__(self, xdf_reader):
        styled_print("🚀", "Initializing BIDSDataset Class", "yellow", panel=True)

        self.xdf_reader = xdf_reader
        self.sub_id = xdf_reader.sub_id
        self.ses_id = xdf_reader.ses_id
        self.eeg = xdf_reader.eeg
        self.filename = f'sub-{self.sub_id}_ses-{self.ses_id}_VowelStudy_run-01'
        
        self.preprocess_eeg()
        self._setup_bidspath()


    def _setup_bidspath(self):
        self.bidspath = BIDSPath(
            subject= self.sub_id, session=self.ses_id,
            task='VCV', run='01', datatype='eeg',
            root=config.BIDS_DIR
        )    

    def preprocess_eeg(self):
        styled_print('', 'Preproocessing EEG', 'green')
        self.eeg = self.eeg.resample(config.EEG_SR)
        #self.eeg.set_eeg_reference(['FCz'])
        '''
        self.eeg.notch_filter(config.NOTCH_FREQ, fir_design='firwin')
        self.eeg.filter(
            l_freq=config.LOW_FREQ, h_freq=config.HIGH_FREQ,
            fir_design='firwin'
        )'
        '''
        
    def create_bids_files(self):
        self._create_bids_file_eeg()
        #self._create_bids_file_audio()

    def _create_bids_file_eeg(self):
        styled_print('', 'Creating BIDS File for EEG', 'green')
        unique_annotations = set(self.eeg.annotations.description)
        event_id = {desc: i+1 for i, desc in enumerate(unique_annotations)}

        write_raw_bids(
            self.eeg, bids_path = self.bidspath,
            allow_preload=True, format="EDF", 
            overwrite=True, event_id=event_id
        )

    def _create_bids_file_audio(self):
        styled_print('', 'Creating BIDS File for Audio', 'green')
        
        audio = self.xdf_reader.audio.get_data()

        output_dir = Path( config.BIDS_DIR, f'sub-{self.sub_id}',
            f'ses-{self.ses_id}' , 'audio'
        )
        os.makedirs(output_dir, exist_ok=True)

        audio = audio.flatten()
        audio = audio / np.max(np.abs(audio))
        audio = (audio * 32765).astype(np.int16)
        
        filename = f'sub-{self.sub_id}_ses-{self.ses_id}_task-VCV_run-01'

        audio_filepath = Path(output_dir, f'{filename}_audio.wav')
        
        write(str(audio_filepath), config.AUDIO_SR,audio)

        events_fileapth = Path(output_dir, f'{filename}_events.tsv')
        annotations = self.eeg.annotations

        with open(events_fileapth, "w", newline="") as tsvFile:
            writer = csv.writer(tsvFile,  delimiter='\t')
            writer.writerow(['onset', 'duration', 'description'])
            for onset, duration, description in zip(
                    annotations.onset, annotations.duration, 
                    annotations.description
                ):
                writer.writerow([onset, duration, description])
     


def create_bids_dataset(dataset_details):
    for details in dataset_details:
        filepath = details[0]
        sub_id = details[1]
        ses_id = details[2]
        xdf_reader = XDFDataReader(
            filepath=filepath,
            sub_id=sub_id,
            ses_id=ses_id
        )
        bids = BIDSDataset(xdf_reader=xdf_reader)
        bids.create_bids_files()
