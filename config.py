import os
from pathlib import Path


# Preprocessing Parameters
EEG_FILTER = {"l_freq": 0.1, "h_freq": 40.0}
ICA_PARAMS = {"n_components": 50, "random_state": 97}
EEG_REFERENCE = ['FCz']
EEG_MONTAGE = "standard_1020"
EOG_CHANNELS = ['EOG1', 'EOG2']

#Flags for Functionality Running
CREATE_BIDS_DATASET = False
P_100_ANALYSIS = True





CURR_DIR = os.getcwd()
RAW_DATA_DIR = Path(CURR_DIR, 'Data')
BIDS_DIR = Path(CURR_DIR, 'BIDS')
IMAGES_DIR = Path(CURR_DIR, 'Images')

# Raw XDF Parameters
EEG_SR = 1000
AUDIO_SR = 48000

# Preprocessing Parameters
NOTCH_FREQ = [50, 100]
LOW_FREQ = 0.5
HIGH_FREQ = 170

T_MIN = -0.2
T_MAX = 1.0




filepaths1 = [   
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf', '01', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P002/ses-S001/eeg/sub-P002_ses-S001_task-Default_run-001_eeg.xdf', '02', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P003/ses-S002/eeg/sub-P003_ses-S002_task-Default_run-001_eeg.xdf', '03', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P004/ses-S001/eeg/sub-P004_ses-S001_task-Default_run-001_eeg.xdf', '04', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P004/ses-S002/eeg/sub-P004_ses-S002_task-Default_run-001_eeg.xdf', '04', '02'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P005/ses-S001/eeg/sub-P005_ses-S001_task-Default_run-001_eeg.xdf', '05', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P005/ses-S002/eeg/sub-P005_ses-S002_task-Default_run-001_eeg.xdf', '05', '02'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P006/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf', '06', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P006/ses-S003/eeg/sub-P006_ses-S003_task-Default_run-001_eeg.xdf', '06', '02'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P007/ses-S002_Elbueno/eeg/sub-P007_ses-S002_task-Default_run-001_eeg.xdf', '07', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P008/ses-S001/eeg/sub-P008_ses-S001_task-Default_run-001_eeg.xdf', '08', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P009/ses-S001/eeg/sub-P009_ses-S001_task-Default_run-001_eeg.xdf', '09', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P009/ses-S003/eeg/sub-P009_ses-S003_task-Default_run-001_eeg.xdf', '09', '02'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P010/ses-S001/eeg/sub-P010_ses-S001_task-Default_run-001_eeg.xdf', '10', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P011/ses-S001/eeg/sub-P011_ses-S001_task-Default_run-001_eeg.xdf', '11', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P012/ses-S001/eeg/sub-P012_ses-S001_task-Default_run-001_eeg.xdf', '12', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P012/ses-S002/eeg/sub-P011_ses-S002_task-Default_run-001_eeg.xdf', '12', '02'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P013/ses-S001/eeg/sub-P013_ses-S001_task-Default_run-001_eeg.xdf', '13', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P013/ses-S002/eeg/sub-P013_ses-S002_task-Default_run-001_eeg.xdf', '13', '02'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P014/ses-S001/eeg/sub-P014_ses-S001_task-Default_run-001_eeg.xdf', '14', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P015/ses-S001/eeg/sub-P015_ses-S001_task-Default_run-001_eeg.xdf', '15', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P015/ses-S002/eeg/sub-P015_ses-S002_task-Default_run-001_eeg.xdf', '15', '02']
]

filepaths = [   
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf', '01', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P002/ses-S001/eeg/sub-P002_ses-S001_task-Default_run-001_eeg.xdf', '02', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P003/ses-S002/eeg/sub-P003_ses-S002_task-Default_run-001_eeg.xdf', '03', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P004/ses-S001/eeg/sub-P004_ses-S001_task-Default_run-001_eeg.xdf', '04', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P005/ses-S001/eeg/sub-P005_ses-S001_task-Default_run-001_eeg.xdf', '05', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P006/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf', '06', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P007/ses-S002_Elbueno/eeg/sub-P007_ses-S002_task-Default_run-001_eeg.xdf', '07', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P008/ses-S001/eeg/sub-P008_ses-S001_task-Default_run-001_eeg.xdf', '08', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P009/ses-S001/eeg/sub-P009_ses-S001_task-Default_run-001_eeg.xdf', '09', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P010/ses-S001/eeg/sub-P010_ses-S001_task-Default_run-001_eeg.xdf', '10', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P011/ses-S001/eeg/sub-P011_ses-S001_task-Default_run-001_eeg.xdf', '11', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P012/ses-S001/eeg/sub-P012_ses-S001_task-Default_run-001_eeg.xdf', '12', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P013/ses-S001/eeg/sub-P013_ses-S001_task-Default_run-001_eeg.xdf', '13', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P014/ses-S001/eeg/sub-P014_ses-S001_task-Default_run-001_eeg.xdf', '14', '01'],
    ['/home/owaismujtaba/projects/elsavier/RawData/sub-P015/ses-S001/eeg/sub-P015_ses-S001_task-Default_run-001_eeg.xdf', '15', '01'],
]
'''

trail_model = 'Silent/Real'
trail_unit = 'Syllables/Words'
experiment_mode = 'Practice/Experiment'
trail_boundary = 'Start/End'
trail_type = 'Stimulus/ISI/Fixation/Speech/ITI'
modality = 'Audio/Text/Pictures'

'''



