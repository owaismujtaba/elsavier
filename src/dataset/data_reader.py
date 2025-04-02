import mne
from pathlib import Path
from pyprep import NoisyChannels
from mnelab.io.xdf import read_raw_xdf
from mne_bids import BIDSPath, read_raw_bids
from pyxdf import resolve_streams, match_streaminfos

from src.utils.graphics import styled_print
import config as config

class XDFDataReader:
    def __init__(self, filepath, sub_id='01', ses_id='01'):
        styled_print("🚀", "Initializing XDFDataReader Class", "yellow", panel=True)
        styled_print("👤", f"Subject: {sub_id} | 🗂 Session: {ses_id}", "cyan")

        self.xdf_filepath = filepath
        self.sub_id = sub_id
        self.ses_id = ses_id

        styled_print("📡", "Resolving streams from XDF file...", "magenta")
        self.streams = resolve_streams(self.xdf_filepath)

        self.read_xdf_file()

    def _load_eeg_stream(self):
        styled_print("🧠", "Loading EEG Data...", "blue")
        eeg_stream_id = match_streaminfos(self.streams, [{'type': 'EEG'}])[0]
        self.eeg = read_raw_xdf(self.xdf_filepath, stream_ids=[eeg_stream_id])
        styled_print("✅", "EEG Data Loaded Successfully!", "green")

    def _load_audio_stream(self):
        styled_print("🎧", "Loading Audio Data...", "yellow")
        audio_stream_id = match_streaminfos(self.streams, [{'type': 'Audio'}])[0]
        self.audio = read_raw_xdf(self.xdf_filepath, stream_ids=[audio_stream_id])
        styled_print("✅", "Audio Data Loaded Successfully!", "green")

    def read_xdf_file(self):
        styled_print("📂", "Reading XDF File...", "magenta")
        try:
            self._load_eeg_stream()
        except:
            styled_print("⚠️", "Error reading EEG from XDF", "red", bold=False, panel=True)
        try:
             #self._load_audio_stream()
            pass
        except:
            styled_print("⚠️", "Error reading Audio from XDF", "red", bold=False, panel=True)



class BIDSDatasetReader:
    def __init__(self, sub_id, ses_id):
        styled_print("🚀", "Initializing BIDSDatasetReader Class", "yellow", panel=True)
        self.sub_id = sub_id
        self.ses_id = ses_id
        self.raw = None
        
        self._setup_bidspath()
        self.processed_dir = Path(config.BIDS_DIR) / "derivatives" / "processed_eeg"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.processed_file = self.processed_dir / f"sub-{sub_id}_ses-{ses_id}_processed-raw.fif"
        
        self.read_or_process_data()
    
    def read_or_process_data(self):
        if self.processed_file.exists():
            styled_print("", "Loading Processed EEG Data", color='green')
            self.raw = mne.io.read_raw_fif(self.processed_file, preload=True)
        else:
            self.read_bids_subject_data()
            self.preprocess()
            self.save_processed_data()
    
    def preprocess(self):
        styled_print('', 'Preprocessing EEG', color='red')
        self._set_channel_types()
        self._remove_bad_channels()
        self.raw.filter(l_freq=0.1, h_freq=40.0, fir_design='firwin', verbose=False)
        self.raw.set_eeg_reference(['FCz'])  
        self._artifact_removal()
    
    def _set_channel_types(self):
        styled_print('', 'Setting Channels and Montage', color='cyan')
        eeg = self.raw.copy()
        try:
            eeg.set_channel_types({'EOG1': 'eog', 'EOG2': 'eog'})
        except:
            eeg.rename_channels({'TP9': 'EOG1', 'TP10': 'EOG2'})
            eeg.set_channel_types({'EOG1': 'eog', 'EOG2': 'eog'})
        montage = mne.channels.make_standard_montage("standard_1020")
        eeg.set_montage(montage)
        self.raw = eeg
    
    def _remove_bad_channels(self):
        styled_print('', 'Interpolating Bad Channels', color='cyan')
        eeg = self.raw.copy()
        prep = NoisyChannels(eeg)
        prep.find_bad_by_deviation()
        prep.find_bad_by_correlation()
        eeg.info['bads'] = prep.get_bads()
        eeg.interpolate_bads(reset_bads=True)
        self.raw = eeg
    
    def _artifact_removal(self):
        styled_print('', 'Removing Artifacts using ICA', color='cyan')
        eeg = self.raw.copy()
        ica = mne.preprocessing.ICA(n_components=50, random_state=97)
        ica.fit(eeg)
        eog_indices, _ = ica.find_bads_eog(eeg, ch_name=['EOG1', 'EOG2'])
        ica.exclude = eog_indices
        self.raw = ica.apply(eeg)
    
    def _setup_bidspath(self):
        self.bidspath = BIDSPath(
            subject=self.sub_id, session=self.ses_id,
            task='VCV', run='01', datatype='eeg',
            root=config.BIDS_DIR
        )   
    
    def read_bids_subject_data(self):
        styled_print('', 'Loading Raw Data', color='cyan')
        self.raw = read_raw_bids(self.bidspath, verbose=False)
        self.raw.load_data()
    
    def save_processed_data(self):
        styled_print('', 'Saving Processed EEG Data', color='green')
        self.raw.save(self.processed_file, overwrite=True)


class BIDSDatasetReader1:
    def __init__(self, sub_id, ses_id):
        styled_print("🚀", "Initializing BIDSDatasetReader Class", "yellow", panel=True)
        self.sub_id = sub_id
        self.ses_id = ses_id
        self.raw=None

        self._setup_bidspath()
        self.read_bids_subject_data()
        

    def preprocess(self):
        styled_print('', 'Preprocessing EEG', color='red')
        self._set_channel_types()
        self._remove_bad_channels()
        self.raw.filter(l_freq=0.1, h_freq=40.0, fir_design='firwin', verbose=False)
        self.raw.set_eeg_reference(['FCz'])  
        self._artifact_removal()
    
    def _set_channel_types(self):
        styled_print('', 'Setting Channels and Montage', color='blue')
        eeg = self.raw.copy()
        try:
            eeg.set_channel_types({'EOG1': 'eog', 'EOG2': 'eog'})
        except:
            eeg.rename_channels({'TP9': 'EOG1', 'TP10': 'EOG2'})  # Example if needed
            eeg.set_channel_types({'EOG1': 'eog', 'EOG2': 'eog'})
        montage = mne.channels.make_standard_montage("standard_1020")
        eeg.set_montage(montage)
        self.raw = eeg

    def _remove_bad_channels(self):
        styled_print('', 'Removing Bad Channels', color='blue')
        eeg = self.raw.copy()
        prep = NoisyChannels(eeg)
        prep.find_bad_by_deviation()
        prep.find_bad_by_correlation()
        eeg.info['bads'] = prep.get_bads()
        eeg.interpolate_bads(reset_bads=True)
        self.raw = eeg
    
    def _artifact_removal(self):
        styled_print('', 'Removing Artifacts', color='blue')
        eeg = self.raw.copy()
        ica = mne.preprocessing.ICA(n_components=50, random_state=97)
        ica.fit(eeg)
        eog_indices, _ = ica.find_bads_eog(eeg, ch_name=['EOG1', 'EOG2'])
        ica.exclude = eog_indices
        self.raw = ica.apply(eeg)
                        
    def _setup_bidspath(self):
        self.bidspath = BIDSPath(
            subject= self.sub_id, session=self.ses_id,
            task='VCV', run='01', datatype='eeg',
            root=config.BIDS_DIR
        )   
    
    def read_bids_subject_data(self):
        styled_print('', 'Loading Raw Data', color='blue')
        self.raw = read_raw_bids(self.bidspath, verbose=False)
        self.raw.load_data()


