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
            self._load_audio_stream()
            #pass
        except:
            styled_print("⚠️", "Error reading Audio from XDF", "red", bold=False, panel=True)



class BIDSDatasetReader:
    def __init__(self, sub_id, ses_id):
        self.sub_id = sub_id
        self.ses_id = ses_id

        self._setup_bidspath()
        self.preprocess()

    def preprocess(self):
        self.raw.filter(l_freq=0.1, h_freq=40.0, fir_design='firwin', verbose=False)    
        
    def _setup_bidspath(self):
        self.bidspath = BIDSPath(
            subject= self.sub_id, session=self.ses_id,
            task='VCV', run='01', datatype='eeg',
            root=config.BIDS_DIR
        )   
    def read_bids_subject_data(self):
        self.raw = read_raw_bids(self.bidspath, verbose=False)