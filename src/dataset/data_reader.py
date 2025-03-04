import mne

from src.dataset.utils import normalize_triggers, correct_triggers
from src.dataset.utils import transition_points_triggers, map_events
import pdb
class DatasReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.eeg, self.streams = None, None
        self.setup_raw_data()

    def _read_file(self):
        self.raw_data = mne.io.read_raw_edf(
            self.filepath,
            preload=True
        )

    def setup_raw_data(self):
        self._read_file()
        self.n_channels = self.raw_data.info['nchan']
        self.start_time = self.raw_data.info['meas_date'].timestamp()
        self.ch_names = self.raw_data.ch_names
        self.triggers = self.raw_data['TRIG'][0][0]
        self.timestamps = self.raw_data.times + self.start_time
        self.timestamps = self.timestamps.astype('datetime64[s]')
    
        

        self.triggers_normalized = normalize_triggers(self.triggers)
        self.triggers_corrected =  correct_triggers(self.triggers_normalized)
        self.transition_points_indexs = transition_points_triggers(self.triggers_corrected)
        self.events = map_events(
            self.triggers_corrected,
            self.transition_points_indexs,
            self.timestamps
        )

        for index in range(20):
            print(self.events[index])
        pdb.set_trace()