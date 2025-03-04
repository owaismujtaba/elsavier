


class DataLoader:
    def __init__(self, data_reader):
        self.data = data_reader.raw_data
        self.events = data_reader.events
        self._filter_events()


    def _filter_events(self):
        event_ids = {
            'Fixation' : 1, 
            'ITI' : 2, 
            'StartReading' : 3, 
            'StartSaying' : 4
        }
        filterd_events = []
        for event in self.events:
            if 'Experiment' in event[2] or 'Block' in event[2]:
                continue
            else:
                filterd_events.append([event[0], event[1], event_ids[event[2]]])

        self.filterd_events = filterd_events
        
