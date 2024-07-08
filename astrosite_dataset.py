import os
import numpy as np
import json
import event_stream


class AstrositeDataset:
    def __init__(self, recordings_path):
        self.recordings_path = recordings_path
        self.recording_files = [
            os.path.join(recordings_path, folder) for folder in os.listdir(recordings_path)
            if os.path.isdir(os.path.join(recordings_path, folder))
        ]
    
    def __len__(self):
        return len(self.recording_files)
    
    def __getitem__(self, idx):
        if idx >= len(self.recording_files):
            raise IndexError('Index out of range')
        
        recording_path = self.recording_files[idx]
        
        # Placeholder function to load events.es file
        def load_events_es(file_path):
            decoder = event_stream.Decoder(file_path)
            chunks = [chunk for chunk in decoder]
            return np.concatenate(chunks)
        
        events_es_path = os.path.join(recording_path, 'events.es')
        labelled_events_path = os.path.join(recording_path, 'labelled_events.npy')
        recording_json_path = os.path.join(recording_path, 'recording.json')
        
        events = load_events_es(events_es_path)
        labelled_events = np.load(labelled_events_path, allow_pickle=True)
        with open(recording_json_path, 'r') as f:
            recording_data = json.load(f)
        
        return {
            'events': events,
            'labelled_events': labelled_events,
            'recording_data': recording_data,
            'target_id': recording_data['object']['id'],
        }
