import os
import numpy as np
import json
import event_stream
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class AstrositeDataset:
    """Dataset class for the Astrosite dataset. One sample consists of the events, labelled events,
    and recording data and target ID for a single recording.

    Args:
        recordings_path (str): Path to the directory containing the recordings.
        split (str, optional): The split to use. Can be either 'all' or a list of satellite IDs.
            If 'all', all recordings are used. If a list of satellite IDs, only recordings from
            those satellites are used. Defaults to 'all'.
        min_sat_events (int, optional): The minimum number of events with label -1 that a recording
            must have to be included in the dataset. Defaults to 1000.
    """
    sensor_size = (1280, 720, 2)

    def __init__(self, recordings_path, split="all", min_sat_events=1000, transform=None):
        self.recordings_path = recordings_path
        self.split = split
        self.min_sat_events = min_sat_events
        if split != "all":
            self.recording_files = self.get_split()
        else:
            self.recording_files = [
                os.path.join(recordings_path, folder)
                for folder in os.listdir(recordings_path)
                if os.path.isdir(os.path.join(recordings_path, folder))
            ]

    def __len__(self):
        return len(self.recording_files)

    def get_split(self):
        files = [f for f in glob.glob(self.recordings_path + "/**/recording.json")]
        recording_files = []
        files_per_satellites = {}
        for file in files:
            json_load = open(file)
            dict_file = json.load(json_load)
            satellite_id = dict_file["object"]["id"]
            if satellite_id in self.split:
                file_location = "/".join(file.split("/")[:-1])
                labelled_events = np.load(file_location+"/labelled_events.npy")
                if min(list(set(labelled_events['label']))) >= -1 and len(labelled_events[labelled_events['label'] == -1]) >= self.min_sat_events :
                    recording_files.append(file_location)
                if not(dict_file['object']['id'] in files_per_satellites):
                    files_per_satellites[satellite_id] = {"occurences" : 1 , "locations":[file]}
                else:
                    files_per_satellites[satellite_id]["locations"].append(file)
                    files_per_satellites[satellite_id]["occurences"] += 1
        return recording_files

    def __getitem__(self, idx):
        if idx >= len(self.recording_files):
            raise IndexError("Index out of range")

        recording_path = self.recording_files[idx]

        # Placeholder function to load events.es file
        def load_events_es(file_path):
            decoder = event_stream.Decoder(file_path)
            chunks = [chunk for chunk in decoder]
            return np.concatenate(chunks)

        events_es_path = os.path.join(recording_path, "events.es")
        labelled_events_path = os.path.join(recording_path, "labelled_events.npy")
        recording_json_path = os.path.join(recording_path, "recording.json")

        events = load_events_es(events_es_path)
        labelled_events = np.load(labelled_events_path, allow_pickle=True)
        with open(recording_json_path, "r") as f:
            recording_data = json.load(f)

        labelled_events = labelled_events.view(dtype=np.dtype([('t', '<u8'), ('x', '<u2'), ('y', '<u2'), (('on', 'p'), '?'), ('label', '<i2')]))
        return {
            "events": events,
            "labelled_events": labelled_events,
            "recording_data": recording_data,
            "target_id": recording_data["object"]["id"],
        }


class ClassificationAstrositeDataset(AstrositeDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        return sample["events"], sample['target_id']


class TrackingAstrositeDataset(AstrositeDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sat_events = sample['labelled_events']
        last_timestamp = sample['events'][-1][0]
        first_timestamp = sample['events'][0][0]
        first_event = np.array([(first_timestamp,0,0,True,0)],dtype=np.dtype([('t', '<u8'), ('x', '<u2'), ('y', '<u2'), (('on', 'p'), '?'), ('label', '<i2')]))
        last_event = np.array([(last_timestamp,0,0,True,0)],dtype=np.dtype([('t', '<u8'), ('x', '<u2'), ('y', '<u2'), (('on', 'p'), '?'), ('label', '<i2')]))
        completed_labelled_datasets=(np.concatenate((first_event,sat_events[sat_events['label']==-1],last_event)))
        return completed_labelled_datasets, sample['target_id']
    
class MergedDataset(DataLoader):
    def __init__(self, dataset1, dataset2):
        assert len(dataset1) == len(dataset2)
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        
    def __len__(self):
        return len(self.dataset1)
    
    def __getitem__(self, idx):
        assert self.dataset1[idx][1] == self.dataset2[idx][1]
        return self.dataset1[idx][0], self.dataset2[idx][0], self.dataset1[idx][1]
    