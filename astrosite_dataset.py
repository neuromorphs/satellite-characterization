import os
import numpy as np
import math
import json
import event_stream
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tonic.slicers import SliceByTime
from tonic import SlicedDataset, transforms
import matplotlib.pyplot as plt
import scipy
#from events_analysis import events_to_spectrogram


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

    def __init__(self, recordings_path, split="all", min_sat_events=1000, transform=None, test=False):
        self.recordings_path = recordings_path
        self.split = split
        self.test = test
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
                    if not(dict_file['object']['id'] in files_per_satellites):
                        files_per_satellites[satellite_id] = {"occurences" : 1 , "locations":[file_location]}
                    else:
                        files_per_satellites[satellite_id]["locations"].append(file_location)
                        files_per_satellites[satellite_id]["occurences"] += 1
            json_load.close()
        for satellite_id in files_per_satellites.keys():
            length = files_per_satellites[satellite_id]['occurences']
            if self.test :
                recording_files += files_per_satellites[satellite_id]['locations'][int(0.8*length):]
            else :
                recording_files += files_per_satellites[satellite_id]['locations'][:int(0.8*length)]
        return recording_files

    def __getitem__(self, idx):
        if idx >= len(self.recording_files):
            raise IndexError("Index out of range")

        recording_path = self.recording_files[idx]
        # Placeholder function to load events.es file
            
        events_es_path = os.path.join(recording_path, "events.es")
        labelled_events_path = os.path.join(recording_path, "labelled_events.npy")
        recording_json_path = os.path.join(recording_path, "recording.json")

        with event_stream.Decoder(events_es_path) as decoder:
            chunks = [chunk for chunk in decoder]
            events = np.concatenate(chunks)

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


class TumblingDataset:
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

    def __init__(self, recordings_path, split="all", min_sat_events=1000, transform=None, test=False):
        self.recordings_path = recordings_path
        self.split = split
        self.test = test
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
        files = [f for f in glob.glob(self.recordings_path + "/**/metadata.json")]
        recording_files = []
        files_per_satellites = {}
        for file in files:
            json_load = open(file)
            dict_file = json.load(json_load)
            satellite_id = dict_file['calculated_properties']["objects"][0]["id"]
            if satellite_id in self.split:
                file_location = "/".join(file.split("/")[:-1])
                if os.path.isfile(file_location+"/labelled_events.npy"):
                    labelled_events = np.load(file_location+"/labelled_events.npy")
                    if len(labelled_events['label'])>0 :
                        if min(list(set(labelled_events['label']))) >= -1 and len(labelled_events[labelled_events['label'] == -1]) >= self.min_sat_events :
                            if not(satellite_id in files_per_satellites):
                                files_per_satellites[satellite_id] = {"occurences" : 1 , "locations":[file_location]}
                            else:
                                files_per_satellites[satellite_id]["locations"].append(file_location)
                                files_per_satellites[satellite_id]["occurences"] += 1
            json_load.close()
        for satellite_id in files_per_satellites.keys():
            length = files_per_satellites[satellite_id]['occurences']
            if self.test :
                recording_files += files_per_satellites[satellite_id]['locations'][int(0.8*length):]
            else :
                recording_files += files_per_satellites[satellite_id]['locations'][:int(0.8*length)]
        return recording_files

    def __getitem__(self, idx):
        if idx >= len(self.recording_files):
            raise IndexError("Index out of range")

        recording_path = self.recording_files[idx]
        # Placeholder function to load events.es file
        

        events_es_paths = [f for f in glob.glob(recording_path + "/*.es")]
        labelled_events_path = os.path.join(recording_path, "labelled_events.npy")
        recording_json_path = os.path.join(recording_path, "metadata.json")

        assert len(events_es_paths) == 1 
        with event_stream.Decoder(events_es_paths[0]) as decoder:
            chunks = [chunk for chunk in decoder]
            events = np.concatenate(chunks)

        labelled_events = np.load(labelled_events_path, allow_pickle=True)
        with open(recording_json_path, "r") as f:
            recording_data = json.load(f)

        labelled_events = labelled_events.view(dtype=np.dtype([('t', '<u8'), ('x', '<u2'), ('y', '<u2'), (('on', 'p'), '?'), ('label', '<i2')]))
        return {
            "events": events,
            "labelled_events": labelled_events,
            "recording_data": recording_data,
            "target_id": recording_data['calculated_properties']["objects"][0]["id"],
        }


class ClassificationAstrositeDataset(AstrositeDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        return sample["events"], sample['target_id']

class OnlySatellitesAstrositeDataset(AstrositeDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sat_events = sample['labelled_events']
        sat_events = sat_events[sat_events['label'] == -1]
        return sat_events, sample['target_id']

class LabelledEventsAstrositeDataset(AstrositeDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        return sample['labelled_events'], sample['target_id']
    
class LabelledEventsTumblingDataset(TumblingDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        return sample['labelled_events'], sample['target_id']
    
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
    
class SpectrogramDataset(AstrositeDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        one_hot_label = torch.zeros((len(self.split)), dtype=torch.float)
        sat_events = sample['labelled_events']
        last_timestamp = sample['events'][-1][0]
        first_timestamp = sample['events'][0][0]
        if last_timestamp-first_timestamp < 2e7:
            print("sample duration smaller than 20sec")
            return self.__getitem__(index+1)
        else :
            last_timestamp = 2e7
        first_timestamp = sample['events'][0][0]
        first_event = np.array([(first_timestamp,0,0,True,0)],dtype=np.dtype([('t', '<u8'), ('x', '<u2'), ('y', '<u2'), (('on', 'p'), '?'), ('label', '<i2')]))
        last_event = np.array([(last_timestamp,0,0,True,0)],dtype=np.dtype([('t', '<u8'), ('x', '<u2'), ('y', '<u2'), (('on', 'p'), '?'), ('label', '<i2')]))
        completed_labelled_datasets=(np.concatenate((first_event,sat_events[sat_events['label']==-1],last_event)))
        activity = []
        ts = []
        event_counter = 0
        event_activity = 0.0
        previous_t = 0  # µs
        t0 = completed_labelled_datasets[0][0]
        tf = completed_labelled_datasets[-1][0]
        tau = 100 #us
        event_count = 0
        time_step = 100
        for t in range(t0, tf, time_step):
            delta_t = t - previous_t
            event_activity *= math.exp(-float(time_step) / tau) # Leak
            while t > completed_labelled_datasets[event_count][0]:
                event_activity += 1                               # Integrate
                event_count += 1
            ts.append(t)
            activity.append(event_activity)
        #spectrogram = np.fft.fft(activity, )
        #spectrogram = scipy.signal.spectrogram(np.array(activity), fs=0.01, nfft=1024)
        #plt.imshow(spectrogram[2])
        spec = plt.specgram(activity, Fs=0.1, NFFT=1024)
        #plt.show()
        one_hot_label[np.where(np.array(self.split) == sample['target_id'])] = 1
        return torch.tensor(spec[0],dtype=torch.float).unsqueeze(0),one_hot_label

class MergedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        assert len(dataset1) == len(dataset2)
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        
    def __len__(self):
        return len(self.dataset1)
    
    def __getitem__(self, idx):
        assert self.dataset1[idx][1] == self.dataset2[idx][1]
        return torch.tensor(self.dataset1[idx][0]), torch.tensor(self.dataset2[idx][0]), self.dataset1[idx][1]
    
def build_merge_dataset(dataset_path, split=['all'],metadata_paths =['metadata/1','metadata/2'] ) :
    dataset1 = ClassificationAstrositeDataset(dataset_path, split=split)
    dataset2 = TrackingAstrositeDataset(dataset_path, split=split)

    assert len(dataset1) == len(dataset2)

    slicer = SliceByTime(time_window=1e6, include_incomplete=False)
    frame_transform = transforms.ToFrame(sensor_size=dataset1.sensor_size, time_window=1e5, include_incomplete=True)

    sliced_dataset1 = SlicedDataset(dataset1, slicer=slicer, metadata_path=metadata_paths[0], transform=frame_transform)
    sliced_dataset2 = SlicedDataset(dataset2, slicer=slicer, metadata_path=metadata_paths[1], transform=frame_transform)

    return MergedDataset(sliced_dataset1, sliced_dataset2)
    