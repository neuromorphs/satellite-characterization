import os
import numpy as np
import json
import event_stream
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tonic.slicers import SliceByTime
from tonic import SlicedDataset, transforms

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def generate_heatmap(y,x, size=(10,15), sigma=5):
    #print("Generate heatmap", target.shape) #(10, 2, 40, 60)
    #heatmap, center, radius, k=1
    #centernet draw_umich_gaussian for not mse_loss
    k=1
    heatmap = np.zeros(size)
    height, width = size
    diameter = sigma#2 * radius + 1
    radius = sigma//2
    #c = norse.torch.functional.receptive_field.covariance_matrix(sigma, sigma)
    #heatmap = norse.torch.functional.receptive_field.gaussian_kernel(size, c)
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    tops = [[x,y]]


    for top in tops:
        x,y = top

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    heatmap[heatmap>1] = 1
    return np.expand_dims(heatmap,0)


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
            with open(file, "r") as f:
                dict_file = json.load(f)
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

        events_es_path = os.path.join(recording_path, "events.es")
        labelled_events_path = os.path.join(recording_path, "labelled_events.npy")
        recording_json_path = os.path.join(recording_path, "recording.json")

        # load events.es file
        with event_stream.Decoder(events_es_path) as decoder:
            chunks = [chunk for chunk in decoder]
            events = np.concatenate(chunks)

        with open(recording_json_path, "r") as f:
            recording_data = json.load(f)

        labelled_events = np.load(labelled_events_path, allow_pickle=True)

        labelled_events = labelled_events.view(dtype=np.dtype([('t', '<u8'), ('x', '<u2'), ('y', '<u2'), (('on', 'p'), '?'), ('label', '<i2')]))
        return {
            "events": events,
            "labelled_events": labelled_events,
            "recording_data": recording_data,
            "target_id": recording_data["object"]["id"]}


class ClassificationAstrositeDataset(AstrositeDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        return sample["events"], sample['target_id']


class BinaryClassificationAstrositeDataset(AstrositeDataset):
    def __init__(
            self, recordings_path, split="all", min_sat_events=1000,
            transform=None, perm_seed: int = 0):
        super().__init__(
            recordings_path, split=split, min_sat_events=min_sat_events,
            transform=transform)
        # create permutation
        rng_state = torch.get_rng_state()
        torch.manual_seed(perm_seed)
        self.is_satellite = torch.rand(len(self)) >= 0.5
        torch.set_rng_state(rng_state)

    def __len__(self):
        return super().__len__() * 2

    def __getitem__(self, index):
        sample = super().__getitem__(index//2)
        sat_events = sample['labelled_events']
        if index%2 == 0 :
            return sat_events, 1
        else :
            mask = sat_events["label"] < 0
            return sat_events[~mask], 0
    
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

class MergedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        assert len(dataset1) == len(dataset2)
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.c,self.h,self.w = self.dataset1[0][0].shape[1:]

    def __len__(self):
        return len(self.dataset1)
    
    def __getitem__(self, idx):
        #if self.dataset1[idx][0].shape[0] != 10 or self.dataset2[idx][0].shape[0] != 10 :
        #    return self.__getitem__(idx+1)
        events = np.nonzero(self.dataset2[idx][0])
        if len(events[0]) == 0 :
            return self.dataset1[idx][0], np.zeros((1,self.h//4,self.w//4))
        y,x = int(np.mean(events[2])//4), int(np.mean(events[3])//4)
        return self.dataset1[idx][0], generate_heatmap(y,x, size=(self.h//4, self.w//4)) #, self.dataset1[idx][1]
    
def build_merge_dataset(dataset_path, split=['all'], metadata_paths =['metadata/3','metadata/4'], crop=False,) :
    dataset1 = ClassificationAstrositeDataset(dataset_path, split=split)
    dataset2 = TrackingAstrositeDataset(dataset_path, split=split)

    assert len(dataset1) == len(dataset2)
    n_slices = 0
    event_density = 0
    slice_sample = 1e7
    slice_bin = 1e6
    for sample in dataset1 :
        duration = sample[0][-1][0] - sample[0][0][0]
        event_density += len(sample[0])
        n_slices+=duration//slice_sample + 1
    print("Expected size:", n_slices)
    print("Average number of events per bin:", event_density/(10*n_slices))

    slicer = SliceByTime(time_window=slice_sample, overlap=0 ,include_incomplete=False)
    if crop:
        preprocessing_transform = transforms.Compose(
        [
            #transforms.Denoise(filter_time=1000),
            transforms.Downsample(spatial_factor=0.1),
            transforms.CenterCrop(sensor_size=(128,72,2), size=(60,40)),
            transforms.ToFrame(sensor_size=(60,40,2), time_window=slice_bin, include_incomplete=True)
        ])
    else:
        preprocessing_transform = transforms.Compose(
        [
            transforms.Denoise(filter_time=1000),
            transforms.Downsample(spatial_factor=0.2),
            transforms.ToFrame(sensor_size=(256,144,2), time_window=slice_bin, include_incomplete=True)
        ])

    sliced_dataset1 = SlicedDataset(dataset1, slicer=slicer, metadata_path=metadata_paths[0], transform=preprocessing_transform)
    sliced_dataset2 = SlicedDataset(dataset2, slicer=slicer, metadata_path=metadata_paths[1], transform=preprocessing_transform)
    count = 0
    count_incomplete = 0
    for s1 in sliced_dataset1:
        if len(s1[0]) :
            if np.sum(s1[0]) == 0:
                count +=1
            else:
                if s1[0].shape[0] != 10 :
                    print(s1[0].shape)
        else :
            count += 1
    print("Empty sample ratio:", (count/len(sliced_dataset1)))

    return MergedDataset(sliced_dataset1, sliced_dataset2)

class EgoMotionDataset(torch.utils.data.Dataset):
    def __init__(
            self, size, width, height, velocity: list, obj_size: int = 3,
            n_objects: int = 10, noise_level: int = 0.01, shift: int = 10,
            period: int = 500, period_sim = 10000, label=1):
        self.size = size
        self.obj_size = obj_size
        self.velocity = velocity
        self.width = width
        self.height = height
        self.n_objects = n_objects
        self.noise_level = noise_level
        self.shift = shift
        self.period = period
        self.period_sim = period_sim
        self.label = label

    def __len__(self):
        return int(self.size)

    def _generate_sample(self):

        time_slices = []
        coherent_noise = np.random.uniform(
            size=(self.height + 2*self.obj_size, self.width + 2*self.obj_size)
            ) < self.noise_level
        for _ in range(int(self.period_sim / self.period)):
            coherent_noise = np.roll(coherent_noise, shift=self.shift, axis=1)
            time_slices.append(torch.tensor(coherent_noise))
        noise = torch.stack(time_slices)

        objects = torch.zeros_like(noise)
        sat = torch.zeros_like(noise)

        rv = np.random.rand()
        rd = np.random.uniform(size=2) * (
                np.random.randint(0, 2, size=2)*2-1)
        
        for n in range(self.n_objects):
            if self.label==1:
                if n == 0:
                    direction = np.random.uniform(size=2) * (
                        np.random.randint(0, 2, size=2)*2-1)
                    direction = direction / np.sqrt(direction[0]**2 + direction[1]**2)
                    offset = [
                        np.random.randint(int(self.width*.4),high=int(self.width*.6)),
                        np.random.randint(int(self.height*.4),high=int(self.height*.6))]
                    velocity = self.velocity[0] + np.random.rand()*.5 / (
                        self.velocity[1] - self.velocity[0])
                    self.sat_offset = offset
                    
                else:
                    direction = rd
                    direction = direction / np.sqrt(direction[0]**2 + direction[1]**2)
                    offset = [
                        np.random.randint(self.obj_size, high=self.width),
                        np.random.randint(self.obj_size, high=self.height)]

                    velocity = self.velocity[0] + rv / (
                        self.velocity[1] - self.velocity[0])
            else:
                direction = rd
                direction = direction / np.sqrt(direction[0]**2 + direction[1]**2)
                offset = [
                    np.random.randint(self.obj_size, high=self.width),
                    np.random.randint(self.obj_size, high=self.height)]

                velocity = self.velocity[0] + rv / (
                    self.velocity[1] - self.velocity[0])
                
            Y, X = np.ogrid[:self.obj_size, :self.obj_size]
            center = (int(self.obj_size / 2), int(self.obj_size / 2))
            dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
            mask = torch.tensor(dist_from_center <= self.obj_size // 2)

            
            self.sat_trajectory = []
            for i, o in enumerate(objects if n!=0 else sat):
                x = direction[0] * i * velocity + offset[0]
                x = int(direction[0] * i * velocity + offset[0])
                x = max(0, min(x, self.width + self.obj_size))
                y = int(direction[1] * i * velocity + offset[1])
                y = max(0, min(y, self.height + self.obj_size))
                o[y: y + self.obj_size, x: x + self.obj_size][mask] = 1

        sample = objects + sat
        sample[sample>1] = 1

        #sample = objects
        #print(sat.shape)
        #print(np.nonzero(sat[-1].numpy()))
        coords = np.nonzero(sat[-1].numpy())
        sample = np.expand_dims(sample[:,10:50,20:80],axis=1)
        coord_y, coord_x = int(np.mean(coords[0])-10)//4, int(np.mean(coords[1])-20)//4
        #print(sample.shape)
        #print(generate_heatmap((self.sat_trajectory[0][1]-20)//4,(self.sat_trajectory[0][0]-10)//4,size=((self.height-20)//4, (self.width-40)//4)).shape)
        #print(coord_x, coord_y)
        return np.concatenate((sample,sample), axis=1), generate_heatmap(coord_y, coord_x, size=((self.height-20)//4, (self.width-40)//4))

    def __getitem__(self, index: int):
        sample = self._generate_sample()
        return sample