import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import tonic
from tonic import SlicedDataset
from tonic.slicers import SliceByTime
import numpy as np
import itertools


class EventsLoader(DataLoader):
    def __init__(self, dataset, bins_per_sample=8, sample_time=1e6, down_sample=1) :
        super(EventsLoader)
        self.bins_per_sample = bins_per_sample
        self.sample_time = sample_time
        self.down_sample = down_sample
        self.dataset = dataset
        self.samples = []
        self.files_idx = []
        self.item_count = 0
        self.dimx, self.dimy = 1280, 720 #Gen4 (346,260) #DAVIS346
        for recording in dataset:
            events = recording['events']
            duration = events[-1][0] - events[0][0]
            num_samples = duration//sample_time
            self.item_count += num_samples
            self.files_idx.append(num_samples)

    
    def __len__(self) :
        return self.item_count

    def __getitem__(self, index):
        file_idx = self.find_start_index(index, 0, len(self.files_idx)-1)
        file = self.dataset[file_idx]
        #timestamp = file['events'][0][0] + (index-file_idx)*self.sample_time
        events = file['events']
        events_attributes = np.array(list(map(list,zip(*events))))
        print(len(events_attributes))
        print(events_attributes.shape)
        events_attributes_ts = events_attributes[0]
        events_attributes = np.concatenate(events_attributes[:3], events_attributes_ts)
        print(events_attributes.shape)
        

        slicing_time_window = 50000  # microseconds
        slicer = SliceByTime(time_window=slicing_time_window)
        sliced_dataset = SlicedDataset(
            events_attributes, slicer=slicer, metadata_path="./metadata/nmnist"
        )

        #labelled_events = file['labelled_events']
        #duration = (events[-1][0]-events[0][0]) //self.sample_time
        #file_sampled = torch.zeros((self.bins_per_sample,self.dimx//self.down_sample, self.dimy//self.down_sample, 2))
        events, labels = self.load_sample(file_idx, index)
        print(type(events))
        print(type(events[0]))
        events, labels = torch.stack(events), torch.stack(labels)
        events[:,0] = (events[:,0] - events[0,0])//self.sample_time
        labels[:,0] = (labels[:,0] - labels[0,0])//self.sample_time
        events_bin = torch.zeros((self.bins_per_sample,self.dimx//self.down_sample, self.dimy//self.down_sample, 2))
        labels_bin = torch.zeros((self.bins_per_sample,self.dimx//self.down_sample, self.dimy//self.down_sample, 2,1))
        events_bin[events] = 1

        labels_bin[labels] = 1
        labels_bin = labels_bin[torch.where(labels_bin[:,:,:,:] == -1)][:4]
        return events, labels

    def find_start_index(self,index, start, end):
        if index >= self.files_idx[end]:
            return end
        elif end - start > 1:
            middle = (start + end) // 2
            if index < self.files_idx[middle]:
                return self.find_start_index(index, start, middle)
            else:
                return self.find_start_index(index, middle, end)
        else:
            return start
        
    def find_event_indexes(self,events, index, start, end):
        if index >= events[end][0]:
            return end
        elif end - start > 1:
            middle = (start + end) // 2
            if  index < events[middle][0]:
                return self.find_event_indexes(events, index, start, middle)
            else:
                return self.find_event_indexes(events, index, middle, end)
        else:
            return start
        
    def load_sample(self, file_idx, index):
        file = self.dataset[file_idx]
        timestamp = file['events'][0][0] + (index-file_idx)*self.sample_time
        events = file['events']
        labelled_events = file['labelled_events']
        print(file_idx)
        start_event_index = self.find_event_indexes(events,timestamp, 0,len(events)-1)
        end_event_index = self.find_event_indexes(events,timestamp + self.sample_time, start_event_index, len(events)-1)
        start_label_event_index = self.find_event_indexes(labelled_events,(index-file_idx)*self.sample_time, 0,len(labelled_events)-1)
        end_label_event_index = self.find_event_indexes(labelled_events,(index-file_idx+1)*self.sample_time, start_label_event_index, len(labelled_events)-1)
        print(start_event_index)
        print(end_event_index)
        print(events[end_event_index] - events[start_event_index])
        events_ds = events[start_event_index:end_event_index]
        labels_ds = labelled_events[start_label_event_index:end_label_event_index]
        return events_ds, labels_ds





