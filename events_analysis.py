import numpy as np
import json
import math
import matplotlib.pyplot as plt
from scipy import fft
from astrosite_dataset import ClassificationAstrositeDataset, AstrositeDataset, TrackingAstrositeDataset


dataset_path = '../dataset/recordings'

target_list = ['50574', '47851'] #, '37951', '39533', '43751', '32711', '27831', '45465',
       #'46826', '42942', '42741', '41471', '43873', '40982', '41725', '43874',
       #'27711', '40892', '50005', '44637']

dataset1 = ClassificationAstrositeDataset(dataset_path, split=target_list)
dataset2 = TrackingAstrositeDataset(dataset_path, split=target_list)

def events_to_spectrogram(sample, tau) :
    file, id = sample
    activity = []
    ts = []
    event_counter = 0
    event_activity = 0.0
    previous_t = 0  # µs
    t0 = file[0][0]
    duration = file[-1][0] - file[0][0]
    for event in file:
        t = event[0]
        delta_t = t - previous_t
        event_activity *= math.exp(-float(delta_t) / tau) # Leak
        event_activity += 1                               # Integrate
        previous_t = t
        ts.append(t)
        activity.append(event_activity)
    spectrogram = np.fft.fft(activity)
    freq = np.fft.fftfreq(t.shape[-1], d=1/1100)

    return spectrogram

def plot_spectrogram(dataset, tau=100.0, n_files=20):
    figure, axis = plt.subplots(5, 4) 
    assert len(dataset)>=n_files
    for k in range(n_files): 
        print(len(dataset[k][0]))
        file, id = dataset[k]
        activity = []
        ts = []
        event_counter = 0
        event_activity = 0.0
        previous_t = 0  # µs
        t0 = file[0][0]
        tf = file[-1][0]
        duration = file[-1][0] - file[0][0]
        print(duration)
        event_count = 0
        time_step = 100
        for t in range(t0, tf, time_step):
            delta_t = t - previous_t
            event_activity *= math.exp(-float(time_step) / tau) # Leak
            while t > file[event_count][0]:
                event_activity += 1                               # Integrate
                event_count += 1
            ts.append(t)
            activity.append(event_activity)
        #spectrogram = np.fft.fft(activity, )
        spec = plt.specgram(activity, Fs=0.1, NFFT=1024)
        print(spec[0].shape)

        """  N = 256
        S = []
        for j in range(0, len(activity)+1, N):
            x = fft.fftshift(fft.fft(activity[j:j+N], n=N))[N//2:N]
            # assert np.allclose(np.imag(x*np.conj(x)), 0)
            Pxx = 10*np.log10(np.real(x*np.conj(x)))
            S.append(Pxx)
        S = np.array(S)
        print(S.shape) """
        axis[k//4, k%4].specgram(activity, Fs = 2)
        axis[k//4, k%4].axes.get_xaxis().set_visible(False)
        axis[k//4, k%4].set_title(id)
    plt.show()

#spectrogram = np.fft.fft(activity)
#freq = np.fft.fftfreq(int(ts[-1]//1000000))
""" plt.plot(ts, activity)
plt.xlabel("Timestamps (microseconds)")
plt.ylabel("Activity")
plt.title("Plot of Global Activity")
plt.show() """


plot_spectrogram(dataset2)
satellite_events = [event for event in events if event[-1]==-1]
print(len(satellite_events))
list_idx = []
for idx in satellite_events:
    if not(idx[-1] in list_idx):
        list_idx.append(idx[-1])
