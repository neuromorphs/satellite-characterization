import numpy as np
import json
import math
import matplotlib.pyplot as plt

json_load = open("../dataset/files_per_satellites.json")
dict_file = json.load(json_load)
satellite_files = dict_file[list(dict_file.keys())[0]]["locations"]
all_events = []
for filename in satellite_files:
    print(filename)
    npy_name = filename.split('/')[:-1]
    npy_name = "../dataset/" + "/".join(npy_name) + "/labelled_events.npy"
    events = np.load(npy_name)
    flip_events = list(zip(*events))
    max_ts = flip_events[0][-1]
    all_events.append(events)

#figure, axis = plt.subplots(5, 4) 

event_activity = 0.0
previous_t = 0  # µs
tau = 100.0     # µs

print(len(all_events))
for k in range(1): #len(all_events)):
    activity = []
    ts = []
    event_counter = 0
    event_activity = 0.0
    for t in range(all_events[k][0][0], all_events[k][-1][0]):
        event_activity *= math.exp(-1 / tau)
        if t == all_events[k][event_counter][0] :
            print(event_counter)
            event_activity += 1
            event_counter +=1
            ts.append(t)
            activity.append(event_activity)
    plt.plot(activity)
    plt.specgram(activity)
    plt.show()
    #print(np.mean(activity))
    print(len(activity))
    print(len(activity)/(ts[-1]-ts[0]))
    print("Time ratio: ", (ts[-1]-ts[0])/(all_events[k][-1][0]-all_events[k][0][0]))
    #print(k)
    #axis[k//4, k%4].plot(activity)
    axis[k//4, k%4].specgram(activity)
    #axis[k//4, k%4].xlabel("Frequencies")
    #axis[k//4, k%4].ylabel("Activity")
    axis[k//4, k%4].axes.get_xaxis().set_visible(False)
    axis[k//4, k%4].set_title(satellite_files[k].split("/")[-2])
print(len(ts))
#spectrogram = np.fft.fft(activity)
print(ts[-1])
#freq = np.fft.fftfreq(int(ts[-1]//1000000))
""" plt.plot(ts, activity)
plt.xlabel("Timestamps (microseconds)")
plt.ylabel("Activity")
plt.title("Plot of Global Activity")
plt.show() """


plt.show()
satellite_events = [event for event in events if event[-1]==-1]
print(len(satellite_events))
list_idx = []
for idx in satellite_events:
    if not(idx[-1] in list_idx):
        list_idx.append(idx[-1])
