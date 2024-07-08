import numpy as np
import json
import math
import matplotlib.pyplot as plt

json_load = open("./files_per_satellites.json")
dict_file = json.load(json_load)
satellite_files = dict_file[list(dict_file.keys())[0]]["locations"]
all_events = []
for filename in satellite_files:
    print(filename)
    npy_name = filename.split('/')[:-1]
    npy_name = "/".join(npy_name) + "/labelled_events.npy"
    events = np.load(npy_name)
    flip_events = list(zip(*events))
    max_ts = flip_events[0][-1]
    all_events.append(events)

figure, axis = plt.subplots(5, 4) 

event_activity = 0.0
previous_t = 0  # µs
tau = 100.0     # µs

ts = []

print(len(events))
for k in range(len(all_events)):
    activity = []
    for chunk in all_events[k]:
        t, x, y, p,i = chunk
        if i == -1 :
            delta_t = t - previous_t
            event_activity *= math.exp(-float(delta_t) / tau)  # Leak
            event_activity += 1                                # Integrate
            previous_t = t
            ts.append(t)
            activity.append(event_activity)
    print(ts[-1]-ts[0])
    print(k)
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
