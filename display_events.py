import cv2
import numpy as np
from astrosite_dataset import AstrositeDataset

def display_recordings(sample, satellite_only=False, frame_rate=10, accumulation_time=1000000):
    events = sample['events']
    labelled_events = sample['labelled_events']
    duration = events[-1][0]-events[0][0]
    frame = np.zeros((720,1280,3))
    assert min(set(list(zip(*labelled_events))[-1]))==-1
    time_count = 0
    if satellite_only :
        for event in labelled_events :
            ts, x, y, p, idx = event
            assert p == True
            if idx == -1 :
                frame[y][x][int(p)] = 255
                if ts-time_count > accumulation_time :
                    cv2.imshow("Events from satellite", frame)
                    cv2.waitKey(1000//frame_rate)
                    frame.fill(0)
                    time_count = event[0]
    else :
        for event in events :
            ts, x, y, p = event
            frame[y][x][int(p)] = 255
            if ts-time_count > accumulation_time :
                cv2.imshow("Events from satellite", frame)
                cv2.waitKey(1000//frame_rate)
                frame.fill(0)
                time_count = event[0]
