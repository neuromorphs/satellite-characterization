import shutil, errno
import numpy as np
import json
import math
import matplotlib.pyplot as plt
from astrosite_dataset import ClassificationAstrositeDataset, AstrositeDataset, SpectrogramDataset


import os

dataset_path = '../dataset/recordings'
assert os.path.isdir(dataset_path)

target_list = ['50574', '47851', '37951', '39533', '43751', '32711', '27831', '45465',
       '46826', '42942', '42741', '41471', '43873', '40982', '41725', '43874', 
       '27711', '40892', '50005', '44637']

train_dataset = SpectrogramDataset(dataset_path, split=target_list, test=False)
test_dataset = SpectrogramDataset(dataset_path, split=target_list, test=True)
def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise

for file in test_dataset.recording_files :
    copyanything(file,"../filtered_dataset/" + "/".join(file.split("/")[2:]))