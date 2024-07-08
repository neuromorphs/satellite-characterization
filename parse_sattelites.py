import json
import glob
import os

files = [f for f in glob.glob("recordings/**/recording.json")]
files_per_satellites = {}
satellites_complete = {}
for file in files:
    json_load = open(file)
    dict_file = json.load(json_load)
    satellite_id = dict_file['object']['id']
    if not(dict_file['object']['id'] in files_per_satellites) :
        files_per_satellites[satellite_id] = {"occurences" : 1 , "locations":[file]}
        satellites_complete[satellite_id]=  {file:dict_file['object']}
    else :
        files_per_satellites[satellite_id]["locations"].append(file)
        files_per_satellites[satellite_id]["occurences"]+=1
        satellites_complete[satellite_id][file] =dict_file['object']

with open("files_per_satellites.json", "w") as outfile: 
    json.dump(files_per_satellites, outfile, indent=4)
