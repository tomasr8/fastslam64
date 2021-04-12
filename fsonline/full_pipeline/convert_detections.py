import matplotlib.pyplot as plt
import numpy as np
import json

with open("detections.json") as f:
    detections = json.load(f)

detections_np = np.zeros((len(detections), 3), dtype=np.float32)
TIME_START = 1614630690

for i, d in enumerate(detections):
    stamp = d['stamp']

    d['stamp'] = (stamp['secs'] - TIME_START) * 1000 + np.round(stamp['nsecs'] / 1e6)

with open("detections_converted.json", "w") as f:
    json.dump(detections, f)
