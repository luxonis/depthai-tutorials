import argparse

import consts.resource_paths
import cv2
import depthai
import numpy as np

if not depthai.init_device(consts.resource_paths.device_cmd_fpath):
    raise RuntimeError("Error initializing device. Try to reset it.")

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', default=0.5, type=float, help="Maximum difference between packet timestamps to be considered as synced")
parser.add_argument('-f', '--fps', default=30, type=int, help="FPS of the cameras")
args = parser.parse_args()

p = depthai.create_pipeline(config={
    "streams": [
        "left", "right", "previewout"
        # {'name': 'left', 'max_fps': args.fps},
        # {'name': 'right', 'max_fps': args.fps},
        # {'name': 'previewout', 'max_fps': args.fps},
        # {'name': 'disparity_color', 'max_fps': 2.0},
    ],
    'depth':
    {
        'calibration_file': consts.resource_paths.calib_fpath,
        'padding_factor': 0.3,
        'depth_limit_m': 10.0, # In meters, for filtering purpose during x,y,z calc
        'confidence_threshold' : 0.5, #Depth is calculated for bounding boxes with confidence higher than this number
    },
    "ai": {
        "blob_file": consts.resource_paths.blob_fpath,
        "blob_file_config": consts.resource_paths.blob_config_fpath
    },
    'camera': {
        'mono': {
            'resolution_h': 720, 'fps': 30
        },
    },
})

if p is None:
    raise RuntimeError("Error initializing pipelne")

latest_packets = {}

while True:
    data_packets = p.get_available_data_packets()

    for packet in data_packets:
        print(packet.stream_name, packet.getMetadata().getTimestamp(), packet.getMetadata().getSequenceNum(), packet.getMetadata().getCameraName())
        if packet.stream_name == 'previewout':
            data = packet.getData()
            data0 = data[0, :, :]
            data1 = data[1, :, :]
            data2 = data[2, :, :]
            frame = cv2.merge([data0, data1, data2])
        elif packet.stream_name == 'left' or packet.stream_name == 'right' or packet.stream_name == 'disparity_color':
            frame = packet.getData()
        else:
            continue

        if packet.stream_name == "left":
            latest_packets['left'] = {'frame': frame, 'packet': packet}
            latest_packets.pop('right', None)
            latest_packets.pop('previewout', None)
        elif packet.stream_name == "right" and 'left' in latest_packets and packet.getMetadata().getSequenceNum() == latest_packets['left']['packet'].getMetadata().getSequenceNum():
            latest_packets['right'] = {'frame': frame, 'packet': packet}
        elif packet.stream_name == 'previewout' and 'left' in latest_packets and abs(packet.getMetadata().getTimestamp() - latest_packets['left']['packet'].getMetadata().getTimestamp()) < args.threshold:
            latest_packets["previewout"] = {'frame': frame, 'packet': packet}

        if len(latest_packets) == 3:
            for key, value in list(latest_packets.items()):
                cv2.imshow(value['packet'].stream_name, value['frame'])
                latest_packets.pop(key, None)

    if cv2.waitKey(1) == ord('q'):
        break

del p
depthai.deinit_device()