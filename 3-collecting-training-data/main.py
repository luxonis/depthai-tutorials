import argparse

import consts.resource_paths
import cv2
import depthai
import numpy as np

if not depthai.init_device(consts.resource_paths.device_cmd_fpath):
    raise RuntimeError("Error initializing device. Try to reset it.")

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', default=0.1, type=float, help="Maximum difference between packet timestamps to be considered as synced")
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

latest_left = None
lr_pairs = {}

# https://stackoverflow.com/a/7859208/5494277
def step_norm(value):
    return round(value / args.threshold) * args.threshold
def seq(packet):
    return packet.getMetadata().getSequenceNum()
def tst(packet):
    return packet.getMetadata().getTimestamp()

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
            latest_left = packet
        elif packet.stream_name == "right" and latest_left is not None and seq(latest_left) == seq(packet):
            lr_pairs[step_norm(tst(packet))] = (latest_left, packet)
        elif packet.stream_name == 'previewout':
            timestamp_normalized = step_norm(tst(packet))
            pair = lr_pairs.pop(timestamp_normalized, None)
            if pair is not None:
                cv2.imshow('left', pair[0].getData())
                cv2.imshow('right', pair[1].getData())
                cv2.imshow('previewout', frame)
            else:
                for key in list(lr_pairs.keys()):
                    if key < timestamp_normalized:
                        del lr_pairs[key]

    if cv2.waitKey(1) == ord('q'):
        break

del p
depthai.deinit_device()