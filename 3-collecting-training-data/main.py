import argparse
from pathlib import Path
from multiprocessing import Process
from uuid import uuid4

import consts.resource_paths
import cv2
import depthai

if not depthai.init_device(consts.resource_paths.device_cmd_fpath):
    raise RuntimeError("Error initializing device. Try to reset it.")

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', default=0.03, type=float, help="Maximum difference between packet timestamps to be considered as synced")
parser.add_argument('-p', '--path', default="data", type=str, help="Path where to store the captured data")
parser.add_argument('-d', '--dirty', action='store_true', default=False, help="Allow the destination path not to be empty")
args = parser.parse_args()

dest = Path(args.path).resolve().absolute()
if dest.exists() and len(list(dest.glob('*'))) != 0 and not args.dirty:
    raise ValueError(f"Path {dest} contains {len(list(dest.glob('*')))} files. Either specify new path or use \"--dirty\" flag to use current one")
dest.mkdir(parents=True, exist_ok=True)

p = depthai.create_pipeline(config={
    "streams": ["left", "right", "previewout"],
    'depth':
    {
        'calibration_file': consts.resource_paths.calib_fpath,
        'padding_factor': 0.3,
        'depth_limit_m': 10.0,
        'confidence_threshold' : 0.5,
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
procs = []

# https://stackoverflow.com/a/7859208/5494277
def step_norm(value):
    return round(value / args.threshold) * args.threshold
def seq(packet):
    return packet.getMetadata().getSequenceNum()
def tst(packet):
    return packet.getMetadata().getTimestamp()
def store_frames(left, right, rgb):
    global procs
    frames_path = dest / Path(str(uuid4()))
    frames_path.mkdir(parents=False, exist_ok=False)
    new_procs = [
        Process(target=cv2.imwrite, args=(str(frames_path / Path("left.png")), left)),
        Process(target=cv2.imwrite, args=(str(frames_path / Path("right.png")), right)),
        Process(target=cv2.imwrite, args=(str(frames_path / Path("rgb.png")), rgb)),
    ]
    for proc in new_procs:
        proc.start()
    procs += new_procs

while True:
    data_packets = p.get_available_data_packets()

    for packet in data_packets:
        print(packet.stream_name, packet.getMetadata().getTimestamp(), packet.getMetadata().getSequenceNum(), packet.getMetadata().getCameraName())
        if packet.stream_name == "left":
            latest_left = packet
        elif packet.stream_name == "right" and latest_left is not None and seq(latest_left) == seq(packet):
            lr_pairs[step_norm(tst(packet))] = (latest_left, packet)
        elif packet.stream_name == 'previewout':
            data = packet.getData()
            data0 = data[0, :, :]
            data1 = data[1, :, :]
            data2 = data[2, :, :]
            frame = cv2.merge([data0, data1, data2])

            timestamp_normalized = step_norm(tst(packet))
            pair = lr_pairs.pop(timestamp_normalized, None)
            if pair is not None:
                store_frames(pair[0].getData(), pair[1].getData(), frame)
                cv2.imshow('left', pair[0].getData())
                cv2.imshow('right', pair[1].getData())
                cv2.imshow('previewout', frame)
            else:
                for key in list(lr_pairs.keys()):
                    if key < timestamp_normalized:
                        del lr_pairs[key]

    if cv2.waitKey(1) == ord('q'):
        break

for proc in procs:
    proc.join()
del p
depthai.deinit_device()