#!/usr/bin/python3

import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # access the camera and its data packets
import consts.resource_paths  # load paths to depthai resources
from pathlib import Path

if not depthai.init_device(consts.resource_paths.device_cmd_fpath):
    raise RuntimeError("Error initializing device. Try to reset it.")

# Create the pipeline using the 'previewout' stream, establishing the first connection to the device.
pipeline = depthai.create_pipeline(config={
    'streams': ['previewout', 'metaout'],
    'ai': {
        # The paths below are based on the tutorial steps.
        "blob_file": str((Path(__file__).parent / Path("face-detection-retail-0004.bin")).absolute()),
        "blob_file_config": str((Path(__file__).parent / Path("face-detection-retail-0004.json")).absolute()),
    }
})

if pipeline is None:
    raise RuntimeError('Pipeline creation failed!')

entries_prev = []

while True:
    # Retrieve data packets from the device.
    # A data packet contains the video frame data.
    nnet_packets, data_packets = pipeline.get_available_nnet_and_data_packets()

    for _, nnet_packet in enumerate(nnet_packets):
        entries_prev = []
        for _, e in enumerate(nnet_packet.entries()):
            if e[0]['id'] == -1.0 or e[0]['confidence'] == 0.0:
                break
            if e[0]['confidence'] > 0.5:
                entries_prev.append(e[0])

    for packet in data_packets:
        # By default, DepthAI adds other streams (notably 'meta_2dh'). Only process `previewout`.
        if packet.stream_name == 'previewout':
            data = packet.getData()
            # change shape (3, 300, 300) -> (300, 300, 3)
            data0 = data[0, :, :]
            data1 = data[1, :, :]
            data2 = data[2, :, :]
            frame = cv2.merge([data0, data1, data2])

            img_h = frame.shape[0]
            img_w = frame.shape[1]

            for e in entries_prev:
                pt1 = int(e['left'] * img_w), int(e['top'] * img_h)
                pt2 = int(e['right'] * img_w), int(e['bottom'] * img_h)

                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)

            cv2.imshow('previewout', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# The pipeline object should be deleted after exiting the loop. Otherwise device will continue working.
# This is required if you are going to add code after exiting the loop.
del pipeline
