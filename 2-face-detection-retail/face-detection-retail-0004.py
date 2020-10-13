from pathlib import Path

import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # access the camera and its data packets

device = depthai.Device('', False)

# Create the pipeline using the 'previewout' stream, establishing the first connection to the device.
pipeline = device.create_pipeline(config={
    'streams': ['previewout', 'metaout'],
    'ai': {
        "blob_file": str(Path("./face-detection-retail-0004.blob").resolve().absolute()),
        "blob_file_config": str(Path("./face-detection-retail-0004.json").resolve().absolute()),
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
        entries_prev = list(nnet_packet.getDetectedObjects())

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
                pt1 = int(e.x_min * img_w), int(e.y_min * img_h)
                pt2 = int(e.x_max * img_w), int(e.y_max * img_h)

                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)

            cv2.imshow('previewout', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# The pipeline object should be deleted after exiting the loop. Otherwise device will continue working.
# This is required if you are going to add code after exiting the loop.
del pipeline
del device
