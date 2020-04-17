import numpy as np # numpy - manipulate the packet data returned by depthai
import cv2 # opencv - display the video stream
import depthai # access the camera and its data packets
import consts.resource_paths # load paths to depthai resources

# Load the device cmd file
if not depthai.init_device(consts.resource_paths.device_cmd_fpath):
    print("Error initializing device. Try to reset it.")
    exit(1)

# Create the pipeline using the 'metaout' and 'previewout' streams, establishing the first connection to the device.
p = depthai.create_pipeline(
    {
        # metaout - contains neural net output
        # previewout - color video
        'streams': ['metaout','previewout'],
        'ai':
        {
            # The paths below are based on the tutorial steps.
            'blob_file': "face-detection-retail-0004.blob",
            'blob_file_config': "face-detection-retail-0004.json"
        }
    }
)

if p is None:
    print('Pipeline was not created.')
    exit(2)

# Maintains a list of detections that will be applied to to the previewout stream.
# Only the most recent detections are applied.
entries_prev = []

while True:

    nnet_packets, data_packets = p.get_available_nnet_and_data_packets()

    for i, nnet_packet in enumerate(nnet_packets):
        # https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_retail_0004_description_face_detection_retail_0004.html#outputs
        # Shape: [1, 1, N, 7], where N is the number of detected bounding boxes
        for i, e in enumerate(nnet_packet.entries()):
            if e[0]['conf'] == 0.0:
                break

            # Clear previous detections if this is the first new detection.
            if i == 0:
                entries_prev.clear()

            # save entry for further usage (as image package may arrive not the same time as nnet package)
            entries_prev.append(e[0])

    for packet in data_packets:
        if packet.stream_name == 'previewout':
            data = packet.getData()
            # The format of previewout image is CHW (Chanel, Height, Width), but OpenCV needs HWC, so we
            # change shape (3, 300, 300) -> (300, 300, 3).
            data0 = data[0,:,:]
            data1 = data[1,:,:]
            data2 = data[2,:,:]
            frame = cv2.merge([data0, data1, data2])

            img_h = frame.shape[0]
            img_w = frame.shape[1]

            # iterate threw pre-saved entries & draw rectangle on image:
            for e in entries_prev:
                # The lower confidence threshold, the more false positives
                # label == 1.0 is a face
                if e['label'] == 1.0 and e['conf'] > 0.5:
                    # Determine rectangle bounds, then draw the image on the frame.
                    x1 = int(e['x_min'] * img_w)
                    y1 = int(e['y_max'] * img_h)

                    pt1 = x1, y1
                    pt2 = int(e['x_max'] * img_w), int(e['y_min'] * img_h)
                    cv2.rectangle(frame, pt1, pt2, (0, 0, 255))

                    pt_t1 = x1, y1 + 20
                    conf_text = "{:.2f}%".format(e['conf'] * 100)
                    cv2.putText(frame, conf_text, pt_t1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            frame = cv2.resize(frame, (300, 300))
            cv2.imshow('previewout', frame)

    # Exit if 'q' key is pressed with focus on the previewout video stream
    if cv2.waitKey(1) == ord('q'):
        break

del p  # in order to stop the pipeline object should be deleted, otherwise device will continue working. This is required if you are going to add code after the main loop, otherwise you can ommit it.
