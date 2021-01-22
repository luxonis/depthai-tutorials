#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
from utils import anchors, non_max_suppression_fast

threshold = 0.5
sigmoid_threshold = np.log(threshold/(1-threshold))
width = 256
height = 256

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(width, height)
cam_rgb.setInterleaved(False)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(str((Path(__file__).parent / Path('face_detection_back.blob')).resolve().absolute()))
cam_rgb.preview.link(detection_nn.input)

# Create outputs
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# Pipeline defined, now the device is assigned and pipeline is started
device = dai.Device(pipeline)
device.startPipeline()

# Output queues will be used to get the rgb frames and nn data from the outputs defined above
q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

frame = None
bboxes = []


def to_tensor_result(packet):
    return {
        tensor.name: np.array(packet.getLayerFp16(tensor.name)).reshape(tensor.dims)
        for tensor in packet.getRaw().tensors
    }


# nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
def frame_norm(frame, bbox):
    return (np.clip(np.array(bbox), 0, 1) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)


def decode_nn_packet(in_nn):
    tensors = to_tensor_result(in_nn)
    box_tensor = np.squeeze(np.concatenate([tensors['StatefulPartitionedCall/functional_1/tf_op_layer_regressors_1/regressors_1'], tensors['StatefulPartitionedCall/functional_1/tf_op_layer_regressors_2/regressors_2']], axis=1)).transpose(1, 0)
    score_tensor = np.squeeze(np.concatenate([tensors['StatefulPartitionedCall/functional_1/tf_op_layer_classificators_1/classificators_1'], tensors['StatefulPartitionedCall/functional_1/tf_op_layer_classificators_2/classificators_2']], axis=1))

    good_detections = np.where(score_tensor > sigmoid_threshold)[0]
    good_scores = 1.0 /(1.0 + np.exp(-score_tensor[good_detections]))

    numGoodDetections = good_detections.shape[0]

    keypoints = np.zeros((numGoodDetections, 6, 2))
    boxes = np.zeros((numGoodDetections, 4))
    for idx, detectionIdx in enumerate(good_detections):
        anchor = anchors[detectionIdx]

        sx = box_tensor[detectionIdx, 0]
        sy = box_tensor[detectionIdx, 1]
        w = box_tensor[detectionIdx, 2]
        h = box_tensor[detectionIdx, 3]

        cx = sx + anchor.x_center * width
        cy = sy + anchor.y_center * height

        cx /= width
        cy /= height
        w /= width
        h /= height

        for j in range(6):
            lx = box_tensor[detectionIdx, 4 + (2 * j) + 0]
            ly = box_tensor[detectionIdx, 4 + (2 * j) + 1]
            lx += anchor.x_center * width
            ly += anchor.y_center * height
            lx /= width
            ly /= height
            keypoints[idx, j, :] = np.array([lx, ly])

        boxes[idx, :] = np.array([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5])

    return boxes


while True:
    # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
    in_rgb = q_rgb.tryGet()
    in_nn = q_nn.tryGet()

    if in_rgb is not None:
        # if the data from the rgb camera is available, transform the 1D data into a HxWxC frame
        shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
        frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

    if in_nn is not None:
        bboxes = decode_nn_packet(in_nn)

    if frame is not None:
        # if the frame is available, draw bounding boxes on it and show the frame
        for raw_bbox in bboxes:
            bbox = frame_norm(frame, raw_bbox)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.imshow("rgb", frame)

    if cv2.waitKey(1) == ord('q'):
        break
