import numpy as np # numpy - manipulate the packet data returned by depthai
import cv2 # opencv - display the video stream
import depthai # access the camera and its data packets
import consts.resource_paths # load paths to depthai resources

if not depthai.init_device(consts.resource_paths.device_cmd_fpath):
    print("Error initializing device. Try to reset it.")
    exit(1)

# Create the pipeline using the 'previewout' stream, establishing the first connection to the device.
p = depthai.create_pipeline(config={
    'streams': ['previewout'],
    'ai': {'blob_file': consts.resource_paths.blob_fpath}
})

if p is None:
    print('Error creating pipeline.')
    exit(2)

while True:
    # Retrieve data from the device
    # data is stored in packets
    data_packets = p.get_available_data_packets()

    for packet in data_packets:
          data = packet.getData()
          print("Received %s packet with shape=%s" % packet.stream_name, str(data.shape))

    if cv2.waitKey(1) == ord('q'):
        break

# In order to stop the pipeline object should be deleted, otherwise device will continue working.
# This is required if you are going to add code on exit.
del p
