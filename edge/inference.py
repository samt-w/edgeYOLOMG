"""
this script takes trained YOLOMG .pt weights and uses them to 
detect drones in video input. It measures the performance of the detection
using mAP and FPS.
"""

# read video
# allow buffer to be initialised (to warm up GPU and for motion masks)
# begin timer
# extract frame
# compute motion mask for each frame
# make prediction
# end timer
# discard frame in buffer
# report accuracy and frame time
# convert accuracy and frame time to mAP and FPS

if __name__ == "__main__":
    pass