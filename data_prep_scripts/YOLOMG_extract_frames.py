import cv2
import os

# Set video folder path
video_folder = './videos/'
image_folder = './images/'
video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

# Iterate through all video files
for video_file in video_files:
    # Create an image folder for each video
    video_image_folder = os.path.join(image_folder, video_file.split('.')[0])
    if not os.path.exists(video_image_folder):
        os.makedirs(video_image_folder)

    # Start processing the video file
    vc = cv2.VideoCapture(os.path.join(video_folder, video_file))
    c = 0
    rval = vc.isOpened()

    while rval:
        c += 1
        rval, frame = vc.read()
        if rval:
            # Write each frame to the corresponding folder
            name0 = str(c)
            name = name0.zfill(4)
            cv2.imwrite(os.path.join(video_image_folder, video_file.split('.')[0] + '_' + name + '.jpg'), frame)
            print(f'extract frame from {video_file}:', name)
        else:
            break

    vc.release()