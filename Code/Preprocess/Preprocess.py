import cv2
import os
import numpy as np

def process_frames(subfolder_path, save_folder, net, output_layers):
    # Loop over each frame in the subfolder
    for frame_file in sorted(os.listdir(subfolder_path)):
        if frame_file.endswith(('.jpg', '.png')):
            # Read the frame
            frame_path = os.path.join(subfolder_path, frame_file)
            frame = cv2.imread(frame_path)

            # Pre-process the image for YOLOv3 model
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            # Obtain model predictions
            outs = net.forward(output_layers)

            # Initialize a mask to save detected areas
            mask = np.zeros_like(frame)
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    # Filter detections with confidence higher than threshold
                    if confidence > 0.2:
                        # Get the bounding box coordinates
                        center_x = int(detection[0] * frame.shape[1])
                        center_y = int(detection[1] * frame.shape[0])
                        w = int(detection[2] * frame.shape[1])
                        h = int(detection[3] * frame.shape[0])
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        mask[y:y + h, x:x + w] = 255

            # Extract the detected object from the original image
            foreground = cv2.bitwise_and(frame, frame, mask=mask[:, :, 0])
            # Save the processed frame
            save_frame_path = os.path.join(save_folder, frame_file)
            cv2.imwrite(save_frame_path, foreground)

# Load YOLOv3 model
weights_path = "yolov3.weights"
cfg_path = "yolov3.cfg"

net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Paths to input dataset and where to save processed data
dataset_path = 'Path_to_your_dataset'
processed_path = 'Path_to_your_output'

if not os.path.exists(processed_path):
    os.makedirs(processed_path)

folders = ['Folder_name_of_Fall', 'Folder_name_of_Nonfall']

# Loop over each folder and subfolder, process frames
for folder in folders:
    folder_path = os.path.join(dataset_path, folder)
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            label = 1 if 'Fall' in subfolder else 0
            save_folder = os.path.join(processed_path, str(label), subfolder)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            process_frames(subfolder_path, save_folder, net, output_layers)

print("Frame processing complete!")
