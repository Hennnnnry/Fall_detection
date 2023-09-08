import cv2
import numpy as np
import os


def is_black_frame(frame):
    """
    Check if the given frame is entirely black.
    """
    return np.all(frame == 0)


def remove_black_frames_from_folder(folder_path, save_folder):
    """
    Remove all black frames from the specified folder.
    """
    # Check if the provided path is a directory
    if not os.path.isdir(folder_path):
        return

    # List all the image frames within the folder
    frames = sorted(
        [os.path.join(folder_path, frame) for frame in os.listdir(folder_path) if frame.endswith(('.jpg', '.png'))])

    for i, frame_path in enumerate(frames):
        frame = cv2.imread(frame_path)

        # If the frame is black, remove it from the directory
        if is_black_frame(frame):
            os.remove(frame_path)


def compute_optical_flow(prev_frame, curr_frame):
    """
    Compute dense optical flow using Gunner Farneback's algorithm.
    """
    # Convert frames to grayscale for optical flow computation
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow using Farneback's method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Convert the computed flow to an RGB image for visualization
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    flow_image = np.zeros((curr_frame.shape[0], curr_frame.shape[1], 3), dtype=np.float32)
    flow_image[..., 0] = angle * (180 / np.pi / 2)
    flow_image[..., 1] = 255
    flow_image[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(flow_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return flow_rgb


def extract_optical_flow_from_folder(folder_path, save_folder):
    """
    Compute optical flow for frames in the given folder and save them to the designated folder.
    """
    # Check if the provided path is a directory
    if not os.path.isdir(folder_path):
        return []

    # List all the frames in the folder
    frames = sorted([os.path.join(folder_path, frame) for frame in os.listdir(folder_path)])
    prev_frame = None

    # Compute optical flow for each pair of consecutive frames
    for i, frame_path in enumerate(frames):
        curr_frame = cv2.imread(frame_path)

        if prev_frame is not None:
            flow_rgb = compute_optical_flow(prev_frame, curr_frame)
            save_path = os.path.join(save_folder, f"flow_{i}.png")
            cv2.imwrite(save_path, flow_rgb)

        prev_frame = curr_frame


# Paths for preprocessed data and where to save optical flow data
processed_path = 'Path_to_your_Preprocess_processed_path'
flow_dataset_path = 'Path_to_your_output'

# Create the directory for optical flow data if it doesn't exist
if not os.path.exists(flow_dataset_path):
    os.makedirs(flow_dataset_path)

# Define labels - 0 for non-fall and 1 for fall
labels = [0, 1]

# Loop through each label, compute and save optical flow
for label in labels:
    label_folder_path = os.path.join(processed_path, str(label))
    subfolders = [os.path.join(label_folder_path, subfolder) for subfolder in os.listdir(label_folder_path)]

    for subfolder in subfolders:
        # First, remove any black frames
        remove_black_frames_from_folder(subfolder, flow_dataset_path)

        # Then compute optical flow for the frames
        subfolder_name = os.path.basename(subfolder)
        save_folder = os.path.join(flow_dataset_path, str(label), subfolder_name)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        flow_data = extract_optical_flow_from_folder(subfolder, save_folder)

print("Feature extraction for DL complete")