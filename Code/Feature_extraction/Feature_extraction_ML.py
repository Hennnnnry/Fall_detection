import cv2
import os
import numpy as np
from multiprocessing import Pool, cpu_count


def is_black_frame(frame):
    """Check if the given frame is completely black."""
    return np.all(frame == 0)


def compute_optical_flow(prev_frame, curr_frame):
    """Compute the optical flow between two consecutive frames using the Farneback algorithm."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


def extract_flow_features(flow):
    """Extract average magnitude and angle from the computed optical flow."""
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    avg_magnitude = np.mean(magnitude)
    avg_angle = np.mean(angle)
    return avg_magnitude, avg_angle


def process_subfolder(args):
    """Process a subfolder of frames, compute the optical flow features and aggregate them."""
    subfolder_path, label = args

    # Ensure the path is a directory
    if not os.path.isdir(subfolder_path):
        return None

    # List all image frames in the subfolder
    frames = sorted(
        [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.endswith(('.jpg', '.png'))])

    magnitudes = []
    angles = []

    # Compute optical flow for each consecutive frame pair in the subfolder
    for i in range(len(frames) - 1):
        prev_frame = cv2.imread(frames[i])
        curr_frame = cv2.imread(frames[i + 1])

        # Skip the pair if either frame is black
        if is_black_frame(prev_frame) or is_black_frame(curr_frame):
            continue
        flow = compute_optical_flow(prev_frame, curr_frame)
        avg_magnitude, avg_angle = extract_flow_features(flow)
        magnitudes.append(avg_magnitude)
        angles.append(avg_angle)

    # Aggregate optical flow feature values
    min_magnitude, max_magnitude, mean_magnitude, std_magnitude = np.min(magnitudes), np.max(magnitudes), np.mean(
        magnitudes), np.std(magnitudes)
    min_angle, max_angle, mean_angle, std_angle = np.min(angles), np.max(angles), np.mean(angles), np.std(angles)
    aggregated_features = (
    min_magnitude, max_magnitude, mean_magnitude, std_magnitude, min_angle, max_angle, mean_angle, std_angle)

    # Display progress
    print(f"Processed {subfolder_path}")
    return (subfolder_path.split('/')[-1], label, aggregated_features)


if __name__ == "__main__":
    dataset_path = 'Path_to_your_Preprocess_processed_path'
    labels = ['0', '1']
    features = []
    tasks = []

    # Gather all the tasks (subfolder paths) to process
    for label in labels:
        label_path = os.path.join(dataset_path, label)
        subfolders = [s for s in os.listdir(label_path) if os.path.isdir(os.path.join(label_path, s))]
        for subfolder in subfolders:
            subfolder_path = os.path.join(label_path, subfolder)
            tasks.append((subfolder_path, label))

    # Use multi-processing to speed up feature extraction from the frames
    with Pool(cpu_count()) as pool:
        results = pool.map(process_subfolder, tasks)

    # Filter out any None values from the results
    results = [r for r in results if r]
    features.extend(results)

    # Write the extracted features to a text file
    with open('flow_features_test.txt', 'w') as f:
        for item in features:
            f.write(f"{item[0]},{item[1]},{','.join(map(str, item[2]))}\n")

    # Notify the user when the process is completed
    print("Feature extraction for ML completed!")
