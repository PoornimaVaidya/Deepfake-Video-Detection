import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Function to extract frames from a video
def extract_frames(video_path, output_size=(299, 299), frame_count=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < frame_count:
        print(f"[WARNING] Only {total_frames} frames available in: {video_path}")

    step = max(total_frames // frame_count, 1) 

    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, output_size)
        frame = frame.astype('float32') / 255.0
        frames.append(frame)

    cap.release()

    while len(frames) < frame_count:
        frames.append(np.zeros((output_size[0], output_size[1], 3), dtype=np.float32))

    return np.array(frames)

# Prediction function
def predict_video(video_path, model, output_size=(299, 299), frame_count=10):
    print(f"[INFO] Processing: {video_path}")
    frames = extract_frames(video_path, output_size, frame_count)
    frames = np.expand_dims(frames, axis=0)  # Shape: (1, 10, 299, 299, 3)
    print(f"[INFO] Input shape to model: {frames.shape}")
    prediction = model.predict(frames)
    label = "FAKE" if np.argmax(prediction) == 1 else "REAL"
    confidence = prediction[0][np.argmax(prediction)]
    print(f"Prediction: {label} (Confidence: {confidence:.2f})\n")

# Load the model for real-time detection
loaded_model = load_model('/Users/poornimavaidya/Desktop/Masters/Spring2025/Adv_AI/ProjectCode/Model/deepfake_detection_model_final.keras')

# Test video paths
fake_sample_path = "/Users/poornimavaidya/Downloads/archive/Fake/DFD_manipulated_sequences/01_02__exit_phone_room__YVGY8LOK.mp4"
real_sample_path = "/Users/poornimavaidya/Desktop/Masters/Spring2025/Adv_AI/ProjectCode/Data_set_100/archive/Real/26__exit_phone_room.mp4"
#"/Users/poornimavaidya/Desktop/Masters/Spring2025/Adv_AI/ProjectCode/Data_set_100/archive/Fake/01_02__exit_phone_room__YVGY8LOK.mp4"
#"/Users/poornimavaidya/Desktop/Masters/Spring2025/Adv_AI/ProjectCode/Data_set_100/archive/Real/26__exit_phone_room.mp4"

# Run predictions
print("Real Video Prediction:")
predict_video(real_sample_path, loaded_model)

print("Fake Video Prediction:")
predict_video(fake_sample_path, loaded_model)
