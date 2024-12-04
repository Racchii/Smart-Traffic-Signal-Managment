import cv2
import torch
import numpy as np
import time

# Load your custom-trained models
ambulance_model = torch.hub.load(
    'ultralytics/yolov5', 'custom',
    path='C:/Users/ASUS/Desktop/project/yolov5/runs/train/ambulance_detection2/weights/best.pt',
    force_reload=True
)
motorbike_rikshaw_model = torch.hub.load(
    'ultralytics/yolov5', 'custom',
    path='D:/pendrive/motor_detection/yolov5/runs/train/exp/weights/best.pt',
    force_reload=True
)
car_truck_model = torch.hub.load(
    'ultralytics/yolov5', 'custom',
    path='D:/car_truck model/yolov5/runs/train/exp10/weights/best.pt',
    force_reload=True
)

# Video files for four lanes
video_files = [
    'C:/Users/ASUS/Downloads/r1/vid1.mp4',  # Lane 1
    'C:/Users/ASUS/Downloads/r1/vid2.mp4',  # Lane 2
    'C:/Users/ASUS/Downloads/r1/vid3.mp4',  # Lane 3
    'C:/Users/ASUS/Downloads/r1/vid4.mp4'   # Lane 4
]
caps = [cv2.VideoCapture(video) for video in video_files]

# Initialize lane data
lanes = {
    f"lane{i+1}": {
        "total_count": 0,
        "vehicle_counts": {"car": 0, "motorcycle": 0, "rickshaw": 0, "truck": 0},
        "ambulance": False,
        "signal": "red",
        "green_time": 0,  # Default green time
        "time_left": 0,  # Time left for the current signal
        "frozen": False  # New key to track whether the lane is frozen
    } for i in range(4)
}

# Startup phase settings
startup_start_time = time.time()
startup_phase = True

# Initialize lane management variables
start_time = None
current_green_lane = None

# Resize frames for grid display (adjusted for better screen fit)
frame_width = 640
frame_height = 360
stacked_frame_width = 1280  # Adjust the width of the stacked frame
stacked_frame_height = 720  # Adjust the height of the stacked frame

# Function to process a single frame
def process_frame(frame, lane_name):
    global lanes, current_green_lane
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Reset vehicle-specific counts
    lanes[lane_name]["vehicle_counts"] = {"car": 0, "motorcycle": 0, "rickshaw": 0, "truck": 0}
    ambulance_present = False

    # Apply ambulance detection only for Lane 2
    if lane_name == "lane2":
        # Use ambulance detection model
        ambulance_results = ambulance_model(frame_rgb)
        for det in ambulance_results.xyxy[0]:  # Iterate through detections
            x1, y1, x2, y2, conf, cls = det.tolist()
            label = ambulance_model.names[int(cls)]
            if conf >= 0.3 and "ambulance" in label.lower():
                ambulance_present = True
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, "Ambulance", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Continue with other detections (motorbike, rickshaw, car, truck) for all lanes
    motorbike_rikshaw_results = motorbike_rikshaw_model(frame_rgb)
    for det in motorbike_rikshaw_results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det.tolist()
        label = motorbike_rikshaw_model.names[int(cls)]
        if conf >= 0.2:
            if "motorbike" in label.lower() or "bike" in label.lower() or "motorcycle" in label.lower():
                lanes[lane_name]["vehicle_counts"]["motorcycle"] += 1
            elif "rickshaw" in label.lower() or "auto" in label.lower():
                lanes[lane_name]["vehicle_counts"]["rickshaw"] += 1
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Use car and truck model for all lanes
    car_truck_results = car_truck_model(frame_rgb)
    for det in car_truck_results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det.tolist()
        label = car_truck_model.names[int(cls)]
        if conf >= 0.3:
            if "car" in label.lower():
                lanes[lane_name]["vehicle_counts"]["car"] += 1
            elif "truck" in label.lower():
                lanes[lane_name]["vehicle_counts"]["truck"] += 1
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Update lane data
    lanes[lane_name]["ambulance"] = ambulance_present
    lanes[lane_name]["total_count"] = sum(lanes[lane_name]["vehicle_counts"].values())

    # Display lane-specific details
    signal_color = (0, 255, 0) if lanes[lane_name]["signal"] == "green" else (0, 0, 255)
    cv2.putText(frame, f"Signal: {lanes[lane_name]['signal']}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, signal_color, 2)
    cv2.putText(frame, f"Total: {lanes[lane_name]['total_count']}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    for idx, (veh, count) in enumerate(lanes[lane_name]["vehicle_counts"].items()):
        cv2.putText(frame, f"{veh.capitalize()}: {count}", (10, 120 + idx * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display green time for every lane (whether it's red or green)
    cv2.putText(frame, f"Green Time Left: {lanes[lane_name]['time_left']}s", (10, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame


# Paths to extra videos
extra_videos = [
    "C:/Users/ASUS/Downloads/extra_videos/vid6.mp4",
    "C:/Users/ASUS/Downloads/extra_videos/vid5.mp4",
    "C:/Users/ASUS/Downloads/aammmbbb.mp4"
]

# Function to play an extra video
def play_extra_video(video_path):
    extra_cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = extra_cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (stacked_frame_width, stacked_frame_height))
        cv2.imshow('Extra Video', frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    extra_cap.release()

# Define priorities for each lane
lane_priorities = {
    "lane1": 1,  # Lowest priority
    "lane2": 2,
    "lane3": 3,
    "lane4": 4   # Highest priority
}

# Function to play an extra video for a specific lane
def play_extra_video(video_path):
    extra_cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = extra_cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (stacked_frame_width, stacked_frame_height))
        cv2.imshow('Extra Video', frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    extra_cap.release()

# Define priorities for each lane
lane_priorities = {
    "lane1": 1,  # Lowest priority
    "lane2": 2,
    "lane3": 3,
    "lane4": 4   # Highest priority
}

# Global variable to track ambulance detection time
ambulance_detected_start_time = None

# Initializing a dictionary for video index tracking
lane_extra_video_index = {f"lane{idx + 1}": 0 for idx in range(4)}  # One video index per lane


def display_traffic_signals():
    global current_green_lane, start_time, startup_phase, ambulance_detected_start_time

    # Define specific extra videos for each lane
    lane_extra_videos = {
        "lane1": "C:/Users/ASUS/Downloads/extra_videos/vid1.mp4",
        "lane2": "C:/Users/ASUS/Downloads/aammmbbb.mp4",  # Specific video for lane 2
        "lane3": "C:/Users/ASUS/Downloads/extra_videos/vid5.mp4",
        "lane4": "C:/Users/ASUS/Downloads/extra_videos/vid6.mp4",
    }

    # Add a variable to track if ambulance video is over
    ambulance_video_over = False

    while True:
        frames = []  # Collect processed frames

        for idx, cap in enumerate(caps):
            lane_name = f"lane{idx + 1}"

            # Check if green time is expiring and switch to the specific extra video
            if lanes[lane_name]["signal"] == "green" and lanes[lane_name]["time_left"] == 1:
                extra_video = lane_extra_videos.get(lane_name, None)
                if extra_video:
                    caps[idx] = cv2.VideoCapture(extra_video)  # Load the specific extra video for the lane
                    if not caps[idx].isOpened():
                        print(f"Error: Unable to open video for {lane_name}")

            # Read the frame from the video feed
            ret, frame = cap.read()
            if not ret:  # If no frame is read, show a blank frame
                print(f"Failed to read frame for {lane_name}")
                frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)  # Blank frame if read fails
            else:
                frame = process_frame(frame, lane_name)

            # Check for ambulance presence in lane 2
            if lane_name == "lane2":
                if lanes["lane2"]["ambulance"]:
                    if ambulance_detected_start_time is None:
                        ambulance_detected_start_time = time.time()  # Start the timer
                    elif time.time() - ambulance_detected_start_time > 8:
                        # Set lane 2 to have infinite green time
                        lanes["lane2"]["green_time"] = float("inf")
                        lanes["lane2"]["signal"] = "green"  # Keep lane 2 green
                        current_green_lane = "lane2"  # Ensure lane 2 is the active green lane
                        start_time = None  # Stop regular green signal countdown

                        cv2.putText(
                            frame,  # The video frame
                            "Ambulance Detected",  # Text to display
                            (10, 50),  # Position on the frame (x, y)
                            cv2.FONT_HERSHEY_SIMPLEX,  # Font style
                            1,  # Font size
                            (0, 0, 255),  # Text color in BGR (red)
                            2,  # Text thickness
                            cv2.LINE_AA  # Line type for better clarity
                        )
                        
                        # Set all other lanes' signals to red
                        for lane in lanes:
                            if lane != "lane2":  # Set other lanes to red when ambulance is detected in lane 2
                                lanes[lane]["signal"] = "red"
                                lanes[lane]["time_left"] = 0
                                lanes[lane]["frozen"] = True

                else:
                    ambulance_detected_start_time = None  # Reset the timer if ambulance is not detected

                # Check if the ambulance video is finished
                if caps[idx].isOpened() and not ret:
                    ambulance_video_over = True
                    # Stop the video feed and display a black frame
                    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)  # Blank (blackout) frame

            # Resize and add to frames list
            frame_resized = cv2.resize(frame, (frame_width, frame_height))
            frames.append(frame_resized)

        # If ambulance video is over, blackout the screen
        if ambulance_video_over:
            blackout_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)  # Black frame
            frames = [blackout_frame] * 4  # Apply blackout to all lanes

        # Regular traffic signal management (for lanes that don't have ambulance active)
        if not (lanes["lane2"]["ambulance"] and lanes["lane2"]["green_time"] == float("inf")):
            if startup_phase:
                if time.time() - startup_start_time >= 20:  # Startup phase ends after 20 seconds
                    startup_phase = False

                    # Calculate green times based on vehicle counts
                    for lane_name, lane_data in lanes.items():
                        lane_data["green_time"] = 10 + \
                            (lane_data["vehicle_counts"]["car"] * 2) + \
                            (lane_data["vehicle_counts"]["rickshaw"] * 2) + \
                            (lane_data["vehicle_counts"]["motorcycle"] * 1) + \
                            (lane_data["vehicle_counts"]["truck"] * 3)

                    # Select the highest green time lane first, then apply priority
                    current_green_lane = max(
                        lanes,
                        key=lambda lane: (lanes[lane]["green_time"], lane_priorities[lane])
                    )
                    lanes[current_green_lane]["signal"] = "green"  # Set the selected lane to green
                    start_time = time.time()  # Reset timer

            else:  # Regular phase
                if start_time:  # Ensure timer exists
                    elapsed_time = int(time.time() - start_time)
                    lanes[current_green_lane]["time_left"] = max(0, lanes[current_green_lane]["green_time"] - elapsed_time)

                    if lanes[current_green_lane]["time_left"] == 0:  # Green time over
                        lanes[current_green_lane]["signal"] = "red"
                        lanes[current_green_lane]["time_left"] = 0
                        lanes[current_green_lane]["frozen"] = True

                        # Reorder lanes based on their green time and priority
                        if current_green_lane == "lane4":
                            current_green_lane = "lane3"
                        elif current_green_lane == "lane3":
                            current_green_lane = "lane2"
                        elif current_green_lane == "lane2":
                            current_green_lane = "lane1"
                        elif current_green_lane == "lane1":
                            current_green_lane = "lane4"

                        lanes[current_green_lane]["signal"] = "green"
                        start_time = time.time()

        # Stacked display of all lane frames
        stacked_frame = np.vstack([np.hstack(frames[:2]), np.hstack(frames[2:])])
        stacked_frame_resized = cv2.resize(stacked_frame, (stacked_frame_width, stacked_frame_height))
        cv2.imshow('Traffic Signal Management', stacked_frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_traffic_signals()
