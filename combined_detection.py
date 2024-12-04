import torch
import cv2

# Load your custom-trained model for ambulances
model_ambulance = torch.hub.load('ultralytics/yolov5', 'custom', 
                                  path='C:/Users/Ayush/Desktop/finaal/yolov5/runs/train/ambulance_detection/weights/best.pt')

# Load the YOLOv5 pretrained COCO model
model_coco = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# COCO class names (reduced for targeted detection)
COCO_CLASSES = {
    2: "Car",
    5: "Bus",
    7: "Truck",
    3: "Motorbike"
}

# Custom model class names
CUSTOM_CLASSES = {
    0: "Ambulance"  # Replace with the actual class name(s) for your custom model
}

# Filter results to include specific classes (for COCO model)
def filter_results(results, target_classes):
    filtered = []
    for result in results.xyxy[0]:  # xyxy format results
        class_id = int(result[5])  # class ID is the 6th element in YOLOv5's output format
        if class_id in target_classes:
            filtered.append(result)
    return filtered

# Combine results from both models with class names
def combine_results_with_labels(frame):
    # Detect ambulances
    results_ambulance = model_ambulance(frame)
    
    # Detect cars, buses, trucks, and bikes
    results_coco = model_coco(frame)
    
    # Filter COCO results for specific vehicle classes
    target_classes = [2, 5, 7, 3]  # Car, Bus, Truck, Motorbike
    filtered_coco_results = filter_results(results_coco, target_classes)
    
    combined_results = []
    
    # Add results from the ambulance model
    for result in results_ambulance.xyxy[0]:
        x1, y1, x2, y2, confidence, class_id = result[:6]
        class_id = int(class_id)
        class_name = CUSTOM_CLASSES.get(class_id, f"Class {class_id}")
        combined_results.append((x1, y1, x2, y2, confidence, class_name))
    
    # Add filtered results from the COCO model
    for result in filtered_coco_results:
        x1, y1, x2, y2, confidence, class_id = result[:6]
        class_id = int(class_id)
        class_name = COCO_CLASSES.get(class_id, f"Class {class_id}")
        combined_results.append((x1, y1, x2, y2, confidence, class_name))
    
    return combined_results

# Process video frame-by-frame
def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Video writer to save the output
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get combined detection results with labels
        combined_results = combine_results_with_labels(frame)

        # Draw bounding boxes and labels on the frame
        for result in combined_results:
            x1, y1, x2, y2, confidence, class_name = result[:6]
            label = f"{class_name}: {confidence:.2f}"
            color = (0, 255, 0) if class_name == "Ambulance" else (255, 0, 0)  # Green for ambulance, blue for others
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the frame with detections
        out.write(frame)

    cap.release()
    out.release()
    print("Video processing complete. Output saved to", output_path)

# Run the video processing function
if _name_ == "_main_":
    video_path = 'C:/Users/Ayush/Downloads/WhatsApp Video 2024-11-15 at 2.03.20 PM.mp4'  # Replace with your video path
    output_path = "C:/Users/Ayush/Desktop/finaal/yolov5/dataset/output.mp4"  # Path to save the output
    process_video(video_path, output_path)