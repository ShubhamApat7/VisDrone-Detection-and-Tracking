from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO('C:/Users/admin/Downloads/VisDrone/Object detection in Image/weights/best.pt')

# Define paths
input_video_path = r'C:/Users/admin/Downloads/uav0000349_02668_s.mp4'
output_video_path = r'C:/Users/admin/Downloads/tracking/video1.mp4'

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Reduce the frame size
scale_factor = 0.5
new_width = int(width * scale_factor)
new_height = int(height * scale_factor)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID' or 'MJPG' as well
out = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height))

# Define class labels
class_labels = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor"
}

# Variable to store the vehicle ID of interest
selected_vehicle_id = None
vehicle_positions = {}

def select_vehicle(event, x, y, flags, param):
    global selected_vehicle_id
    if event == cv2.EVENT_LBUTTONDOWN:
        for box in param['boxes']:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_vehicle_id = int(box[4])
                print(f"Selected vehicle ID: {selected_vehicle_id}")
                break

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', select_vehicle, param={'boxes': []})

last_known_position = None

def reselect_vehicle(current_vehicles):
    global selected_vehicle_id, last_known_position
    if selected_vehicle_id not in [int(box.id) for box in current_vehicles]:
        min_distance = float('inf')
        new_vehicle_id = None
        for box in current_vehicles:
            if last_known_position is None:
                continue
            x_center, y_center = int(box.xywh[0][0]), int(box.xywh[0][1])
            distance = np.linalg.norm(np.array((x_center, y_center)) - np.array(last_known_position))
            if distance < min_distance:
                min_distance = distance
                new_vehicle_id = box.id
        selected_vehicle_id = new_vehicle_id

try:
    frame_count = 0
    frame_skip = 5  # Process every 5th frame
    for result in model.track(source=input_video_path, stream=True):
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        frame = result.orig_img

        if result.boxes is None:
            print("No boxes detected in the current frame.")
            continue

        frame = cv2.resize(frame, (new_width, new_height))
        boxes = []
        for box in result.boxes:
            if box.xywh[0][0] is not None and box.xywh[0][1] is not None and box.xywh[0][2] is not None and box.xywh[0][3] is not None:
                box_id = int(box.id)
                boxes.append([int(box.xywh[0][0] * scale_factor - box.xywh[0][2] * scale_factor / 2), int(box.xywh[0][1] * scale_factor - box.xywh[0][3] * scale_factor / 2),
                              int(box.xywh[0][0] * scale_factor + box.xywh[0][2] * scale_factor / 2), int(box.xywh[0][1] * scale_factor + box.xywh[0][3] * scale_factor / 2),
                              box_id])
                # Debugging print
                print(f"Detected box ID: {box_id}")
            else:
                print(f"Skipping invalid box: {box.xywh}")

        cv2.setMouseCallback('frame', select_vehicle, param={'boxes': boxes})

        if selected_vehicle_id is None or not any(int(box.id) == selected_vehicle_id for box in result.boxes):
            reselect_vehicle(result.boxes)

        for box in result.boxes:
            box_id = int(box.id)
            if box_id == selected_vehicle_id:
                x_center, y_center, width, height = box.xywh[0]
                x_center = int(x_center * scale_factor)
                y_center = int(y_center * scale_factor)
                width = int(width * scale_factor)
                height = int(height * scale_factor)
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)

                label = class_labels.get(box_id, 'unknown')

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ID {selected_vehicle_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                last_known_position = (x_center, y_center)
                print(f"Tracking vehicle ID {selected_vehicle_id} at ({x_center}, {y_center})")

        cv2.imshow('frame', frame)
        cv2.waitKey(1)

        out.write(frame)
        frame_count += 1

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
