import cv2
import time
import socket
import json
import numpy as np
import tensorflow as tf
from collections import deque
from collections import defaultdict, deque

# UDP setup
UDP_IP = "127.0.0.1"
UDP_PORT = 5006
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Load MoveNet
interpreter = tf.lite.Interpreter(model_path="1.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

movingFast = 0
cap_front = cv2.VideoCapture(0)  #   front camera

# Define durations (in seconds) for each state
TRACKING_DURATION = 10
DESTROYING_DURATION = 15  # longer to emphasize the damage visually
HEALING_DURATION = 15    # longer healing to reward gentle interaction

# Threshold for velocity
VELOCITY_THRESHOLD = 0.7 # normalized screen movement per second.  increase if want that more abrupt movements are considered as fast and not fast when arms are slow
HISTORY_DURATION = TRACKING_DURATION   # seconds
INTERVAL_TRACKING = 0.5  # seconds between samples
TRACKING_MOVEMENT_SECONDS = 2
CONSECUTIVE_MOVEMENT_COUNT = 2  # 4 consecutive entries => 2s
PERCENTAGE_FOR_DESTROYING = 0

velocity_history = defaultdict(lambda: {
    "left": deque(maxlen=int(CONSECUTIVE_MOVEMENT_COUNT)),  # Store last 4 = 2s of data if you're sampling every 0.5s
    "right": deque(maxlen=int(CONSECUTIVE_MOVEMENT_COUNT))
})

# History buffer per person
movement_histories = {}
confirmation_timers = {}
healing_timers = {}
last_sample_time = time.time()
last_labels = {}  # Store last known labels for drawing

INPUT_WIDTH = 256
INPUT_HEIGHT = 160       # 16:9 aspect ratio input size for MoveNet needs to be multiple of 32 so 160 so its rather 16:10 aspect ratio


# Helper functions
def calculate_velocity(prev_point, curr_point, dt, width, height):
    if prev_point is None or curr_point is None or dt == 0:
        return 0.0
    dx = (curr_point[0] - prev_point[0]) / width
    dy = (curr_point[1] - prev_point[1]) / height
    velocity = ((dx ** 2 + dy ** 2) ** 0.5) / dt
   
    return velocity

def get_valid_keypoint(person, key1, key2):
    k1 = person.get(key1)
    k2 = person.get(key2)
    if k1 and k1['score'] > 0.5:
        return k1['x'], k1['y']
    elif k2 and k2['score'] > 0.5:
        return k2['x'], k2['y']
    return None

def draw_keypoints(frame, person, label=None):
    keypoint_connections = [
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_wrist'),
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip'),
        ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'),
        ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle')
    ]

    # Draw circles
    for name, kp in person.items():
        if kp['score'] > 0.3:
            x, y = int(kp['x']), int(kp['y'])
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    # Draw connections
    for kp1, kp2 in keypoint_connections:
        if kp1 in person and kp2 in person:
            p1, p2 = person[kp1], person[kp2]
            if p1['score'] > 0.3 and p2['score'] > 0.3:
                pt1 = (int(p1['x']), int(p1['y']))
                pt2 = (int(p2['x']), int(p2['y']))
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

    # Draw label
    if label and 'nose' in person and person['nose']['score'] > 0.3:
        x, y = int(person['nose']['x']) -40, int(person['nose']['y']) - 50
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

def parse_person_keypoints(raw_keypoints, input_width, input_height, frame_width, frame_height):
    persons = []
    scale_x = frame_width / input_width
    scale_y = frame_height / input_height
    for p in raw_keypoints:
        person = {}
        for i, name in enumerate([
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle']):
            y, x, score = p[i * 3], p[i * 3 + 1], p[i * 3 + 2]
            person[name] = {
                'x': int(x * input_width * scale_x),
                'y': int(y * input_height * scale_y),
                'score': score
            }
        persons.append(person)
    return persons


def prepare_frame_for_movenet(frame):
    # Resize frame to 256x144 (no padding, might distort if original isn't 16:9)
    resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    input_image = tf.convert_to_tensor(resized)
    input_image = tf.expand_dims(input_image, axis=0)
    return tf.cast(input_image, dtype=tf.uint8)

# Main loop
prev_keypoints = []
prev_time = time.time()

# Prepare the input shape once
input_shape = [1, INPUT_HEIGHT, INPUT_WIDTH, 3]  # (1,160,256,3)
interpreter.resize_tensor_input(input_details[0]['index'], input_shape)
interpreter.allocate_tensors()


# Initialize state
tracking_active = True
global_phase = None
global_phase_end_time = 0
tracking_start_time = time.time()

def has_consecutive_fast_movements(history, times):
    count = 0
    for status in history:
        if status:  # True means fast
            count += 1
            if count >= times:
                return True
        else:
            count = 0
    return False

def filter_valid_persons(persons, score_threshold=0.2):
    valid_persons = []
    for person in persons:
        scores = [kp['score'] for kp in person.values()]
        avg_score = sum(scores) / len(scores)
        if avg_score >= score_threshold:
            valid_persons.append(person)
    return valid_persons

def calculate_fast_movement_percentage(histories: dict, consecutive_threshold: int) -> float:
    """
    Calculates the percentage of people with fast movements.
    :param histories: Dictionary of movement histories per person.
    :param consecutive_threshold: Number of consecutive fast entries to qualify as "fast".
    :return: Percentage of people with fast movements (0 to 100).
    """
    if not histories:
        return 0.0

    total_people = len(histories)
    fast_movers = sum(
        1 for history in histories.values() if has_consecutive_fast_movements(history, consecutive_threshold)
    )

    return (fast_movers / total_people) * 100

# Global dictionary to store recent nose x positions
nose_x_history = {}


def update_nose_position(person_index, person, frame_width, frame_height):
    """
    Updates and stores the last 2 normalized nose x and y positions for one person.
    - x: 0 = left, 1 = right
    - y: 0 = top, 1 = bottom
    """
    global nose_position_history  # Use a dict like {person_index: [(x1, y1), (x2, y2)]}

    if 'nose' in person and person['nose']['score'] > 0.3:
        normalized_x = person['nose']['x'] / frame_width
        normalized_y = person['nose']['y'] / frame_height
        position = (round(normalized_x, 2), round(normalized_y, 2))
    else:
        position = None

    if person_index not in nose_position_history:
        nose_position_history[person_index] = []

    nose_position_history[person_index].append(position)

    # Keep only the last 2 values
    if len(nose_position_history[person_index]) > 2:
        nose_position_history[person_index] = nose_position_history[person_index][-2:]


motion_status_history = {}
def update_motion_status(person_index, motion_status):
    global motion_status_history

    if person_index not in motion_status_history:
        motion_status_history[person_index] = []

    motion_status_history[person_index].append(motion_status)

    # Keep only the last value (stored in a list still)
    if len(motion_status_history[person_index]) > 1:
        motion_status_history[person_index] = motion_status_history[person_index][-1:]


person_tracking_data = {}
def update_person_tracking(person_index, person, frame_width, frame_height, motion_status):
    """
    Stores the latest:
    - Nose position (normalized x, y)
    - Motion status (1 = fast, 0 = not fast)
    """
    global person_tracking_data

    if person_index not in person_tracking_data:
        person_tracking_data[person_index] = {
            'nose': None,
            'fastMotion': 0
        }

    # Update nose position
    if 'nose' in person and person['nose']['score'] > 0.3:
        normalized_x = round(person['nose']['x'] / frame_width, 2)
        normalized_y = round(person['nose']['y'] / frame_height, 2)
        position = (normalized_x, normalized_y)
    else:
        position = None

    person_tracking_data[person_index]['nose'] = position
    person_tracking_data[person_index]['fastMotion'] = motion_status


def send_people_json(person_tracking_data, global_phase, udp_socket, udp_ip, udp_port):
    people_data = []

    for person_id, values in person_tracking_data.items():
        nose_pos = values.get('nose')
        fast_motion = values.get('fastMotion', 0)

        if nose_pos is None:
            continue

        person_info = {
            "xy_pos": nose_pos,
            "mov_fast": fast_motion
        }
        people_data.append(person_info)

    payload = {
        "tracking_dur": TRACKING_DURATION,
        "destroying_dur": DESTROYING_DURATION,
        "healing_dur": HEALING_DURATION,
        "state": global_phase,
        "ppl": people_data    
    }

    json_string = json.dumps(payload)
    #print(payload)
    udp_socket.sendto(json_string.encode(), (udp_ip, udp_port))


while cap_front.isOpened():
    ret_front, frame_front = cap_front.read()
    if not ret_front:
        print("Error reading from the camera")
        break

    frame_front = cv2.flip(frame_front, 1)
    frame_rgb = cv2.cvtColor(frame_front, cv2.COLOR_BGR2RGB)
    input_img = prepare_frame_for_movenet(frame_rgb)

    interpreter.set_tensor(input_details[0]['index'], input_img.numpy())
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    curr_keypoints = parse_person_keypoints(output, INPUT_WIDTH, INPUT_HEIGHT, frame_front.shape[1], frame_front.shape[0])
    valid_people = filter_valid_persons(curr_keypoints, score_threshold=0.3)

    curr_time = time.time()
    dt = curr_time - prev_time
    prev_time = curr_time

    # Default text to show on frame (empty by default)
    phase_text = ""
    phase_color = (0, 255, 0)  # Default green

    if len(valid_people) == 0:
        if global_phase != "idle":
            global_phase = "idle"
            print("Idle Phase: No people detected.")
        phase_text = "Idle State: Waiting for people..."
        tracking_active = False
    else:
        if global_phase == "idle":
            print("People detected. Resuming tracking.")
            global_phase = "tracking"
            tracking_active = True
            tracking_start_time = curr_time
            nose_x_history.clear()
            movement_histories.clear()
            last_labels.clear()
            person_tracking_data.clear()

    if global_phase in ("healing", "destroying"):
        if curr_time < global_phase_end_time:
            remaining_time = int(global_phase_end_time - curr_time)
            phase_text = f"{global_phase.capitalize()} State: {remaining_time}s"
            phase_color = (0, 255, 0) if global_phase == "healing" else (0, 0, 255)

            frame_height, frame_width = frame_front.shape[:2]

            # Update nose positions from previous frame
            for person_id, person in enumerate(curr_keypoints):
                draw_keypoints(frame_front, person, label=global_phase.capitalize())
                if person_id >= len(prev_keypoints):
                    continue
                prev_person = prev_keypoints[person_id]
                update_nose_position(person_id, prev_person, frame_width, frame_height)

            # Show fast movement percentage
            cv2.putText(frame_front, f"Total fast movement percentage: {fast_percentage:.2f}%", 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        else:
            print(f"{global_phase.capitalize()} phase ended. Resuming tracking.")
            global_phase = "tracking"
            tracking_active = True
            tracking_start_time = curr_time
            nose_x_history.clear()
            movement_histories.clear()
            last_labels.clear()
            person_tracking_data.clear()
    
    if global_phase == "tracking":
        remaining_time = int(tracking_start_time + TRACKING_DURATION - curr_time)
        phase_text = f"Tracking Phase: {remaining_time}s"
        phase_color = (0, 255, 0)

        if curr_time - tracking_start_time >= TRACKING_DURATION:
            print("--- tracking window passed, evaluating ---")
            fast_percentage = calculate_fast_movement_percentage(movement_histories, CONSECUTIVE_MOVEMENT_COUNT)
            someone_fast = any(has_consecutive_fast_movements(history, CONSECUTIVE_MOVEMENT_COUNT) 
                              for history in movement_histories.values())

            if fast_percentage > PERCENTAGE_FOR_DESTROYING:
                print("Destroying phase triggered")
                global_phase = "destroying"
                global_phase_end_time = curr_time + DESTROYING_DURATION
                tracking_active = False
            else:
                print("Healing phase triggered")
                global_phase = "healing"
                global_phase_end_time = curr_time + HEALING_DURATION
                tracking_active = False

    # Run movement and nose position updates on interval
    if curr_time - last_sample_time >= INTERVAL_TRACKING:
        for i, person in enumerate(curr_keypoints):
            if i >= len(prev_keypoints):
                continue
            prev_person = prev_keypoints[i]
            frame_height, frame_width = frame_front.shape[:2]
           

            # wrist to elbow velocity 
            left_curr = get_valid_keypoint(person, 'left_wrist', 'left_elbow')
            left_prev = get_valid_keypoint(prev_person, 'left_wrist', 'left_elbow')
            right_curr = get_valid_keypoint(person, 'right_wrist', 'right_elbow')
            right_prev = get_valid_keypoint(prev_person, 'right_wrist', 'right_elbow')
           
            # elbow-to-shoulder velocity
            left_upper_curr = get_valid_keypoint(person, 'left_elbow', 'left_shoulder')
            left_upper_prev = get_valid_keypoint(prev_person, 'left_elbow', 'left_shoulder')
            right_upper_curr = get_valid_keypoint(person, 'right_elbow', 'right_shoulder')
            right_upper_prev = get_valid_keypoint(prev_person, 'right_elbow', 'right_shoulder')

            left_shoulder = prev_keypoints[i]['left_shoulder']
            left_wrist = prev_keypoints[i]['left_wrist']
            right_shoulder = prev_keypoints[i]['right_shoulder']
            right_wrist = prev_keypoints[i]['right_wrist']

            # Check visibility and whether wrists are above shoulders
            left_wrist_raised = left_wrist['score'] > 0.2 and left_shoulder['score'] > 0.2 and left_wrist['y'] < left_shoulder['y']
            right_wrist_raised = right_wrist['score'] > 0.2 and right_shoulder['score'] > 0.2 and right_wrist['y'] < right_shoulder['y']
            
           
            update_person_tracking(i, prev_person, frame_width, frame_height, movingFast)

            # Apply velocity check only if wrist is raised
            if left_wrist_raised or right_wrist_raised:
             
                v_left_upper = calculate_velocity(left_upper_prev, left_upper_curr, dt, frame_front.shape[1], frame_front.shape[0])
                v_right_upper = calculate_velocity(right_upper_prev, right_upper_curr, dt, frame_front.shape[1], frame_front.shape[0])
                v_left = calculate_velocity(left_prev, left_curr, dt, frame_front.shape[1], frame_front.shape[0])
                v_right = calculate_velocity(right_prev, right_curr, dt, frame_front.shape[1], frame_front.shape[0])

                # Average all components (optional)
                v_left_total = (v_left + v_left_upper) / 2
                v_right_total = (v_right + v_right_upper) / 2

                velocity_history[i]["left"].append(v_left_total)
                velocity_history[i]["right"].append(v_right_total)

                avg_v_left = sum(velocity_history[i]["left"]) / len(velocity_history[i]["left"])
                avg_v_right = sum(velocity_history[i]["right"]) / len(velocity_history[i]["right"])
        
                movingFast = avg_v_left > VELOCITY_THRESHOLD or avg_v_right > VELOCITY_THRESHOLD
                update_person_tracking(i, prev_person, frame_width, frame_height, movingFast)

                status = 'fast' if movingFast else 'slow'
            else:

                movingFast = False
                status = 'fast' if movingFast else 'slow'

            if i not in movement_histories:
                movement_histories[i] = deque(maxlen=int(TRACKING_DURATION / INTERVAL_TRACKING))
            movement_histories[i].append(movingFast)
            last_labels[i] = status.capitalize()

        last_sample_time = curr_time
      
 
    # Send JSON data once per loop
    send_people_json(person_tracking_data, global_phase, sock, UDP_IP, UDP_PORT)

    # Draw labels and keypoints
    for person_id, person in enumerate(curr_keypoints):
        label = last_labels.get(person_id, None)
        draw_keypoints(frame_front, person, label=label)

    # Display phase text on frame
    if phase_text:
        cv2.putText(frame_front, phase_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, phase_color, 2, cv2.LINE_AA)

    # Show frame once
    cv2.imshow('Front Camera', frame_front)
    # cv2.imshow('Ceiling Camera', frame_ceil) # If needed

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_keypoints = curr_keypoints

cap_front.release()
cv2.destroyAllWindows()
