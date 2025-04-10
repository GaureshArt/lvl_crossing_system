import RPi.GPIO as GPIO
import time
from adafruit_servokit import ServoKit
import serial
import cv2
import numpy as np
from datetime import datetime, timedelta
import threading
import platform

# GPIO and LoRa communication setup
ser = serial.Serial(
    port='/dev/ttyAMA0',
    baudrate=9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1
)

Kit = ServoKit(channels=16)
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
buzzer=22
rPin=26   #37
gPin=19  #35
bPin=13  #33
rPin2=21  #40
gPin2=20  #38
bPin2=16  #36
ir1=17    #11
ir2=27   #13
GPIO.setup(ir1,GPIO.IN,pull_up_down=GPIO.PUD_UP)
GPIO.setup(ir2,GPIO.IN,pull_up_down=GPIO.PUD_UP)
GPIO.setup(rPin,GPIO.OUT)
GPIO.setup(gPin,GPIO.OUT)
GPIO.setup(bPin,GPIO.OUT)
GPIO.setup(rPin2,GPIO.OUT)
GPIO.setup(gPin2,GPIO.OUT)
GPIO.setup(bPin2,GPIO.OUT)
GPIO.setup(buzzer,GPIO.OUT)

GPIO.output(buzzer,GPIO.LOW)
# Global variables for train and communication
train_detected = False
last_triggered = None
Train_address = None
ack_sent = False
train_no = None
gate_status = None
pending_ack = False
current_msg_tosent = ""
last_sent_msg = ""
object_detected = False
last_sent_time = 0
ack_wait_time = 10
data = None
detector_running = False
detector_thread = None
video_stream = None
exit_detector = False

Kit.servo[0].angle=90
Kit.servo[3].angle=90
time.sleep(1)

# Object Detection Configuration
class Config:
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    MIN_AREA = 500
    ALARM_SECONDS = 4
    BG_LEARNING_SECONDS = 5
    MAX_TRACKED_OBSTACLES = 3
    ROIS = [
        {
            "points": np.array([[100, 100], [540, 100], [540, 380], [100, 380]]),
            "color": (0, 255, 0)  # Green
        }
    ]
    LK_PARAMS = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    MAX_OPTICAL_FLOW_POINTS = 30
    TRACKING_THRESHOLD = 50
    MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    BG_THRESHOLD = 30
    MIN_CONTOUR_ASPECT_RATIO = 0.3
    MAX_CONTOUR_ASPECT_RATIO = 3.0

# Video stream class
class VideoStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            with self.lock:
                self.grabbed, self.frame = grabbed, frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.grabbed else None

    def stop(self):
        self.stopped = True

# Obstacle tracker class
alarm_active=False
class ObstacleTracker:
    def __init__(self):
        self.obstacles = {}
        self.next_id = 0
        
        self.prev_gray = None
        self.start_time = datetime.now()
        self.bg_learned = False
        self.bg_model = None
        self.first_frame = True
        self.avg_brightness = None
        self.brightness_threshold = 50
        self.alarm_active = False

    def init_optical_flow(self, frame):
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def create_tracker(self):
        tracker_types = ['CSRT', 'KCF', 'MOSSE']
        for tracker_type in tracker_types:
            try:
                if tracker_type == 'CSRT':
                    return cv2.TrackerCSRT_create()
                elif tracker_type == 'KCF':
                    return cv2.TrackerKCF_create()
                elif tracker_type == 'MOSSE':
                    return cv2.legacy.TrackerMOSSE_create()
            except:
                continue
        return cv2.legacy.TrackerMOSSE_create()

    def is_inside_any_roi(self, point):
        return any(cv2.pointPolygonTest(roi["points"], point, False) >= 0 for roi in Config.ROIS)

    def is_existing_object(self, new_center):
        for data in self.obstacles.values():
            distance = np.hypot(new_center[0]-data["center"][0], new_center[1]-data["center"][1])
            if distance < Config.TRACKING_THRESHOLD:
                return True
        return False

    def create_new_tracker(self, frame, x, y, w, h):
        if len(self.obstacles) >= Config.MAX_TRACKED_OBSTACLES:
            oldest_id = min(self.obstacles, key=lambda k: self.obstacles[k]["start_time"])
            del self.obstacles[oldest_id]

        tracker = self.create_tracker()
        tracker.init(frame, (x, y, w, h))
        
        roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        points = cv2.goodFeaturesToTrack(
            roi_gray, 
            maxCorners=Config.MAX_OPTICAL_FLOW_POINTS,
            qualityLevel=0.2,
            minDistance=5,
            blockSize=5
        )
        
        if points is not None:
            points = points.reshape(-1, 2).astype(np.float32)
            points += np.array([x, y], dtype=np.float32)
            points = points.reshape(-1, 1, 2)
        else:
            points = np.empty((0, 1, 2), dtype=np.float32)

        self.obstacles[self.next_id] = {
            "tracker": tracker,
            "start_time": datetime.now(),
            "center": (x + w//2, y + h//2),
            "points": points,
            "box": (x, y, w, h),
            "valid_frames": 0,
            "alarm_triggered": False
        }
        self.next_id += 1

    def update_optical_flow(self, frame, obj_data):
        if self.prev_gray is None or obj_data["points"].size == 0:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, 
                obj_data["points"], 
                None,
                **Config.LK_PARAMS
            )
        except cv2.error:
            return None
        
        self.prev_gray = gray.copy()
        return new_points[status.squeeze() == 1]

    def update_object_data(self, obj_id, box, frame):
        x, y, w, h = map(int, box)
        data = self.obstacles[obj_id]
        
        if data["points"].size > 0:
            new_points = self.update_optical_flow(frame, data)
            if new_points is not None and len(new_points) > 4:
                data["points"] = new_points.reshape(-1, 1, 2)
                avg_move = np.mean(new_points - data["points"], axis=0)
                x += int(avg_move[0, 0])
                y += int(avg_move[0, 1])
        
        data["center"] = (x + w//2, y + h//2)
        data["box"] = (x, y, w, h)
        data["valid_frames"] += 1

    def process_frame(self, frame):
        #global object_detected
        
        if self.prev_gray is None:
            self.init_optical_flow(frame)

        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate current frame brightness
        current_brightness = cv2.mean(gray_frame)[0]
        if self.avg_brightness is None:
            self.avg_brightness = current_brightness
        else:
            self.avg_brightness = 0.9 * self.avg_brightness + 0.1 * current_brightness
        
        # Skip processing if lighting change is too drastic
        if abs(current_brightness - self.avg_brightness) > self.brightness_threshold:
            return frame

        # Learn background for first seconds
        if elapsed_time < Config.BG_LEARNING_SECONDS:
            if self.first_frame:
                self.bg_model = gray_frame.copy().astype(np.float32)
                self.first_frame = False
            else:
                cv2.accumulateWeighted(gray_frame, self.bg_model, 0.5)
            
            cv2.putText(frame, f"Learning BG: {Config.BG_LEARNING_SECONDS - elapsed_time:.1f}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return frame
        
        # After BG learning period
        if not self.bg_learned:
            if self.bg_model is not None:
                self.bg_model = cv2.convertScaleAbs(self.bg_model)
                self.bg_learned = True
                print("Background learning completed!")
            else:
                self.bg_model = gray_frame
                self.bg_learned = True
                print("Using fallback background model")

        # Compare with learned background
        diff = cv2.absdiff(gray_frame, self.bg_model)
        _, fg_mask = cv2.threshold(diff, Config.BG_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # Enhanced morphological operations
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, Config.MORPH_KERNEL)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, Config.MORPH_KERNEL)
        
        # Find contours with stricter criteria
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        potential_obstacles = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < Config.MIN_AREA:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # Filter by aspect ratio to remove line-like artifacts
            if (aspect_ratio < Config.MIN_CONTOUR_ASPECT_RATIO or 
                aspect_ratio > Config.MAX_CONTOUR_ASPECT_RATIO):
                continue
                
            # Calculate solidity (area/convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area)/hull_area if hull_area > 0 else 0
            
            # Filter by solidity to remove sparse contours
            if solidity < 0.5:
                continue
                
            # Additional check - compare with color variance in the region
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue
                
            color_stddev = np.std(roi, axis=(0,1)).mean()
            if color_stddev < 10:  # Too uniform to be a real obstacle
                continue
                
            potential_obstacles.append((x, y, w, h))

        # Process potential obstacles
        for (x, y, w, h) in potential_obstacles:
            center = (x + w//2, y + h//2)
            
            if self.is_inside_any_roi(center) and not self.is_existing_object(center):
                if len(self.obstacles) < Config.MAX_TRACKED_OBSTACLES:
                    self.create_new_tracker(frame, x, y, w, h)

        # Update trackers with additional validation
        self.alarm_active = False
        to_delete = []
        
        # Reset object_detected at the beginning of each frame
        #bject_detected = False
        
        for obj_id, data in list(self.obstacles.items()):
            success, box = data["tracker"].update(frame)
            if not success:
                to_delete.append(obj_id)
                continue

            x, y, w, h = map(int, box)
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                to_delete.append(obj_id)
                continue
                
            if not self.is_inside_any_roi((x + w//2, y + h//2)):
                to_delete.append(obj_id)
                global object_detected
                object_detected=False
                continue

            # Additional validation - check if the region still differs from background
            roi = gray_frame[y:y+h, x:x+w]
            bg_roi = self.bg_model[y:y+h, x:x+w]
            if roi.shape != bg_roi.shape:
                to_delete.append(obj_id)
                continue
                
            diff = cv2.absdiff(roi, bg_roi)
            _, roi_mask = cv2.threshold(diff, Config.BG_THRESHOLD, 255, cv2.THRESH_BINARY)
            if cv2.countNonZero(roi_mask) < (w * h * 0.2):  # Less than 20% different
                to_delete.append(obj_id)
                continue

            self.update_object_data(obj_id, box, frame)
            elapsed = (datetime.now() - data["start_time"]).total_seconds()
            
            if elapsed >= Config.ALARM_SECONDS:
                print("line 361")
                data["alarm_triggered"] = True
                self.alarm_active = True
                # Set object_detected flag if any valid object is detected
                
                object_detected = True

        for obj_id in to_delete:
            del self.obstacles[obj_id]

        return self.draw_annotations(frame)

    def draw_annotations(self, frame):
        for roi in Config.ROIS:
            cv2.polylines(frame, [roi["points"]], True, roi["color"], 2)
        
        for obj_id, data in self.obstacles.items():
            x, y, w, h = data["box"]
            elapsed = (datetime.now() - data["start_time"]).total_seconds()
            
            if data["alarm_triggered"]:
                color = (0, 0, 255)  # Red for alarm triggered
            elif elapsed >= Config.ALARM_SECONDS:
                color = (0, 165, 255)  # Orange for alarm ready
            else:
                color = (0, 255, 0)  # Green for new object
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{elapsed:.1f}s", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Show system status
        status_text = "TRACKING MODE"
        if (datetime.now() - self.start_time).total_seconds() < Config.BG_LEARNING_SECONDS:
            status_text = "LEARNING BACKGROUND"
        cv2.putText(frame, status_text, (10, frame.shape[0]-40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show object detection status
        if object_detected:
            detection_status = "OBSTACLE DETECTED!"
            cv2.putText(frame, detection_status, (10, frame.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame

# Communication functions
def receive_message():
    global data
    global Train_address
    global train_no
    if ser.in_waiting > 0:
        response = ser.readline().decode().strip()
        ser.reset_input_buffer()
        if "+RCV" in response:
            data = response.split(",")[2]  # Extract payload
            if data != "ACK":
                Train_address = response.split("=")[1].split(",")[0]
                train_no = data[0:5]
            print(f"Received: {data}, {Train_address}")

def send_message(message):
    if Train_address is not None:
        command = f"AT+SEND={Train_address},{len(message)},{message}\r\n"
        ser.write(command.encode())
        ser.flush()

def default_rgb_state():
    GPIO.output(rPin, GPIO.HIGH)  # Red ON (Set 1)
    GPIO.output(gPin, GPIO.LOW)
    GPIO.output(bPin, GPIO.LOW)
    
    GPIO.output(rPin2, GPIO.LOW)
    GPIO.output(gPin2, GPIO.HIGH)  # Green ON (Set 2)
    GPIO.output(bPin2, GPIO.LOW)

def send_message_ack(message):
    global last_sent_msg, pending_ack, last_sent_time
    if Train_address is None:
        return
    
    command1 = f"AT+SEND={Train_address},{len(message)},{message}\r\n"
    ser.write(command1.encode())
    ser.flush()
    last_sent_msg = message
    last_sent_time = time.time()
    pending_ack = True
            
def handle_ack():
    global pending_ack, data
    if Train_address is not None:
        receive_message()
        if data == "ACK":
            pending_ack = False
            print("ACK received from Arduino")

def check_ack_time():
    global pending_ack, last_sent_time, last_sent_msg
    if pending_ack and (time.time() - last_sent_time >= ack_wait_time):
        send_message_ack(last_sent_msg)

# Object detection thread function
def run_object_detection():
    global video_stream, exit_detector, object_detected
    
    print("Starting object detection...")
    video_stream = VideoStream(0).start()
    time.sleep(1.0)  # Allow camera to warm up
    
    frame = video_stream.read()
    if frame is None:
        print("Error reading video source")
        return
    
    tracker = ObstacleTracker()
    last_time = time.time()
    
    while not exit_detector:
        frame = video_stream.read()
        if frame is None:
            break

        frame = tracker.process_frame(frame)
        
        fps = 1 / (time.time() - last_time)
        last_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0]-70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Railway Obstacle Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up resources when detection stops
    if video_stream is not None:
        video_stream.stop()
    cv2.destroyAllWindows()
    print("Object detection stopped")

# Start the object detection thread
def start_detection():
    global detector_thread, detector_running, exit_detector
    
    if not detector_running:
        exit_detector = False
        detector_thread = threading.Thread(target=run_object_detection)
        detector_thread.daemon = True
        detector_thread.start()
        detector_running = True
        print("Detection thread started")

# Stop the object detection thread
def stop_detection():
    global detector_running, exit_detector, object_detected
    
    if detector_running:
        exit_detector = True
        # Reset object_detected flag when stopping detection
        object_detected = False
        detector_running = False
        print("Detection thread stopping...")

# Main railway function
def railway():
    global train_detected, last_triggered, ack_sent, gate_status
    global pending_ack, current_msg_tosent, last_sent_msg, object_detected
    global last_sent_time, data, detector_running, Train_address
    
    irstate1 = GPIO.input(ir1)
    irstate2 = GPIO.input(ir2)
    
    # Train detection logic
    if irstate1 == 0 and last_triggered is None:
        train_detected = True
        last_triggered = "ir1"
        receive_message()
        print(f"IR1 triggered, {last_triggered}, {train_no}")
        # Start object detection when train is detected
        if not detector_running:
            start_detection()
        
    elif irstate2 == 0 and last_triggered is None:
        train_detected = True
        last_triggered = "ir2"
        receive_message()
        print(f"IR2 triggered, {last_triggered}, {train_no}")
        # Start object detection when train is detected
        if not detector_running:
            start_detection()
            
    if train_detected:
        # Change signals
        GPIO.output(rPin, GPIO.LOW)
        GPIO.output(gPin, GPIO.HIGH)  # Green ON (Set 1)
        GPIO.output(bPin, GPIO.LOW)
        
        GPIO.output(rPin2, GPIO.HIGH)  # Red ON (Set 2)
        GPIO.output(gPin2, GPIO.LOW)
        GPIO.output(bPin2, GPIO.LOW)
        
        if not ack_sent:
            time.sleep(2)
            
            send_message("ACK")
            print("ACK sent")
            time.sleep(2)
            for i in range(90,-1,-10):
                GPIO.output(buzzer,GPIO.HIGH)
                Kit.servo[0].angle=i
                Kit.servo[3].angle=i
                time.sleep(.1)
            GPIO.output(buzzer,GPIO.LOW)
            
            send_message("GATE CLOSED")
            time.sleep(2)
            
            #gate_status = "0"
            #current_msg_tosent = gate_status
            #send_message_ack(current_msg_tosent)
            if last_triggered == "ir1":
                msg1 = f"{last_triggered}>ir2"
                send_message(msg1)
                time.sleep(2)
            else:
                msg2 = f"{last_triggered}>ir1"
                send_message(msg2)
                time.sleep(2)
            
            ack_sent = True
            
        if ack_sent:
            if not (object_detected):
                gate_status="0"
                GPIO.output(buzzer,GPIO.LOW)
                if current_msg_tosent != gate_status:
                    current_msg_tosent = gate_status
                    send_message_ack(current_msg_tosent)
                    
            if object_detected:
                GPIO.output(buzzer,GPIO.HIGH)
                gate_status = "1"
                if current_msg_tosent != gate_status:
                    current_msg_tosent = gate_status
                    send_message_ack(current_msg_tosent)
                
        handle_ack()
        check_ack_time()
            
    # Train has exited - IR2 triggered after IR1
    if irstate2 == 0 and last_triggered == "ir1":
        train_detected = False
        last_triggered = None
        ack_sent = False
        gate_status = None
        pending_ack = False
        current_msg_tosent = ""
        last_sent_msg = ""
        object_detected = False
        last_sent_time = 0
        data = None
        for i in range(0,91,10):
            Kit.servo[0].angle=i
            Kit.servo[3].angle=i
            time.sleep(.1)
        default_rgb_state()
        time.sleep(2)
        send_message("exited")
        time.sleep(2)
        Train_address = None
        GPIO.output(buzzer,GPIO.LOW)
        
        # Stop object detection when train has exited
        if detector_running:
            stop_detection()
    
    # Train has exited - IR1 triggered after IR2
    if irstate1 == 0 and last_triggered == "ir2":
        train_detected = False
        last_triggered = None
        ack_sent = False
        gate_status = None
        pending_ack = False
        current_msg_tosent = ""
        last_sent_msg = ""
        object_detected = False
        data = None
        last_sent_time = 0
        for i in range(0,91,10):
            Kit.servo[0].angle=i
            Kit.servo[3].angle=i
            time.sleep(.1)
        
        default_rgb_state()
        time.sleep(2)
        send_message("exited")
        time.sleep(2)
        Train_address = None
        GPIO.output(buzzer,GPIO.LOW)
        
        # Stop object detection when train has exited
        if detector_running:
            stop_detection()
    
    time.sleep(0.5)

# Main function
def main():
    default_rgb_state()
    tracker=ObstacleTracker()
    try:
        print("Railway Monitoring System Started")
        print("Waiting for train detection...")
        
        while True:
            railway()
            
    except KeyboardInterrupt:
        print("\nExiting program...")
    finally:
        if detector_running:
            stop_detection()
        if video_stream is not None:
            video_stream.stop()
        GPIO.cleanup()
        ser.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

