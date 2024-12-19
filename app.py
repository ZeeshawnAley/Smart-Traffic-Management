from flask import Flask, render_template, jsonify, Response
import threading
import cv2
import time

app = Flask(__name__)

# Global variable to store car counts
car_counts = {"North": 0, "South": 0, "East": 0, "West": 0}
lock = threading.Lock()

latest_frame = None  # For video streaming

# Function to run car detection using OpenCV
def car_detection():
    global car_counts
    car_cascade = cv2.CascadeClassifier('haarcascade/cars.xml')
    cap = cv2.VideoCapture("traffic_video.mp4")  # Use webcam or a video file

    prev_car_positions = []  # To track previous positions of cars
    car_directions = {}  # Track direction for each car (North, South, East, West)

    min_move_threshold = 10  # Minimum pixel movement to count a direction change

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))

        new_car_positions = []  # Store current positions of cars for the next frame
        current_car_directions = {}  # Temporary storage for car directions in this frame

        # Check the position of each car and update the direction counts
        for (x, y, w, h) in cars:
            car_center = (x + w // 2, y + h // 2)

            # If it's the first frame or no previous positions are tracked, skip direction calculation
            if len(prev_car_positions) == 0:
                prev_car_positions.append(car_center)
                continue

            # Find if the current car matches any previous car by checking distance
            matched_car = None
            for i, prev_pos in enumerate(prev_car_positions):
                if abs(car_center[0] - prev_pos[0]) < 40 and abs(car_center[1] - prev_pos[1]) < 40:
                    matched_car = i
                    break

            # If no match found, add new car to the list
            if matched_car is None:
                prev_car_positions.append(car_center)
                current_car_directions[len(prev_car_positions) - 1] = None  # No direction yet
                continue

            # If matched, determine the direction of the car's movement
            car_id = matched_car
            prev_x, prev_y = prev_car_positions[car_id]
            car_direction = car_directions.get(car_id, None)

            # Determine if the movement is large enough to be counted
            if abs(car_center[0] - prev_x) > min_move_threshold or abs(car_center[1] - prev_y) > min_move_threshold:
                # Direction logic (count only once for each movement)
                if car_center[1] < prev_y and car_direction != "North":  # Moving up
                    current_car_directions[car_id] = "North"
                    with lock:
                        car_counts["North"] += 1
                elif car_center[1] > prev_y and car_direction != "South":  # Moving down
                    current_car_directions[car_id] = "South"
                    with lock:
                        car_counts["South"] += 1
                elif car_center[0] < prev_x and car_direction != "West":  # Moving left
                    current_car_directions[car_id] = "West"
                    with lock:
                        car_counts["West"] += 1
                elif car_center[0] > prev_x and car_direction != "East":  # Moving right
                    current_car_directions[car_id] = "East"
                    with lock:
                        car_counts["East"] += 1

        # Update the direction tracking for each car (marking the direction it moved)
        car_directions.update(current_car_directions)

        # Update the previous positions list to include the current positions
        prev_car_positions = new_car_positions

        # Draw rectangles on detected cars (Optional for debugging)
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Encode frame to JPEG and store as bytes
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            global latest_frame
            latest_frame = jpeg.tobytes()

        time.sleep(0.03)  # To simulate real-time frame rate (adjust if needed)

    cap.release()

# Traffic signal duration variables (used in the priority function)
base_time = 60  # Minimum time for a signal
time_per_car = 10  # Additional time per car
max_signal_time = 300  # Maximum allowed signal time in seconds

# Function to provide traffic signal priority based on car counts
def priority():
    with lock:
        # Sort car counts in descending order
        sorted_traffic = sorted(car_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create a prioritized sequence of directions with their signal times
        prioritized_sequence = [
            {
                "side": side,
                "car_count": count,
                "signal_time": min(base_time + count * time_per_car, max_signal_time)
            }
            for side, count in sorted_traffic
        ]
        
        return prioritized_sequence

# Route to provide traffic data with prioritization
@app.route('/traffic-data')
def traffic_data():
    prioritized_sequence = priority()
    total = car_counts["North"] + car_counts["South"] + car_counts["East"] + car_counts["West"]
    return jsonify({
        "North": car_counts["North"],
        "South": car_counts["South"],
        "East": car_counts["East"],
        "West": car_counts["West"],
        "Total": total,
        "prioritized_sequence": prioritized_sequence
    })

# Stream video frames to frontend
@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            # Wait until there is a new frame
            while latest_frame is None:
                time.sleep(0.1)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for frontend
@app.route('/')
def index():
    return render_template('index.html')

# Run Flask server and OpenCV detection concurrently
if __name__ == '__main__':
    # Start OpenCV detection in a separate thread
    detection_thread = threading.Thread(target=car_detection, daemon=True)
    detection_thread.start()

    # Run Flask server
    app.run(debug=True)