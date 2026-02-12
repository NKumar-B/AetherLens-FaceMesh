#  AetherLens - Face Mesh Mode
#  Libraries to be Installed 
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#  1. DETECTOR CONFIGURATION 
# Path to the face landmarker model file (478 3D landmarks).
# Ensure 'face_landmarker.task' is downloaded and in your project folder.
model_path = 'face_landmarker.task' 
base_options = python.BaseOptions(model_asset_path=model_path)

# Configure Face Landmarker Options:
#  output_face_blendshapes: Enables detection of facial expressions (smiling, blinking, etc.)
#  running_mode: Set to IMAGE for real-time frame-by-frame processing.
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    running_mode=vision.RunningMode.IMAGE)

# Initialize the Face Landmarker detector
detector = vision.FaceLandmarker.create_from_options(options)

#  2. CAMERA SETUP 
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break

    # Flip the frame for a natural mirror-view experience
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Convert BGR (OpenCV) to RGB (MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    #  3. RUN INFERENCE 
    # Detect facial landmarks in the current frame
    result = detector.detect(mp_image)

    #  4. VISUALIZE FACE MESH 
    if result.face_landmarks:
        # Iterate through detected faces (supports multiple faces if configured)
        for face_landmarks in result.face_landmarks:
            for landmark in face_landmarks:
                # Convert normalized coordinates (0.0 to 1.0) to pixel coordinates
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                # Draw a tiny green dot for each of the 478 facial landmarks
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Display the output window
    cv2.imshow("AetherLens - Face Mesh Mode", frame)
    
    # Press 'q' to exit the application
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Release resources
cap.release()
cv2.destroyAllWindows()