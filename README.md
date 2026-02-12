#  AetherLens - Face Mesh (3D Landmark Mapping)

**AetherLens - Face Mesh** is a high-fidelity computer vision application that performs real-time 3D facial geometry tracking. By leveraging the **MediaPipe Face Landmarker** task, it maps **478 unique 3D landmarks** onto the human face, providing a dense mesh suitable for facial analysis, augmented reality (AR) filters, and virtual avatars.

<img width="797" height="637" alt="FaceMesh" src="https://github.com/user-attachments/assets/02f97392-e255-43f3-8382-a3f84a53e035" />

---

##  Key Features

* **478 3D Facial Landmarks**: Detects a high-density mesh including eye contours, lips, and facial silhouettes.
* **Expression Detection (Blendshapes)**: Capable of outputting 52 blendshape scores to recognize facial expressions like smiling, blinking, or brow movement.
* **Real-Time Performance**: Optimized for sub-millisecond processing on standard CPU/GPU hardware.
* **Mirror Mode**: Horizontally flipped feed for an intuitive user experience.

---

##  Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/NKumar-B/AetherLens-FaceMesh.git
cd AetherLens-FaceMesh

```

### 2. Set Up a Virtual Environment (Recommended)

```bash
python -m venv .venv
.\.venv\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Download the Model Bundle

You must download the **Face Landmarker** model bundle from Google and place it in your project root:

* **Model Name**: `face_landmarker.task`
* **Download Link**: [MediaPipe Face Landmarker Guide](https://www.google.com/search?q=https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker%23models)

---

##  How to Use

1. Run the application:
```bash
python FaceLandmarker.py

```


2. **Face Mesh Visualization**: Look into the camera to see the green landmark mesh track your face in real-time.
3. **Exit**: Press **'q'** to close the window.

---

##  Technical Overview

The system uses a multi-stage ML pipeline:

1. **Face Detection**: A short-range BlazeFace model identifies the presence of a face.
2. **Landmark Estimation**: A 3D landmark model predicts 478 landmarks via regression on the detected face region.
3. **Coordinate Mapping**: Normalized coordinates  are converted to pixel values based on the live camera resolution.

---

##  License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

##  Acknowledgments

* **[Google MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide)**: For providing the robust Face Landmarker Tasks API and pre-trained `.task` models.
* **[OpenCV (Open Source Computer Vision Library)](https://opencv.org/)**: For the powerful real-time image processing and visualization tools.
* **[The COCO Dataset Team](https://cocodataset.org/)**: For their foundational work in standardizing computer vision training data.
* **[NumPy](https://numpy.org/)**: For the efficient numerical processing required for coordinate mapping.

---

