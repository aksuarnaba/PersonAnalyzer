# Person Analyzer

Real-time person analysis with OpenCV and MediaPipe. It works on a webcam, a phone used as a webcam (DroidCam), a video file, or a single image.

**Face analysis**
- Face detection (YuNet)
- Gender and age estimation
- Emotion recognition

**Hand analysis**
- 21-point hand landmarks with a drawn skeleton
- Finger counting
- Open / closed / partial hand
- Built-in gestures: Thumbs up, Thumbs down, Peace, OK, Fist, Open hand, Pointing, Rock on
- Custom gestures you record yourself
- Static sign language (for example the ASL alphabet), recorded live or imported from photos

---

## Project structure

```
Person_Analyzer/
├── app.py                 # main program
├── hands.py               # hand tracking, gestures, classifier
├── import_signs.py        # build the dataset from images
├── models/
│   ├── age/
│   │   ├── age_deploy.prototxt
│   │   └── age_net.caffemodel
│   ├── emotion/
│   │   └── emotion-ferplus-8.onnx
│   ├── face_detector/
│   │   └── face_detection_yunet_2023mar.onnx
│   ├── gender/
│   │   ├── gender_deploy.prototxt
│   │   └── gender_net.caffemodel
│   └── hand/
│       └── hand_landmarker.task      # downloaded automatically on first run
└── data/                  # created automatically
    ├── custom.csv         # your custom gesture samples
    └── signs.csv          # your sign language samples
```

---

## Installation

Python 3.12 is recommended, because MediaPipe supports fewer Python versions than OpenCV.

```
uv add mediapipe
uv add "opencv-contrib-python>=4.10,<5"
uv add cv2-enumerate-cameras
```

Or with pip:

```
pip install mediapipe "opencv-contrib-python<5" cv2-enumerate-cameras
```

> **Important: OpenCV must be 4.x.**
> OpenCV 5 removed the Caffe importer, so the age and gender models (`.caffemodel`) fail to load with
> `Caffe importer has been removed`. Install only `opencv-contrib-python` and make sure `opencv-python` is not installed alongside it.
>
> Check with: `python -c "import cv2; print(cv2.__version__)"`

The face, age, gender and emotion model files go in `models/` as shown above. The hand model downloads itself the first time you run the app.

---

## Running

```
python Person_Analyzer\app.py
```

| Option | Default | Description |
|---|---|---|
| `--source` | `700` | Camera index, video file path, URL, image file, or `auto` |
| `--every` | `3` | Run the age/gender/emotion models every N frames (hands run every frame) |
| `--record LABEL` | none | Record training samples for a gesture or letter |
| `--set` | `custom` | Dataset to record into: `custom` or `signs` |

Examples:

```
python Person_Analyzer\app.py                          # DroidCam (700)
python Person_Analyzer\app.py --source 0               # built-in webcam
python Person_Analyzer\app.py --source auto            # search for DroidCam, then any camera
python Person_Analyzer\app.py --source video.mp4       # video file
python Person_Analyzer\app.py --source photo.jpg       # single image
```

Press **Q** or **Esc** to quit.

### DroidCam / camera indices

OpenCV camera indices can include a backend: `700` means DirectShow camera 0, and `1400` means Media Foundation camera 0. To see what your system exposes:

```
python -c "from cv2_enumerate_cameras import enumerate_cameras as e; [print(c.index, c.backend, c.name) for c in e()]"
```

Use the index printed next to the DroidCam entry as `--source`. If the camera won't open, check that the DroidCam desktop client is connected to your phone and that no other program is using the camera.

---

## Built-in gestures

The overlay shows `Fingers: N | Open/Closed/Partial` for each hand, plus the gesture name when one matches.

| Gesture | Rule |
|---|---|
| Thumbs up / down | Thumb out, other four fingers curled, thumb mostly vertical |
| Peace | Index and middle extended, ring and pinky curled |
| OK | Thumb tip touching index tip, other three fingers extended |
| Fist | All four fingers curled |
| Open hand | All five fingers extended |
| Pointing | Only the index finger extended |
| Rock on | Index and pinky extended |

To add your own rule, add an `if` block to `classify_gesture()` in `hands.py`.

---

## Custom gestures

Record a few poses and the app learns to recognize them (nearest-neighbor matching, no training step).

```
python Person_Analyzer\app.py --record "Spock" --set custom
```

1. Hold the pose in front of the camera.
2. Press **SPACE** to start capturing, and move or rotate your hand slightly while it records.
3. Press **SPACE** again to pause.
4. Press **Q** to save and quit.

Aim for 100–200 samples per gesture. Repeat with a different label for each gesture. Recognized gestures appear as `Custom: Spock 100%`.

---

## Sign language

Sign language uses the same mechanism, stored in its own dataset (`data/signs.csv`). Recognized signs appear as `Sign: A 100%`.

### Option 1: record live

```
python Person_Analyzer\app.py --record A --set signs
```

Do this once per letter, using the same controls as custom gestures.

### Option 2: import from photos

Organize images with one sub-folder per label:

```
signs_images/
├── A/  a1.jpg  a2.jpg ...
├── B/  b1.jpg  ...
```

or put the label in the file name (`A.jpg`, `C_1.jpg`, `C_2.jpg`; trailing numbers are ignored). Then run:

```
python Person_Analyzer\import_signs.py Person_Analyzer\signs_images
```

The script prints the number of samples per label and lists images where no hand was found. You can combine imported photos with live recordings.

Tips for photos:
- Use real photos of hands. Drawings and cartoon charts often fail to detect.
- A single chart image with many letters must be cropped into one image per sign.
- 20–100 varied photos per letter works much better than one.

### Limitations

- Only **static handshapes** are recognized. Letters that need motion (such as J and Z) and full word-level signing would need a sequence model trained on video.
- Similar-looking signs (A/S/T/M/N/E) are easily confused unless you provide plenty of varied samples.

---

## Tuning

| Problem | What to change |
|---|---|
| Thumb or fingers misdetected | The `1.05` factors in `finger_states()` in `hands.py` |
| Custom or sign predictions never appear | Raise `max_dist` in `GestureKNN.predict` (default `0.8`) |
| Wrong custom or sign matches | Lower `max_dist`, or add more varied samples |
| Overlay is slow | Increase `--every` |

To start a dataset over, delete `data/custom.csv` or `data/signs.csv`.

---

## Troubleshooting

| Message | Cause and fix |
|---|---|
| `Caffe importer has been removed` | OpenCV 5 is installed. Install `opencv-contrib-python<5`. |
| `Cannot open source: 700` | DroidCam is not connected, the camera is in use, or the index differs. See the DroidCam section above. |
| `Missing model file: ...` | A model file is not in the `models/` folder shown above. |
| `W0000 ... Feedback manager ...` / `INFO: Created TensorFlow Lite XNNPACK delegate` | Harmless MediaPipe log messages. |
| `No module named 'hands'` | Run the script from inside `Person_Analyzer`, or call it by its full path. |
| `can't find '__main__' module` | You passed a folder to `python` instead of a script. The script comes first, then the folder as its argument. |

---

## Notes

- Age, gender and emotion predictions are rough estimates from small, older models. Treat them as a demo, not as reliable facts about a person.
- All processing runs locally. Nothing is uploaded, except that the hand model is downloaded once from Google's MediaPipe model storage.