# Tom & Jerry – Hand Gesture Mouse Control

Control your computer mouse with hand gestures, built with **MediaPipe** and **scikit-learn**.

## Features
- Hand landmark detection with MediaPipe
- Gesture classification (RandomForest / MLP, best one is saved)
- Cursor movement, left/right click, drag, scroll
- Volume control and play/pause with gestures (macOS)
- Sound effects on actions

## Gestures

| Gesture | Action |
|---------|--------|
| `one` | Move the cursor |
| `peace` | Left click |
| `little_finger` | Right click |
| `two_up` | Scroll (move hand up/down) |
| `grabbing` | Select / drag |
| `like` | Volume up (move hand up) |
| `dislike` | Volume down (move hand down) |
| `fist` | Play/pause (hold ~1 second) |

Press `SPACE` to toggle control on/off (starts OFF) and `ESC` to quit.

## Project Structure

```
hand-mouse/
├── src/
│   ├── common.py      # paths, class names, normalize()
│   ├── convert.py     # HaGRID annotations -> train/val/test CSV
│   ├── record.py      # record your own samples from the camera
│   ├── train.py       # train the classifier
│   ├── mouse.py       # run the mouse controller
│   └── view_all.py    # plot example gestures
├── data/              # HaGRID sample + camera.csv
├── models/            # trained model (model.joblib)
├── outputs/           # generated images
├── requirements.txt
└── README.md
```

## Dataset

The training data comes from **HaGRID (HAnd Gesture Recognition Image Dataset)**:
- Repository: https://github.com/hukenovs/hagrid
- Paper: https://arxiv.org/abs/2206.08219

The `data/` folder contains a small sample of the HaGRID annotations. For the full dataset, visit the original repository.

## Installation

MediaPipe supports Python 3.9 - 3.12, so use one of those versions.

```bash
git clone https://github.com/Yousef-mousa1/tom-jerry.git
cd tom-jerry
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run everything from the project root:

```bash
python src/convert.py   # build train.csv / val.csv / test.csv
python src/record.py    # (optional) record your own samples
python src/train.py     # train and save models/model.joblib
python src/mouse.py     # start controlling the mouse
```

On macOS, allow **Camera** and **Accessibility** access for your terminal / VS Code in System Settings > Privacy & Security.

## Credits
- Dataset: [HaGRID](https://github.com/hukenovs/hagrid) by Kapitanov et al.
- Hand tracking: [MediaPipe](https://developers.google.com/mediapipe)

## Author
Youssef Mousa