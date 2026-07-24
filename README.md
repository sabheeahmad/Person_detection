# P_check — Person Entry/Exit Counter

P_check uses YOLOv8 to detect and track people in camera frames. When a person crosses the configured doorway lines, the project records an `ENTRY` or `EXIT` event and updates the count of people inside.

This project is designed for places such as shops, offices, and building entrances where automatic visitor counting is useful. It processes a sequence of images from a fixed camera instead of requiring a live video stream.

Each detected person receives a tracking ID so the system can follow their movement across multiple frames. Two virtual lines are used to confirm the direction of movement: crossing them in one order means `ENTRY`, while crossing them in the opposite order means `EXIT`. This helps prevent small movements near the doorway from being counted as real events.

For every confirmed crossing, the project saves an annotated image and adds a row to a CSV file. These results make it easy to review individual detections and see the total number of entries, exits, and people currently inside.

## Sample Detection

![P_check entry detection sample](detection_output/incident_5f138750-97cd-4f1a-be62-6c77d7930504_20260305-015638_2_4_trimmed_crossing_ENTRY_frame_000012_frame_000014.jpg)

The yellow box marks the detected person, and the red lines define the entry/exit boundary.

## How It Works

1. YOLOv8 detects people in each frame.
2. The tracker follows each person between frames.
3. Crossing the two configured lines creates an `ENTRY` or `EXIT` event.

## Setup

This project requires Python 3.10+ and an NVIDIA GPU with CUDA.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add captured frames to a folder inside `frames/`:

```text
frames/
└── example/
    ├── frame_000001.jpg
    ├── frame_000002.jpg
    └── ...
```

## Run

Process every frame folder:

```bash
python main.py
```

Process only one folder:

```bash
P_CHECK_FRAMES_DIR=frames/example python main.py
```

## Output

- `detection_output/counts.csv` — entry, exit, and occupancy records
- `detection_output/*.jpg` — saved images for detected crossings
- `debug_output/` — optional debug images

Detection settings are in `config.py`. The region of interest and crossing lines are configured by the JSON files in `mask/`.
