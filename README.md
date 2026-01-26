# People Tracking with Face Recognition

This project implements a people tracking system using face recognition powered by the InsightFace library.

It captures video from a webcam, detects faces, extracts facial features, and matches them against a database of known individuals.

The system also includes a Streamlit-based interface for managing the database of known faces.
With the UI, you can view, add a name, or remove individuals from the database.
You can also merge two identities if the system mistakenly creates duplicates.

## Setup Instructions

```bash

virtualenv -p python3.12 venv
source venv/bin/activate # or venv\Scripts\activate.bat (Windows)


pip install insightface onnxruntime-gpu opencv-python numpy streamlit
```

### To run the app:

First initialize the database:

```bash
python db.py
```

To detect and track people using your webcam, run:

```bash
python main.py
```

For the Streamlit-based manager interface, use:

```bash
streamlit run manager.py
```
