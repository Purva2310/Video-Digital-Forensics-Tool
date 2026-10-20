# Video Digital Forensics Tool

A Python application developed as a mini project for
Cyber Security and Digital Forensics.

The tool supports video inspection, file integrity comparison,
frame extraction, basic visual change analysis, and PDF reporting.

## Features

- Upload and preview videos.
- Calculate SHA-256 fingerprints and compare reference hashes.
- Extract video and audio metadata.
- Extract a frame near a selected timestamp.
- Extract frames at regular intervals and download a ZIP.
- Flag large visual changes between consecutive frames.
- Download a PDF examination report.

## Technologies Used

- Python
- Streamlit — application interface
- OpenCV — video reading and frame processing
- NumPy — numerical calculations
- FFprobe — metadata extraction
- ReportLab — PDF generation

## Installation

Python 3.12.9 was used during development.

1. Download or clone this repository.
2. Open the project folder in VS Code.
3. Create a virtual environment:

```powershell
python -m venv .venv
```

4. Install the dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

5. Install a Windows FFmpeg distribution containing `ffprobe.exe`.

6. Update `FFPROBE_PATH` in `app.py` to match your installation:

```python
FFPROBE_PATH = Path(r"D:\ffmpeg\bin\ffprobe.exe")
```

FFprobe is installed separately from the Python packages.

## Run the Application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:
http://localhost:8501

## How to Use

1. Enter the case ID and examiner name.
2. Select a video and review the preview.
3. View its SHA-256 fingerprint and optionally compare a reference.
4. Extract metadata.
5. Extract individual frames or an interval batch.
6. Run visual change analysis.
7. Download the PDF report.

## Output Files

- PNG — extracted frame
- ZIP — interval frame collection
- PDF — examination report

## Limitations

- Visual flags may result from normal cuts, motion, or lighting.
- Flags do not prove tampering.
- No flags does not establish authenticity.
- Frame seeking is approximate.
- Metadata may be missing or modified.
- SHA-256 comparison checks file integrity against a reference;
  it does not prove that the recorded content is genuine.
- This academic tool is not a validated deepfake detector or
  a complete chain-of-custody system.

## Author

Purva Thakare
