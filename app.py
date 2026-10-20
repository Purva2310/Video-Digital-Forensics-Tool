import streamlit as st
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import cv2

import io
import zipfile

import numpy as np
import time

from datetime import datetime, timezone
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)

#Metadata function
FFPROBE_PATH= Path(r"D:\ffmpeg\bin\ffprobe.exe")

def extract_metadata(uploaded_file):
    """Read metadata while the temporary video still exists."""

    if not FFPROBE_PATH.is_file():
        raise FileNotFoundError(
            f"FFprobe was not found at: {FFPROBE_PATH}"
        )

    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.TemporaryDirectory() as temporary_directory:
        video_path = Path(temporary_directory) / f"input{suffix}"

        video_path.write_bytes(uploaded_file.getvalue())

        if not video_path.is_file():
            raise FileNotFoundError(
                f"Temporary video was not created: {video_path}"
            )

        command = [
            str(FFPROBE_PATH),
            "-v", "error",
            "-show_format",
            "-show_streams",
            "-of", "json",
            str(video_path),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=True,
        )

        metadata = json.loads(result.stdout)

    return metadata

#Frame extraction

def extract_frame(uploaded_file, requested_seconds):
    """Extract a PNG frame near the requested video time."""

    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.TemporaryDirectory() as temporary_directory:
        video_path = Path(temporary_directory) / f"input{suffix}"
        video_path.write_bytes(uploaded_file.getvalue())

        capture = cv2.VideoCapture(str(video_path))

        try:
            if not capture.isOpened():
                raise ValueError("OpenCV could not open this video.")

            # Convert seconds to milliseconds for OpenCV.
            seek_ok = capture.set(
                cv2.CAP_PROP_POS_MSEC,
                requested_seconds * 1000,
            )

            if not seek_ok:
                raise ValueError(
                    "The video reader could not seek to this time."
                )

            success, frame = capture.read()

            if not success or frame is None:
                raise ValueError(
                    "No frame could be read here. Try an earlier time."
                )

            encoded_ok, png_buffer = cv2.imencode(".png", frame)

            if not encoded_ok:
                raise ValueError("Could not create the PNG image.")

            return {
                "png_bytes": png_buffer.tobytes(),
                "requested_seconds": requested_seconds,
            }

        finally:
            capture.release()

#Analyze Visual Changes
def analyze_visual_changes(
    uploaded_file,
    threshold,
    max_frames=3000,
    time_limit=60,
):
    """Compare consecutive decoded frames using grayscale differences."""

    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.TemporaryDirectory() as temporary_directory:
        video_path = Path(temporary_directory) / f"input{suffix}"
        video_path.write_bytes(uploaded_file.getvalue())

        capture = cv2.VideoCapture(str(video_path))

        try:
            if not capture.isOpened():
                raise ValueError("OpenCV could not open this video.")

            previous_gray = None
            scores = []
            flagged_frames = []
            frames_read = 0
            started_at = time.monotonic()
            stop_reason = "Frame limit reached; analysis may be partial."

            while frames_read < max_frames:
                if time.monotonic() - started_at >= time_limit:
                    stop_reason = (
                        "Time limit reached; analysis is partial."
                    )
                    break

                success, frame = capture.read()

                if not success or frame is None:
                    stop_reason = (
                        "Reader stopped: end of video or decode failure. "
                        "Full decoding is not independently verified."
                    )
                    break

                # Use a fixed analysis size to keep processing manageable.
                resized = cv2.resize(frame, (320, 180))
                gray = cv2.cvtColor(
                    resized,
                    cv2.COLOR_BGR2GRAY,
                )

                # Frame indices start at zero.
                frame_index = frames_read
                frames_read += 1

                if previous_gray is not None:
                    difference = cv2.absdiff(gray, previous_gray)
                    score = float(np.mean(difference))

                    scores.append({
                        "Frame index": frame_index,
                        "Difference score": score,
                    })

                    if score >= threshold:
                        flagged_frames.append({
                            "Previous frame": frame_index - 1,
                            "Current frame": frame_index,
                            "Difference score": round(score, 3),
                        })

                previous_gray = gray

            if frames_read < 2:
                raise ValueError(
                    "At least two readable frames are needed."
                )

            return {
                "frames_read": frames_read,
                "scores": scores,
                "flags": flagged_frames,
                "threshold": threshold,
                "stop_reason": stop_reason,
            }

        finally:
            capture.release()

#Generate report
def generate_pdf_report(
    case_id,
    examiner_name,
    filename,
    file_size,
    sha256_hash,
    metadata,
    analysis,
    frame_result,
    batch,
):
    """Build a PDF from the currently available examination results."""

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    story = []

    def paragraph(value, style="BodyText"):
        # Escape user text so it is not treated as formatting markup.
        return Paragraph(escape(str(value)), styles[style])

    def heading(title):
        story.append(Spacer(1, 12))
        story.append(paragraph(title, "Heading2"))

    def add_table(rows):
        cells = [
            [paragraph(label), paragraph(value)]
            for label, value in rows
        ]

        table = Table(
            cells,
            colWidths=[155, document.width - 155],
            hAlign="LEFT",
        )

        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f6")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))

        story.append(table)

    story.append(paragraph(
        "Video Digital Forensics Examination Report",
        "Title",
    ))

    story.append(paragraph(
        "Academic video examination tool - application version 1.0"
    ))

    heading("1. Case and File Details")

    generated_at = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    add_table([
        ("Case ID", case_id or "Not provided"),
        ("Examiner", examiner_name or "Not provided"),
        ("Report generated", generated_at),
        ("Uploaded filename", filename),
        ("File size", f"{file_size} bytes"),
        ("OpenCV version", cv2.__version__),
    ])

    heading("2. SHA-256 Fingerprint")

    # Split the hash visually to prevent a long unbroken line.
    story.append(Paragraph(
        escape(sha256_hash[:32])
        + "<br/>"
        + escape(sha256_hash[32:]),
        styles["Code"],
    ))

    story.append(paragraph(
        "The two lines above form one continuous SHA-256 value. "
        "This records the uploaded file's current bytes. It does not "
        "establish authenticity. Reference-hash comparison is not "
        "included in this report."
    ))

    heading("3. Metadata")

    if metadata is None:
        story.append(paragraph(
            "No successful metadata extraction is available."
        ))
    else:
        format_info = metadata.get("format", {})
        streams = metadata.get("streams", [])

        add_table([
            (
                "Container",
                format_info.get("format_long_name", "Unavailable"),
            ),
            (
                "Container duration (s)",
                format_info.get("duration", "Unavailable"),
            ),
            (
                "Reported total bitrate (bits/s)",
                format_info.get("bit_rate", "Unavailable"),
            ),
            ("Stream count", len(streams)),
        ])

        for stream in streams:
            stream_type = stream.get("codec_type", "Unknown")
            stream_index = stream.get("index", "Unknown")

            story.append(Spacer(1, 8))
            story.append(paragraph(
                f"Stream {stream_index}: {stream_type}",
                "Heading3",
            ))

            rows = [
                ("Codec", stream.get("codec_name", "Unavailable")),
            ]

            if stream_type == "video":
                rows.extend([
                    ("Width (pixels)", stream.get("width", "Unavailable")),
                    ("Height (pixels)", stream.get("height", "Unavailable")),
                    (
                        "Average frame rate (ratio)",
                        stream.get("avg_frame_rate", "Unavailable"),
                    ),
                    (
                        "Reported frame count",
                        stream.get("nb_frames", "Unavailable"),
                    ),
                ])

            elif stream_type == "audio":
                rows.extend([
                    (
                        "Sample rate (Hz)",
                        stream.get("sample_rate", "Unavailable"),
                    ),
                    ("Channels", stream.get("channels", "Unavailable")),
                ])

            add_table(rows)

    heading("4. Visual Change Analysis")

    if analysis is None:
        story.append(paragraph(
            "No successful visual change analysis is available."
        ))
    else:
        add_table([
            ("Frames read", analysis["frames_read"]),
            ("Frame pairs compared", len(analysis["scores"])),
            ("Threshold used", analysis["threshold"]),
            ("Flagged pairs", len(analysis["flags"])),
            ("Stopping condition", analysis["stop_reason"]),
            ("Analysis image size", "320 x 180 pixels, grayscale"),
            ("Processing limits", "3000 frames; 60 seconds checked between reads"),
        ])

        story.append(Spacer(1, 8))
        story.append(paragraph(
            "Method: mean absolute grayscale pixel difference between "
            "consecutive resized frames. Scores range from 0 to 255. "
            "Frame indices start at zero."
        ))

        if analysis["flags"]:
            story.append(paragraph(
                "Flagged frame pairs (up to the first 100 shown):",
                "Heading3",
            ))

            for flag in analysis["flags"][:100]:
                story.append(paragraph(
                    f"Frames {flag['Previous frame']} to "
                    f"{flag['Current frame']}: "
                    f"score {flag['Difference score']}"
                ))

            if len(analysis["flags"]) > 100:
                story.append(paragraph(
                    "Additional flags are omitted from this PDF."
                ))
        else:
            story.append(paragraph(
                "No pairs exceeded the selected threshold in the "
                "analyzed portion. This does not establish authenticity."
            ))

    heading("5. Frame Extraction")

    if frame_result is None:
        story.append(paragraph(
            "No single-frame extraction result is available."
        ))
    else:
        story.append(paragraph(
            "Single frame requested near "
            f"{frame_result['requested_seconds']:.3f} seconds. "
            "Timing is approximate."
        ))

        image = Image(io.BytesIO(frame_result["png_bytes"]))

        scale = min(
            document.width / image.imageWidth,
            240 / image.imageHeight,
        )

        image.drawWidth = image.imageWidth * scale
        image.drawHeight = image.imageHeight * scale

        story.append(Spacer(1, 8))
        story.append(image)

    story.append(Spacer(1, 8))

    if batch is None:
        story.append(paragraph(
            "No interval extraction result is available."
        ))
    else:
        story.append(paragraph(
            f"Interval extraction: {batch['count']} frames, "
            f"requested at {batch['interval']:.3f}-second intervals "
            "starting at zero. Images are available separately in "
            "the ZIP download."
        ))

        if batch.get("stop_message"):
            story.append(paragraph(batch["stop_message"]))

    heading("6. Interpretation and Limitations")

    story.append(paragraph(
        "This report supports academic examination and manual review. "
        "Metadata may be absent or modified. Visual changes can result "
        "from normal scene cuts, movement, or lighting changes. "
        "The method is not a validated tampering or deepfake detector. "
        "The tool processes temporary copies and does not overwrite "
        "the source video. This report is not an authenticity "
        "certificate or a complete chain-of-custody record."
    ))

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(
            A4[0] - 40,
            22,
            f"Page {doc.page}",
        )
        canvas.restoreState()

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    return buffer.getvalue()

#Configure the browser page
st.set_page_config(
    page_title= "Video Forensics Tool",
    page_icon= "🎞️",
    layout= "wide",
)

st.title("Video Digital Forensics Tool")
st.write(
    "Select a video to begin examining its technical details and frame."
)

st.info(
    "This academic tool supports video examination."
    "It does not certify whether a video is authentic."
)

#Collect basic examination details.
st.subheader("1. Examination Details")

case_id= st.text_input(
    "Case ID",
    placeholder= "Example: CASE-001",
)

examiner_name= st.text_input(
    "Examiner Name",
    placeholder= "Enter your name",
)

#Accept a video file.
st.subheader("2. Select a Video")

uploaded_video= st.file_uploader(
    "Choose a video file",
    type= ["mp4","avi","mov","mkv"],
)

if uploaded_video is not None:
    st.success("Video received.")

    st.write("**Filename:**", uploaded_video.name)
    st.write(
        "**File size:**",
        f"{uploaded_video.size/ (1024*1024):2f} MiB",
    )

    if case_id:
        st.write("**Case ID:**", case_id)

    if examiner_name:
        st.write("**Examiner:**", examiner_name)

    st.subheader("3. Video Preview")
    st.video(uploaded_video)

    st.caption(
        "Preview playback depends on your browser's support for"
        "the video's encoding. The file has not yet been validated"
        "for analysis."
    )


    #Fingerprint- SHA-256

    st.subheader("4. File Fingerprint- SHA-256")

    #getbuffer() gives access to the uploaded file's bytes
    #without depending on the file's current read positions.
    video_buffer= uploaded_video.getbuffer()

    sha256_hash= hashlib.sha256(video_buffer).hexdigest()

    #Release the buffer view after calculating the hash
    del video_buffer

    st.write("SHA-256 of the uploaded file:")
    st.code(sha256_hash, language=None)

    st.caption(
        "This fingerprint identifies the file's current bytes."
        "It does not prove that the video is authentic or unedited."
    )

    reference_hash= st.text_input(
        "Optional: enter an earlier SHA-256 hash for comparison",
        help="Paste a 64-character SHA-256 hash from a trusted reference.",
    )

    if reference_hash.strip():
        cleaned_hash= reference_hash.strip().lower()

        is_valid_hash= (
            len(cleaned_hash)==64
            and all(
                character in "0123456789abcdef"
                for character in cleaned_hash
            )
        )

        if not is_valid_hash:
            st.error(
                "Enter a valid SHA-256 hash: "
                "exactly 64 hexadecimal characters."
            )
        elif cleaned_hash== sha256_hash:
            st.success(
                "Hashes match. The uploaded file's fingerprint "
                "matches the supplied reference."
            )
        else:
            st.warning(
                "Hashes do not match. Check that you supplied "
                "the correct reference for this file."
            )

    #Metadata

    st.subheader("5. Video Metadata")

    if st.session_state.get("metadata_file_hash")!= sha256_hash:
        st.session_state["metadata_file_hash"]=sha256_hash
        st.session_state.pop("video_metadata", None)

    if st.button("Extract Metadata"):
        try:
            with st.spinner("Reading video metadata..."):
                metadata= extract_metadata(uploaded_video)

                st.session_state["video_metadata"]=metadata

        except FileNotFoundError as error:
            st.error(str(error))

        except subprocess.TimeoutExpired:
            st.error(
                "Metadata extraction exceeded the 60-second limit."
            )

        except subprocess.CalledProcessError as error:
            st.error("FFprobe could not inspect this file.")
            st.text((error.stderr or "No error details available.")[:2000])

        except json.JSONDecodeError:
            st.error("FFprobe returned an unreadable metadata response.")

        except OSError as error:
            st.error(f"Could not process the temporary copy: {error}")

    metadata= st.session_state.get("video_metadata")

    if metadata is not None:
        format_info= metadata.get("format",{})
        streams= metadata.get("streams",[])

        video_streams= []
        audio_streams= []

        for stream in streams:
            stream_type= stream.get("codec_type")

            if stream_type== "video":
                disposition= stream.get("disposition",{})
                attached_pic= disposition.get("attached_pic",0)

                #Exclude embedded cover images.
                if attached_pic in (0,"0", False, None):
                    video_streams.append(stream)

            elif stream_type== "audio":
                audio_streams.append(stream)

        if not video_streams:
            st.warning("FFprobe found no regular video stream.")
        else:
            # Summarize the first regular video stream.
            video_info = video_streams[0]

            duration = format_info.get("duration")
            duration_display = "Unavailable"

            try:
                duration_display = f"{float(duration):.2f} seconds"
            except (TypeError, ValueError):
                pass

            frame_rate = video_info.get("avg_frame_rate")
            frame_rate_display = "Unavailable"

            try:
                numerator, denominator = frame_rate.split("/")
                fps = float(numerator) / float(denominator)

                if fps > 0:
                    frame_rate_display = f"{fps:.3f} FPS"
            except (AttributeError, ValueError, ZeroDivisionError):
                pass

            width = video_info.get("width")
            height = video_info.get("height")

            resolution = (
                f"{width} × {height} pixels"
                if width and height
                else "Unavailable"
            )

            summary = [
                {
                    "Property": "Container",
                    "Value": format_info.get(
                        "format_long_name", "Unavailable"
                    ),
                },
                {
                    "Property": "Container duration",
                    "Value": duration_display,
                },
                {
                    "Property": "Resolution",
                    "Value": resolution,
                },
                {
                    "Property": "Video codec",
                    "Value": video_info.get(
                        "codec_name", "Unavailable"
                    ),
                },
                {
                    "Property": "Reported average frame rate",
                    "Value": frame_rate_display,
                },
                {
                    "Property": "Video streams",
                    "Value": str(len(video_streams)),
                },
                {
                    "Property": "Audio streams",
                    "Value": str(len(audio_streams)),
                },
            ]

            st.table(summary)

            st.caption(
                "Video properties summarize the first regular video "
                "stream. Average FPS does not establish a constant "
                "frame rate. Metadata does not prove authenticity."
            )

        with st.expander("View complete FFprobe metadata"):
            st.json(metadata)

    #Extracting single frame

    st.subheader("6. Extract a Frame")

    # Remove the previous result when a different video is uploaded.
    if st.session_state.get("frame_file_hash") != sha256_hash:
        st.session_state["frame_file_hash"] = sha256_hash
        st.session_state.pop("extracted_frame", None)

    requested_seconds = st.number_input(
        "Frame position in seconds",
        min_value=0.0,
        value=0.0,
        step=0.1,
        format="%.3f",
        help="Enter 2.0 to request a frame near 2 seconds.",
    )

    if st.button("Extract Frame"):
        st.session_state.pop("extracted_frame", None)

        try:
            with st.spinner("Extracting frame..."):
                frame_result = extract_frame(
                    uploaded_video,
                    requested_seconds,
                )

            st.session_state["extracted_frame"] = frame_result

        except (ValueError, OSError, cv2.error) as error:
            st.error(f"Frame extraction failed: {error}")

    frame_result = st.session_state.get("extracted_frame")

    if frame_result is not None:
        extracted_at = frame_result["requested_seconds"]

        st.image(
            frame_result["png_bytes"],
            caption=f"Frame requested near {extracted_at:.3f} seconds",
        )

        st.caption(
            "The selected time is approximate. Video seeking may "
            "return a nearby frame rather than the exact timestamp."
        )

        st.download_button(
            label="Download Frame as PNG",
            data=frame_result["png_bytes"],
            file_name=f"frame_requested_{extracted_at:.3f}s.png",
            mime="image/png",
        )

    #Extracting multiple frames
    st.subheader("7. Extract Frames at Intervals")

    # Clear previous batch results when the video changes.
    if st.session_state.get("batch_file_hash") != sha256_hash:
        st.session_state["batch_file_hash"] = sha256_hash
        st.session_state.pop("frame_batch", None)

    interval_seconds = st.number_input(
        "Interval between frames (seconds)",
        min_value=0.5,
        value=2.0,
        step=0.5,
    )

    frame_limit = st.number_input(
        "Maximum number of frames",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
        help="Limits processing time and the size of the download.",
    )

    st.caption(
        "Extraction starts at 0 seconds. It stops at the selected "
        "limit or when a frame cannot be read."
    )

    if st.button("Extract Multiple Frames"):
        st.session_state.pop("frame_batch", None)

        extracted_frames = []
        stop_message = None
        progress = st.progress(0.0)

        try:
            with st.spinner("Extracting frames..."):
                for index in range(int(frame_limit)):
                    requested_time = index * interval_seconds

                    try:
                        result = extract_frame(
                            uploaded_video,
                            requested_time,
                        )
                    except (ValueError, OSError, cv2.error) as error:
                        stop_message = (
                            f"Stopped at requested time "
                            f"{requested_time:.3f}s: {error}"
                        )
                        break

                    extracted_frames.append(result)

                    progress.progress(
                        (index + 1) / int(frame_limit)
                    )

            if extracted_frames:
                zip_buffer = io.BytesIO()

                with zipfile.ZipFile(
                    zip_buffer,
                    mode="w",
                    compression=zipfile.ZIP_STORED,
                ) as archive:
                    for index, result in enumerate(extracted_frames):
                        requested_time = result["requested_seconds"]

                        filename = (
                            f"frame_{index:03d}_requested_"
                            f"{requested_time:.3f}s.png"
                        )

                        archive.writestr(
                            filename,
                            result["png_bytes"],
                        )

                st.session_state["frame_batch"] = {
                    "zip_bytes": zip_buffer.getvalue(),
                    "count": len(extracted_frames),
                    "interval": interval_seconds,
                    "stop_message": stop_message,
                }
            else:
                st.error(stop_message or "No frames were extracted.")

        finally:
            progress.empty()

    batch = st.session_state.get("frame_batch")

    if batch is not None:
        st.success(
            f"Extracted {batch['count']} frames using a "
            f"{batch['interval']:.3f}-second interval."
        )

        if batch["stop_message"]:
            st.warning(batch["stop_message"])

        st.download_button(
            label="Download Frames as ZIP",
            data=batch["zip_bytes"],
            file_name="extracted_frames.zip",
            mime="application/zip",
        )

        st.caption(
            "Filenames record requested times. Actual frame timing "
            "is approximate. A read failure may indicate the end of "
            "the video or a decoding problem."
        )

    
    #Visual Change Analysis
    st.subheader("8. Visual Change Analysis")

    st.write(
        "Compare consecutive frames and flag large visual changes "
        "for manual review."
    )

    if st.session_state.get("analysis_file_hash") != sha256_hash:
        st.session_state["analysis_file_hash"] = sha256_hash
        st.session_state.pop("visual_analysis", None)

    threshold = st.slider(
        "Difference threshold",
        min_value=1,
        max_value=255,
        value=30,
        help=(
            "Lower values usually produce more flags. "
            "This is a pixel-difference threshold, not a probability."
        ),
    )

    st.caption(
        "Analysis uses resized grayscale frames (320 × 180), "
        "with limits of 3,000 frames and approximately 60 seconds "
        "of processing. The time limit is checked between reads."
    )

    if st.button("Analyze Visual Changes"):
        st.session_state.pop("visual_analysis", None)

        try:
            with st.spinner("Comparing video frames..."):
                analysis = analyze_visual_changes(
                    uploaded_video,
                    threshold,
                )

            st.session_state["visual_analysis"] = analysis

        except (ValueError, OSError, cv2.error) as error:
            st.error(f"Analysis failed: {error}")

    analysis = st.session_state.get("visual_analysis")

    if analysis is not None:
        st.write("**Frames read:**", analysis["frames_read"])
        st.write("**Threshold used:**", analysis["threshold"])
        st.write("**Flagged frame pairs:**", len(analysis["flags"]))
        st.info(analysis["stop_reason"])

        st.line_chart(
            analysis["scores"],
            x="Frame index",
            y="Difference score",
        )

        if analysis["flags"]:
            st.dataframe(analysis["flags"], hide_index=True)
        else:
            st.write(
                "No frame pairs exceeded the selected threshold "
                "in the analyzed portion."
            )

        st.warning(
            "Flags can result from normal scene cuts, camera movement, "
            "or lighting changes. No flags does not establish "
            "authenticity. This method is not a validated tampering "
            "or deepfake detector."
        )
    #PDF Generation
    st.subheader("9. Download Examination Report")

    st.caption(
        "The PDF includes results currently available for this video. "
        "Run metadata extraction and visual analysis first for a "
        "more complete report."
    )

    # Only include results associated with this video's hash.
    report_metadata = (
        st.session_state.get("video_metadata")
        if st.session_state.get("metadata_file_hash") == sha256_hash
        else None
    )

    report_analysis = (
        st.session_state.get("visual_analysis")
        if st.session_state.get("analysis_file_hash") == sha256_hash
        else None
    )

    report_frame = (
        st.session_state.get("extracted_frame")
        if st.session_state.get("frame_file_hash") == sha256_hash
        else None
    )

    report_batch = (
        st.session_state.get("frame_batch")
        if st.session_state.get("batch_file_hash") == sha256_hash
        else None
    )

    try:
        pdf_bytes = generate_pdf_report(
            case_id=case_id,
            examiner_name=examiner_name,
            filename=uploaded_video.name,
            file_size=uploaded_video.size,
            sha256_hash=sha256_hash,
            metadata=report_metadata,
            analysis=report_analysis,
            frame_result=report_frame,
            batch=report_batch,
        )

        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="video_examination_report.pdf",
            mime="application/pdf",
        )

    except Exception as error:
        st.error(f"Could not generate the PDF report: {error}")

else:
    st.caption("Select a video to display its details.")
