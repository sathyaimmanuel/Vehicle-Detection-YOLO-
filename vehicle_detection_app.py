# vehicle_detection_app.py
import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from ultralytics import YOLO
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Traffic Vehicle Detection System",
    page_icon="🚗",
    layout="wide"
)

# Initialize session state
if 'media_path' not in st.session_state:
    st.session_state.media_path = None
if 'media_type' not in st.session_state:
    st.session_state.media_type = None
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = []
if 'output_video_path' not in st.session_state:
    st.session_state.output_video_path = None
if 'vehicle_counts' not in st.session_state:
    st.session_state.vehicle_counts = defaultdict(int)
if 'frame_data' not in st.session_state:
    st.session_state.frame_data = []

# Load YOLO model
@st.cache_resource
def load_model():
    try:
        model = YOLO('best.pt')  # Try to load custom trained model
        st.sidebar.success("Custom vehicle detection model loaded!")
    except:
        model = YOLO('yolov8m.pt')  # Fallback to pretrained model
        st.sidebar.warning("Using pretrained YOLOv8 medium model")
    return model

model = load_model()

# Vehicle classes to detect
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']

# Sidebar configuration
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
selected_classes = st.sidebar.multiselect(
    "Select Vehicle Classes",
    options=VEHICLE_CLASSES,
    default=['car', 'truck', 'bus']
)

# Main application title
st.title("🚗 Traffic Vehicle Detection System")
st.markdown("""
This application detects and classifies vehicles in traffic scenes using YOLOv8.
Upload an image or video to analyze traffic patterns.
""")

# Function to process image
def process_image(image, conf_thresh, classes):
    """Process single image and return detection results"""
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run detection
    results = model.predict(
        image_rgb, 
        conf=conf_thresh,
        classes=[VEHICLE_CLASSES.index(c) for c in classes] if classes else None
    )
    
    # Get detection data
    detection_data = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            class_name = model.names[int(box.cls[0])]
            if class_name in VEHICLE_CLASSES:  # Only count vehicle classes
                detection_data.append({
                    'class': class_name,
                    'confidence': float(box.conf[0]),
                    'x1': int(box.xyxy[0][0]),
                    'y1': int(box.xyxy[0][1]),
                    'x2': int(box.xyxy[0][2]),
                    'y2': int(box.xyxy[0][3])
                })
    
    # Annotate image
    annotated_image = results[0].plot()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    return annotated_image, detection_data

# Function to process video
def process_video(video_path, conf_thresh, classes):
    """Process video file and return statistics"""
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Prepare output video
    output_path = os.path.join(tempfile.gettempdir(), "output_detected_video.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # Reset counts and frame data
    vehicle_counts = defaultdict(int)
    frame_data = []
    
    # Prepare Streamlit UI elements
    progress_bar = st.progress(0)
    status_text = st.empty()
    video_placeholder = st.empty()
    
    # Process each frame
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        annotated_frame, detections = process_image(frame, conf_thresh, classes)
        
        # Update counts
        for detection in detections:
            vehicle_counts[detection['class']] += 1
        
        frame_data.append({
            'frame': frame_num,
            'detections': len(detections),
            'timestamp': frame_num / fps
        })
        
        # Write to output video
        out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        
        # Display progress
        progress = (frame_num + 1) / frame_count
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_num + 1}/{frame_count} - Detected {len(detections)} vehicles")
        
        # Show sample frame (every 10 frames to reduce UI updates)
        if frame_num % 10 == 0:
            video_placeholder.image(annotated_frame, channels="RGB", caption="Sample Processed Frame")
        
        frame_num += 1
    
    # Clean up
    cap.release()
    out.release()
    
    return vehicle_counts, frame_data, output_path

# Main application flow
upload_type = st.radio("Select Input Type", ["Image", "Video"])

if upload_type == "Image":
    uploaded_file = st.file_uploader("Upload Traffic Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Store in session state
        st.session_state.media_path = uploaded_file.name
        st.session_state.media_type = "image"
        
        # Display original image
        st.subheader("Original Image")
        st.image(image, channels="BGR", use_column_width=True)
        
        # Process and display results
        st.subheader("Detection Results")
        with st.spinner("Processing image..."):
            result_image, detections = process_image(
                image, 
                confidence_threshold, 
                selected_classes
            )
            
            # Store results
            st.session_state.detection_results = detections
            st.session_state.vehicle_counts = defaultdict(int)
            for detection in detections:
                st.session_state.vehicle_counts[detection['class']] += 1
            
            # Display processed image
            st.image(result_image, caption="Processed Image", use_column_width=True)
            
            # Show detection data
            st.subheader("Detection Data")
            if detections:
                df = pd.DataFrame(detections)
                st.dataframe(df)
                
                # Summary statistics
                st.subheader("Vehicle Counts")
                count_df = pd.DataFrame.from_dict(
                    st.session_state.vehicle_counts, 
                    orient='index', 
                    columns=['Count']
                ).reset_index()
                count_df.columns = ['Vehicle Type', 'Count']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(count_df)
                with col2:
                    st.bar_chart(count_df.set_index('Vehicle Type'))
            else:
                st.warning("No vehicles detected with current settings")

else:  # Video processing
    uploaded_file = st.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Create temporary file
        suffix = uploaded_file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as temp_file:
            temp_file.write(uploaded_file.read())
            st.session_state.media_path = temp_file.name
            st.session_state.media_type = "video"
        
        # Display original video
        st.subheader("Original Video")
        st.video(st.session_state.media_path)
        
        # Process video
        st.subheader("Processing Results")
        if st.button("Start Processing"):
            with st.spinner("Processing video..."):
                counts, frame_data, output_path = process_video(
                    st.session_state.media_path,
                    confidence_threshold,
                    selected_classes
                )
                
                # Store results
                st.session_state.vehicle_counts = counts
                st.session_state.frame_data = frame_data
                st.session_state.output_video_path = output_path
                
                # Show summary
                st.success("Video processing complete!")
                st.subheader("Traffic Summary")
                
                # Vehicle counts
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Total Vehicle Counts**")
                    count_df = pd.DataFrame.from_dict(
                        counts, 
                        orient='index', 
                        columns=['Count']
                    )
                    st.dataframe(count_df)
                
                # Traffic flow chart
                with col2:
                    st.markdown("**Traffic Flow Over Time**")
                    flow_df = pd.DataFrame(frame_data)
                    st.line_chart(flow_df.set_index('timestamp')['detections'])
                
                # Display processed video
                st.subheader("Processed Video with Detections")
                st.video(output_path)
                
                # Export results
                st.download_button(
                    label="Download Detection Data",
                    data=pd.DataFrame(frame_data).to_csv(index=False).encode('utf-8'),
                    file_name='traffic_analysis.csv',
                    mime='text/csv'
                )

# Add footer
st.markdown("---")
st.markdown("""
### Traffic Analysis Features:
- **Vehicle Detection**: Identifies cars, trucks, buses, motorcycles, and bicycles
- **Customizable Detection**: Adjust confidence threshold and select vehicle classes
- **Traffic Statistics**: Provides counts and temporal patterns of vehicle detections
""")