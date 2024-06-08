import streamlit as st
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

def ingest_data(file_path, device_type):
    if device_type == 'Tobii':
        data = pd.read_csv(file_path)
    elif device_type == 'WebGazer':
        data = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported device type")
    return data

def batch_process(data_dir, device_type):
    all_data = []
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        data = ingest_data(file_path, device_type)
        all_data.append(data)
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data

def overlay_bounding_boxes(image_path, gaze_data, bounding_boxes):
    image = cv2.imread(image_path)
    for bbox in bounding_boxes:
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for _, row in gaze_data.iterrows():
        cv2.circle(image, (int(row['x']), int(row['y'])), 5, (0, 255, 0), -1)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def interactive_ui(image_path, gaze_data):
    st.title("Eye-Tracking Data Visualization")
    
    bounding_boxes = []

    def draw_bounding_box():
        nonlocal bounding_boxes
        x = st.slider('X Coordinate', 0, image_width)
        y = st.slider('Y Coordinate', 0, image_height)
        w = st.slider('Width', 0, image_width - x)
        h = st.slider('Height', 0, image_height - y)
        bounding_boxes.append((x, y, w, h))
        overlay_bounding_boxes(image_path, gaze_data, bounding_boxes)

    if st.button('Add Bounding Box'):
        draw_bounding_box()

    if st.button('Export Data'):
        export_data(gaze_data, bounding_boxes)

def export_data(gaze_data, bounding_boxes):
    gaze_data.to_csv('gaze_data.csv', index=False)
    with open('bounding_boxes.txt', 'w') as f:
        for bbox in bounding_boxes:
            f.write(f'{bbox}\n')
    st.success('Data Exported Successfully!')

def export_visualization(image_path, bounding_boxes, output_path='output.png'):
    image = cv2.imread(image_path)
    for bbox in bounding_boxes:
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imwrite(output_path, image)
    st.success(f'Visualization exported to {output_path}')

def export_processed_data(gaze_data, bounding_boxes):
    gaze_data.to_csv('processed_gaze_data.csv', index=False)
    with open('bounding_boxes.csv', 'w') as f:
        f.write('x,y,w,h\n')
        for bbox in bounding_boxes:
            f.write(f'{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n')
    st.success('Processed data exported successfully')

# Streamlit app
st.title("Eye-Tracking Data Analysis Tool")

uploaded_files = st.file_uploader("Upload your eye-tracking data files", accept_multiple_files=True)
device_type = st.selectbox("Select device type", ['Tobii', 'WebGazer'])
image_path = st.text_input("Enter path to the stimulus image")

if st.button('Process Data'):
    all_data = []
    for uploaded_file in uploaded_files:
        data = ingest_data(uploaded_file, device_type)
        all_data.append(data)
    combined_data = pd.concat(all_data, ignore_index=True)
    st.dataframe(combined_data)
    
    interactive_ui(image_path, combined_data)