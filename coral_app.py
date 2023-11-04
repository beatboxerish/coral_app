import streamlit as st
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import cv2
import numpy as np


st.set_page_config(layout="wide", page_title="Coral Segmentation App")

st.write("## Segment your Corals and Ref box")
st.write(
    ":dog: Upload an image with corals and center reference white tile. Final image can be downloaded from the sidebar."
)
st.sidebar.write("## Upload and download :gear:")

MAX_FILE_SIZE = 7 * 1024 * 1024  # 7MB


# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def get_segmentations(upload):
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image)

    col2.write("Segmentations :wrench:")
    image = model_pipeline(image, model)
    col2.image(image)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image", convert_image(image), "segmentations.png", "image/png")


def model_pipeline(image, model):
    cv_image = np.array(image)
    resized_image = resize_image(cv_image, (640, 490))
    segmented_image = inference(resized_image, cv_image, model)
    segmented_image = Image.fromarray(segmented_image)
    return segmented_image


def load_model():
    model = YOLO('yolo_model.pt')
    return model


def resize_image(img, size):
    resized_image = cv2.resize(img, size)
    return resized_image


def inference(resized_image, original_image, model):
    result = model.predict(resized_image)
    segmented_image = draw_bboxes_xyxyn(result[0].boxes.xyxyn, original_image)
    return segmented_image


def draw_bboxes_xyxyn(bboxes, img):
    colors = [(150, 150, 150)]
    drawn_img = img.copy()
    for i, box in enumerate(bboxes):
        x, y, x1, y1 = box
        x, x1 = x*img.shape[1], x1*img.shape[1]
        y, y1 = y*img.shape[0], y1*img.shape[0]
        cv2.rectangle(drawn_img, (int(x), int(y)), (int(x1), int(y1)), colors[0], 10)
    return drawn_img

with st.spinner('Please wait while we load the model...'):
    model = load_model()


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 7MB.")
    else:
        get_segmentations(upload=my_upload)
else:
    st.error("Please upload a valid image.")