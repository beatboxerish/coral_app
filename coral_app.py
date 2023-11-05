import streamlit as st
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import os
import urllib
from segment_anything import sam_model_registry, SamPredictor
import supervision as sv



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
    sam_model, yolo_model = load_models()
    original_image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(original_image)

    sam_image, yolo_image = model_inference(original_image, yolo_model, sam_model)

    col2.write("Yolo Boxes :white_square_button:")
    col2.image(yolo_image)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download Image with YOLO Boxes", convert_image(yolo_image), "yolo.png", "image/png")

    col3.write("Segmented Image :robot_face:")
    col3.image(sam_image)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download Segmented image", convert_image(sam_image), "segmentations.png", "image/png")


def model_inference(image, yolo_model, sam_model):
    cv_image = np.array(image)
    resized_image = resize_image(cv_image, (640, 490))
    yolo_image, bb_result = yolo_inference(resized_image, cv_image, yolo_model)
    sam_image = sam_inference(sam_model, bb_result, resized_image, cv_image)
    sam_image, yolo_image = Image.fromarray(sam_image), Image.fromarray(yolo_image)
    return sam_image, yolo_image


def load_models():
    # load yolo_model
    if "yolo_model" not in st.session_state.keys():
        st.session_state["yolo_model"] = YOLO('yolo_v6_best.pt')
    yolo_model = st.session_state["yolo_model"]

    # load sam_model
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with st.spinner('Downloading SAM model...'):
        if not os.path.exists("sam_weights"):
            os.makedirs('sam_weights')
            urllib.request.urlretrieve(url, "sam_weights/sam_vit_b_01ec64.pth")
    sam_model = get_sam_predictor(url, device)
    return sam_model, yolo_model


@st.cache_resource()
def get_sam_predictor(url, device):
    sam = sam_model_registry["vit_b"](checkpoint="sam_weights/sam_vit_b_01ec64.pth")
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor



def resize_image(img, size):
    resized_image = cv2.resize(img, size)
    return resized_image


def yolo_inference(resized_image, original_image, model):
    result = model.predict(resized_image)
    yolo_image = draw_bboxes_xyxyn(result[0].boxes.xyxyn, original_image)
    return yolo_image, result[0]


def sam_inference(sam_model, bb_result, resized_image, original_image):
    original_image_size = original_image.shape
    masks, class_ids = get_sam_masks(bb_result, sam_model, resized_image)

    # resizing segmentation map
    big_masks = [torch.nn.functional.interpolate(i.to(torch.float32).unsqueeze(0),
                                             size=(original_image_size[0], original_image_size[1])).to(bool)
                                             for i in masks]
    big_masks = torch.stack(big_masks).squeeze(1)
    detections = create_detections(big_masks, class_ids)
    segmented_image = draw_masks_image(original_image, detections)
    return segmented_image


def get_sam_masks(yolo_result, sam, resized_image):
    # multiple bounding boxes as input for a single image
    input_boxes = yolo_result.boxes.xyxy
    class_ids = yolo_result.boxes.cls.cpu().numpy()

    transformed_boxes = sam.transform.apply_boxes_torch(input_boxes, resized_image.shape[:2])
    sam.set_image(resized_image)
    masks, iou_predictions, low_res_masks = sam.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False
    )
    return masks, class_ids


def draw_bboxes_xyxyn(bboxes, img):
    colors = [(150, 150, 150)]
    drawn_img = img.copy()
    for i, box in enumerate(bboxes):
        x, y, x1, y1 = box
        x, x1 = x * img.shape[1], x1 * img.shape[1]
        y, y1 = y * img.shape[0], y1 * img.shape[0]
        cv2.rectangle(drawn_img, (int(x), int(y)), (int(x1), int(y1)), colors[0], 10)
    return drawn_img


def create_detections(masks, class_ids):
    # creating Detections object for all the masks
    xyxys = np.array([sv.mask_to_xyxy(masks=i.cpu()) for i in masks])
    xyxys = xyxys.squeeze(1)
    numpy_masks = masks.cpu().numpy().squeeze(1)
    detections = sv.Detections(
          class_id = class_ids,
          xyxy=xyxys,
          mask=numpy_masks
    )
    return detections


def draw_masks_image(image_bgr, detections):
    # bounding boxes and segmented areas
    box_annotator = sv.BoxAnnotator(color=sv.Color.red(), thickness=10)
    mask_annotator = sv.MaskAnnotator(color=sv.Color.red())
    source_image = image_bgr.copy()
    segmented_image = image_bgr.copy()

    source_image = box_annotator.annotate(scene=source_image,
                                          detections=detections,
                                          skip_label=False)
    segmented_image = mask_annotator.annotate(scene=segmented_image,
                                              detections=detections)

    return segmented_image


col1, col2, col3 = st.columns(3)
my_upload = st.sidebar.file_uploader(
    "Upload an image", type=["png", "jpg", "jpeg"]
)

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 7MB.")
    else:
        get_segmentations(upload=my_upload)
else:
    pass