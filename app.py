import streamlit as st
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from utils import clean_image, get_prediction, make_results
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import cv2
import uuid

# Custom toast with unique dismissible button using session state
def custom_toast(message):
    toast_key = f"toast_{uuid.uuid4()}"

    if toast_key not in st.session_state:
        st.session_state[toast_key] = True

    if st.session_state[toast_key]:
        close_btn_col, toast_col, _ = st.columns([1, 10, 1])
        with close_btn_col:
            if st.button("🔁", key=f"close_{toast_key}", help="Generate another result", type="secondary"):
                st.session_state[toast_key] = False

        with toast_col:
            st.markdown(f"""
                <div style="
                    margin-top: 20px;
                    background-color: rgba(34, 139, 34, 0.95);
                    padding: 20px 30px;
                    color: white;
                    font-size: 24px;
                    border-radius: 12px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                    text-align: center;
                ">
                    🌿 {message}
                </div>
            """, unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model(path):
    xception_model = tf.keras.models.Sequential([
        tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    densenet_model = tf.keras.models.Sequential([
        tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    inputs = tf.keras.Input(shape=(512, 512, 3))
    xception_output = xception_model(inputs)
    densenet_output = densenet_model(inputs)
    outputs = tf.keras.layers.average([densenet_output, xception_output])
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.load_weights(path)
    return model

model = load_model("eggplant_model.h5")

# Hide Streamlit branding
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<h1>Eggplant Disease Detection</h1>", unsafe_allow_html=True)
st.write("Upload an image or scan live using your camera to check eggplant health.")

# Select input method
option = st.radio("Select Input Method:", ('Upload Image', 'Live Camera Scan'))

# Upload Image Mode
if option == 'Upload Image':
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(image, caption="Uploaded Image", use_container_width=True)
        with st.spinner("Predicting..."):
            image = clean_image(image)
            predictions, predictions_arr = get_prediction(model, image)
            result = make_results(predictions, predictions_arr)
            custom_toast(f"The eggplant {result['status']} with {result['prediction']} prediction.")

            # Show diagnosis button at bottom center
            if result.get("link"):
                st.markdown(f"""
                    <style>
                        .diagnosis-btn {{
                            position: fixed;
                            bottom: 30px;
                            left: 50%;
                            transform: translateX(-50%);
                            z-index: 9999;
                        }}
                        .diagnosis-btn button {{
                            padding: 12px 30px;
                            font-size: 20px;
                            background-color: #6c63ff;
                            color: white;
                            border: none;
                            border-radius: 10px;
                            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                            cursor: pointer;
                        }}
                    </style>
                    <div class="diagnosis-btn">
                        <a href="{result['link']}" target="_blank">
                            <button>🔍 View Diagnosis</button>
                        </a>
                    </div>
                """, unsafe_allow_html=True)

# Live Camera Scan Mode
elif option == 'Live Camera Scan':
    st.write("Enable webcam below and click 'Scan Eggplant'")

    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.latest_frame = None

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            self.latest_frame = img
            return img

    ctx = webrtc_streamer(
        key="high-res-cam",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={
            "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}
        },
        async_processing=True,
    )

    if ctx.video_transformer:
        if st.button("Scan Eggplant 🔍", help="Click this button to scan the eggplant"):
            frame = ctx.video_transformer.latest_frame
            if frame is not None:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img_rgb)
                st.image(image, caption="Captured Frame", use_container_width=True)
                with st.spinner("Predicting..."):
                    image = clean_image(image)
                    predictions, predictions_arr = get_prediction(model, image)
                    result = make_results(predictions, predictions_arr)
                    custom_toast(f"The eggplant {result['status']} with {result['prediction']} prediction.")

                    if result.get("link"):
                        st.markdown(f"""
                            <style>
                                .diagnosis-btn {{
                                    position: fixed;
                                    bottom: 30px;
                                    left: 50%;
                                    transform: translateX(-50%);
                                    z-index: 9999;
                                }}
                                .diagnosis-btn button {{
                                    padding: 12px 30px;
                                    font-size: 20px;
                                    background-color: #6c63ff;
                                    color: white;
                                    border: none;
                                    border-radius: 10px;
                                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                                    cursor: pointer;
                                }}
                            </style>
                            <div class="diagnosis-btn">
                                <a href="{result['link']}" target="_blank">
                                    <button>🔍 View Diagnosis</button>
                                </a>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Please wait for the camera to initialize and capture a frame.")
