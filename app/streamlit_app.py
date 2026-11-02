import sys
import torch
import hashlib
import streamlit as st

from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mobilenet import create_mobilenet
from src.evaluation.gradcam import draw_gradcam_contours, generate_gradcam_map
from src.data.preprocessing import create_image_transform

st.set_page_config(
    page_title="AutoInspect AI",
    page_icon="^",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f5f8fc;
        color: #23344d;
    }

    [data-testid="stHeader"] {
        background: rgba(37, 68, 105, 0.045);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #203653;
        letter-spacing: -0.025em;
    }

    [data-testid="stMetric"],
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px solid #e1eaf4;
        border-radius: 16px;
        padding: 1rem 1.15rem;
        box-shadow: 0 5px 18px rgba(37, 68, 105, 0.045);
    }

    [data-testid="stFileUploader"] {
        background: #fafdff;
        border-style: dashed;
    }

    [data-testid="stImage"] img {
        border-radius: 14px;
    }

    [data-testid="stProgressBar"] > div > div {
        background: #5b91a8;
    }

    div.stButton > button {
        min-height: 2.8rem;
        border-radius: 12px;
        font-weight: 600;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = (PROJECT_ROOT / "models" / "checkpoints" / "best_mobilenet.pth")
CLASS_NAMES = ["Low", "Medium", "High"]

CLASS_DESCRIPTIONS = {
    "Low": (
        "Minor visible damage, such as small scratches, "
        "paint chips or shallow dents."
    ),
    "Medium": (
        "Moderate visible damage, such as larger dents, "
        "damaged bumpers or cracked vehicle components."
    ),
    "High": (
        "Extensive visible damage, such as major body "
        "deformation or substantial panel damage."
    ),
}

transform = create_image_transform()

@st.cache_resource
def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")

    model = create_mobilenet(
        num_classes=len(CLASS_NAMES),
        pretrained=False
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model

def predict(model, image):
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(probabilities.argmax().item())

    return (
        predicted_index,
        probabilities.cpu().numpy(),
        image_tensor,
    )

st.markdown(
    """
    <div style="
        padding: 1.7rem 2rem;
        margin-bottom: 1.25rem;
        border: 1px solid #dce8f3;
        border-radius: 20px;
        background: rgba(37, 68, 105, 0.045);
    ">
        <div style="
            margin-bottom: .45rem;
            color: #547a91;
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .12em;
            text-transform: uppercase;
        ">Speedy claims with Automatic Assessment</div>
        <h1 style="margin: 0; color: #203653;">AutoInspect AI</h1>
        <p style="margin: .45rem 0 0; color: #53677d; font-size: 1.08rem;">
            A vehicle damage assessment, with a little more context.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "Upload a vehicle photo to estimate visible damage severity. "
    "This is a helpful starting point—not an assessment of hidden structural "
    "damage, roadworthiness, or repair costs."
)

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a vehicle damage image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.getvalue()
        image_key = hashlib.sha256(image_bytes).hexdigest()
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("The uploaded file could not be read as an image.")
        st.stop()

    if st.session_state.get("gradcam_image_key") != image_key:
        st.session_state.pop("gradcam_image_key", None)
        st.session_state.pop("gradcam_visualization", None)

    left, right = st.columns(2)
    with left:
        st.subheader("Uploaded Image")
        st.image(
            image,
            caption="Vehicle image",
            width="stretch",
        )
        
    with right:
        st.subheader("Damage Assessment")

        with st.spinner("Analyzing vehicle image..."):
            predicted_index, probabilities, image_tensor = predict(
                model,
                image,
            )

        predicted_class = CLASS_NAMES[predicted_index]

        st.metric(
            "Predicted Damage Severity",
            predicted_class,
        )

        st.write(CLASS_DESCRIPTIONS[predicted_class])
        st.write("Class probabilities")

        for index, class_name in enumerate(CLASS_NAMES):
            probability = float(probabilities[index])
            st.write(f"**{class_name}:** {probability:.2%}")
            st.progress(probability)

    st.divider()
    st.header("A closer look")
    st.markdown(
        "Grad-CAM highlights image regions that influenced the model's "
        "classification. It is an explanation aid, not proof that the "
        "model's prediction is correct. The contour outlines the strongest "
        "areas of influence; it is not a damage boundary."
    )

    if st.button(
        "Generate Grad-CAM visualization", key="generate_gradcam", type="primary"):
        with st.spinner("Generating Grad-CAM visualization..."):
            try:
                grayscale_cam = generate_gradcam_map(
                    model=model,
                    image_tensor=image_tensor,
                    target_index=predicted_index
                )

                st.session_state["gradcam_visualization"] = draw_gradcam_contours(
                    image,
                    grayscale_cam
                )
                st.session_state["gradcam_image_key"] = image_key

            except ImportError as error:
                st.error(
                    "Grad-CAM dependencies are missing. Install them with: "
                    "pip install grad-cam"
                )
                st.exception(error)

            except Exception as error:
                st.error("Grad-CAM generation failed.")
                st.exception(error)

    if st.session_state.get("gradcam_image_key") == image_key:
        st.success("Your visualization is ready.")
        original_col, contour_col = st.columns(2)

        with original_col:
            st.image(
                image.resize((224, 224)),
                caption="Original image",
                width="stretch",
            )

        with contour_col:
            st.image(
                st.session_state["gradcam_visualization"],
                caption=f"Influence contours — {predicted_class}",
                width="stretch",
            )

else:
    st.session_state.pop("gradcam_image_key", None)
    st.session_state.pop("gradcam_visualization", None)
    st.caption("Upload an image to see its damage assessment and Grad-CAM option.")

st.divider()
st.caption("AutoInspect AI | Deep Learning Portfolio Project")
