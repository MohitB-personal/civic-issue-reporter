import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import os
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Civic Issue Reporter", layout="centered")

st.title("🛣️ Civic Issue Reporter")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("model.pt")

model = load_model()

# ---------------- STORAGE SETUP ----------------
DB_FILE = "issues.csv"
UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

if not os.path.exists(DB_FILE):
    df = pd.DataFrame(columns=[
        "timestamp",
        "image_path",
        "prediction",
        "confidence",
        "authority",
        "status"
    ])
    df.to_csv(DB_FILE, index=False)

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["📸 Report Issue", "🧑‍💼 Admin Dashboard"])

# =========================================================
# ===================== USER TAB ==========================
# =========================================================
with tab1:
    st.subheader("Report a Civic Issue")

    st.write("Upload an image **or** take a photo directly")

    uploaded_file = st.file_uploader(
        "Upload road image",
        type=["jpg", "jpeg", "png"]
    )

    camera_photo = st.camera_input("Take a photo")

    image = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

    elif camera_photo is not None:
        image = Image.open(camera_photo)

    if image is not None:
        st.image(image, caption="Captured Image", width=500)

        if st.button("Analyze Issue"):
            with st.spinner("Analyzing image..."):
                # Save image
                timestamp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = os.path.join(
                    UPLOAD_DIR, f"{timestamp_id}.jpg"
                )
                image.save(image_path)

                # Run model
                results = model(image_path)
                pred_class = results[0].names[results[0].probs.top1]
                confidence = float(results[0].probs.top1conf)

                if pred_class == "potholes":
                    authority = "Municipal Road Department"
                else:
                    authority = "No action required"

            # Save to CSV
            df = pd.read_csv(DB_FILE)

            if "status" not in df.columns:
                df["status"] = "Reported"

            df.loc[len(df)] = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                image_path,
                pred_class,
                round(confidence, 3),
                authority,
                "Reported"
            ]

            df.to_csv(DB_FILE, index=False)

            st.success("✅ Issue reported successfully!")
            st.write(f"**Prediction:** {pred_class}")
            st.write(f"**Confidence:** {round(confidence, 3)}")
            st.write(f"**Authority:** {authority}")
            st.write("**Status:** Reported")

# =========================================================
# =================== ADMIN TAB ===========================
# =========================================================
with tab2:
    st.subheader("🧑‍💼 Admin Dashboard")

    df = pd.read_csv(DB_FILE)

    if df.empty:
        st.info("No issues reported yet.")
    else:
        status_options = ["Reported", "In Progress", "Resolved"]

        for i in range(len(df)):
            st.markdown("---")

            col1, col2, col3 = st.columns([2, 3, 2])

            with col1:
                if os.path.exists(df.loc[i, "image_path"]):
                    st.image(df.loc[i, "image_path"], width=200)
                else:
                    st.warning("Image not found")

            with col2:
                st.write(f"🕒 **Time:** {df.loc[i, 'timestamp']}")
                st.write(f"🧠 **Prediction:** {df.loc[i, 'prediction']}")
                st.write(f"📊 **Confidence:** {df.loc[i, 'confidence']}")
                st.write(f"🏛️ **Authority:** {df.loc[i, 'authority']}")

            with col3:
                new_status = st.selectbox(
                    "Status",
                    status_options,
                    index=status_options.index(df.loc[i, "status"]),
                    key=f"status_{i}"
                )
                df.loc[i, "status"] = new_status

        if st.button("Save Status Updates"):
            df.to_csv(DB_FILE, index=False)
            st.success("Status updates saved successfully!")
