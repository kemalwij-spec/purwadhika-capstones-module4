from io import BytesIO

import streamlit as st
from PIL import Image
from ultralytics import YOLO

from ppe_analysis import (
    Detection,
    analyze_detections,
    summarize,
    summary_text,
    draw_annotated,
    OVERLAP_THRESHOLD,
)

MODEL_PATH = "best.pt"  # weights hasil training di folder yang sama saat deploy

st.set_page_config(page_title="Pengecekan APD Pekerja Konstruksi", page_icon="🦺", layout="centered")

@st.cache_resource
def load_model(path):
    return YOLO(path)


def run_pipeline(model, image: Image.Image, conf: float, overlap_threshold: float):
    results = model.predict(image, conf=conf, verbose=False)[0]
    names = results.names
    detections = [
        Detection(names[int(cls)], float(c), tuple(map(float, box)))
        for box, cls, c in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf)
    ]
    statuses = analyze_detections(detections, overlap_threshold)
    summary = summarize(statuses)
    return statuses, summary


def main():
    st.title("🦺 Construction Safety Checker")
    st.write(
        "Upload foto lokasi konstruksi untuk mendeteksi pekerja dan mengecek "
        "apakah masing-masing sudah memakai helm dan rompi keselamatan."
    )

    with st.sidebar:
        st.header("Pengaturan")
        conf = st.slider("Confidence threshold deteksi", 0.05, 0.95, 0.25, 0.05)
        overlap_threshold = st.slider(
            "Threshold pencocokan gear-ke-orang", 0.1, 0.9, OVERLAP_THRESHOLD, 0.05,
            help="Seberapa besar box helm/rompi harus overlap dengan box orang supaya dianggap miliknya.",
        )
        st.caption(
            "Kasus yang tidak yakin ditandai 'perlu dicek'"
        )

    model = load_model(MODEL_PATH)

    uploaded = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload gambar untuk mulai.")
        return

    image = Image.open(uploaded).convert("RGB")

    with st.spinner("Menjalankan deteksi..."):
        statuses, summary = run_pipeline(model, image, conf, overlap_threshold)
        annotated = draw_annotated(image, statuses)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gambar Asli")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Hasil Anotasi")
        st.image(annotated, use_container_width=True)
        st.caption("🟩 lengkap   🟥 APD tidak lengkap   🟧 belum jelas / perlu dicek")

    st.subheader("Ringkasan")
    st.write(summary_text(summary))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pekerja terdeteksi", summary["total_workers"])
    m2.metric("APD Lengkap", summary["fully_equipped"])
    m3.metric(" APD Tidak Lengkap", summary["confirmed_missing_gear"])
    m4.metric("Perlu dicek", summary["needs_review"])

    if summary["confirmed_missing_gear"] > 0:
        st.error(f"⚠️ {summary['confirmed_missing_gear']} pekerja terkonfirmasi APD tidak lengkap.")
    if summary["needs_review"] > 0:
        st.warning(
            f"🔍 {summary['needs_review']} pekerja tidak bisa diklasifikasikan/belum jelas  "
            f"dan perlu dicek manual."
        )

    buf = BytesIO()
    annotated.save(buf, format="PNG")
    st.download_button(
        "Download gambar hasil anotasi",
        data=buf.getvalue(),
        file_name="annotated_result.png",
        mime="image/png",
    )

    with st.expander("Detail per pekerja"):
        for i, s in enumerate(statuses):
            head_conf = f" ({s.head_conf:.2f})" if s.head_conf else ""
            body_conf = f" ({s.body_conf:.2f})" if s.body_conf else ""
            st.write(f"**Pekerja {i + 1}** — kepala: `{s.head_status}`{head_conf}, badan: `{s.body_status}`{body_conf}")


if __name__ == "__main__":
    main()
