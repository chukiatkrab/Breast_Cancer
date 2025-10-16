import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


# --- Page config ---
st.set_page_config(
    page_title="Decision Tree – Breast Cancer (Feature Ablation)",
    page_icon="🌳",
    layout="centered",
)

# --- Constants ---
CSV_FILE = Path("breast_cancer_bd.csv")
TARGET = "Class"
ID_COLS = ["id", "Sample code number"]
ALL_FEATURES = [
    "Clump Thickness", 
    "Uniformity of Cell Size", 
    "Uniformity of Cell Shape",
    "Marginal Adhesion", 
    "Single Epithelial Cell Size", 
    "Bare Nuclei",
    "Bland Chromatin", 
    "Normal Nucleoli", 
    "Mitoses",
]

# =========================
# Helper functions
# =========================
@st.cache_data(show_spinner="กำลังโหลดข้อมูล CSV...")
def load_and_prep_data(csv_path: Path, features: list[str]) -> tuple[pd.DataFrame, pd.Series] | tuple[None, None]:
    """โหลด CSV, ทำความสะอาด, คืน X (เฉพาะฟีเจอร์ที่เลือก) และ y (0/1)."""
    if not csv_path.exists():
        return None, None
    try:
        df = pd.read_csv(csv_path, na_values=["?"])
        # ตัดคอลัมน์ไอดีถ้ามี
        df = df.drop(columns=[col for col in ID_COLS if col in df.columns], errors="ignore")

        if TARGET not in df.columns:
            st.error(f"ไม่พบคอลัมน์เป้าหมาย '{TARGET}' ในไฟล์ CSV")
            return None, None

        X = df.drop(columns=[TARGET]).copy()

        # ทำ case-insensitive ให้ชื่อคอลัมน์ตรง ALL_FEATURES
        rename_map = {}
        X.columns = [col.strip() for col in X.columns]
        for feat in ALL_FEATURES:
            for col in X.columns:
                if feat.lower() == col.lower():
                    rename_map[col] = feat
                    break
        X = X.rename(columns=rename_map)

        # เติมคอลัมน์ที่ขาดให้ครบก่อน แล้วเลือกเฉพาะ features ที่ผู้ใช้ต้องการ
        for feat in ALL_FEATURES:
            if feat not in X.columns:
                X[feat] = np.nan

        X = X[features].apply(pd.to_numeric, errors="coerce")

        # map target: 2 -> 0 (benign), 4 -> 1 (malignant)
        y = df[TARGET].map({2: 0, 4: 1})

        return X, y
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ CSV: {e}")
        return None, None


def get_model_pipeline(features: list[str]) -> Pipeline:
    """สร้าง Pipeline (Imputer median + DecisionTree) สำหรับชุดฟีเจอร์ที่เลือก."""
    preprocessor = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), features)],
        remainder="drop",
    )
    classifier = DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced",
        criterion="gini",
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", classifier),
    ])


@st.cache_resource(show_spinner="กำลังเตรียมโมเดล...")
def load_model(uploaded_file: bytes | None, features: tuple[str, ...]) -> tuple[Pipeline | None, str]:
    X, y = load_and_prep_data(CSV_FILE, list(features))
    if X is not None and y is not None:
        try:
            model = get_model_pipeline(list(features))
            model.fit(X, y)
            return model, f"trained_from_{CSV_FILE.name}"
        except Exception as e:
            st.error(f"เทรนโมเดลเริ่มต้นจาก CSV ไม่สำเร็จ: {e}")
            return None, "no_model"

    return None, "no_model"


def predict_one(model: Pipeline, values: dict, features: list[str], threshold: float) -> tuple[int, float]:
    """พยากรณ์รายเดียว (robust ต่อคอลัมน์หาย/ชนิดข้อมูล)"""
    row = {col: values.get(col, np.nan) for col in features}
    df_one = pd.DataFrame([row], columns=features).apply(pd.to_numeric, errors="coerce")

    try:
        proba = float(model.predict_proba(df_one)[0][1])
    except AttributeError:
        try:
            score = float(model.decision_function(df_one)[0])
            proba = 1.0 / (1.0 + np.exp(-score))
        except AttributeError:
            pred_label = int(model.predict(df_one)[0])
            return pred_label, float(pred_label)

    pred_label = 1 if proba >= threshold else 0
    return pred_label, proba


# =========================
# App
# =========================
def main():
    st.title("🌳 Decision Tree – พยากรณ์มะเร็งเต้านม")

    # Sidebar removed — provide sensible defaults so rest of app is unaffected
    selected_features = ALL_FEATURES.copy()
    uploaded_joblib = None
    threshold = 0.5

    # ----- Load/Train model ตามชุดฟีเจอร์ -----
    model, src = load_model(uploaded_joblib.read() if uploaded_joblib else None, tuple(selected_features))
    if model is None:
        st.error("ไม่สามารถโหลดหรือสร้างโมเดลได้ โปรดตรวจสอบไฟล์ CSV หรือโมเดลที่อัปโหลด")
        st.stop()

    status_message = "จากไฟล์ที่อัปโหลด" if "upload" in src else f"จาก {CSV_FILE.name}"
    st.success(f"โหลดโมเดลสำเร็จ ({status_message}) — ใช้ฟีเจอร์ {len(selected_features)} ตัว")

    # ----- กรอกค่าฟีเจอร์ (อินพุตรายเดียว) -----
    st.subheader("กรอกค่าฟีเจอร์ (1–10)")
    input_values = {}
    cols = st.columns(2)
    for i, feat in enumerate(selected_features):
        with cols[i % 2]:
            # ค่าตั้งต้นแบบ benign-ish
            if feat == "Clump Thickness":
                default_value = 5
            elif feat == "Single Epithelial Cell Size":
                default_value = 2
            elif feat == "Bland Chromatin":
                default_value = 3
            else:
                default_value = 1
            input_values[feat] = st.number_input(feat, min_value=1, max_value=10, value=default_value, step=1)

    # ----- พยากรณ์รายเดียว -----
    if st.button("🔮 พยากรณ์", use_container_width=True):
        pred_label, proba = predict_one(model, input_values, selected_features, threshold)
        result_text = "เป็นมะเร็ง" if pred_label == 1 else "ไม่เป็นมะเร็ง"
        (st.error if pred_label == 1 else st.success)(f"**ผลลัพธ์: {result_text}**")
        st.progress(proba)

    # ----- ดูข้อมูล + คำนวณ Accuracy -----
    with st.expander("📄 ดูข้อมูลและประเมินความแม่นยำของโมเดล"):
        X, y = load_and_prep_data(CSV_FILE, selected_features)
        if X is None:
            st.warning("ไม่พบไฟล์ CSV สำหรับแสดงข้อมูลหรือคำนวณ Accuracy")
        else:
            st.write(f"แหล่งข้อมูล: **{CSV_FILE.name}**")
            st.dataframe(pd.concat([X, y.rename(TARGET)], axis=1), use_container_width=True)

            colA, colB = st.columns(2)
            with colA:
                if st.button("คำนวณ Accuracy (Hold-out 20%) — ชุดฟีเจอร์ที่เลือก"):
                    with st.spinner("กำลังเทรน/ทดสอบ..."):
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42, stratify=y
                        )
                        eval_model = get_model_pipeline(selected_features)
                        eval_model.fit(X_train, y_train)
                        acc = accuracy_score(y_test, eval_model.predict(X_test))
                        st.info(f"**Accuracy (selected features): {acc * 100:.2f}%**")




if __name__ == "__main__":
    main()

