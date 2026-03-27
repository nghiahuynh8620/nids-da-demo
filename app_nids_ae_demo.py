from __future__ import annotations

import inspect
import io
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import h5py
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="NIDS-DA AE Predictor", layout="wide")

ATTACKS = ["FGSM", "BIM", "PGD", "JSMA", "DeepFool"]
DATASETS = ["NSL-KDD", "UNSW-NB15", "CICIDS2017"]


# ---------- utils ----------
def label_text(class_id: int) -> str:
    return "Normal" if int(class_id) == 0 else "Attack/Adversarial"


def normalize_name(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())


def safe_load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


# ---------- model builders for weight fallback ----------
def build_classifier_fallback(input_dim: int) -> tf.keras.Model:
    tf.keras.backend.clear_session()
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(25, activation="relu"),
            tf.keras.layers.Dense(13, activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )


def build_dae_fallback(dataset_name: str, input_dim: int) -> tf.keras.Model:
    tf.keras.backend.clear_session()
    ds = dataset_name.upper()
    if "NSL" in ds and input_dim == 24:
        hidden = [20, 15, 8, 15, 20]
    elif "UNSW" in ds and input_dim == 29:
        hidden = [15, 8, 15]
    elif "CIC" in ds and input_dim == 10:
        hidden = [9, 8, 5, 8, 9]
    else:
        bottleneck = max(4, min(8, max(2, input_dim // 2)))
        h1 = max(bottleneck + 2, int(round(input_dim * 0.75)))
        h2 = max(bottleneck + 1, int(round(input_dim * 0.5)))
        hidden = [h1, h2, bottleneck, h2, h1]

    layers: list[tf.keras.layers.Layer] = [tf.keras.layers.Input(shape=(input_dim,))]
    for units in hidden:
        layers.append(tf.keras.layers.Dense(units, activation="relu"))
    layers.append(tf.keras.layers.Dense(input_dim, activation="sigmoid"))
    return tf.keras.Sequential(layers)


# ---------- keras loading ----------
def _load_model_direct(model_path: str):
    sig = inspect.signature(tf.keras.models.load_model)
    kwargs = {"compile": False}
    if "safe_mode" in sig.parameters:
        kwargs["safe_mode"] = False
    return tf.keras.models.load_model(model_path, **kwargs)


def _extract_archive_members(model_path: str) -> tuple[str, Optional[str]]:
    tmpdir = tempfile.mkdtemp(prefix="keras_extract_")
    config_path = None
    with zipfile.ZipFile(model_path, "r") as zf:
        names = set(zf.namelist())
        if "model.weights.h5" not in names:
            raise RuntimeError("File .keras không chứa model.weights.h5.")
        zf.extract("model.weights.h5", path=tmpdir)
        if "config.json" in names:
            zf.extract("config.json", path=tmpdir)
            config_path = str(Path(tmpdir) / "config.json")
    return str(Path(tmpdir) / "model.weights.h5"), config_path


def _build_model_from_keras_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        return tf.keras.models.model_from_json(raw)
    except Exception:
        cfg = json.loads(raw)
        if isinstance(cfg, dict) and "config" in cfg and isinstance(cfg["config"], dict):
            inner = cfg["config"]
            if "layers" in inner:
                return tf.keras.Sequential.from_config(inner)
        raise


def _peek_keras_metadata(model_path: str) -> dict:
    if not str(model_path).lower().endswith(".keras"):
        return {}
    try:
        with zipfile.ZipFile(model_path, "r") as zf:
            if "metadata.json" in zf.namelist():
                return json.loads(zf.read("metadata.json"))
    except Exception:
        return {}
    return {}


@st.cache_resource(show_spinner=False)
def load_keras_model(model_path: str, kind: str, dataset_name: str, input_dim: int):
    errors: list[str] = []
    path_obj = Path(model_path)
    metadata = _peek_keras_metadata(model_path)

    try:
        return _load_model_direct(model_path)
    except Exception as e:
        errors.append(f"load_model trực tiếp thất bại: {e}")

    if path_obj.suffix.lower() == ".keras":
        try:
            weights_path, config_path = _extract_archive_members(model_path)
            if config_path:
                try:
                    model = _build_model_from_keras_config(config_path)
                    model.load_weights(weights_path)
                    return model
                except Exception as e:
                    errors.append(f"rebuild từ config.json + load_weights thất bại: {e}")
            model = build_classifier_fallback(input_dim) if kind == "classifier" else build_dae_fallback(dataset_name, input_dim)
            model.load_weights(weights_path)
            return model
        except Exception as e:
            errors.append(f"fallback extract+load_weights thất bại: {e}")

    raise RuntimeError(
        "Không load được model Keras.\n"
        f"- file: {model_path}\n"
        + (f"- metadata .keras: {metadata}\n" if metadata else "")
        + "- các lỗi:\n"
        + "\n".join(f"  • {msg}" for msg in errors)
    )


# ---------- data loading ----------
@st.cache_data(show_spinner=False)
def load_npy(path: str):
    return np.load(path, allow_pickle=False)


@st.cache_data(show_spinner=False)
def load_json_cached(path: str):
    return safe_load_json(Path(path))


def classifier_predict(model: tf.keras.Model, x: np.ndarray):
    probs = np.asarray(model.predict(x, verbose=0))
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    if probs.shape[1] == 1:
        positive_prob = float(probs[0, 0])
        pred = int(positive_prob >= 0.5)
        prob_vec = np.array([1.0 - positive_prob, positive_prob], dtype=float)
        return pred, float(prob_vec[pred]), prob_vec
    pred = int(np.argmax(probs[0]))
    prob_vec = probs[0].astype(float)
    return pred, float(prob_vec[pred]), prob_vec


def compute_errors(x_nf: np.ndarray, recon_nf: np.ndarray, metric: str) -> np.ndarray:
    diff = x_nf - recon_nf
    metric_norm = (metric or "paper_l2").lower()
    if metric_norm in {"paper_l2", "l2", "euclidean"}:
        return np.linalg.norm(diff, axis=1)
    if metric_norm in {"mse", "source_mse"}:
        return np.mean(np.square(diff), axis=1)
    if metric_norm in {"rmse", "source_rmse"}:
        return np.sqrt(np.mean(np.square(diff), axis=1))
    return np.linalg.norm(diff, axis=1)


def select_nf_indices(selected_feature_names: list[str], feature_groups: dict) -> list[int]:
    candidates = (
        feature_groups.get("non_functional")
        or feature_groups.get("non_functional_features")
        or feature_groups.get("nf_features")
        or []
    )
    name_to_idx = {normalize_name(name): i for i, name in enumerate(selected_feature_names)}
    nf_idx: list[int] = []
    for name in candidates:
        idx = name_to_idx.get(normalize_name(name))
        if idx is not None:
            nf_idx.append(idx)
    return sorted(set(nf_idx))


def locate_artifacts(root: Path, dataset_name: str):
    ds = root / dataset_name
    prep_dir = ds / "02_preprocessing"
    clf_dir = ds / "04_classifier"
    dae_dir = ds / "06_dae"
    adv_dir = ds / "05_adversarial"

    classifier_path = find_existing(
        [
            clf_dir / "substitute_classifier.keras",
            clf_dir / "substitute_classifier.h5",
            clf_dir / "classifier.keras",
            clf_dir / "classifier.h5",
        ]
    )
    dae_path = find_existing(
        [
            dae_dir / "nf_dae.keras",
            dae_dir / "nf_dae.h5",
            dae_dir / "dae.keras",
            dae_dir / "dae.h5",
        ]
    )

    return {
        "dataset_root": ds,
        "prep_dir": prep_dir,
        "clf_dir": clf_dir,
        "dae_dir": dae_dir,
        "adv_dir": adv_dir,
        "group_path": ds / "03_feature_groups" / "feature_groups.json",
        "manifest_path": prep_dir / "preprocessing_manifest.json",
        "classifier_path": classifier_path or (clf_dir / "substitute_classifier.keras"),
        "dae_path": dae_path or (dae_dir / "nf_dae.keras"),
        "threshold_path": dae_dir / "threshold.json",
    }


def get_sample_from_artifacts(adv_dir: Path, attack_name: str, sample_index: int):
    path = adv_dir / f"{attack_name.lower()}_adv_success.npy"
    if not path.exists():
        raise FileNotFoundError(f"Không thấy file AE: {path}")
    arr = load_npy(str(path)).astype(np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if len(arr) == 0:
        raise ValueError(f"File {path.name} không có mẫu thành công nào.")
    sample_index = max(0, min(int(sample_index), len(arr) - 1))
    return path, arr, arr[sample_index : sample_index + 1], sample_index


def get_sample_from_upload(uploaded_file, sample_index: int):
    name = uploaded_file.name.lower()
    if name.endswith(".npy"):
        arr = np.load(uploaded_file, allow_pickle=False).astype(np.float32)
    elif name.endswith(".csv"):
        arr = pd.read_csv(uploaded_file).to_numpy(dtype=np.float32)
    else:
        raise ValueError("Chỉ hỗ trợ .npy hoặc .csv")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if len(arr) == 0:
        raise ValueError("File upload không có dữ liệu.")
    sample_index = max(0, min(int(sample_index), len(arr) - 1))
    return arr, arr[sample_index : sample_index + 1], sample_index


def run_single_prediction(
    x: np.ndarray,
    classifier: tf.keras.Model,
    dae: tf.keras.Model,
    nf_idx: list[int],
    threshold: float,
    threshold_metric: str,
):
    pred_before, conf_before, prob_vec = classifier_predict(classifier, x)
    result = {
        "pred_before": pred_before,
        "conf_before": conf_before,
        "prob_vec": prob_vec,
        "gate_to_dae": pred_before == 0,
        "flag_adversarial": False,
        "reconstruction_error": None,
        "threshold": float(threshold),
        "threshold_metric": threshold_metric,
        "final_decision": pred_before,
    }

    if pred_before == 0:
        if not nf_idx:
            raise ValueError("Không tìm được non-functional feature indices từ manifest + feature_groups.")
        x_nf = x[:, nf_idx]
        recon_nf = dae.predict(x_nf, verbose=0)
        err = float(compute_errors(x_nf, recon_nf, threshold_metric)[0])
        flagged = err > threshold
        result["reconstruction_error"] = err
        result["flag_adversarial"] = bool(flagged)
        result["final_decision"] = 1 if flagged else 0
    return result


def build_explanation(result: dict) -> dict:
    pred_label = label_text(result["pred_before"])
    final_label = label_text(result["final_decision"])
    gate_text = "Có" if result["gate_to_dae"] else "Không"
    flag_text = "Có" if result["flag_adversarial"] else "Không"
    conf_pct = result["conf_before"] * 100.0
    err = result["reconstruction_error"]
    threshold = result["threshold"]
    ratio = (err / threshold) if (err is not None and threshold > 0) else None

    if result["gate_to_dae"] and result["flag_adversarial"]:
        summary = "Mẫu này đã đánh lừa được classifier ở bước đầu, nhưng bị NF-DAE phát hiện lại là adversarial/bất thường."
    elif result["gate_to_dae"] and not result["flag_adversarial"]:
        summary = "Mẫu này được classifier xem là normal và NF-DAE cũng không phát hiện bất thường vượt ngưỡng."
    else:
        summary = "Mẫu này đã bị classifier xem là attack ngay từ đầu nên không cần đưa qua NF-DAE gate."

    steps = [
        f"1. **Pred trước detector:** classifier dự đoán mẫu là **{pred_label}** với confidence khoảng **{conf_pct:.2f}%**.",
        f"2. **Gate vào DAE:** **{gate_text}**. Theo logic paper/rewrite, chỉ các mẫu classifier dự đoán là **Normal** mới được đưa qua NF-DAE.",
    ]
    if err is not None:
        ratio_text = f", lớn hơn threshold khoảng **{ratio:.2f} lần**" if ratio is not None else ""
        compare_text = "vượt ngưỡng" if err > threshold else "không vượt ngưỡng"
        steps.append(
            f"3. **So sánh error và threshold:** reconstruction error = **{err:.8f}**, threshold = **{threshold:.8f}**. Giá trị này **{compare_text}**{ratio_text}."
        )
    else:
        steps.append("3. **So sánh error và threshold:** không áp dụng vì mẫu không đi qua DAE.")
    steps.append(
        f"4. **Kết luận cuối:** hệ thống kết luận mẫu là **{final_label}**. Trạng thái flag adversarial = **{flag_text}**."
    )

    if result["gate_to_dae"] and err is not None:
        demo_script = (
            f"Mẫu này trước detector bị classifier dự đoán là {pred_label} với confidence khoảng {conf_pct:.2f}%, "
            f"nên được đưa qua NF-DAE để kiểm tra. Reconstruction error là {err:.8f}, so với threshold {threshold:.8f}. "
            f"Vì vậy hệ thống kết luận cuối là {final_label}."
        )
    else:
        demo_script = (
            f"Mẫu này đã bị classifier dự đoán là {pred_label} ngay từ đầu với confidence khoảng {conf_pct:.2f}%, "
            f"nên không cần đưa qua NF-DAE. Kết luận cuối vẫn là {final_label}."
        )

    return {"summary": summary, "steps": steps, "demo_script": demo_script}


# ---------- UI ----------
st.title("NIDS-DA Interactive AE Predictor")
st.caption("Load 1 adversarial example rồi dự đoán bằng substitute classifier và NF-DAE detector.")

with st.sidebar:
    st.header("Cấu hình")
    mode = st.radio("Nguồn dữ liệu", ["Dùng thư mục artifacts", "Upload file AE"], index=0)
    dataset_name = st.selectbox("Dataset", DATASETS, index=1)
    attack_name = st.selectbox("Attack", ATTACKS, index=3)
    sample_index = st.number_input("Chỉ số mẫu", min_value=0, value=0, step=1)
    artifact_root_str = st.text_input(
        "Artifact root",
        value="artifacts",
        help="Thư mục gốc chứa NSL-KDD / UNSW-NB15 / CICIDS2017 cùng các thư mục 02_preprocessing, 03_feature_groups, 04_classifier, 05_adversarial, 06_dae.",
    )
    uploaded_ae = None
    if mode == "Upload file AE":
        uploaded_ae = st.file_uploader("Upload AE (.npy hoặc .csv)", type=["npy", "csv"])

col1, col2 = st.columns([1.1, 1.0])
artifact_root = Path(artifact_root_str)
art = locate_artifacts(artifact_root, dataset_name)

with col1:
    st.subheader("Mô hình và artefact")
    if mode == "Dùng thư mục artifacts":
        st.code(
            "\n".join(
                [
                    f"classifier: {art['classifier_path']}",
                    f"dae       : {art['dae_path']}",
                    f"threshold : {art['threshold_path']}",
                    f"ae file   : {art['adv_dir'] / (attack_name.lower() + '_adv_success.npy')}",
                ]
            )
        )
    else:
        st.info("Bạn vẫn cần classifier, DAE, threshold và feature_groups trong artifact root; chỉ riêng AE được upload thủ công.")
    run = st.button("Predict 1 AE", type="primary", use_container_width=True)

if run:
    try:
        required = [art["classifier_path"], art["dae_path"], art["threshold_path"], art["group_path"], art["manifest_path"]]
        missing = [str(p) for p in required if not Path(p).exists()]
        if missing:
            raise FileNotFoundError("Thiếu artefact bắt buộc:\n- " + "\n- ".join(missing))

        threshold_meta = load_json_cached(str(art["threshold_path"]))
        groups = load_json_cached(str(art["group_path"]))
        manifest = load_json_cached(str(art["manifest_path"]))
        selected_feature_names = manifest.get("selected_feature_names") or manifest.get("feature_names")
        if not selected_feature_names:
            raise ValueError("preprocessing_manifest.json không có selected_feature_names.")
        nf_idx = select_nf_indices(selected_feature_names, groups)
        if not nf_idx:
            raise ValueError("Không map được non-functional feature indices từ feature_groups.json.")

        threshold = float(threshold_meta["threshold"])
        threshold_metric = threshold_meta.get("threshold_metric", "paper_l2")

        if mode == "Dùng thư mục artifacts":
            ae_path, ae_all, x, real_index = get_sample_from_artifacts(art["adv_dir"], attack_name, int(sample_index))
            source_desc = f"{ae_path.name} | tổng mẫu: {len(ae_all)} | index dùng: {real_index}"
        else:
            if uploaded_ae is None:
                raise ValueError("Bạn chưa upload file AE.")
            ae_all, x, real_index = get_sample_from_upload(uploaded_ae, int(sample_index))
            source_desc = f"upload: {uploaded_ae.name} | tổng mẫu: {len(ae_all)} | index dùng: {real_index}"

        if x.shape[1] != len(selected_feature_names):
            raise ValueError(
                f"Số chiều mẫu = {x.shape[1]}, nhưng classifier cần {len(selected_feature_names)} feature."
            )

        classifier = load_keras_model(str(art["classifier_path"]), "classifier", dataset_name, len(selected_feature_names))
        dae = load_keras_model(str(art["dae_path"]), "dae", dataset_name, len(nf_idx))

        result = run_single_prediction(x, classifier, dae, nf_idx, threshold, threshold_metric)
        explanation = build_explanation(result)

        with col1:
            st.subheader("Kết quả dự đoán")
            m1, m2, m3 = st.columns(3)
            m1.metric("Pred trước detector", label_text(result["pred_before"]))
            m2.metric("Gate vào DAE", "Có" if result["gate_to_dae"] else "Không")
            m3.metric("Kết luận cuối", label_text(result["final_decision"]))

            m4, m5, m6 = st.columns(3)
            m4.metric("Confidence", f"{result['conf_before']:.4f}")
            m5.metric("Threshold", f"{result['threshold']:.8f}")
            rec_text = "-" if result["reconstruction_error"] is None else f"{result['reconstruction_error']:.8f}"
            m6.metric("Reconstruction error", rec_text)

            st.write("**Flag adversarial:**", "Có" if result["flag_adversarial"] else "Không")
            st.write("**Threshold metric:**", result["threshold_metric"])
            st.write("**Nguồn mẫu:**", source_desc)

            st.subheader("Giải thích kết quả")
            st.info(explanation["summary"])
            for step in explanation["steps"]:
                st.markdown(step)
            with st.expander("Đoạn nói nhanh để demo"):
                st.write(explanation["demo_script"])

            prob_labels = ["Normal", "Attack"] if len(result["prob_vec"]) == 2 else [str(i) for i in range(len(result["prob_vec"]))]
            prob_df = pd.DataFrame(
                {
                    "class_id": list(range(len(result["prob_vec"]))),
                    "probability": result["prob_vec"],
                    "label": prob_labels,
                }
            )
            st.dataframe(prob_df, use_container_width=True)

        with col2:
            st.subheader("Chi tiết feature")
            nf_set = set(nf_idx)
            feature_df = pd.DataFrame(
                {
                    "feature": selected_feature_names,
                    "value": x[0],
                    "is_non_functional": [i in nf_set for i in range(len(selected_feature_names))],
                }
            )
            st.dataframe(feature_df, use_container_width=True, height=520)
            if result["gate_to_dae"]:
                st.info("Theo logic paper/rewrite, chỉ các mẫu classifier dự đoán là normal mới được đưa qua NF-DAE để kiểm tra reconstruction error.")
            else:
                st.info("Mẫu đã bị classifier xem là attack, nên không cần qua DAE gate.")

    except Exception as e:
        st.error(str(e))

st.divider()
with st.expander("Cấu trúc thư mục mong đợi"):
    st.code(
        """artifacts/
├── NSL-KDD/
│   ├── 02_preprocessing/
│   │   └── preprocessing_manifest.json
│   ├── 03_feature_groups/
│   │   └── feature_groups.json
│   ├── 04_classifier/
│   │   └── substitute_classifier.keras
│   ├── 05_adversarial/
│   │   └── jsma_adv_success.npy
│   └── 06_dae/
│       ├── nf_dae.keras
│       └── threshold.json
├── UNSW-NB15/
└── CICIDS2017/
"""
    )
    st.write(
        "Nếu bạn đã chạy notebook rewrite, các file chính thường được lưu ở 04_classifier/substitute_classifier.keras, "
        "05_adversarial/<attack>_adv_success.npy, 06_dae/nf_dae.keras và 06_dae/threshold.json."
    )
