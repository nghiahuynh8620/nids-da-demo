from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title='NIDS-DA AE Predictor', layout='wide')

ATTACKS = ["FGSM", "BIM", "PGD", "JSMA", "DeepFool"]
DATASETS = ["NSL-KDD", "UNSW-NB15", "CICIDS2017"]


def normalize_name(s: str) -> str:
    return ''.join(ch.lower() for ch in s if ch.isalnum())


def find_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def safe_load_json(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_keras_model(model_path: str):
    return tf.keras.models.load_model(model_path)


@st.cache_data(show_spinner=False)
def load_npy(path: str):
    return np.load(path, allow_pickle=False)


@st.cache_data(show_spinner=False)
def load_json(path: str):
    return safe_load_json(Path(path))


def classifier_predict(model: tf.keras.Model, x: np.ndarray) -> Tuple[int, float, np.ndarray]:
    probs = model.predict(x, verbose=0)
    probs = np.asarray(probs)
    if probs.ndim == 1:
        positive_prob = float(probs[0])
        pred = int(positive_prob >= 0.5)
        return pred, positive_prob, np.array([1.0 - positive_prob, positive_prob], dtype=float)
    if probs.ndim == 2 and probs.shape[1] == 1:
        positive_prob = float(probs[0, 0])
        pred = int(positive_prob >= 0.5)
        return pred, positive_prob, np.array([1.0 - positive_prob, positive_prob], dtype=float)
    pred = int(np.argmax(probs[0]))
    positive_prob = float(probs[0, pred])
    return pred, positive_prob, probs[0].astype(float)



def compute_errors(x_nf: np.ndarray, recon_nf: np.ndarray, metric: str) -> np.ndarray:
    diff = x_nf - recon_nf
    if metric == 'paper_l2':
        return np.linalg.norm(diff, axis=1)
    # fallback: rmse / mse-like source metric used in some rewrite cells
    return np.sqrt(np.mean(np.square(diff), axis=1))



def select_nf_indices(selected_feature_names: list[str], feature_groups: dict) -> list[int]:
    candidates = (
        feature_groups.get('non_functional')
        or feature_groups.get('non_functional_features')
        or feature_groups.get('nf_features')
        or []
    )
    name_to_idx = {normalize_name(name): i for i, name in enumerate(selected_feature_names)}
    nf_idx = []
    for name in candidates:
        idx = name_to_idx.get(normalize_name(name))
        if idx is not None:
            nf_idx.append(idx)
    return nf_idx



def locate_artifacts(root: Path, dataset_name: str):
    ds = root / dataset_name
    prep_dir = ds / '02_preprocessing'
    clf_dir = ds / '04_classifier'
    dae_dir = ds / '06_dae'
    adv_dir = ds / '05_adversarial'
    group_path = ds / '03_feature_groups' / 'feature_groups.json'

    manifest_path = prep_dir / 'preprocessing_manifest.json'
    classifier_path = clf_dir / 'substitute_classifier.keras'
    dae_path = dae_dir / 'nf_dae.keras'
    threshold_path = dae_dir / 'threshold.json'

    return {
        'dataset_root': ds,
        'prep_dir': prep_dir,
        'clf_dir': clf_dir,
        'dae_dir': dae_dir,
        'adv_dir': adv_dir,
        'group_path': group_path,
        'manifest_path': manifest_path,
        'classifier_path': classifier_path,
        'dae_path': dae_path,
        'threshold_path': threshold_path,
    }



def get_sample_from_artifacts(adv_dir: Path, attack_name: str, sample_index: int):
    path = adv_dir / f'{attack_name.lower()}_adv_success.npy'
    if not path.exists():
        raise FileNotFoundError(f'Không thấy file AE: {path}')
    arr = load_npy(str(path)).astype(np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if len(arr) == 0:
        raise ValueError(f'File {path.name} không có mẫu thành công nào.')
    sample_index = max(0, min(sample_index, len(arr) - 1))
    return path, arr, arr[sample_index : sample_index + 1], sample_index



def get_sample_from_upload(uploaded_file, sample_index: int):
    name = uploaded_file.name.lower()
    if name.endswith('.npy'):
        arr = np.load(uploaded_file, allow_pickle=False).astype(np.float32)
    elif name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        arr = df.to_numpy(dtype=np.float32)
    else:
        raise ValueError('Chỉ hỗ trợ .npy hoặc .csv')
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if len(arr) == 0:
        raise ValueError('File upload không có dữ liệu.')
    sample_index = max(0, min(sample_index, len(arr) - 1))
    return arr, arr[sample_index : sample_index + 1], sample_index



def run_single_prediction(x: np.ndarray, classifier: tf.keras.Model, dae: tf.keras.Model,
                          nf_idx: list[int], threshold: float, threshold_metric: str):
    pred_before, conf_before, prob_vec = classifier_predict(classifier, x)

    result = {
        'pred_before': pred_before,
        'conf_before': conf_before,
        'prob_vec': prob_vec,
        'gate_to_dae': pred_before == 0,
        'flag_adversarial': False,
        'reconstruction_error': None,
        'threshold': threshold,
        'final_decision': pred_before,
        'threshold_metric': threshold_metric,
    }

    if pred_before == 0:
        if not nf_idx:
            raise ValueError('Không tìm được non-functional feature indices từ preprocessing_manifest + feature_groups.')
        x_nf = x[:, nf_idx]
        recon_nf = dae.predict(x_nf, verbose=0)
        errors = compute_errors(x_nf, recon_nf, threshold_metric)
        err = float(errors[0])
        flagged = err > threshold
        result['reconstruction_error'] = err
        result['flag_adversarial'] = bool(flagged)
        result['final_decision'] = 1 if flagged else pred_before
    return result



def build_explanation(result: dict) -> dict:
    pred_label = label_text(result['pred_before'])
    final_label = label_text(result['final_decision'])
    gate_text = 'Có' if result['gate_to_dae'] else 'Không'
    flag_text = 'Có' if result['flag_adversarial'] else 'Không'
    conf_pct = result['conf_before'] * 100.0

    if result['reconstruction_error'] is not None and result['threshold'] > 0:
        ratio = result['reconstruction_error'] / result['threshold']
    else:
        ratio = None

    if result['gate_to_dae'] and result['flag_adversarial']:
        summary = (
            f"Mẫu này đã đánh lừa được classifier ở bước đầu (bị dự đoán là {pred_label}), "
            f"nhưng bị NF-DAE phát hiện lại nên kết luận cuối là {final_label}."
        )
    elif result['gate_to_dae'] and not result['flag_adversarial']:
        summary = (
            f"Mẫu này được classifier dự đoán là {pred_label}, sau đó đi qua NF-DAE nhưng không vượt ngưỡng, "
            f"nên kết luận cuối vẫn là {final_label}."
        )
    else:
        summary = (
            f"Mẫu này đã bị classifier chặn ngay từ đầu với nhãn {pred_label}, nên không cần đưa qua NF-DAE; "
            f"kết luận cuối là {final_label}."
        )

    steps = [
        f"1. **Pred trước detector:** `{pred_label}` với confidence khoảng **{conf_pct:.2f}%**.",
        f"2. **Gate vào DAE:** `{gate_text}`.",
    ]

    if result['gate_to_dae']:
        err = result['reconstruction_error']
        thr = result['threshold']
        if ratio is not None:
            compare_text = f"Lỗi tái tạo **{err:.8f}** so với threshold **{thr:.8f}**, tức khoảng **{ratio:.2f}x** ngưỡng."
        else:
            compare_text = f"Lỗi tái tạo **{err:.8f}** so với threshold **{thr:.8f}**."
        if result['flag_adversarial']:
            meaning = "Vì reconstruction error lớn hơn threshold, mẫu bị gắn cờ là adversarial/bất thường."
        else:
            meaning = "Vì reconstruction error không vượt threshold, mẫu chưa bị detector xem là adversarial."
        steps.append(f"3. **Kiểm tra NF-DAE:** {compare_text} {meaning}")
    else:
        steps.append("3. **Không qua NF-DAE:** do classifier đã xem mẫu là attack từ đầu.")

    steps.append(f"4. **Kết luận cuối:** `{final_label}`. Flag adversarial: `{flag_text}`.")

    demo_script = (
        f"Mẫu này trước detector bị classifier dự đoán là {pred_label} với confidence khoảng {conf_pct:.2f}%. "
        + (
            f"Do mẫu bị xem là normal nên hệ thống đưa tiếp qua NF-DAE. Reconstruction error là "
            f"{result['reconstruction_error']:.8f}, threshold là {result['threshold']:.8f}"
            + (f", tức khoảng {ratio:.2f} lần ngưỡng" if ratio is not None else "")
            + ". "
            + ("Vì error vượt ngưỡng nên mẫu bị gắn cờ adversarial và kết luận cuối là Attack/Adversarial."
               if result['flag_adversarial'] else
               "Vì error không vượt ngưỡng nên mẫu không bị flag adversarial và giữ nguyên kết luận ban đầu.")
            if result['gate_to_dae'] else
            f"Do classifier đã xem đây là attack ngay từ đầu nên mẫu không cần qua NF-DAE, và kết luận cuối vẫn là {final_label}."
        )
    )

    return {
        'summary': summary,
        'steps': steps,
        'demo_script': demo_script,
    }


def label_text(label: int) -> str:
    return 'Attack/Adversarial' if int(label) == 1 else 'Normal'


st.title('NIDS-DA Interactive AE Predictor')
st.caption('Load 1 adversarial example rồi dự đoán bằng substitute classifier và NF-DAE detector.')

with st.sidebar:
    st.header('Cấu hình')
    mode = st.radio('Nguồn dữ liệu', ['Dùng thư mục artifacts', 'Upload file AE'], index=0)
    dataset_name = st.selectbox('Dataset', DATASETS, index=1)
    attack_name = st.selectbox('Attack', ATTACKS, index=3)
    sample_index = st.number_input('Chỉ số mẫu', min_value=0, value=0, step=1)
    artifact_root_str = st.text_input(
        'Artifact root',
        value='artifacts',
        help='Trỏ tới thư mục chứa NSL-KDD/UNSW-NB15/CICIDS2017 và các thư mục 02_preprocessing, 04_classifier, 05_adversarial, 06_dae',
    )
    uploaded_ae = None
    if mode == 'Upload file AE':
        uploaded_ae = st.file_uploader('Upload AE (.npy hoặc .csv)', type=['npy', 'csv'])

col1, col2 = st.columns([1.1, 1.0])

with col1:
    st.subheader('Mô hình và artefact')
    artifact_root = Path(artifact_root_str)
    art = locate_artifacts(artifact_root, dataset_name)
    if mode == 'Dùng thư mục artifacts':
        st.code('\n'.join([
            f"classifier: {art['classifier_path']}",
            f"dae       : {art['dae_path']}",
            f"threshold : {art['threshold_path']}",
            f"ae file   : {art['adv_dir'] / (attack_name.lower() + '_adv_success.npy')}",
        ]))
    else:
        st.info('Bạn vẫn cần classifier, DAE, threshold và feature_groups trong artifact root; chỉ riêng AE được upload thủ công.')

    run = st.button('Predict 1 AE', type='primary', use_container_width=True)

if run:
    try:
        required = [art['classifier_path'], art['dae_path'], art['threshold_path'], art['group_path'], art['manifest_path']]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError('Thiếu artefact bắt buộc:\n- ' + '\n- '.join(missing))

        classifier = load_keras_model(str(art['classifier_path']))
        dae = load_keras_model(str(art['dae_path']))
        threshold_meta = load_json(str(art['threshold_path']))
        groups = load_json(str(art['group_path']))
        manifest = load_json(str(art['manifest_path']))
        selected_feature_names = manifest['selected_feature_names']
        nf_idx = select_nf_indices(selected_feature_names, groups)
        threshold = float(threshold_meta['threshold'])
        threshold_metric = threshold_meta.get('threshold_metric', 'paper_l2')

        if mode == 'Dùng thư mục artifacts':
            ae_path, ae_all, x, real_index = get_sample_from_artifacts(art['adv_dir'], attack_name, int(sample_index))
            source_desc = f'{ae_path.name} | tổng mẫu: {len(ae_all)} | index dùng: {real_index}'
        else:
            if uploaded_ae is None:
                raise ValueError('Bạn chưa upload file AE.')
            ae_all, x, real_index = get_sample_from_upload(uploaded_ae, int(sample_index))
            source_desc = f'upload: {uploaded_ae.name} | tổng mẫu: {len(ae_all)} | index dùng: {real_index}'

        if x.shape[1] != len(selected_feature_names):
            raise ValueError(
                f'Số chiều của mẫu là {x.shape[1]}, nhưng classifier đang cần {len(selected_feature_names)} feature.'
            )

        result = run_single_prediction(x, classifier, dae, nf_idx, threshold, threshold_metric)

        st.success('Dự đoán xong.')

        with col1:
            st.subheader('Kết quả dự đoán')
            m1, m2, m3 = st.columns(3)
            m1.metric('Pred trước detector', label_text(result['pred_before']))
            m2.metric('Gate vào DAE', 'Có' if result['gate_to_dae'] else 'Không')
            m3.metric('Kết luận cuối', label_text(result['final_decision']))

            m4, m5, m6 = st.columns(3)
            m4.metric('Confidence', f"{result['conf_before']:.4f}")
            m5.metric('Threshold', f"{result['threshold']:.8f}")
            rec_text = '-' if result['reconstruction_error'] is None else f"{result['reconstruction_error']:.8f}"
            m6.metric('Reconstruction error', rec_text)

            st.write('**Flag adversarial:**', 'Có' if result['flag_adversarial'] else 'Không')
            st.write('**Threshold metric:**', result['threshold_metric'])
            st.write('**Nguồn mẫu:**', source_desc)

            explanation = build_explanation(result)
            st.subheader('Giải thích kết quả')
            st.info(explanation['summary'])
            for step in explanation['steps']:
                st.markdown(step)

            with st.expander('Đoạn nói nhanh để demo'):
                st.write(explanation['demo_script'])

            prob_df = pd.DataFrame({
                'class_id': list(range(len(result['prob_vec']))),
                'probability': result['prob_vec'],
                'label': ['Normal', 'Attack'][: len(result['prob_vec'])],
            })
            st.dataframe(prob_df, use_container_width=True)

        with col2:
            st.subheader('Chi tiết feature')
            feature_df = pd.DataFrame({
                'feature': selected_feature_names,
                'value': x[0],
                'is_non_functional': [i in set(nf_idx) for i in range(len(selected_feature_names))],
            })
            st.dataframe(feature_df, use_container_width=True, height=520)

            if result['gate_to_dae']:
                st.info(
                    'Theo logic paper/rewrite, chỉ các mẫu mà classifier dự đoán là normal mới được đưa qua NF-DAE để kiểm tra reconstruction error.'
                )
            else:
                st.info('Mẫu đã bị classifier xem là attack, nên không cần qua DAE gate.')

    except Exception as e:
        st.error(str(e))

st.divider()
with st.expander('Cấu trúc thư mục mong đợi'):
    st.code(
        '''artifacts/
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
'''
    )
    st.write(
        'Nếu bạn đã chạy notebook rewrite, các file chính thường được lưu ở 04_classifier/substitute_classifier.keras, '
        '05_adversarial/<attack>_adv_success.npy, 06_dae/nf_dae.keras và 06_dae/threshold.json.'
    )
