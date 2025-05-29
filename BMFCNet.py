import os
import numpy as np
import scipy.io as sio
import pandas as pd
from mne import create_info
from mne.io import RawArray
from mne.preprocessing import ICA

# === CONFIG ===
SAMPLE_RATE = 256  # Hz
WINDOW_SEC = 4
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SEC  # 1024 samples
OVERLAP = 0.75
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))

CHANNELS = ['Fp1', 'F3', 'F7', 'Fz', 'Fp2', 'F4', 'F8',
            'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1',
            'O2', 'T3', 'T4', 'T5', 'T6', 'Oz']

def load_mat_file(filepath):
    mat = sio.loadmat(filepath)
    data = mat['data']  # adjust key if needed
    return data

def apply_ica(data):
    info = create_info(CHANNELS, SAMPLE_RATE, ch_types='eeg')
    raw = RawArray(data, info)
    raw.filter(l_freq=1.0, h_freq=100.0, fir_design='firwin')
    raw.notch_filter(freqs=50.0)
    ica = ICA(n_components=15, random_state=97, max_iter='auto')
    ica.fit(raw)
    raw = ica.apply(raw)
    return raw.get_data()

def segment_and_normalize(eeg):
    segments = []
    for start in range(0, eeg.shape[1] - WINDOW_SIZE + 1, STEP_SIZE):
        segment = eeg[:, start:start + WINDOW_SIZE]
        segment = (segment - np.mean(segment, axis=1, keepdims=True)) / np.std(segment, axis=1, keepdims=True)
        segments.append(segment)
    return np.array(segments)

def process_file(filepath, label):
    data = load_mat_file(filepath)
    if data.shape[0] > len(CHANNELS):
        data = data[:len(CHANNELS), :]
    data = apply_ica(data)
    segments = segment_and_normalize(data)
    labels = np.full((segments.shape[0],), label)
    return segments, labels

def process_dataset(root_dir, df_labels):
    X_all, y_all = [], []
    # Extract numeric id from Data Id for matching
    df_labels['id'] = df_labels['Data Id'].astype(str).str.extract('(\d+)')[0]
    label_dict = dict(zip(df_labels['id'], df_labels['label']))

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".mat"):
                file_id = os.path.splitext(file)[0].lower()
                numeric_id = ''.join(filter(str.isdigit, file_id))
                if numeric_id not in label_dict:
                    print(f"Skipping {file}: no label found in DataFrame.")
                    continue
                label = label_dict[numeric_id]
                filepath = os.path.join(subdir, file)
                print(f"Processing {file} (Label: {label})...")
                segments, labels = process_file(filepath, label)
                X_all.append(segments)
                y_all.append(labels)

    if not X_all or not y_all:
        raise ValueError("No data processed: labels missing or mismatched.")

    return np.vstack(X_all), np.hstack(y_all)

if __name__ == "__main__":
    data_dir = r"C:\Users\hp\OneDrive\research project\EEG-Depression-Dataset-main\Depression dataset"
    label_file = r"C:\Users\hp\OneDrive\research project\EEG-Depression-Dataset-main\labels_processed.csv"

    df_labels = pd.read_csv(label_file)
    X, y = process_dataset(data_dir, df_labels)

    print("Shape of X:", X.shape)  # (samples, 20, 1024)
    print("Shape of y:", y.shape)  # (samples,)
    print("Unique labels:", np.unique(y))  # To see all assigned labels