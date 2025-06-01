import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Concatenate, Add, Multiply, Dense, GlobalAveragePooling1D, \
    BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping


# === Define BMFCNet Model ===
def residual_inception_block(x, filters=[32, 64, 128]):
    conv1 = Conv1D(filters[0], kernel_size=3, padding='same', activation='relu')(x)
    conv2 = Conv1D(filters[1], kernel_size=5, padding='same', activation='relu')(x)
    conv3 = Conv1D(filters[2], kernel_size=7, padding='same', activation='relu')(x)
    merged = Concatenate()([conv1, conv2, conv3])
    shortcut = Conv1D(sum(filters), kernel_size=1, padding='same')(x)
    return Add()([merged, shortcut])


def build_bmfcnet(input_shape):
    inp = Input(shape=input_shape)

    # Low-level features
    ll = Conv1D(64, kernel_size=3, padding='same', activation='relu')(inp)
    ll = BatchNormalization()(ll)
    ll = Conv1D(64, kernel_size=3, padding='same', activation='relu')(ll)
    ll_ri = residual_inception_block(ll)

    # High-level features
    hl = Conv1D(128, kernel_size=3, padding='same', activation='relu')(inp)
    hl = BatchNormalization()(hl)
    hl = Conv1D(128, kernel_size=3, padding='same', activation='relu')(hl)
    hl_ri = residual_inception_block(hl)

    # Fusion (Add, Multiply, Concat)
    add_feat = Add()([ll_ri, hl_ri])
    mul_feat = Multiply()([ll_ri, hl_ri])
    concat_feat = Concatenate()([ll_ri, hl_ri])

    # Global pooling
    add_pool = GlobalAveragePooling1D()(add_feat)
    mul_pool = GlobalAveragePooling1D()(mul_feat)
    concat_pool = GlobalAveragePooling1D()(concat_feat)

    # Final fusion + Dense
    fusion = Concatenate()([add_pool, mul_pool, concat_pool])
    dense = Dense(64, activation='relu')(fusion)
    out = Dense(1, activation='sigmoid')(dense)

    return Model(inputs=inp, outputs=out)


# === Training for each state ===
states = ['data', 'dataOpen', 'dataClose', 'all']
data_path = "processed_data"

for state in states:
    print(f"\n=== Training & Evaluating for EEG State: {state.upper()} ===")

    x_path = os.path.join(data_path, f"X_{state}.npy")
    y_path = os.path.join(data_path, f"y_{state}.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"Missing preprocessed files for state '{state}'. Skipping...")
        continue

    # === Load data ===
    X = np.load(x_path)
    y = np.load(y_path)

    # === Transpose to (samples, time, channels) ===
    X = np.transpose(X, (0, 2, 1))

    # === Split data ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # === Build & Compile model ===
    model = build_bmfcnet(X.shape[1:])
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])

    # === Train model ===
    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=2
    )

    # === Evaluate ===
    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    print(f"\n--- Classification Report for {state.upper()} ---")
    print(classification_report(y_test, y_pred))

    # === Optional: Save model ===
    model.save(f"bmfcnet_{state}.h5")
    print(f"Saved model: bmfcnet_{state}.h5")
