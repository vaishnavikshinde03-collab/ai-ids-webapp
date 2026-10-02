import os
import glob
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def load_and_preprocess_data():
    all_files = glob.glob(os.path.join(DATASET_DIR, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in directory: {DATASET_DIR}")

    df_list = []
    for f in all_files:
        temp_df = pd.read_csv(f)
        temp_df.columns = temp_df.columns.str.strip()
        df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)

    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if 'Label' in df.columns:
        df['Label'] = df['Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
        y = df['Label'].values
        X = df.drop(columns=['Label'])
    else:
        raise KeyError("'Label' column missing from dataset.")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(MODELS_DIR, "minmax_scaler.pkl"))
    print("MinMax Scaler saved successfully.")

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, X_train.shape[1]

def build_cnn(input_dim):
    model = Sequential([
        tf.keras.layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def build_lstm(input_dim):
    model = Sequential([
        tf.keras.layers.Reshape((1, input_dim), input_shape=(input_dim,)),
        LSTM(64, return_sequences=True, activation='tanh'),
        LSTM(32, activation='tanh'),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    enc = Dense(64, activation='relu')(input_layer)
    enc = Dense(32, activation='relu')(enc)
    bottleneck = Dense(16, activation='relu')(enc)
    dec = Dense(32, activation='relu')(bottleneck)
    dec = Dense(64, activation='relu')(dec)
    output_layer = Dense(input_dim, activation='sigmoid')(dec)
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0005),
                        loss='mean_squared_error')
    return autoencoder

if __name__ == "__main__":
    print("Loading and preprocessing dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test, input_dim = load_and_preprocess_data()

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    print("\n--- Training CNN Model ---")
    cnn_model = build_cnn(input_dim)
    cnn_model.fit(X_train, y_train, epochs=15, batch_size=256,
                  validation_data=(X_val, y_val), verbose=1,
                  callbacks=[early_stop])
    cnn_model.save(os.path.join(MODELS_DIR, "cnn_ids.keras"))
    print("CNN Model saved as cnn_ids.keras")

    print("\n--- Training LSTM Model ---")
    lstm_model = build_lstm(input_dim)
    lstm_model.fit(X_train, y_train, epochs=15, batch_size=256,
                   validation_data=(X_val, y_val), verbose=1,
                   callbacks=[early_stop])
    lstm_model.save(os.path.join(MODELS_DIR, "lstm_ids.keras"))
    print("LSTM Model saved as lstm_ids.keras")

    print("\n--- Training Autoencoder Model ---")
    X_train_benign = X_train[y_train == 0]
    X_val_benign = X_val[y_val == 0]

    autoencoder_model = build_autoencoder(input_dim)
    autoencoder_model.fit(X_train_benign, X_train_benign,
                          epochs=15, batch_size=256,
                          validation_data=(X_val_benign, X_val_benign), verbose=1,
                          callbacks=[early_stop])
    autoencoder_model.save(os.path.join(MODELS_DIR, "autoencoder_ids.keras"))
    print("Autoencoder Model saved as autoencoder_ids.keras")

    print("\nAll models trained and saved to:", MODELS_DIR)