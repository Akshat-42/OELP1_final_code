import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from joblib import dump

# --- Configuration ---
DATA_PATH = "test_pts_new.csv"
MODEL_SAVE_PATH = "cricket_trajectory_model.keras"
SCALER_X_PATH = "scaler_X.joblib"
SCALER_Y_PATH = "scaler_y.joblib"

# --- Categorization Bins ---
V_BINS = [0, 22, 26, np.inf]
V_LABELS = ['Low', 'Medium', 'High']

W_MAG_BINS = [0, 150, 220, np.inf]
W_MAG_LABELS = ['Low', 'Medium', 'High']

# NEW: Spin Angle Categories (Left Spin, Straight, Right Spin)
# -10 to 10 degrees is considered "Straight-ish"
W_ANGLE_BINS = [-np.inf, -10, 10, np.inf]
W_ANGLE_LABELS = ['Negative_Spin', 'Neutral_Spin', 'Positive_Spin']

def load_and_clean_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    data = pd.read_csv(filepath)
    # Parse '[x, y]' string to columns
    data["p_f"] = data["p_f"].apply(lambda x: list(map(float, x.strip("[]").split(',')))[0:2])
    p_f_coords = data['p_f'].apply(pd.Series)
    p_f_coords.columns = ['land_x', 'land_y']
    return pd.concat([data.drop('p_f', axis=1), p_f_coords], axis=1)

def engineer_features(data):
    data['v_cat'] = pd.cut(data['v_mag'], bins=V_BINS, labels=V_LABELS, right=False)
    data['w_mag_cat'] = pd.cut(data['w_mag'], bins=W_MAG_BINS, labels=W_MAG_LABELS, right=False)
    
    # NEW: Categorize the Spin Direction
    data['w_angle_cat'] = pd.cut(data['w_angle'], bins=W_ANGLE_BINS, labels=W_ANGLE_LABELS)
    return data

def build_model(input_dim, output_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        Dense(512, activation='relu'),
        BatchNormalization(),
        
        Dense(256, activation='relu'),
        
        Dense(output_dim, activation='linear')
    ])
    return model

def main():
    print("Loading Data...")
    df = load_and_clean_data(DATA_PATH)
    df = engineer_features(df)
    
    # Inputs: Now includes w_angle_cat to resolve ambiguity
    feature_cols = ['land_x', 'land_y', 'p_x', 'p_y', 'p_z']
    cat_cols = ['v_cat', 'w_mag_cat', 'w_angle_cat']
    
    target_cols = ['v_mag', 'phi', 'theta', 'w_mag', 'w_angle']
    
    X = df[feature_cols + cat_cols]
    y = df[target_cols]

    print(f"Features: {feature_cols + cat_cols}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), feature_cols),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )
    
    scaler_y = StandardScaler()
    
    print("Preprocessing data...")
    X_processed = preprocessor.fit_transform(X)
    y_processed = scaler_y.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.1, random_state=42)
    
    model = build_model(X_train.shape[1], y_train.shape[1])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    early_stopper = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    print("Starting Training...")
    model.fit(
        X_train, y_train, 
        epochs=300, 
        batch_size=64, 
        validation_data=(X_test, y_test), 
        callbacks=[early_stopper, lr_scheduler]
    )
    
    print("Saving artifacts...")
    model.save(MODEL_SAVE_PATH)
    dump(preprocessor, SCALER_X_PATH)
    dump(scaler_y, SCALER_Y_PATH)
    print("Done.")

if __name__ == "__main__":
    main()