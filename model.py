import pandas as pd
import numpy as np
import optuna
import shap
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.combine import SMOTETomek
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Input, Dropout, BatchNormalization, AdditiveAttention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Nadam, AdamW, RMSprop
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Load cleaned dataset
df = pd.read_csv("/content/drive/MyDrive/dataset/heart_disease_cleaned.csv")

# Splitting features and target
X = df.drop(columns=["num"])
y = df["num"].apply(lambda x: 1 if x > 0 else 0)  # Binary classification

# Handle class imbalance
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

# Feature Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Feature Selection
selector = SelectKBest(mutual_info_classif, k=15)
X_selected = selector.fit_transform(X_scaled, y_resampled)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Optimized CNN+LSTM Model using Functional API
def create_cnn_lstm(trial):
    filters = trial.suggest_int('filters', 32, 128)
    lstm_units = trial.suggest_int('lstm_units', 32, 128)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    optimizer_choice = trial.suggest_categorical('optimizer', ['adam', 'nadam', 'adamw', 'rmsprop'])

    optimizer_dict = {
        'adam': Adam(learning_rate=0.001),
        'nadam': Nadam(learning_rate=0.001),
        'adamw': AdamW(learning_rate=0.001),
        'rmsprop': RMSprop(learning_rate=0.001)
    }

    inputs = Input(shape=(X_train.shape[1], 1))
    x = Conv1D(filters=filters, kernel_size=3, activation='swish')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout_rate)(x)
    x = LSTM(lstm_units, activation='swish', return_sequences=True)(x)
    attention_out = AdditiveAttention()([x, x])
    x = LSTM(lstm_units // 2, activation='swish')(attention_out)
    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(16, activation='swish')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer_dict[optimizer_choice], loss='binary_crossentropy', metrics=['accuracy'])
    return model

def objective(trial):
    model = create_cnn_lstm(trial)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    model.fit(X_train.reshape(-1, X_train.shape[1], 1), y_train, epochs=50, batch_size=32,
              validation_data=(X_test.reshape(-1, X_test.shape[1], 1), y_test),
              callbacks=[early_stopping, lr_scheduler], verbose=1)

    _, accuracy = model.evaluate(X_test.reshape(-1, X_test.shape[1], 1), y_test, verbose=0)
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
best_cnn_lstm = create_cnn_lstm(study.best_trial)
best_cnn_lstm.fit(X_train.reshape(-1, X_train.shape[1], 1), y_train, epochs=50, batch_size=32, verbose=1)

# Extract features from the CNN+LSTM model
X_train_encoded = best_cnn_lstm.predict(X_train.reshape(-1, X_train.shape[1], 1))
X_test_encoded = best_cnn_lstm.predict(X_test.reshape(-1, X_test.shape[1], 1))

# Define base models
rf = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42)
xgb = XGBClassifier(learning_rate=0.05, n_estimators=500, eval_metric='logloss', random_state=42)
svm = SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
lgbm = LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=42)
catboost = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, verbose=0, random_state=42)

# Stacking ensemble model with additional models
stacking_model = StackingClassifier(
    estimators=[('rf', rf), ('xgb', xgb), ('svm', svm), ('lgbm', lgbm), ('catboost', catboost)],
    final_estimator=XGBClassifier(eval_metric='logloss', random_state=42),
    stack_method='auto'
)

# Cross-validation for ensemble model
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_val_score(stacking_model, X_train_encoded, y_train, cv=cv, scoring='accuracy')
print(f"Cross-validation accuracy: {cv_results.mean()}")

# Train hybrid model
stacking_model.fit(X_train_encoded, y_train)

# Predictions
y_pred = stacking_model.predict(X_test_encoded)
y_pred_proba = stacking_model.predict_proba(X_test_encoded)[:, 1]

# Explainability with SHAP using KernelExplainer for the stacked model
explainer = shap.KernelExplainer(stacking_model.predict, X_train_encoded)
shap_values = explainer.shap_values(X_test_encoded)
shap.summary_plot(shap_values, X_test_encoded, feature_names=['CNN-LSTM Output'])

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))
