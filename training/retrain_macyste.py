#!/usr/bin/env python3
"""
retrain_macyste.py — Tek komutla tam pipeline retrain
Kullanım: python3 retrain_macyste.py --dataset <csv_path>
"""
import argparse, pandas as pd, numpy as np, pickle, json
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras

FEATURES = ['rate_of_turn','speed_knots','lat','lon','pump_rpm','flow_rate',
            'oil_temp','oil_pressure','heading_sin','heading_cos','rate_of_turn_rstd']
SEQ_LEN = 10
MODEL_DIR = '/root/models'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--suffix', default='live')
args = parser.parse_args()

print(f"Dataset: {args.dataset}")
df = pd.read_csv(args.dataset)
if 'rate_of_turn_rstd' not in df.columns:
    df['rate_of_turn_rstd'] = df['rate_of_turn'].rolling(10,min_periods=1).std().fillna(0)
if 'heading' in df.columns and 'heading_sin' not in df.columns:
    import numpy as np
    df['heading_sin'] = np.sin(np.radians(df['heading']))
    df['heading_cos'] = np.cos(np.radians(df['heading']))

print("Label dağılımı:")
print(df['label'].value_counts())

# 1. Scaler — sadece normal veri
normal = df[df['label']=='normal'][FEATURES].fillna(0)
sc = MinMaxScaler()
sc.fit(normal)
scaler_path = f'{MODEL_DIR}/scaler_macyste_{args.suffix}.pkl'
pickle.dump(sc, open(scaler_path,'wb'), protocol=4)
print(f"Scaler kaydedildi: {scaler_path}")

X = sc.transform(df[FEATURES].fillna(0))
labels = df['label'].values

# 2. LSTM-AE — normal veri üzerinde train
X_normal = sc.transform(normal)
seqs_normal = np.array([X_normal[i:i+SEQ_LEN] for i in range(len(X_normal)-SEQ_LEN)])
n = len(seqs_normal)
X_tr, X_val = seqs_normal[:int(n*0.8)], seqs_normal[int(n*0.8):]

inp = keras.Input(shape=(SEQ_LEN, len(FEATURES)), name='encoder_input')
x = keras.layers.LSTM(64, return_sequences=True, name='enc_lstm1')(inp)
x = keras.layers.LSTM(32, return_sequences=False, name='enc_lstm2')(x)
x = keras.layers.RepeatVector(SEQ_LEN, name='bottleneck')(x)
x = keras.layers.LSTM(32, return_sequences=True, name='dec_lstm1')(x)
x = keras.layers.LSTM(64, return_sequences=True, name='dec_lstm2')(x)
out = keras.layers.TimeDistributed(keras.layers.Dense(len(FEATURES)), name='reconstruction')(x)
model = keras.Model(inp, out, name=f'LSTM_AE_MaCySTe_{args.suffix}')
model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')

cb = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss')]
print("LSTM-AE training...")
history = model.fit(X_tr, X_tr, validation_data=(X_val,X_val),
                    epochs=50, batch_size=32, callbacks=cb, verbose=1)

pred_val = model.predict(X_val, batch_size=256, verbose=0)
mse_val = np.mean(np.power(X_val-pred_val, 2), axis=(1,2))
p99 = float(np.percentile(mse_val, 99))
tau = round(p99*1.2, 8)
print(f"Best val_loss: {min(history.history['val_loss']):.6f}")
print(f"P99={p99:.6f} tau_live={tau:.6f}")

lstm_path = f'{MODEL_DIR}/detective_lstm_macyste_{args.suffix}.keras'
model.save(lstm_path)
th_path = f'{MODEL_DIR}/threshold_macyste_{args.suffix}.json'
with open(th_path,'w') as f:
    json.dump({'threshold':tau,'p99':p99,'mean':float(mse_val.mean()),'std':float(mse_val.std())},f)
print(f"LSTM-AE kaydedildi: {lstm_path}")

# 3. XGBoost — tüm veri
le = LabelEncoder()
y = le.fit_transform(labels[SEQ_LEN:])
print(f"XGBoost classes: {le.classes_}")

X_k2 = X[SEQ_LEN:]
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_k2, y, test_size=0.2, random_state=42, stratify=y)

clf = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                    random_state=42, eval_metric='mlogloss', verbosity=0)
clf.fit(X_tr2, y_tr2)
print(classification_report(y_te2, clf.predict(X_te2), target_names=le.classes_))

clf_path = f'{MODEL_DIR}/detective_classifier_macyste_{args.suffix}.pkl'
le_path  = f'{MODEL_DIR}/label_encoder_macyste_{args.suffix}.pkl'
pickle.dump(clf, open(clf_path,'wb'), protocol=4)
pickle.dump(le,  open(le_path,'wb'),  protocol=4)
print(f"XGBoost kaydedildi: {clf_path}")
print(f"\n=== Tüm modeller hazır (suffix={args.suffix}) ===")
