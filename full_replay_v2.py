#!/usr/bin/env python3
"""
Full Pipeline Replay v2 — K1+K2+K3+K4+XAI
macyste_dataset_v2.csv (normal + T0856 + T0814)
"""
import pandas as pd, numpy as np, pickle, json
import tensorflow as tf

DATASET = "/root/MaCySTe/grad_rl_data/macyste_dataset_v2.csv"
STATS   = "/root/MaCySTe/grad_rl_data/final_stats_v2.json"
XAI_LOG = "/root/MaCySTe/grad_rl_data/final_xai_v2.jsonl"

FEATURES = ['rate_of_turn','speed_knots','lat','lon','pump_rpm','flow_rate',
            'oil_temp','oil_pressure','heading_sin','heading_cos','rate_of_turn_rstd']

# ── K3 Maritime Topology ──────────────────────────────────────────────
# MaCySTe graph: Bridge(NMEA)→Control(Autopilot)→Field(SGCS PLCs)→SIEM
# Betweenness centrality pre-computed (Brandes, offline)
MITRE_MAP = {
    't0856_heading_180':'T0856','t0814_asterix_dos':'T0814','normal':'Normal'
}
# CVSS v4.0 base scores (journal Eq.4, ws=1.5)
CVSS_BASE = {'T0856':7.5, 'T0814':5.0, 'Normal':0.0}
# Betweenness centrality — MaCySTe topology
CB_NORM   = {'T0856':0.25, 'T0814':0.25, 'Normal':0.0}
# IEC 62443 SL mapping
SL_MAP = lambda r: "SL4" if r>=9 else "SL3" if r>=7 else "SL2" if r>=4 else "SL1"
# K3 propagation paths
K3_PATHS = {
    'T0856': {
        'entry_point': 'NMEA mux — Bridge network',
        'attack_vector': 'WebSocket inject_heading → BridgeCommand NMEA feed',
        'affected_zones': ['Bridge Zone','Navigation System'],
        'critical_node': 'ECDIS/OpenCPN heading display',
        'physical_consequence': 'Operator navigates on false heading; collision/grounding risk',
        'mitre_tactic': 'Inhibit Response Function',
        'cwe': 'CWE-290 Authentication Bypass by Spoofing',
    },
    'T0814': {
        'entry_point': 'ASTERIX multicast (239.0.1.2:8600)',
        'attack_vector': 'UDP flood → ASTERIX radar converter overload',
        'affected_zones': ['Bridge Zone','Radar System'],
        'critical_node': 'macyste-radar-converter-asterix',
        'physical_consequence': 'Radar picture lost; collision avoidance degraded',
        'mitre_tactic': 'Denial of Control',
        'cwe': 'CWE-400 Uncontrolled Resource Consumption',
    },
}
# K4 action mapping
K4_ACTIONS = {
    'T0856': 'ISOLATE_NAV_SYSTEM — isolate NMEA feed; switch to backup compass',
    'T0814': 'ISOLATE_NAV_SYSTEM — isolate radar subsystem; switch to AIS/ARPA backup',
    'Normal': 'DO_NOTHING',
}

print("Modeller yükleniyor...")
sc    = pickle.load(open('/root/models/scaler_macyste_v2.pkl','rb'))
model = tf.keras.models.load_model('/root/models/detective_lstm_macyste_v2.keras')
clf   = pickle.load(open('/root/models/detective_classifier_macyste_v2.pkl','rb'))
le    = pickle.load(open('/root/models/label_encoder_macyste_v2.pkl','rb'))
th_data = json.load(open('/root/models/threshold_macyste_v2.json'))
th    = th_data['threshold']
print(f"Threshold (P99×1.2): {th:.6f}")

df = pd.read_csv(DATASET)
import math as _math
if 'heading_sin' not in df.columns:
    df['heading_sin'] = df['heading'].apply(lambda h: _math.sin(_math.radians(float(h or 0))))
    df['heading_cos'] = df['heading'].apply(lambda h: _math.cos(_math.radians(float(h or 0))))
if 'rate_of_turn_rstd' not in df.columns:
    df['rate_of_turn_rstd'] = df['rate_of_turn'].rolling(10,min_periods=1).std().fillna(0)
if 'rate_of_turn_rstd' not in df.columns:
    df['rate_of_turn_rstd'] = df['rate_of_turn'].rolling(10,min_periods=1).std().fillna(0)

X = sc.transform(df[FEATURES].fillna(0))
labels = df['label'].values
seq_len = 10
total = len(df)

print(f"Dataset: {total} satır | Label: {dict(pd.Series(labels).value_counts())}")
print("K1 batch inference...")
seqs = np.array([X[i:i+seq_len] for i in range(total-seq_len)])
labels_seq = labels[seq_len:]
pred = model.predict(seqs, batch_size=512, verbose=1)
mse_all = np.mean(np.power(seqs - pred, 2), axis=(1,2))
feat_err_all = np.power(seqs[:,-1,:] - pred[:,-1,:], 2)

print("K2 batch inference...")
k2_raw = le.inverse_transform(clf.predict(X[seq_len:]))
k2_proba = clf.predict_proba(X[seq_len:])

print("K3+K4+XAI hesaplanıyor...")
tp=fp=tn=fn=0
attack_start={}; attack_detected={}; mttd={}
per_class = {l:{'total':0,'tp':0,'fp':0,'fn':0} for l in ['t0856_heading_180','t0814_asterix_dos','normal']}

with open(XAI_LOG,'w') as xai_f:
    for i in range(len(seqs)):
        label    = labels_seq[i]
        is_attack = label != 'normal'
        mse      = float(mse_all[i])
        is_anomaly = mse > th
        feat_err = feat_err_all[i]

        # K1 — culprit sensor
        culprit_idx  = int(np.argmax(feat_err))
        culprit_feat = FEATURES[culprit_idx] if culprit_idx < len(FEATURES) else 'unknown'
        top3 = sorted(enumerate(feat_err), key=lambda x: x[1], reverse=True)[:3]
        k1_top = [{'feature':FEATURES[j],'mse_contribution':round(float(v),6),
                   'pct':round(float(v)/max(float(mse),1e-9)*100,1)} for j,v in top3]

        # K2 — classification + confidence
        mitre_id = MITRE_MAP.get(k2_raw[i], k2_raw[i])
        proba    = k2_proba[i]
        conf     = round(float(max(proba))*100, 1)
        k2_uncertain = max(proba) < 0.70
        k2_probas = {c:round(float(p),4) for c,p in zip(le.classes_, proba)}

        # K3 — differentiable risk tensor R(v,t)
        i_base = CVSS_BASE.get(mitre_id, 0.0) if is_anomaly else 0.0
        cb     = CB_NORM.get(mitre_id, 0.0)
        alpha  = max(0.0, (mse-th)/th) if mse > th else 0.0
        risk   = round(min(i_base*(1+cb)*(1+alpha), 10.0), 2)
        sl     = SL_MAP(risk)
        k3     = K3_PATHS.get(mitre_id, {})
        requires_human = risk >= 9.0

        # K4 — action
        if requires_human:
            action = 'AWAITING_HUMAN_APPROVAL'
        elif is_anomaly:
            action = K4_ACTIONS.get(mitre_id, 'ISOLATE_NAV_SYSTEM')
        else:
            action = 'DO_NOTHING'

        # XAI summary
        if k2_uncertain and is_anomaly:
            xai_summary = f"⚠️ NOVEL/UNCERTAIN — K2 conf {conf}% < 70%. Graph fallback: CVSS {risk} {sl}"
        elif requires_human:
            xai_summary = f"⚠️ IEC 62443 {sl} — HUMAN APPROVAL: {mitre_id} CVSS {risk} conf {conf}%"
        elif is_anomaly:
            xai_summary = f"🚨 IEC 62443 {sl} — AUTO RESPONSE: {mitre_id} CVSS {risk} conf {conf}%"
        else:
            xai_summary = f"✅ Normal — MSE {mse:.4f} < τ {th:.4f}"

        # Confusion matrix
        if is_attack and is_anomaly:       tp+=1
        elif is_attack and not is_anomaly: fn+=1
        elif not is_attack and is_anomaly: fp+=1
        else:                              tn+=1

        # Per-class
        if label in per_class:
            per_class[label]['total']+=1
            if is_attack and is_anomaly: per_class[label]['tp']+=1
            elif is_attack:              per_class[label]['fn']+=1
            elif is_anomaly:             per_class[label]['fp']+=1

        # MTTD
        if is_attack and label not in attack_start:
            attack_start[label]=i
        if is_attack and is_anomaly and label not in attack_detected:
            attack_detected[label]=i
            mttd[label]=i-attack_start.get(label,i)

        # XAI log
        entry = {
            'row':i, 'timestamp':df['timestamp'].iloc[i+seq_len] if 'timestamp' in df else '',
            'label':label, 'mitre':mitre_id,
            'k1':{'is_anomaly':bool(is_anomaly),'mse':float(mse),'mse':round(mse,6),'threshold':th,
                  'culprit':culprit_feat,'top_features':k1_top},
            'k2':{'technique':mitre_id,'confidence_pct':conf,'uncertain':bool(k2_uncertain),
                  'all_probas':k2_probas,'mitre_tactic':k3.get('mitre_tactic','')},
            'k3':{'cvss':risk,'iec62443_sl':sl,'entry_point':k3.get('entry_point',''),
                  'attack_vector':k3.get('attack_vector',''),
                  'affected_zones':k3.get('affected_zones',[]),
                  'physical_consequence':k3.get('physical_consequence',''),
                  'cwe':k3.get('cwe',''),'requires_human':bool(requires_human)},
            'k4':{'action':action,'recommended':K4_ACTIONS.get(mitre_id,'')},
            'xai_summary':xai_summary,
        }
        xai_f.write(json.dumps(entry)+'\n')

        if i % 5000 == 0:
            print(f"[{i:6d}/{len(seqs)}] TP={tp} FP={fp} TN={tn} FN={fn}")

# ── Final metrics ─────────────────────────────────────────────────────
n = tp+fp+tn+fn
acc  = (tp+tn)/n
prec = tp/(tp+fp) if (tp+fp) else 0
rec  = tp/(tp+fn) if (tp+fn) else 0
f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0
fpr  = fp/(fp+tn) if (fp+tn) else 0
fnr  = fn/(fn+tp) if (fn+tp) else 0

# Per-class F1
per_class_f1 = {}
for lbl, m in per_class.items():
    if lbl == 'normal': continue
    p = m['tp']/(m['tp']+m.get('fp',0)) if (m['tp']+m.get('fp',0)) else 0
    r = m['tp']/(m['tp']+m['fn']) if (m['tp']+m['fn']) else 0
    per_class_f1[lbl] = round(2*p*r/(p+r) if (p+r) else 0, 4)

stats = {
    'dataset':'macyste_dataset_v2.csv','threshold':th,'seq_len':seq_len,
    'total_rows':n,'tp':tp,'fp':fp,'tn':tn,'fn':fn,
    'accuracy':round(acc,4),'precision':round(prec,4),
    'recall':round(rec,4),'f1':round(f1,4),
    'fpr':round(fpr,4),'fnr':round(fnr,4),
    'mttd_rows':mttd,'per_class_f1':per_class_f1,
    'per_class_detail':per_class,
}
with open(STATS,'w') as f:
    json.dump(stats, f, indent=2)

print()
print('='*55)
print('GRAD-RL MaCySTe Full Pipeline — Offline Evaluation')
print('='*55)
print(f'Accuracy:  {acc:.4f} ({acc*100:.2f}%)')
print(f'Precision: {prec:.4f}')
print(f'Recall:    {rec:.4f}')
print(f'F1 (macro):{f1:.4f}')
print(f'FPR:       {fpr:.4f} ({fpr*100:.2f}%)')
print(f'FNR:       {fnr:.4f} ({fnr*100:.2f}%)')
print(f'TP={tp} FP={fp} TN={tn} FN={fn}')
print()
print('Per-class F1:')
for k,v in per_class_f1.items():
    print(f'  {k}: {v}')
print()
print(f'MTTD (satır): {mttd}')
print(f'Stats: {STATS}')
print(f'XAI log: {XAI_LOG}')
