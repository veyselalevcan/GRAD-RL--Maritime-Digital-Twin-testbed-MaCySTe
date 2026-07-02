# GRAD-RL--Maritime-Digital-Twin-testbed-MaCySTe
GRAD-RL framework validation on MaCySTe maritime FSRU digital twin —  K1 LSTM-AE + K2 XGBoost + K3 Graph Risk + K4 PPO | MITRE ATT&amp;CK for ICS
# GRAD-RL MaCySTe Maritime Validation

Validation of the GRAD-RL autonomous cyber-physical defense framework 
on MaCySTe containerized FSRU maritime digital twin.

## Framework
K1 LSTM-AE → K2 XGBoost → K3 Graph Risk Tensor → K4 PPO → IEC 62443 Gate

## Key Results
| Metric | Value |
|--------|-------|
| Accuracy | 98.28% |
| FPR | 0.88% |
| T0856 F1 | 0.6424 |
| T0814 F1 | 0.6607 |
| MTTD T0856 | 1s |
| MTTD T0814 | 0s |
| LSTM-AE val_loss | 0.000277 |
| SL4 detections | 57/63 (90.5%) |

## Dataset
Zenodo: [10.5281/zenodo.21130212]

## Quick Start
pip install -r requirements.txt
uvicorn api.api_detective_macyste:app --port 8000
python collector/macyste_collector.py --label normal --duration 300 --output data.csv

## Attack Scenarios
python attacks/attack_t0856_heading_drift.py   # T0856
python attacks/attack_t0814_asterix_highfreq.py # T0814
python attacks/attack_t0831_nats_live.py        # T0831

## Citation
