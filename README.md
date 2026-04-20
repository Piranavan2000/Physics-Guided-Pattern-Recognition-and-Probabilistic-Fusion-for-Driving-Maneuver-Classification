# Physics-Guided Pattern Recognition and Probabilistic Fusion for Driving Maneuver Classification

[![Research Status](https://img.shields.io/badge/Research-Finalized-success.svg)](https://github.com/Piranavan2000/Physics-Guided-Pattern-Recognition)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Ensemble-RF_XGB_LGBM-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Bridging the Safety-Accuracy Gap in Autonomous Driving through Newtonian Physics and Probabilistic Soft Fusion.**

This repository serves as a **Research Toolkit** for the replicable analysis of driving maneuvers. It implements a **Physics-Constrained Post-Filtering** framework that anchors AI predictions in physical reality, resolving "AI hallucinations" in maneuver classification.

---

## 🌟 Research Tiers

### 1. Environment & Reproducibility
To ensure exact replication of results, all experiments were conducted in a standardized environment (Python 3.12). 
*   **Requirements**: Install the exact dependencies via `pip install -r requirements.txt`.
*   **Environment Gap Fix**: The `requirements.txt` file explicitly defines versions for `scikit-learn (1.7.2)`, `xgboost (3.1.3)`, and `lightgbm (4.6.0)` used to obtain the journal's reported Macro F1-score of 0.7224.

### 2. Data Acquisition & Acquisition
The framework uses a multi-domain dataset strategy:
*   **Synthetic Domain (CARLA)**: High-resolution telemetry generated in CARLA Simulator (Town01).
*   **Naturalistic Domain (Real-World)**:
    *   [UAH-DriveSet](http://www.robesafe.es/personal/eduardo.romera/uah-driveset/): Public naturalistic dataset.
    *   [Mendeley Driving Dataset](https://data.mendeley.com/datasets/hb74y9vk88/1): Smartphone-based IMU telemetry.

### 3. Preprocessing & Denoising
We implement a **Recursive Kalman Filtering** pipeline ($R=0.1, Q=0.1$) to handle high-frequency sensor noise. 
*   **Module**: `preprocessing.py`
*   **Sliding Window**: $N=30$ samples (1.5 seconds) with a step $S=5$.
*   **Signature Mapping**: Each window is flattened into a 210-length feature vector ($30 \times 7$ sensor channels).

### 4. Vaulted Partitioning Strategy
To prevent data leakage and the Sim-to-Real gap, we employ a **60/30/10 training/validation/consistency split** enhanced by a **40% Sequestered Data Vault** (External Holdout).
*   **Script**: `data_prep.py`
*   **Logic**: The 40% real-world holdout is never seen by the SMOTETomek balancing or the training loop, ensuring empirical validity.

### 5. Probabilistic Soft Fusion Layer
The core contribution is the integration of Newtonian limits as a gentle probabilistic prior ($ \alpha = 0.1 $):
$$ P_{new} = (1 - \alpha) P_{model} + \alpha P_{physics} $$

**Newtonian Boundaries**:
*   **Harsh Brake**: $\min(a_x) < -3.0 m/s^2$
*   **Sharp Turn**: $\max(|a_y|) > 4.0 m/s^2$
*   **Sudden Acceleration**: $\max(a_x) > 2.5 m/s^2$

---

## 📊 Performance Portfolio

| Metric | Base Ensemble | Hybrid (Physics-Guided) | Improvement |
| :--- | :--- | :--- | :--- |
| **Macro F1-Score** | 0.6789 | **0.7224** | **+4.35%** |
| **Expected Calibration Error (ECE)** | 0.142 | **0.041** | **-71.1%** |
| **Inference Latency** | 0.05ms | **0.06ms** | Negligible |

---

## 🚀 Execution Guide

1.  **Prepare Environment**: `pip install -r requirements.txt`
2.  **Partition Data**: `python data_prep.py` (Sequesters the 40% Vault).
3.  **Run Pipeline**: `python run_model.py` (Training + Fusion + Evaluation).

---

## 📝 Citation

```latex
@article{sathiyavannan2026physics,
  title={Physics-Guided Pattern Recognition and Probabilistic Fusion for Driving Maneuver Classification},
  author={Sathiyavannan, Piranavan and Kanagasabai, Thiruthanigesan and Abeywardhana, Lakmini},
  journal={Department of Information Technology, Sri Lanka Institute of Information Technology},
  year={2026}
}
```
