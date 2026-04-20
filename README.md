# Physics-Guided Pattern Recognition and Probabilistic Fusion for Driving Maneuver Classification

[![Research Status](https://img.shields.io/badge/Research-Finalized-success.svg)](https://github.com/Piranavan2000/Physics-Guided-Pattern-Recognition)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Ensemble-RF_XGB_LGBM-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Bridging the Safety-Accuracy Gap in Autonomous Driving through Newtonian Physics and Probabilistic Soft Fusion.**

This repository contains the implementation of a **Physics-Constrained Post-Filtering** framework designed to resolve "AI hallucinations" in maneuver classification. By integrating Newtonian physical constraints at inference time as a gentle probabilistic prior, the system achieves a 71% reduction in Expected Calibration Error while maintaining high-speed edge compatibility.

---

## 🌟 Key Features

*   **Probabilistic Soft Fusion Layer**: Integrates Newtonian constraints ($\alpha = 0.1$) to anchor AI predictions in physical reality.
*   **Edge-Optimized Ensemble**: Hybrid architecture combining Random Forest, XGBoost, and LightGBM for robust time-series classification.
*   **Multi-Domain Validation**: Trained on 91,000+ samples from CARLA (Simulation) and validated on 12,000+ samples from Naturalistic datasets (UAH-DriveSet, Mendeley).
*   **Real-Time Performance**: Average inference latency of **0.06ms**, providing an 833x safety margin for 20Hz telemetry systems.
*   **Probabilistic Honesty**: ECE reduction from **0.142 to 0.041**, ensuring that prediction confidence aligns with empirical frequency.

---

## 🏗️ System Architecture

The framework transitions from raw high-frequency (100Hz) IMU telemetry to kinematically-verified maneuver alerts via a dual-path pipeline:
1.  **Denoising**: Recursive Kalman filtering and 30-step sliding windows.
2.  **Feature Vector**: Extraction of a 210-length signature vector ($f_{210}$) for the AI ensemble.
3.  **Kinematic Prior**: Analysis of Newtonian boundaries ($F=ma$, Centripetal Force).
4.  **Soft Fusion**: Merging of weighted ensemble probabilities with kinematic priors.

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.9+ installed and the following libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn scipy joblib
```

### Running the Core Pipeline
The project logic was developed in a Jupyter environment and exported to a production-ready Python script. 

**1. Generate/Update the Script** (Optional):
The core logic resides in `Model_Comparison_08_02_2026 copy.ipynb`. To generate a fresh script version:
```bash
jupyter nbconvert --to script "Model_Comparison_08_02_2026 copy.ipynb" --output run_model.py
```

**2. Run Evaluation & Benchmarking**:
Load the script and execute the full training, validation, and benchmarking suite:
```bash
python run_model.py
```

This will:
*   Preprocess the CARLA and Real-World datasets.
*   Apply Domain-Stratified SMOTETomek balancing.
*   Train the Weighted Voting Ensemble (RF, XGB, LGBM).
*   Apply the Physics Soft Fusion layer.
*   Generate performance plots and evaluation tables in the `Results_08_02_2026/` directory.

---

## 📊 Research Results

| Metric | Base Ensemble | Hybrid (Physics-Guided) | Improvement |
| :--- | :--- | :--- | :--- |
| **Macro F1-Score** | 0.6789 | **0.7224** | **+4.35%** |
| **Expected Calibration Error (ECE)** | 0.142 | **0.041** | **-71.1%** |
| **Maneuver Precision (External)** | ~81% | **93.0%** | **+12.0%** |
| **Inference Latency** | 0.05ms | **0.06ms** | Negligible Overhead |

### Statistical Significance
Applying **McNemar's Test** confirmed the improvement is statistically significant:
*   **Chi-Square**: 28.03
*   **P-Value**: $< 0.001$

---

## 📂 Project Structure

*   `Model_Comparison_08_02_2026 copy.ipynb`: Main research and development notebook.
*   `run_model.py`: Production-ready script generated from the notebook.
*   `journal/journal.tex`: LaTeX source for the manuscript.
*   `Results_08_02_2026/`: Visualizations, confusion matrices, and ROC curves.
*   `Town01_CITY_STUNT_...csv`: Synthetic CARLA telemetry data.

---

## 📝 Citation

If you use this research or code in your work, please cite the corresponding paper:

```latex
@article{sathiyavannan2026physics,
  title={Physics-Guided Pattern Recognition and Probabilistic Fusion for Driving Maneuver Classification},
  author={Sathiyavannan, Piranavan and Kanagasabai, Thiruthanigesan and Abeywardhana, Lakmini},
  journal={Department of Information Technology, Sri Lanka Institute of Information Technology},
  year={2026}
}
```

---
*Developed as part of Final Year Research at SLIIT Computing.*
