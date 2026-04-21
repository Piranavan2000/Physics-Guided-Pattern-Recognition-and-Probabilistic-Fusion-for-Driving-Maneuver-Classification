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
*   **Synthetic Domain (CARLA)**: High-resolution telemetry generated in CARLA Simulator (Town04).
*   **Naturalistic Domain (Real-World)**:
    *   [UAH-DriveSet](https://github.com/jair-jr/driverBehaviorDataset): Public naturalistic dataset.
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

### Step 1: Install CARLA Simulator

CARLA is the open-source simulator used to generate the synthetic driving telemetry dataset.

#### System Requirements
| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **OS** | Windows 10 (64-bit) | Windows 10/11 (64-bit) |
| **GPU** | NVIDIA GTX 1060 (6 GB) | NVIDIA RTX 2070+ |
| **RAM** | 8 GB | 16 GB+ |
| **Disk** | 20 GB free | SSD with 30 GB+ free |

#### Installation Steps

1.  **Download CARLA**: Visit [https://github.com/carla-simulator/carla/releases](https://github.com/carla-simulator/carla/releases) and download the latest stable release (e.g., `CARLA_0.9.15.zip`).

2.  **Extract**: Unzip the downloaded archive to a directory of your choice, e.g.:
    ```
    C:\CARLA\CARLA_0.9.15\
    ```

3.  **Install Python API**: Navigate to the CARLA Python API directory and install it:
    ```bash
    cd C:\CARLA\CARLA_0.9.15\PythonAPI\carla\dist
    pip install carla-0.9.15-cp39-cp39-win_amd64.whl
    ```
    > **Note**: Choose the `.whl` file that matches your Python version (e.g., `cp39` for Python 3.9, `cp310` for Python 3.10).

4.  **Verify Installation**:
    ```bash
    python -c "import carla; print(carla.__version__)"
    ```
    This should print the CARLA version without errors.

---

### Step 2: Generate the Synthetic Dataset with `carla.py`

The `carla.py` script connects to a running CARLA server, spawns a Tesla Model 3 ego-vehicle with a 6-DOF IMU sensor, and automatically executes five driving maneuvers to collect labeled telemetry data.

#### How to Run

1.  **Launch the CARLA server** by running `CarlaUE4.exe`:
    ```bash
    cd C:\CARLA\CARLA_0.9.15\
    CarlaUE4.exe
    ```
    Wait until the 3D world is fully loaded (you will see the simulator window).

2.  **Run the data generation script** in a separate terminal:
    ```bash
    python carla.py
    ```

#### What the Script Does

| Stage | Description |
| :--- | :--- |
| **Connection** | Connects to the CARLA server at `localhost:2000` |
| **Environment** | Sets synchronous mode at **20 Hz** sampling rate (`fixed_delta_seconds = 0.05`) |
| **Vehicle** | Spawns a Tesla Model 3 at spawn point 50 |
| **IMU Sensor** | Attaches a 6-DOF IMU at the vehicle's center of mass |
| **Noise Injection** | Adds spectral engine noise (5 Hz + 12 Hz harmonics) and Gaussian road jitter to simulate real smartphone IMU readings |
| **Maneuver Execution** | Performs 5 maneuvers sequentially: Normal Driving, Sudden Acceleration, Harsh Brake, Sharp Turn, and Sudden Lane Change |
| **Data Collection** | Collects **2,000 rows per class** (10,000 total) with columns: `timestamp`, `accel_x`, `accel_y`, `accel_z`, `gyro_x`, `gyro_y`, `gyro_z`, `speed`, `event_type` |
| **Output** | Saves the dataset as `carla_perfect_noisy_dataset.csv` |

#### Maneuver Parameters

| Maneuver | Throttle | Brake | Steer | Entry Speed |
| :--- | :--- | :--- | :--- | :--- |
| Normal Driving | 0.3 | 0.0 | 0.0 | 36 km/h |
| Sudden Acceleration | 1.0 | 0.0 | 0.0 | 7 km/h |
| Harsh Brake | 0.0 | 1.0 | 0.0 | 65 km/h |
| Sharp Turn | 0.3 | 0.0 | 0.9 | 36 km/h |
| Sudden Lane Change | 0.6 | 0.0 | ±0.6 | 50 km/h |

> **Expected Runtime**: ~10–15 minutes depending on hardware performance.

---

### Step 3: Run the Model Comparison Notebook

The `Model_Comparison_08_02_2026 copy.ipynb` Jupyter notebook implements the full model training, evaluation, and comparison pipeline.

#### How to Run

1.  **Install Jupyter** (if not already installed):
    ```bash
    pip install jupyter
    ```

2.  **Launch the notebook**:
    ```bash
    jupyter notebook "Model_Comparison_08_02_2026 copy.ipynb"
    ```

3.  **Execute all cells** sequentially (`Kernel → Restart & Run All`).

#### What the Notebook Does

| Stage | Description |
| :--- | :--- |
| **Data Loading** | Loads the generated CARLA dataset and naturalistic datasets (UAH-DriveSet, Mendeley) |
| **Preprocessing** | Applies Kalman filtering ($R=0.1, Q=0.1$) and sliding window ($N=30$, $S=5$) feature extraction to produce 210-length feature vectors |
| **Class Balancing** | Applies Domain-Stratified SMOTETomek to balance the training set |
| **Model Training** | Trains three base classifiers: Random Forest, XGBoost, and LightGBM |
| **Weighted Ensemble** | Combines predictions using soft voting weights: $w_{RF}=2, w_{XGB}=1, w_{LGBM}=1$ |
| **Physics Fusion** | Applies the Probabilistic Soft Fusion layer ($\alpha=0.1$) with Newtonian boundaries |
| **10-Run Stability** | Repeats the full pipeline across 10 random seeds to verify reproducibility ($\sigma = 0.0050$) |
| **External Validation** | Evaluates on the sequestered Mendeley Data Vault (40% holdout) |
| **Outputs** | Generates confusion matrices, ROC curves, reliability diagrams, alpha sensitivity plots, and the master performance table |

#### Key Results Produced

| Metric | Value |
| :--- | :--- |
| **Macro F1-Score (Mean ± Std)** | $0.7224 \pm 0.0050$ |
| **Macro Precision** | 0.8165 |
| **ECE (Calibration)** | 0.041 |
| **Brier Score** | 0.024 |
| **Inference Latency** | 0.06 ms |
| **McNemar's χ²** | 28.03 ($p < 0.001$) |


---

### Step 4: Process Real-World Data (UAH-DriveSet)

To validate the model on naturalistic data, use the UAH-DriveSet processor.

#### 1. Download the Dataset
Download or clone the driver behavior dataset from GitHub:
[https://github.com/jair-jr/driverBehaviorDataset](https://github.com/jair-jr/driverBehaviorDataset)

#### 2. Prepare the Directory Structure
Navigate to the `rel world data cpmbine/` directory and set up the raw data:
1. Create a folder named `raw_data`.
2. Extract the trip folders (e.g., `16`, `17`, `20`, `21`) from the downloaded dataset into `raw_data/`.

Your structure should look like this:
```
rel world data cpmbine/
├── raw_data/
│   ├── 16/
│   │   ├── acelerometro_terra.csv
│   │   ├── giroscopio_terra.csv
│   │   └── groundTruth.csv
│   ├── 17/
│   └── ...
└── realworld.ipynb
```

#### 3. Run the Processor
1. Open `rel world data cpmbine/realworld.ipynb` in Jupyter.
2. Execute all cells to process and synchronize the sensor data.
3. The script will generate `real_world_validation.csv` which is used by the main comparison notebook.
