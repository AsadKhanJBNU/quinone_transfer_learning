# Quinone Transfer Learning

Transfer learning for **quinone molecules**: fine-tune a pretrained graph neural network (GNN) on a small set of quinone derivatives from QM9 to predict **HOMO energy (eV)**.

## Overview

- **Goal:** Predict HOMO (highest occupied molecular orbital) energy in eV for quinone-containing molecules.
- **Approach:** A pretrained RGNN model (trained on a larger molecular dataset for HOMO) is loaded and fine-tuned on 30 quinone molecules filtered from the QM9 dataset, using progressive unfreezing and Optuna-tuned hyperparameters.
- **Data:** Quinones are identified in QM9 via the SMARTS pattern `C(=O)C=CC(=O)` (quinone core). Labels (HOMO, LUMO, bandgap) come from QM9’s DFT-derived properties.

## Project Structure

```
quinone_transfer_learning/
├── README.md
├── quinone_molecules.csv          # 30 quinones (smiles, homo_eV, lumo_eV, bandgap_eV)
├── download_quinone_data.ipynb    # Build quinone set from QM9 → CSV
├── quinone_optuna_tuning.ipynb    # Hyperparameter search (dropout, lr, unfreeze, etc.)
├── train_final.ipynb              # Load pretrained model, fine-tune, evaluate, save results
├── Data_Prep/
│   ├── Graph_Data.py              # Molecule_data: SMILES → PyG graphs, caching
│   └── datacreator.py             # Train/test split + graph dataset creation
├── models/
│   └── GRNNModel.py               # RGNNPredictor (GNN + GRU, molecule-level readout)
├── data/                          # Created at runtime
│   └── processed_dataset/         # Cached train/test .pt graph files
├── results/                       # Training history, predictions, metrics, plots
└── saved_models/                  # Put pretrained weights here (modelHomo.model)
```

## Workflow

1. **Get quinone data**  
   Run `download_quinone_data.ipynb` to download QM9, filter quinones with the SMARTS pattern, and save `quinone_molecules.csv`.

2. **Tune hyperparameters (optional)**  
   Run `quinone_optuna_tuning.ipynb` to search over dropout, learning rate, weight decay, and number of unfrozen layers. Best parameters are then used in `train_final.ipynb`.

3. **Fine-tune and evaluate**  
   Run `train_final.ipynb`: it loads the pretrained checkpoint from `saved_models/modelHomo.model`, builds train/val/test loaders from the processed graphs, fine-tunes with the chosen config, and writes metrics and plots to `results/`.

## Data Pipeline

- **Graph_Data.py:** Converts SMILES to PyTorch Geometric graphs via RDKit; caches them under `data/<savepath>/processed/` as `train_data_set.pt` and `test_data_set.pt`. Molecules with no bonds are skipped and listed in `incorrect_smiles.csv`.
- **datacreator.py:** Splits the dataframe 80/20 (train/test), then builds `Molecule_data` for each split. The training notebook further splits the test set into validation and final test.

## Model: RGNNPredictor

- **File:** `models/GRNNModel.py`
- **Architecture:** Atom features → linear → ResGatedGraphConv + GRU layers (with BatchNorm and dropout) → molecule-level aggregation with virtual nodes, GATConv, and multi-step GRU → final linear layer → scalar HOMO prediction.
- **Transfer learning:** Progressive unfreezing: only the last `unfreeze_layers` atom blocks plus the molecule-level layers and output head are trained; earlier layers stay frozen.

## Dependencies

- Python 3 
- PyTorch, PyTorch Geometric 
- RDKit
- pandas, scikit-learn, numpy
- Optuna
- matplotlib

Install with conda/pip as needed for PyTorch, PyG, and RDKit in your environment.

## Results

After running `train_final.ipynb`, results are saved in `results/`:

- `training_history.csv` — per-epoch train loss, validation RMSE/R²/MAE
- `train_predictions.csv`, `test_predictions.csv` — true vs predicted HOMO (eV)
- `test_metrics.json` — train/test RMSE, R², MAE and best hyperparameters
- `scatter_*.png` — scatter plots of true vs predicted values

With only 30 quinone samples and a small test set, the model can fit the training data very well while test metrics may be unstable; consider collecting more quinone data or using cross-validation for more reliable estimates.
