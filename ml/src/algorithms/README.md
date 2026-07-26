# Ablation study: CNN-LSTM vs CNN-GRU vs CNN-Transformer

This folder adds three runnable variants that keep the existing SisFall preprocessing and CNN feature extractor unchanged, while swapping only the temporal model stage.

## 1) Run the baseline LSTM model (existing model through the new interface)
```bash
python ml/src/algorithms/run_algo.py --algo lstm
```
Expected output metrics: `ml/src/algorithms/results/lstm_metrics.json`

## 2) Run the GRU model standalone
```bash
python ml/src/algorithms/run_algo.py --algo gru
```
Expected output metrics: `ml/src/algorithms/results/gru_metrics.json`

## 3) Run the Transformer model standalone
```bash
python ml/src/algorithms/run_algo.py --algo transformer
```
Expected output metrics: `ml/src/algorithms/results/transformer_metrics.json`

## 4) Run all three and generate one comparison table
```bash
python ml/src/algorithms/compare_models.py
```
Expected outputs:
- Console comparison table
- `ml/src/algorithms/results/comparison_table.csv`

## 5) One-line summary of each model
- LSTM: baseline model using a two-layer LSTM over the CNN feature sequence.
- GRU: same CNN backbone and training setup, but swaps the LSTM for a GRU.
- Transformer: same CNN backbone and training setup, but replaces the LSTM with a small Transformer encoder plus positional encoding and mean pooling.

## 6) Notes
- All models use the same SisFall data split and preprocessing path from the existing training pipeline.
- Re-run with `--force` if you want to retrain and overwrite existing metrics JSON files.
