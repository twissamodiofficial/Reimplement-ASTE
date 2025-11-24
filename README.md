# ASTE: Aspect Sentiment Triplet Extraction

Re-implementation of **"Knowing What, How and Why: A Near Complete Solution for Aspect-Based Sentiment Analysis"** (AAAI 2020).

This implementation achieves competitive results on the SemEval datasets through a two-stage approach:
- **Stage 1**: Multi-task learning for aspect/opinion extraction and sentiment classification
- **Stage 2**: Aspect-opinion pairing with correct baseline training

## 📊 Results

| Dataset | Aspect F1 | Opinion F1 | Sentiment Acc | Pair F1 | Triplet F1 | Baseline |
|---------|-----------|------------|---------------|---------|------------|----------|
| 14res   | 0.8388    | 0.8065     | 0.8632        | 0.5323  | **0.4639** | 0.5189         |
| 14lap   | 0.7929    | 0.7054     | 0.7429        | 0.4692  | **0.3385** | 0.4350         |
| 15res   | 0.7970    | 0.7742     | 0.7976        | 0.4891  | **0.4130** | 0.4679         |
| 16res   | 0.8075    | 0.7995     | 0.8725        | 0.4595  | **0.4324** | 0.5362         |

## 🚀 Quick Start

### Complete Setup (One Command)

```bash
#!/bin/bash

# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_lg

# Download GloVe embeddings (840B version - recommended)
mkdir -p embeddings
cd embeddings
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
cd ..

# Create output directories
mkdir -p models/checkpoints models/plots logs

echo "Setup complete! Ready to train."
```

**Or step-by-step:**

### 1. Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Download Required Models & Embeddings

```bash
# Download spaCy model (required for dependency parsing)
python -m spacy download en_core_web_lg

# Download GloVe embeddings
mkdir -p embeddings
cd embeddings

# Option A: Full GloVe 840B (recommended, 2.2 GB)
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip

# Option B: Smaller GloVe 6B (822 MB) - uncomment if preferred
# wget http://nlp.stanford.edu/data/glove.6B.zip
# unzip glove.6B.zip
# rm glove.6B.zip

cd ..
```

The code will automatically use `glove.840B.300d.txt` if available, otherwise falls back to `glove.6B.300d.txt`.

### 3. Data Setup and Prepare Data Directories

```bash
# Create output directories
mkdir -p models/checkpoints models/plots logs
```

The SemEval datasets should be in `data/` directory:
```bash
git clone https://github.com/xuuuluuu/SemEval-Triplet-data.git
```
Navigate to the ASTE-Data-V1-AAAI2020 folder and set it up in the following format

```
data/
├── 14res/
│   ├── train.txt
│   ├── dev.txt
│   └── test.txt
├── 14lap/
│   ├── train.txt
│   ├── dev.txt
│   └── test.txt
├── 15res/
│   ├── 15rest_train.txt
│   ├── 15rest_dev.txt
│   └── 15rest_test.txt
└── 16res/
    ├── 16rest_train.txt
    ├── 16rest_dev.txt
    └── 16rest_test.txt
```

### 4. Train & Evaluate

```bash
# Full pipeline: train, evaluate, and generate plots
./run_local.sh 14res

# Or run individual commands
python train_aste.py --dataset 14res --batch_size 16 --num_epochs 40
python evaluate_aste.py --dataset 14res --model_dir models
python generate_plots_from_results.py --dataset 14res

# Train on other datasets
./run_local.sh 14lap
./run_local.sh 15res
./run_local.sh 16res
```

**Expected Results on Local Machine (CPU):**
- 14res: Triplet F1 ~ 53.2%
- Training time: ~3-4 minutes per epoch on CPU
- Both Stage 1 and Stage 2 should complete successfully

## 📁 Project Structure

```
ASTE_Reattempt/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── run_local.sh                       # Local training script
├── workflow.sh                        # SLURM batch script
│
├── Core Model Files:
├── aste_model.py                      # Stage 1 & 2 model architectures
├── train_aste.py                      # Training pipeline (both stages)
├── evaluate_aste.py                   # Evaluation script
├── aste_data_loader.py                # Data loading utilities
├── data_prep.py                       # Data preprocessing
│
├── Utilities:
├── generate_plots_from_results.py     # Visualization generation
├── clean_glove.py                     # GloVe preprocessing
│
├── Data & Embeddings:
├── data/                              # SemEval datasets
│   ├── 14res/, 14lap/, 15res/, 16res/
│   └── README.md
├── embeddings/                        # GloVe embeddings
│   ├── glove.840B.300d.txt           # Primary (2.2 GB)
│   └── glove.6B.300d.txt             # Fallback (822 MB)
│
└── Output:
    ├── models/                        # Saved models & results
    │   ├── {dataset}_evaluation_results.json
    │   ├── {dataset}_training_metrics.json
    │   ├── best_model_{dataset}_complete.pt
    │   ├── plots/                     # All visualizations
    │   │   ├── 14res/
    │   │   ├── 14lap/
    │   │   ├── 15res/
    │   │   └── 16res/
    │   └── {dataset}_tensorboard/     # TensorBoard logs
    └── logs/                          # SLURM logs
```

## 🔧 Requirements

### Python Dependencies
```
torch>=1.9.0
numpy>=1.19.0
spacy>=3.0.0
scikit-learn>=0.24.0
tqdm>=4.60.0
matplotlib>=3.3.0
seaborn>=0.11.0
tensorboard>=2.7.0
```

### Additional Downloads

1. **spaCy Large Model** (REQUIRED)
   ```bash
   python -m spacy download en_core_web_lg
   ```
   Size: ~560 MB

2. **GloVe Embeddings** (REQUIRED)
   - **Recommended**: GloVe 840B 300d (2.2 GB)
     ```bash
     wget http://nlp.stanford.edu/data/glove.840B.300d.zip
     ```
   - **Alternative**: GloVe 6B 300d (822 MB)
     ```bash
     wget http://nlp.stanford.edu/data/glove.6B.zip
     ```

## 📖 Usage

### Training

#### Local Training
```bash
# Train on 14res dataset
./run_local.sh 14res

# Train on other datasets
./run_local.sh 14lap
./run_local.sh 15res
./run_local.sh 16res
```

#### SLURM Cluster
```bash
# Train single dataset
sbatch workflow.sh 0  # 0=14res, 1=14lap, 2=15res, 3=16res

# Train all datasets
sbatch workflow.sh all
```

#### Manual Training
```bash
python train_aste.py \
    --dataset 14res \
    --batch_size 16 \
    --num_epochs 40 \
    --learning_rate 0.1 \
    --dropout_rate 0.5 \
    --hidden_size 300
```

### Evaluation

```bash
# Evaluate trained model
python evaluate_aste.py \
    --dataset 14res \
    --model_dir models \
    --batch_size 16
```

Results are saved to `models/{dataset}_evaluation_results.json`:
```json
{
  "aspect_f1": 0.8388,
  "opinion_f1": 0.8065,
  "sentiment_accuracy": 0.8632,
  "pair_f1": 0.5323,
  "triplet_f1": 0.4639
}
```

### Visualization

```bash
# Generate plots from evaluation results
python generate_plots_from_results.py --dataset 14res
```

Generates 5 comprehensive visualizations:
1. **Component Analysis** - Performance of each component
2. **Performance Overview** - Training progress & baseline comparison
3. **Stage One Metrics** - Aspect/opinion/sentiment training curves
4. **Stage Two Metrics** - Pairing performance & threshold optimization
5. **Training Losses** - Loss curves for both stages

Outputs saved to `models/plots/{dataset}/`

## 🏗️ Model Architecture

### Stage 1: Multi-task Learning
- **BiLSTM encoder** with GloVe embeddings
- **Three parallel heads**:
  1. Aspect extraction (BIOES tagging)
  2. Opinion extraction (BIOES tagging)
  3. Sentiment classification (per-token)
- **Hidden size**: 300
- **Dropout**: 0.5

### Stage 2: Aspect-Opinion Pairing
- **Input**: Aspect and opinion spans from Stage 1 predictions
- **Architecture**: Span-based pairing classifier
- **Training**: Uses ground-truth pairs (baseline approach)
- **Inference**: Pairs Stage 1 predicted spans
- **Threshold optimization**: Grid search on validation set

## 📝 Training Details

### Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch size | 16 | Training batch size |
| Learning rate | 0.1 | Initial learning rate (SGD) |
| Optimizer | SGD | Stochastic Gradient Descent |
| Dropout | 0.5 | Dropout rate |
| Weight decay | 0.001 | L2 regularization |
| Hidden size | 300 | LSTM hidden dimensions |
| Embedding dim | 300 | GloVe embedding size |
| Stage 1 epochs | 40 | Multi-task learning epochs |
| Stage 2 epochs | 8 | Pairing classifier epochs |

### Training Pipeline
1. **Stage 1** (40 epochs)
   - Multi-task learning for aspect/opinion/sentiment
   - Evaluation every 10 epochs
   - Early stopping based on validation F1
   
2. **Threshold Optimization**
   - Grid search from 0.2 to 0.7
   - Optimized for recall-weighted score
   - Typical optimal: 0.25
   
3. **Stage 2** (8 epochs)
   - Pairs ground-truth aspects with opinions
   - Trains binary classifier
   - Uses optimal threshold from step 2

## 📊 Output Files

### Training Outputs
- `models/best_model_{dataset}_complete.pt` - Complete model (both stages)
- `models/{dataset}_training_metrics.json` - Training curves & metrics
- `models/{dataset}_tensorboard/` - TensorBoard logs

### Evaluation Outputs
- `models/{dataset}_evaluation_results.json` - Test set metrics
- `models/plots/{dataset}/` - All visualizations (PNG)

### Plots Generated
Each dataset gets 5 plots (PNG):
- `{dataset}_component_analysis.*` - Component performance bars
- `{dataset}_performance_overview.*` - Overall progress
- `{dataset}_stage_one_metrics.*` - Stage 1 training details
- `{dataset}_stage_two_metrics.*` - Stage 2 pairing analysis
- `{dataset}_training_losses.*` - Loss curves

## 🐛 Troubleshooting

### "en_core_web_lg not found"
```bash
python -m spacy download en_core_web_lg
```

### "GloVe embeddings not found"
```bash
cd embeddings
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
```

### "CUDA out of memory"
Reduce batch size:
```bash
python train_aste.py --dataset 14res --batch_size 8
```

### Plots show 0.000 for metrics
Regenerate plots from evaluation results:
```bash
python generate_plots_from_results.py --dataset 14res
```

### No evaluation results
Run evaluation first:
```bash
python evaluate_aste.py --dataset 14res --model_dir models
```

## 🔍 Key Features

### ✅ Baseline Implementation
- Stage 2 trains on ground-truth pairs (as per baseline approach)
- Stage 2 inference uses Stage 1 predictions
- Exact evaluation protocol from original paper

### ✅ Correct Metrics
- Uses actual evaluation results for visualization
- Proper pair F1 calculation
- All component metrics tracked

### ✅ Comprehensive Visualization
- 5 plot types per dataset
- Training curves and final metrics
- Paper comparison benchmarks

### ✅ Modular Design
- Separate training, evaluation, and visualization
- Can regenerate plots without retraining
- Clean code separation

## 📚 Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{peng2020knowing,
  title={Knowing What, How and Why: A Near Complete Solution for Aspect-Based Sentiment Analysis},
  author={Peng, Haiyun and Xu, Lu and Bing, Lidong and Huang, Fei and Lu, Wei and Si, Luo},
  booktitle={AAAI},
  year={2020}
}
```

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the repository maintainer.

## 📄 License

This project is provided as-is for research purposes.

---

**Note**: This implementation is designed for research and educational purposes. Results may vary based on random seeds, hardware, and exact environment setup.
