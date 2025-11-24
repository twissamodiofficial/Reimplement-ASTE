#!/bin/bash
#SBATCH --job-name=aste_workflow
#SBATCH --output=logs/workflow_%j.out
#SBATCH --error=logs/workflow_%j.err
#SBATCH --partition=MGPU-TC2           # REQUIRED - specify partition
#SBATCH --qos=normal                   # REQUIRED - specify QoS
#SBATCH --nodes=1                      # REQUIRED
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=06:00:00


module load anaconda/25.5.1
module load cuda/12.8.0

source ~/.bashrc
conda activate aste

python -c "import spacy; spacy.load('en_core_web_lg')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "en_core_web_lg not found, installing..."
    python -m spacy download en_core_web_lg
fi

echo "=== ASTE Complete Training & Evaluation Workflow ==="
echo " baseline-Compliant Implementation: Stage Two trains on Stage One predictions"
echo " Target: 51.89% triplet F1 on 14res (vs previous 38.46%)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU Info: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

SCRIPT_DIR="/home/msai/twissa001/DL/Idea-3"
cd "$SCRIPT_DIR" || exit 1
echo "Working directory: $(pwd)"

mkdir -p logs models/checkpoints models/plots models/tensorboard

echo "Setting up dataset symlinks..."
if [ -d "data/15res" ]; then
    cd data/15res
    ln -sf 15rest_train.txt train.txt 2>/dev/null || true
    ln -sf 15rest_test.txt test.txt 2>/dev/null || true  
    ln -sf 15rest_dev.txt dev.txt 2>/dev/null || true
    echo " 15res symlinks created"
    cd ../..
fi

if [ -d "data/16res" ]; then
    cd data/16res
    ln -sf 16rest_train.txt train.txt 2>/dev/null || true
    ln -sf 16rest_test.txt test.txt 2>/dev/null || true
    ln -sf 16rest_dev.txt dev.txt 2>/dev/null || true  
    echo " 16res symlinks created"
    cd ../..
fi

declare -a DATASETS=("14res" "14lap" "15res" "16res") 
declare -a DATASET_NAMES=("Restaurant 2014" "Laptop 2014" "Restaurant 2015" "Restaurant 2016")

DATASET_ARG=${1:-${DATASET:-all}}
echo " Dataset mode: '$DATASET_ARG'"
echo ""

if [ "$DATASET_ARG" = "all" ]; then
    echo "Training and evaluating all datasets..."
    for i in 0 1 2 3; do
        DATASET=${DATASETS[i]}
        DATASET_NAME=${DATASET_NAMES[i]}
        
        echo ""
        echo "=== TRAINING: $DATASET ($DATASET_NAME) ==="
        
        python train_aste.py \
            --dataset $DATASET \
            --data_dir "$SCRIPT_DIR/data" \
            --output_dir "$SCRIPT_DIR/models" \
            --batch_size 16 \
            --num_epochs 40 \
            --learning_rate 0.1 \
            --dropout_rate 0.5 \
            --weight_decay 0.001 \
            --hidden_size 300
        
        if [ $? -eq 0 ]; then
            echo " Base training completed for $DATASET"
            
            echo ""
            echo "=== EVALUATION: $DATASET ($DATASET_NAME) ==="
            
            python evaluate_aste.py \
                --dataset $DATASET \
                --data_dir "$SCRIPT_DIR/data" \
                --model_dir "$SCRIPT_DIR/models" \
                --batch_size 16
            
            if [ $? -eq 0 ]; then
                echo " Evaluation completed for $DATASET"
                
                echo ""
                echo "=== GENERATING PLOTS: $DATASET ($DATASET_NAME) ==="
                python generate_plots_from_results.py \
                    --dataset $DATASET \
                    --model_dir "$SCRIPT_DIR/models"
                
                if [ $? -eq 0 ]; then
                    echo " Plots generated for $DATASET"
                else
                    echo " Plot generation failed for $DATASET (but training & evaluation succeeded)"
                fi
            else
                echo " Evaluation failed for $DATASET"
            fi
        else
            echo " Training failed for $DATASET"
        fi
    done
else
    if ! [[ "$DATASET_ARG" =~ ^[0-3]$ ]]; then
        echo " Error: Invalid dataset argument. Use 0-3 for specific dataset or 'all' for all datasets."
        echo "  0 = 14res (Restaurant 2014)"
        echo "  1 = 14lap (Laptop 2014)"
        echo "  2 = 15res (Restaurant 2015)"
        echo "  3 = 16res (Restaurant 2016)"
        exit 1
    fi
    
    DATASET_ID=$DATASET_ARG
    DATASET=${DATASETS[$DATASET_ID]}
    DATASET_NAME=${DATASET_NAMES[$DATASET_ID]}
    
    echo "Training and evaluating: $DATASET ($DATASET_NAME)"
    
    echo ""
    echo "=== TRAINING: $DATASET ($DATASET_NAME) ==="
    
    python train_aste.py \
        --dataset $DATASET \
        --data_dir "$SCRIPT_DIR/data" \
        --output_dir "$SCRIPT_DIR/models" \
        --batch_size 16 \
        --num_epochs 40 \
        --learning_rate 0.1 \
        --dropout_rate 0.5 \
        --weight_decay 0.001 \
        --hidden_size 300
    
    if [ $? -eq 0 ]; then
        echo " Base training completed for $DATASET"
        
        
        echo ""
        echo "=== EVALUATION: $DATASET ($DATASET_NAME) ==="
        
        python evaluate_aste.py \
            --dataset $DATASET \
            --data_dir "$SCRIPT_DIR/data" \
            --model_dir "$SCRIPT_DIR/models" \
            --batch_size 16
        
        if [ $? -eq 0 ]; then
            echo " Evaluation completed for $DATASET"
            
            echo ""
            echo "=== GENERATING PLOTS: $DATASET ($DATASET_NAME) ==="
            python generate_plots_from_results.py \
                --dataset $DATASET \
                --model_dir "$SCRIPT_DIR/models"
            
            if [ $? -eq 0 ]; then
                echo " Plots generated for $DATASET"
                echo ""
                echo " Complete! Check:"
                echo "   - $SCRIPT_DIR/models/${DATASET}_evaluation_results.json for metrics"
                echo "   - $SCRIPT_DIR/models/plots/${DATASET}/ for visualizations"
            else
                echo " Plot generation failed for $DATASET (but training & evaluation succeeded)"
            fi
        else
            echo " Evaluation failed for $DATASET"
        fi
    else
        echo " Training failed for $DATASET"
    fi
fi

echo "Job completed for ASTE workflow"