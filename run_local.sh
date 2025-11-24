#!/bin/bash


echo "=== ASTE baseline-Compliant Training Workflow ==="
echo " Fixed: Stage Two now trains on Stage One predictions (not ground truth)"
echo " Target: Improve from 38.46% to ~51.89% triplet F1 on 14res"
echo ""

mkdir -p logs models/checkpoints models/plots

DATASET=${1:-14res}

echo "Training dataset: $DATASET"
echo ""

echo "=== STAGE 1 & 2 TRAINING ==="
python train_aste.py \
    --dataset $DATASET \
    --batch_size 16 \
    --num_epochs 40 \
    --learning_rate 0.001 \
    --dropout_rate 0.5 \
    --weight_decay 0.001 \
    --hidden_size 300

if [ $? -eq 0 ]; then
    echo " baseline-compliant training completed for $DATASET"
    
    echo ""
    echo "=== EVALUATION ==="
    python evaluate_aste.py \
        --dataset $DATASET \
        --model_dir models \
        --batch_size 16
    
    if [ $? -eq 0 ]; then
        echo " Evaluation completed for $DATASET"
        
        echo ""
        echo "=== GENERATING PLOTS ==="
        python generate_plots_from_results.py --dataset $DATASET --model_dir models
        
        if [ $? -eq 0 ]; then
            echo " Plots generated successfully"
            echo ""
            echo " Complete! Check:"
            echo "   - models/${DATASET}_evaluation_results.json for metrics"
            echo "   - models/plots/${DATASET}/ for visualizations"
        else
            echo " Plot generation failed (but training & evaluation succeeded)"
        fi
    else
        echo " Evaluation failed for $DATASET"
    fi
else
    echo " Training failed for $DATASET"
fi

echo "Workflow completed for $DATASET"