#!/usr/bin/env python3
"""
Generate plots from evaluation results and training metrics
Uses actual final metrics from evaluation_results.json
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_results(dataset_name, model_dir='./models'):
    """Load evaluation results and training metrics"""
    eval_file = os.path.join(model_dir, f'{dataset_name}_evaluation_results.json')
    training_file = os.path.join(model_dir, f'{dataset_name}_training_metrics.json')
    
    with open(eval_file, 'r') as f:
        eval_results = json.load(f)
    
    with open(training_file, 'r') as f:
        training_metrics = json.load(f)
    
    return eval_results, training_metrics

def plot_component_analysis(eval_results, training_metrics, dataset_name, output_dir):
    """Plot component performance analysis using actual results"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Component comparison (using ACTUAL evaluation results)
    components = ['Aspect\nExtraction', 'Opinion\nExtraction', 'Sentiment\nClassification', 'Aspect-Opinion\nPairing']
    
    final_values = [
        eval_results['aspect_f1'],
        eval_results['opinion_f1'],
        eval_results['sentiment_accuracy'],
        eval_results['pair_f1']  # Using actual pair_f1 from evaluation
    ]
    
    bars = ax1.bar(components, final_values, 
                  color=['purple', 'orange', 'teal', 'darkorange'], alpha=0.8)
    ax1.set_title('Component Performance Analysis', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Stage 1 component evolution (from training)
    stage1_metrics = training_metrics['stage_one']
    stage1_epochs = [i*10 for i in range(1, len(stage1_metrics['aspect_f1']) + 1)]
    
    if stage1_epochs:
        ax2.plot(stage1_epochs, stage1_metrics['aspect_f1'], 
                'purple', linewidth=2, marker='o', label='Aspect F1', markersize=5)
        ax2.plot(stage1_epochs, stage1_metrics['opinion_f1'], 
                'orange', linewidth=2, marker='s', label='Opinion F1', markersize=5)
        ax2.plot(stage1_epochs, stage1_metrics['sentiment_acc'], 
                'teal', linewidth=2, marker='^', label='Sentiment Acc', markersize=5)
    
    ax2.set_title('Stage 1: Component Learning Curves', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Model architecture summary
    ax3.text(0.1, 0.9, 'ASTE Model Architecture', fontsize=16, fontweight='bold', transform=ax3.transAxes)
    ax3.text(0.1, 0.8, f'• Stage 1: Multi-task Learning', fontsize=12, transform=ax3.transAxes)
    ax3.text(0.1, 0.7, f'• Stage 2: Aspect-Opinion Pairing', fontsize=12, transform=ax3.transAxes)
    ax3.text(0.1, 0.6, f'• Hidden Size: 300', fontsize=12, transform=ax3.transAxes)
    ax3.text(0.1, 0.5, f'• Embedding: GloVe 300d', fontsize=12, transform=ax3.transAxes)
    ax3.text(0.1, 0.4, f'• Learning Rate: 0.1', fontsize=12, transform=ax3.transAxes)
    ax3.text(0.1, 0.3, f'• Batch Size: 16', fontsize=12, transform=ax3.transAxes)
    ax3.text(0.1, 0.2, f'• Total Parameters: ~5.5M', fontsize=12, transform=ax3.transAxes)
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_component_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved component analysis to {output_dir}")
    plt.close()

def plot_performance_overview(eval_results, training_metrics, dataset_name, output_dir):
    """Plot performance overview using actual results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Stage 1: Training loss
    stage1_metrics = training_metrics['stage_one']
    epochs = list(range(1, len(stage1_metrics['train_loss']) + 1))
    
    ax1.plot(epochs, stage1_metrics['train_loss'], 'darkblue', linewidth=2, marker='o', markersize=3)
    ax1.set_title('Stage 1: Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # Stage 2: Training loss
    stage2_metrics = training_metrics['stage_two']
    stage2_epochs = list(range(1, len(stage2_metrics['train_loss']) + 1))
    
    ax2.plot(stage2_epochs, stage2_metrics['train_loss'], 'darkred', linewidth=2, marker='s', markersize=4)
    ax2.set_title('Stage 2: Training Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    # Final performance metrics comparison
    metrics_names = ['Aspect\nF1', 'Opinion\nF1', 'Sentiment\nAcc', 'Pair\nF1', 'Triplet\nF1']
    metrics_values = [
        eval_results['aspect_f1'],
        eval_results['opinion_f1'],
        eval_results['sentiment_accuracy'],
        eval_results['pair_f1'],
        eval_results['triplet_f1']
    ]
    
    bars = ax3.bar(metrics_names, metrics_values,
                   color=['purple', 'orange', 'teal', 'darkorange', 'darkgreen'], alpha=0.8)
    ax3.set_title('Final Test Performance (All Metrics)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Score')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Paper comparison
    paper_results = {'14res': 0.5189, '14lap': 0.435, '15res': 0.4679, '16res': 0.5362}
    datasets = ['14res', '14lap', '15res', '16res']
    our_result = eval_results['triplet_f1']
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, [paper_results[d] for d in datasets], width, 
                    label='Paper Results', alpha=0.7, color='skyblue')
    bars2 = ax4.bar(x + width/2, [our_result if d == dataset_name else 0 for d in datasets], width,
                    label='Our Result', alpha=0.7, color='lightcoral')
    
    ax4.set_title('Paper Comparison (Triplet F1)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Triplet F1 Score')
    ax4.set_xticks(x)
    ax4.set_xticklabels(datasets)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 0.6)
    
    # Add value labels
    for i, d in enumerate(datasets):
        paper_val = paper_results[d]
        ax4.text(i - width/2, paper_val + 0.01, f'{paper_val:.3f}', 
                ha='center', va='bottom', fontsize=9)
        if d == dataset_name:
            ax4.text(i + width/2, our_result + 0.01, f'{our_result:.3f}', 
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_performance_overview.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved performance overview to {output_dir}")
    plt.close()

def plot_stage_two_metrics(eval_results, training_metrics, dataset_name, output_dir):
    """Plot Stage 2 metrics using validation data and final results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    stage2_metrics = training_metrics['stage_two']
    
    # Validation F1 during training
    eval_epochs = [i for i in range(1, len(stage2_metrics['val_f1']) + 1)]
    
    if eval_epochs:
        ax1.plot(eval_epochs, stage2_metrics['val_f1'], 'darkgreen', linewidth=3, marker='o', markersize=6)
        ax1.axhline(y=eval_results['pair_f1'], color='red', linestyle='--', linewidth=2, label='Final Test Pair F1')
        ax1.set_title('Stage 2: Validation F1 During Training', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('F1 Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Precision and Recall
        ax2.plot(eval_epochs, stage2_metrics['val_precision'], 'darkblue', linewidth=2, marker='s', markersize=5, label='Precision')
        ax2.plot(eval_epochs, stage2_metrics['val_recall'], 'darkred', linewidth=2, marker='^', markersize=5, label='Recall')
        ax2.axhline(y=eval_results['pair_precision'], color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='Test Precision')
        ax2.axhline(y=eval_results['pair_recall'], color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Test Recall')
        ax2.set_title('Stage 2: Precision & Recall', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
    
    # Final test metrics breakdown
    final_metrics = {
        'Triplet\nPrecision': eval_results['triplet_precision'],
        'Triplet\nRecall': eval_results['triplet_recall'],
        'Triplet\nF1': eval_results['triplet_f1'],
        'Pair\nPrecision': eval_results['pair_precision'],
        'Pair\nRecall': eval_results['pair_recall'],
        'Pair\nF1': eval_results['pair_f1']
    }
    
    colors = ['darkblue', 'darkred', 'darkgreen', 'blue', 'red', 'green']
    bars = ax3.bar(final_metrics.keys(), final_metrics.values(), color=colors, alpha=0.7)
    ax3.set_title('Final Test Performance: Triplets & Pairs', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Score')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.xticks(rotation=20)
    
    # Threshold search results
    threshold_data = stage2_metrics.get('threshold_search', [])
    if threshold_data:
        thresholds = [item['threshold'] for item in threshold_data]
        f1_scores = [item['f1'] for item in threshold_data]
        precisions = [item['precision'] for item in threshold_data]
        recalls = [item['recall'] for item in threshold_data]
        
        ax4.plot(thresholds, f1_scores, 'darkgreen', linewidth=2, marker='o', label='F1', markersize=5)
        ax4.plot(thresholds, precisions, 'darkblue', linewidth=2, marker='s', label='Precision', markersize=4)
        ax4.plot(thresholds, recalls, 'darkred', linewidth=2, marker='^', label='Recall', markersize=4)
        
        optimal_threshold = training_metrics.get('optimal_threshold', 0.25)
        ax4.axvline(x=optimal_threshold, color='purple', linestyle='--', linewidth=2, label=f'Optimal ({optimal_threshold})')
        
        ax4.set_title('Threshold Optimization', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_stage_two_metrics.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved stage two metrics to {output_dir}")
    plt.close()

def plot_stage_one_metrics(training_metrics, dataset_name, output_dir):
    """Plot Stage 1 training metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    stage1_metrics = training_metrics['stage_one']
    epochs = list(range(1, len(stage1_metrics['train_loss']) + 1))
    eval_epochs = [i*10 for i in range(1, len(stage1_metrics['aspect_f1']) + 1)]
    
    # Training loss
    ax1.plot(epochs, stage1_metrics['train_loss'], 'darkblue', linewidth=2, marker='o', markersize=3)
    ax1.set_title('Stage 1: Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # Component F1 scores
    ax2.plot(eval_epochs, stage1_metrics['aspect_f1'], 'purple', linewidth=2, marker='o', label='Aspect F1', markersize=5)
    ax2.plot(eval_epochs, stage1_metrics['opinion_f1'], 'orange', linewidth=2, marker='s', label='Opinion F1', markersize=5)
    ax2.plot(eval_epochs, stage1_metrics['sentiment_acc'], 'teal', linewidth=2, marker='^', label='Sentiment Acc', markersize=5)
    ax2.set_title('Stage 1: Component Performance', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Token-level metrics
    ax3.plot(eval_epochs, stage1_metrics['val_f1'], 'darkgreen', linewidth=2, marker='D', label='Token F1', markersize=5)
    ax3.plot(eval_epochs, stage1_metrics['val_precision'], 'darkblue', linewidth=2, marker='s', label='Token Precision', markersize=4)
    ax3.plot(eval_epochs, stage1_metrics['val_recall'], 'darkred', linewidth=2, marker='^', label='Token Recall', markersize=4)
    ax3.set_title('Stage 1: Token-level Metrics', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Final component scores
    final_scores = {
        'Aspect F1': stage1_metrics['aspect_f1'][-1],
        'Opinion F1': stage1_metrics['opinion_f1'][-1],
        'Sentiment Acc': stage1_metrics['sentiment_acc'][-1]
    }
    
    bars = ax4.bar(final_scores.keys(), final_scores.values(),
                   color=['purple', 'orange', 'teal'], alpha=0.8)
    ax4.set_title('Stage 1: Final Component Scores', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Score')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_stage_one_metrics.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved stage one metrics to {output_dir}")
    plt.close()

def plot_training_losses(training_metrics, dataset_name, output_dir):
    """Plot combined training losses"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    stage1_metrics = training_metrics['stage_one']
    stage2_metrics = training_metrics['stage_two']
    
    epochs1 = list(range(1, len(stage1_metrics['train_loss']) + 1))
    epochs2 = list(range(1, len(stage2_metrics['train_loss']) + 1))
    
    # Stage 1 loss
    ax1.plot(epochs1, stage1_metrics['train_loss'], 'darkblue', linewidth=2, marker='o', markersize=3)
    ax1.set_title('Stage 1: Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # Stage 2 loss
    ax2.plot(epochs2, stage2_metrics['train_loss'], 'darkred', linewidth=2, marker='s', markersize=4)
    ax2.set_title('Stage 2: Training Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_training_losses.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved training losses to {output_dir}")
    plt.close()

def main():
    """Generate all plots for a dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate plots from evaluation results')
    parser.add_argument('--dataset', type=str, default='14res', help='Dataset name (14res, 14lap, 15res, 16res)')
    parser.add_argument('--model_dir', type=str, default='./models', help='Directory containing results')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    # Set output directory (use plots/{dataset} structure to match original)
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, 'plots', args.dataset)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating plots for dataset: {args.dataset}")
    print(f"{'='*60}\n")
    
    # Load results
    eval_results, training_metrics = load_results(args.dataset, args.model_dir)
    
    print(f"📊 Evaluation Results:")
    print(f"   - Aspect F1: {eval_results['aspect_f1']:.4f}")
    print(f"   - Opinion F1: {eval_results['opinion_f1']:.4f}")
    print(f"   - Sentiment Acc: {eval_results['sentiment_accuracy']:.4f}")
    print(f"   - Pair F1: {eval_results['pair_f1']:.4f}")
    print(f"   - Triplet F1: {eval_results['triplet_f1']:.4f}")
    print()
    
    # Generate plots
    plot_component_analysis(eval_results, training_metrics, args.dataset, args.output_dir)
    plot_performance_overview(eval_results, training_metrics, args.dataset, args.output_dir)
    plot_stage_one_metrics(training_metrics, args.dataset, args.output_dir)
    plot_stage_two_metrics(eval_results, training_metrics, args.dataset, args.output_dir)
    plot_training_losses(training_metrics, args.dataset, args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"✅ All plots saved to: {args.output_dir}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
