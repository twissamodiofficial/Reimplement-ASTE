#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import logging

from aste_model import StageOneModel, StageTwoModel
from data_prep import SemEvalParser
from train_aste import ASTEDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_optimal_threshold(model_dir, dataset_name, default_threshold=0.25):
    try:
        metrics_file = os.path.join(model_dir, f'{dataset_name}_training_metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                if 'optimal_threshold' in metrics:
                    threshold = metrics['optimal_threshold']
                    logger.info(f"Loaded optimal threshold: {threshold:.3f}")
                    return threshold
        else:
            logger.warning(f"Training metrics file not found: {metrics_file}")
    except Exception as e:
        logger.error(f"Error loading optimal threshold: {e}")
    
    return default_threshold

class ASTEEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def load_models_and_data(self):
        logger.info("Loading models and data")
        
        dataset_name = getattr(self.args, 'dataset', '14res')
        comprehensive_model = os.path.join(self.args.model_dir, f'best_model_{dataset_name}_complete.pt')
        
        if os.path.exists(comprehensive_model):
            logger.info(f"Using comprehensive model: {comprehensive_model}")
            checkpoint = torch.load(comprehensive_model, map_location=self.device, weights_only=False)
            self.vocab_size = checkpoint['vocab_size']
            self.hidden_size = checkpoint['hidden_size']
            self.num_labels = checkpoint['num_labels']
            self.word_to_id = checkpoint['word_to_id']
            self.label_to_id = checkpoint['label_to_id']
            self.id_to_label = {v: k for k, v in self.label_to_id.items()}
            
            self.stage_one_model = StageOneModel(
                vocab_size=self.vocab_size,
                embed_dim=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=1,
                dropout=0.1
            ).to(self.device)
            
            self.stage_two_model = StageTwoModel(
                embed_dim=300,
                hidden_size=self.hidden_size,
                max_distance=100,
                dropout=0.1
            ).to(self.device)
            
            self.stage_one_model.load_state_dict(checkpoint['stage_one_state_dict'])
            self.stage_two_model.load_state_dict(checkpoint['stage_two_state_dict'])
            
        else:
            logger.info("Trying individual models")
            stage_one_path = os.path.join(self.args.model_dir, f'{dataset_name}_stage_one_best.pt')
            stage_two_path = os.path.join(self.args.model_dir, f'{dataset_name}_stage_two_best.pt')
            
            if not os.path.exists(stage_one_path):
                stage_one_path = os.path.join(self.args.model_dir, 'stage_one_best.pt')
                if not os.path.exists(stage_one_path):
                    stage_one_path = os.path.join(self.args.model_dir, 'stage_one_final.pt')
            if not os.path.exists(stage_two_path):
                stage_two_path = os.path.join(self.args.model_dir, 'stage_two_best.pt')
                if not os.path.exists(stage_two_path):
                    stage_two_path = os.path.join(self.args.model_dir, 'stage_two_final.pt')
            
            logger.info(f"Loading Stage One: {stage_one_path}")
            logger.info(f"Loading Stage Two: {stage_two_path}")
            
            if not os.path.exists(stage_one_path):
                raise FileNotFoundError(f"Stage One model not found: {stage_one_path}")
            if not os.path.exists(stage_two_path):
                raise FileNotFoundError(f"Stage Two model not found: {stage_two_path}")
            
            stage_one_checkpoint = torch.load(stage_one_path, map_location=self.device, weights_only=False)
            self.vocab_size = stage_one_checkpoint['vocab_size']
            self.hidden_size = stage_one_checkpoint['hidden_size']
            self.num_labels = stage_one_checkpoint['num_labels']
            self.word_to_id = stage_one_checkpoint['word_to_id']
            self.label_to_id = stage_one_checkpoint['label_to_id']
            self.id_to_label = {v: k for k, v in self.label_to_id.items()}
            
            self.stage_one_model = StageOneModel(
                vocab_size=self.vocab_size,
                embed_dim=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=1,
                dropout=0.1
            ).to(self.device)
            
            self.stage_two_model = StageTwoModel(
                embed_dim=300,
                hidden_size=self.hidden_size,
                max_distance=100,
                dropout=0.1
            ).to(self.device)
            
            self.stage_one_model.load_state_dict(stage_one_checkpoint['model_state_dict'])
            
            stage_two_checkpoint = torch.load(stage_two_path, map_location=self.device, weights_only=False)
            if 'stage_two_state_dict' in stage_two_checkpoint:
                self.stage_two_model.load_state_dict(stage_two_checkpoint['stage_two_state_dict'])
            else:
                self.stage_two_model.load_state_dict(stage_two_checkpoint['model_state_dict'])
        
        self.boundary_id_to_label = {0: 'O', 1: 'B', 2: 'I', 3: 'E', 4: 'S'}
        
        self.stage_one_model.eval()
        self.stage_two_model.eval()
        
        dataset_name = getattr(self.args, 'dataset', '14res')
        data_dir = os.path.join(self.args.data_dir, dataset_name)
        test_file = os.path.join(data_dir, 'test.txt')
        
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        logger.info(f"Loading test data: {test_file}")
        
        from aste_data_loader import ASTEDatasetOfficial, collate_fn_official
        test_dataset = ASTEDatasetOfficial(test_file, self.word_to_id)
        self.test_data = test_dataset
        
        batch_size = getattr(self.args, 'batch_size', 16)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_official
        )
        
        logger.info(f"Vocab size: {self.vocab_size}, Examples: {len(test_dataset)}, Batch size: {batch_size}")
    
    def collate_fn(self, batch):
        max_len = max(len(item['tokens']) for item in batch)
        
        input_ids = []
        labels = []
        attention_masks = []
        dep_matrices = []
        
        for item in batch:
            padded_tokens = item['tokens'] + [0] * (max_len - len(item['tokens']))
            padded_labels = item['labels'] + [0] * (max_len - len(item['labels']))
            attention_mask = [1] * len(item['tokens']) + [0] * (max_len - len(item['tokens']))
            
            dep_matrix = np.zeros((max_len, max_len))
            orig_len = len(item['tokens'])
            dep_matrix[:orig_len, :orig_len] = item['dep_matrix']
            
            input_ids.append(padded_tokens)
            labels.append(padded_labels)
            attention_masks.append(attention_mask)
            dep_matrices.append(dep_matrix)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.float),
            'dep_matrix': torch.tensor(dep_matrices, dtype=torch.float)
        }
    
    def evaluate_comprehensive(self):
        logger.info("Running comprehensive evaluation")
        
        dataset_name = getattr(self.args, 'dataset', '14res')
        optimal_threshold = load_optimal_threshold(self.args.model_dir, dataset_name, default_threshold=0.35)
        logger.info(f"Using optimal threshold: {optimal_threshold}")
        
        all_predictions = []
        all_labels = []
        all_predicted_triplets = []
        all_ground_truth_triplets = []
        
        aspect_preds_list = []
        aspect_labels_list = []
        opinion_preds_list = []
        opinion_labels_list = []
        sentiment_preds_list = []
        sentiment_labels_list = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                opinion_gt_labels = batch['opinion_labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                dep_matrix = batch['dep_matrix'].to(self.device)
                
                lengths = attention_mask.sum(dim=1)
                
                stage_one_outputs = self.stage_one_model(input_ids, dep_matrix, lengths)
                stage_one_logits = stage_one_outputs['unified_logits']
                predictions = torch.argmax(stage_one_logits, dim=-1)
                
                opinion_logits = stage_one_outputs['opinion_logits']
                opinion_predictions = torch.argmax(opinion_logits, dim=-1)
                
                flat_preds = predictions.view(-1)
                flat_labels = labels.view(-1)
                flat_attention = attention_mask.view(-1)
                
                active_positions = flat_attention == 1
                active_preds = flat_preds[active_positions]
                active_labels = flat_labels[active_positions]
                
                all_predictions.extend(active_preds.cpu().numpy())
                all_labels.extend(active_labels.cpu().numpy())
                
                for i in range(len(predictions)):
                    pred_seq = predictions[i]
                    opinion_pred_seq = opinion_predictions[i]
                    label_seq = labels[i]
                    opinion_gt_seq = opinion_gt_labels[i]
                    
                    self.extract_component_predictions(
                        pred_seq, label_seq, opinion_pred_seq, opinion_gt_seq,
                        aspect_preds_list, aspect_labels_list,
                        opinion_preds_list, opinion_labels_list,
                        sentiment_preds_list, sentiment_labels_list
                    )
                    
                    word_embeds = self.stage_one_model.embedding(input_ids[i:i+1])
                    predicted_triplets = self.extract_predicted_triplets_fixed(
                        pred_seq, input_ids[i], attention_mask[i], dep_matrix[i],
                        opinion_pred_seq, word_embeds[0], threshold=optimal_threshold
                    )
                    gt_triplets = self.extract_ground_truth_triplets(label_seq, opinion_gt_seq)
                    
                    all_predicted_triplets.append(predicted_triplets)
                    all_ground_truth_triplets.append(gt_triplets)
        
        results = self.calculate_all_metrics(
            all_predictions, all_labels,
            aspect_preds_list, aspect_labels_list,
            opinion_preds_list, opinion_labels_list,
            sentiment_preds_list, sentiment_labels_list,
            all_predicted_triplets, all_ground_truth_triplets
        )
        
        return results
    
    def extract_component_predictions(self, pred_seq, label_seq, opinion_pred_seq, opinion_gt_seq,
                                    asp_preds, asp_labels,
                                    opi_preds, opi_labels,
                                    sent_preds, sent_labels):
        for pred, label, opinion_pred, opinion_gt in zip(pred_seq, label_seq, opinion_pred_seq, opinion_gt_seq):
            pred_label = self.id_to_label[pred.item()]
            gt_label = self.id_to_label[label.item()]
            
            asp_preds.append(1 if pred_label != 'O' else 0)
            asp_labels.append(1 if gt_label != 'O' else 0)
            
            opinion_pred_label = self.boundary_id_to_label.get(opinion_pred.item(), 'O')
            opinion_gt_id = opinion_gt.item()
            
            opi_preds.append(1 if opinion_pred_label != 'O' else 0)
            opi_labels.append(1 if opinion_gt_id != 0 else 0)
            
            if gt_label != 'O' and pred_label != 'O':
                pred_sent = 'NEU'
                if 'POS' in pred_label:
                    pred_sent = 'POS'
                elif 'NEG' in pred_label:
                    pred_sent = 'NEG'
                elif 'NEU' in pred_label:
                    pred_sent = 'NEU'
                
                gt_sent = 'NEU'
                if 'POS' in gt_label:
                    gt_sent = 'POS'
                elif 'NEG' in gt_label:
                    gt_sent = 'NEG'
                elif 'NEU' in gt_label:
                    gt_sent = 'NEU'
                
                sent_preds.append(pred_sent)
                sent_labels.append(gt_sent)
    
    def extract_spans(self, predictions, target_labels):
        spans = []
        current_span = None
        
        for i, pred in enumerate(predictions):
            label = self.id_to_label[pred.item()]
            
            if label in target_labels:
                if label.startswith('B-'):
                    if current_span:
                        spans.append(current_span)
                    current_span = [i, i]
                elif label.startswith('I-') and current_span:
                    current_span[1] = i
            else:
                if current_span:
                    spans.append(current_span)
                    current_span = None
        
        if current_span:
            spans.append(current_span)
        
        return spans
    
    def extract_predicted_triplets(self, predictions, input_ids, attention_mask, dep_matrix, threshold=None):
        if threshold is None:
            threshold = 0.45
            
        triplets = []
        
        with torch.no_grad():
            seq_len = attention_mask.sum().item()
            input_single = input_ids.unsqueeze(0)
            dep_single = dep_matrix.unsqueeze(0)
            lengths = torch.tensor([seq_len], device=self.device)
            
            stage_one_outputs = self.stage_one_model(input_single, dep_single, lengths)
            
            unified_preds = torch.argmax(stage_one_outputs['unified_logits'], dim=-1)[0]
            aspect_spans = []
            
            for sentiment in ['POS', 'NEG', 'NEU']:
                sent_spans = self.extract_spans(unified_preds, [f'B-{sentiment}', f'I-{sentiment}', f'E-{sentiment}', f'S-{sentiment}'])
                aspect_spans.extend(sent_spans)
            
            opinion_preds = torch.argmax(stage_one_outputs['opinion_logits'], dim=-1)[0]
            opinion_spans = self.extract_opinion_spans_from_boundary_predictions(opinion_preds)
            
            if len(aspect_spans) > 0 and len(opinion_spans) > 0:
                candidate_pairs = []
                for asp_start, asp_end in aspect_spans:
                    for opi_start, opi_end in opinion_spans:
                        if asp_start != opi_start or asp_end != opi_end:
                            candidate_pairs.append([0, asp_start, asp_end, opi_start, opi_end])
                
                if candidate_pairs:
                    aspect_spans_list = []
                    opinion_spans_list = []
                    
                    for _, asp_start, asp_end, opi_start, opi_end in candidate_pairs:
                        aspect_spans_list.append([asp_start, asp_end])
                        opinion_spans_list.append([opi_start, opi_end])
                    
                    aspect_spans_tensor = torch.tensor(aspect_spans_list, device=self.device)
                    opinion_spans_tensor = torch.tensor(opinion_spans_list, device=self.device)
                    
                    num_pairs = len(candidate_pairs)
                    word_embeds = self.stage_one_model.embedding(input_ids)
                    pair_sentence_embeds = word_embeds.repeat(num_pairs, 1, 1)
                    
                    lengths_tensor = torch.tensor([seq_len] * num_pairs, device=self.device)
                    
                    pair_scores = self.stage_two_model(
                        pair_sentence_embeds,
                        aspect_spans_tensor,
                        opinion_spans_tensor,
                        lengths_tensor
                    )
                    
                    valid_pairs = []
                    
                    pair_probs = torch.softmax(pair_scores, dim=-1)[:, 1].cpu().numpy()
                    
                    for i, (_, asp_start, asp_end, opi_start, opi_end) in enumerate(candidate_pairs):
                        if pair_probs[i] > threshold:
                            valid_pairs.append((asp_start, asp_end, opi_start, opi_end))
                    
                    for asp_start, asp_end, opi_start, opi_end in valid_pairs:
                        sentiment = self.determine_sentiment_for_pair(unified_preds, (asp_start, asp_end), (opi_start, opi_end))
                        triplets.append(((asp_start, asp_end), (opi_start, opi_end), sentiment))
        
        return triplets
    
    def extract_opinion_spans_from_boundary_predictions(self, opinion_predictions):
        spans = []
        current_span = None
        
        for i, pred_id in enumerate(opinion_predictions):
            pred_label = self.boundary_id_to_label.get(pred_id.item(), 'O')
            
            if pred_label == 'B':
                if current_span:
                    spans.append((current_span[0], current_span[1]))
                current_span = [i, i]
            elif pred_label == 'I':
                if current_span:
                    current_span[1] = i
            elif pred_label == 'E':
                if current_span:
                    current_span[1] = i
                    spans.append((current_span[0], current_span[1]))
                    current_span = None
            elif pred_label == 'S':
                spans.append((i, i))
            elif pred_label == 'O':
                if current_span:
                    spans.append((current_span[0], current_span[1]))
                    current_span = None
        
        if current_span:
            spans.append((current_span[0], current_span[1]))
        
        return spans

    def determine_sentiment_for_pair(self, predictions, aspect_span, opinion_span):
        sentiment_votes = {'POS': 0, 'NEG': 0, 'NEU': 0}
        
        for i in range(min(aspect_span[0], opinion_span[0]), max(aspect_span[1], opinion_span[1]) + 1):
            if i < len(predictions):
                label = self.id_to_label[predictions[i].item()]
                
                if 'POS' in label:
                    sentiment_votes['POS'] += 1
                elif 'NEG' in label:
                    sentiment_votes['NEG'] += 1
                elif 'NEU' in label:
                    sentiment_votes['NEU'] += 1
        
        if sum(sentiment_votes.values()) == 0:
            return 'NEU'
        
        return max(sentiment_votes, key=sentiment_votes.get)

    def extract_ground_truth_triplets(self, label_seq, opinion_gt_labels=None):
        if opinion_gt_labels is None:
            triplets = []
            for sentiment in ['POS', 'NEG', 'NEU']:
                spans = self.extract_spans(label_seq, [f'B-{sentiment}', f'I-{sentiment}', f'E-{sentiment}', f'S-{sentiment}'])
                for span in spans:
                    triplets.append((tuple(span), tuple(span), sentiment))
            return triplets
        
        triplets = []
        
        aspect_spans = []
        for sentiment in ['POS', 'NEG', 'NEU']:
            sent_spans = self.extract_spans(label_seq, [f'B-{sentiment}', f'I-{sentiment}', f'E-{sentiment}', f'S-{sentiment}'])
            for span in sent_spans:
                aspect_spans.append((span, sentiment))
        
        opinion_spans = self.extract_opinion_spans_from_boundary_sequence(opinion_gt_labels)
        
        for (asp_span, sentiment), opi_span in zip(aspect_spans, opinion_spans):
            if len(aspect_spans) == len(opinion_spans):
                triplets.append((tuple(asp_span), tuple(opi_span), sentiment))
            else:
                triplets.append((tuple(asp_span), tuple(opi_span), sentiment))
        
        return triplets
    
    def extract_opinion_spans_from_boundary_sequence(self, boundary_labels):
        spans = []
        current_span = None
        
        for i, label_id in enumerate(boundary_labels):
            if hasattr(label_id, 'item'):
                label_id = label_id.item()
            
            boundary_label = self.boundary_id_to_label.get(label_id, 'O')
            
            if boundary_label == 'B':
                if current_span:
                    spans.append((current_span[0], current_span[1]))
                current_span = [i, i]
            elif boundary_label == 'I':
                if current_span:
                    current_span[1] = i
            elif boundary_label == 'E':
                if current_span:
                    current_span[1] = i
                    spans.append((current_span[0], current_span[1]))
                    current_span = None
            elif boundary_label == 'S':
                spans.append((i, i))
            elif boundary_label == 'O':
                if current_span:
                    spans.append((current_span[0], current_span[1]))
                    current_span = None
        
        if current_span:
            spans.append((current_span[0], current_span[1]))
        
        return spans
    
    def determine_sentiment(self, predictions, aspect_span, opinion_span):
        asp_center = (aspect_span[0] + aspect_span[1]) / 2
        opi_center = (opinion_span[0] + opinion_span[1]) / 2
        
        sentiment_votes = {'POS': 0, 'NEG': 0, 'NEU': 0}
        
        for i, pred in enumerate(predictions):
            label = self.id_to_label[pred.item()]
            
            if any(sent in label for sent in ['POS', 'NEG', 'NEU']):
                sentiment_type = None
                if 'POS' in label:
                    sentiment_type = 'POS'
                elif 'NEG' in label:
                    sentiment_type = 'NEG'
                elif 'NEU' in label:
                    sentiment_type = 'NEU'
                
                if sentiment_type:
                    distance = min(abs(i - asp_center), abs(i - opi_center))
                    weight = 1.0 / (1.0 + distance)
                    sentiment_votes[sentiment_type] += weight
        
        return max(sentiment_votes, key=sentiment_votes.get)
    
    def extract_predicted_triplets_fixed(self, predictions, input_ids, attention_mask, dep_matrix, opinion_predictions, word_embeds, threshold=None):
        if threshold is None:
            threshold = 0.45
            
        seq_len = attention_mask.sum().item()
        
        aspects = []
        for sentiment in ['POS', 'NEG', 'NEU']:
            sentiment_aspects = self.extract_spans(predictions, [f'B-{sentiment}', f'I-{sentiment}', f'E-{sentiment}', f'S-{sentiment}'])
            for start, end in sentiment_aspects:
                aspects.append((start, end, sentiment))
        
        boundary_id_to_label = {0: 'O', 1: 'B', 2: 'I', 3: 'E', 4: 'S'}
        opinion_label_seq = [boundary_id_to_label.get(pred_id.item(), 'O') for pred_id in opinion_predictions]
        opinion_spans = self.extract_spans_from_boundary_sequence(opinion_label_seq)
        opinions = [(start, end) for start, end in opinion_spans]
        
        if not aspects or not opinions:
            return []
        
        candidate_pairs = []
        for asp_start, asp_end, sentiment in aspects:
            for opi_start, opi_end in opinions:
                if asp_start == opi_start and asp_end == opi_end:
                    continue
                candidate_pairs.append((asp_start, asp_end, opi_start, opi_end, sentiment))
        
        if not candidate_pairs:
            return []
        
        valid_triplets = []
        
        batch_size = min(16, len(candidate_pairs))
        for i in range(0, len(candidate_pairs), batch_size):
            batch_pairs = candidate_pairs[i:i+batch_size]
            
            aspect_spans = torch.tensor([[asp_start, asp_end] for asp_start, asp_end, _, _, _ in batch_pairs], device=self.device)
            opinion_spans = torch.tensor([[opi_start, opi_end] for _, _, opi_start, opi_end, _ in batch_pairs], device=self.device)
            
            batch_word_embeds = word_embeds.unsqueeze(0).repeat(len(batch_pairs), 1, 1)
            batch_lengths = torch.tensor([seq_len] * len(batch_pairs), device=self.device)
            
            with torch.no_grad():
                pair_scores = self.stage_two_model(batch_word_embeds, aspect_spans, opinion_spans, batch_lengths)
                pair_probs = torch.softmax(pair_scores, dim=1)
                
                for j, (asp_start, asp_end, opi_start, opi_end, sentiment) in enumerate(batch_pairs):
                    if pair_probs[j, 1] > threshold:
                        valid_triplets.append(((asp_start, asp_end), (opi_start, opi_end), sentiment))
        
        return valid_triplets
    
    def extract_spans_from_boundary_sequence(self, label_sequence):
        spans = []
        start_idx = None
        
        for idx, label in enumerate(label_sequence):
            if label == 'B':
                start_idx = idx
            elif label == 'S':
                spans.append((idx, idx))
            elif label == 'E' and start_idx is not None:
                spans.append((start_idx, idx))
                start_idx = None
            elif label in ['O', 'B']:
                if start_idx is not None and label == 'B':
                    spans.append((start_idx, idx - 1))
                start_idx = idx if label == 'B' else None
                    
        if start_idx is not None:
            spans.append((start_idx, len(label_sequence) - 1))
            
        return spans
    
    def calculate_all_metrics(self, all_preds, all_labels,
                            asp_preds, asp_labels,
                            opi_preds, opi_labels, 
                            sent_preds, sent_labels,
                            pred_triplets, gt_triplets):
        results = {}
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        results['token_precision'] = precision
        results['token_recall'] = recall
        results['token_f1'] = f1
        
        if asp_labels:
            asp_precision, asp_recall, asp_f1, _ = precision_recall_fscore_support(
                asp_labels, asp_preds, average='binary', zero_division=0
            )
            results['aspect_precision'] = asp_precision
            results['aspect_recall'] = asp_recall
            results['aspect_f1'] = asp_f1
        
        if opi_labels:
            opi_precision, opi_recall, opi_f1, _ = precision_recall_fscore_support(
                opi_labels, opi_preds, average='binary', zero_division=0
            )
            results['opinion_precision'] = opi_precision
            results['opinion_recall'] = opi_recall
            results['opinion_f1'] = opi_f1
        
        if sent_labels:
            sent_acc = accuracy_score(sent_labels, sent_preds)
            results['sentiment_accuracy'] = sent_acc
        
        triplet_f1, triplet_precision, triplet_recall = self.calculate_triplet_f1(
            pred_triplets, gt_triplets
        )
        results['triplet_precision'] = triplet_precision
        results['triplet_recall'] = triplet_recall
        results['triplet_f1'] = triplet_f1
        
        pair_f1, pair_precision, pair_recall = self.calculate_pair_f1(
            pred_triplets, gt_triplets
        )
        results['pair_precision'] = pair_precision
        results['pair_recall'] = pair_recall
        results['pair_f1'] = pair_f1
        
        return results
    
    def calculate_triplet_f1(self, predicted_triplets, ground_truth_triplets):
        total_predicted = 0
        total_ground_truth = 0
        total_correct = 0
        
        for pred_triplets, gt_triplets in zip(predicted_triplets, ground_truth_triplets):
            total_predicted += len(pred_triplets)
            total_ground_truth += len(gt_triplets)
            
            for pred_triplet in pred_triplets:
                pred_asp, pred_opi, pred_sent = pred_triplet
                
                for gt_triplet in gt_triplets:
                    gt_asp, gt_opi, gt_sent = gt_triplet
                    
                    if (pred_asp == gt_asp and pred_opi == gt_opi and pred_sent == gt_sent):
                        total_correct += 1
                        break
        
        precision = total_correct / total_predicted if total_predicted > 0 else 0
        recall = total_correct / total_ground_truth if total_ground_truth > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1, precision, recall
    
    def calculate_pair_f1(self, predicted_triplets, ground_truth_triplets):
        total_predicted = 0
        total_ground_truth = 0
        total_correct = 0
        
        for pred_triplets, gt_triplets in zip(predicted_triplets, ground_truth_triplets):
            pred_pairs = set()
            gt_pairs = set()
            
            for pred_asp, pred_opi, pred_sent in pred_triplets:
                pred_pairs.add((pred_asp, pred_opi))
            
            for gt_asp, gt_opi, gt_sent in gt_triplets:
                gt_pairs.add((gt_asp, gt_opi))
            
            total_predicted += len(pred_pairs)
            total_ground_truth += len(gt_pairs)
            
            for pred_pair in pred_pairs:
                if pred_pair in gt_pairs:
                    total_correct += 1
        
        precision = total_correct / total_predicted if total_predicted > 0 else 0
        recall = total_correct / total_ground_truth if total_ground_truth > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1, precision, recall
    
    def print_results(self, results):
        print("="*60)
        print("ASTE EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nMain Metrics:")
        print(f"  Pair F1:           {results.get('pair_f1', 0)*100:.2f}%")
        print(f"  Pair Precision:    {results.get('pair_precision', 0)*100:.2f}%")
        print(f"  Pair Recall:       {results.get('pair_recall', 0)*100:.2f}%")
        print(f"  Triplet F1:        {results['triplet_f1']*100:.2f}%")
        print(f"  Triplet Precision: {results['triplet_precision']*100:.2f}%")
        print(f"  Triplet Recall:    {results['triplet_recall']*100:.2f}%")
        
        print(f"\nComponent Analysis:")
        print(f"  Aspect F1:         {results.get('aspect_f1', 0)*100:.2f}%")
        print(f"  Opinion F1:        {results.get('opinion_f1', 0)*100:.2f}%")
        print(f"  Sentiment Acc:     {results.get('sentiment_accuracy', 0)*100:.2f}%")
        
        print(f"\nToken-Level Metrics:")
        print(f"  Token F1:          {results['token_f1']*100:.2f}%")
        print(f"  Token Precision:   {results['token_precision']*100:.2f}%")
        print(f"  Token Recall:      {results['token_recall']*100:.2f}%")
        
        print("\n" + "="*60)
        
        print(f"\nPaper Comparison (Table 5):")
        print(f"  Expected Pair F1:   56.10% (14res)")
        print(f"  Our Pair F1:        {results.get('pair_f1', 0)*100:.2f}%")
        print(f"  Expected Triplet F1: 51.89% (14res)")
        print(f"  Our Triplet F1:     {results['triplet_f1']*100:.2f}%")
        
        if results['triplet_f1'] >= 0.42:
            print(f"  Result: Performance within expected range")
        else:
            print(f"  Result: Performance below paper baseline")
        
        print("="*60)

    def diagnose_stage_one_predictions(self):
        print("="*80)
        print("STAGE ONE DIAGNOSTIC")
        print("="*80)
        
        aspect_exact_match = 0
        aspect_partial_match = 0
        aspect_total_pred = 0
        aspect_total_true = 0
        
        opinion_exact_match = 0
        opinion_partial_match = 0
        opinion_total_pred = 0
        opinion_total_true = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                opinion_gt = batch['opinion_labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                dep_matrix = batch['dep_matrix'].to(self.device)
                lengths = attention_mask.sum(dim=1)
                
                outputs = self.stage_one_model(input_ids, dep_matrix, lengths)
                unified_preds = torch.argmax(outputs['unified_logits'], dim=-1)
                opinion_preds = torch.argmax(outputs['opinion_logits'], dim=-1)
                
                for i in range(len(unified_preds)):
                    seq_len = int(lengths[i].item())
                    
                    pred_aspects = self._extract_spans_from_unified(unified_preds[i][:seq_len])
                    true_aspects = self._extract_spans_from_unified(labels[i][:seq_len])
                    
                    pred_opinions = self._extract_spans_from_boundary(opinion_preds[i][:seq_len])
                    true_opinions = self._extract_spans_from_boundary(opinion_gt[i][:seq_len])
                    
                    aspect_total_pred += len(pred_aspects)
                    aspect_total_true += len(true_aspects)
                    
                    for pred_span in pred_aspects:
                        if pred_span in true_aspects:
                            aspect_exact_match += 1
                        elif any(self._spans_overlap(pred_span, true_span) for true_span in true_aspects):
                            aspect_partial_match += 1
                    
                    opinion_total_pred += len(pred_opinions)
                    opinion_total_true += len(true_opinions)
                    
                    for pred_span in pred_opinions:
                        if pred_span in true_opinions:
                            opinion_exact_match += 1
                        elif any(self._spans_overlap(pred_span, true_span) for true_span in true_opinions):
                            opinion_partial_match += 1
        
        print(f"\nAspect Extraction:")
        print(f"  Predicted: {aspect_total_pred}, True: {aspect_total_true}")
        print(f"  Exact matches: {aspect_exact_match} ({aspect_exact_match/aspect_total_true*100:.1f}%)")
        print(f"  Partial matches: {aspect_partial_match} ({aspect_partial_match/aspect_total_true*100:.1f}%)")
        aspect_recall = (aspect_exact_match + aspect_partial_match) / aspect_total_true if aspect_total_true > 0 else 0
        print(f"  Recall: {aspect_recall*100:.1f}%")
        
        print(f"\nOpinion Extraction:")
        print(f"  Predicted: {opinion_total_pred}, True: {opinion_total_true}")
        print(f"  Exact matches: {opinion_exact_match} ({opinion_exact_match/opinion_total_true*100:.1f}%)")
        print(f"  Partial matches: {opinion_partial_match} ({opinion_partial_match/opinion_total_true*100:.1f}%)")
        opinion_recall = (opinion_exact_match + opinion_partial_match) / opinion_total_true if opinion_total_true > 0 else 0
        print(f"  Recall: {opinion_recall*100:.1f}%")
        
        print("\nPerformance Issues:")
        if aspect_recall < 0.7:
            print(f"  Aspect recall low ({aspect_recall*100:.1f}%)")
        if opinion_recall < 0.7:
            print(f"  Opinion recall low ({opinion_recall*100:.1f}%)")
        if aspect_total_pred / aspect_total_true > 2.0:
            print(f"  Too many aspect predictions")
        if opinion_total_pred / opinion_total_true > 2.0:
            print(f"  Too many opinion predictions")
        
        print("="*80)
        
        return {
            'aspect_recall': aspect_recall,
            'opinion_recall': opinion_recall
        }

    def _extract_spans_from_unified(self, predictions):
        spans = []
        current_span = None
        
        for i, pred_id in enumerate(predictions):
            label = self.id_to_label.get(pred_id.item(), 'O')
            
            if label.startswith('B-'):
                if current_span:
                    spans.append(tuple(current_span))
                current_span = [i, i]
            elif label.startswith('I-') or label.startswith('E-'):
                if current_span:
                    current_span[1] = i
            elif label.startswith('S-'):
                spans.append((i, i))
                current_span = None
            else:
                if current_span:
                    spans.append(tuple(current_span))
                    current_span = None
        
        if current_span:
            spans.append(tuple(current_span))
        
        return set(spans)

    def _extract_spans_from_boundary(self, predictions):
        boundary_map = {0: 'O', 1: 'B', 2: 'I', 3: 'E', 4: 'S'}
        spans = []
        current_span = None
        
        for i, pred_id in enumerate(predictions):
            label = boundary_map.get(pred_id.item(), 'O')
            
            if label == 'B':
                if current_span:
                    spans.append(tuple(current_span))
                current_span = [i, i]
            elif label == 'I' or label == 'E':
                if current_span:
                    current_span[1] = i
            elif label == 'S':
                spans.append((i, i))
                current_span = None
            else:
                if current_span:
                    spans.append(tuple(current_span))
                    current_span = None
        
        if current_span:
            spans.append(tuple(current_span))
        
        return set(spans)

    def _spans_overlap(self, span1, span2):
        return not (span1[1] < span2[0] or span2[1] < span1[0])


def main():
    parser = argparse.ArgumentParser(description='ASTE Evaluation')
    parser.add_argument('--model_dir', type=str, default='./models')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='14res')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    
    args = parser.parse_args()
    
    logger.info("ASTE Evaluation")
    
    evaluator = ASTEEvaluator(args)
    evaluator.load_models_and_data()
    
    stage_one_diagnostics = evaluator.diagnose_stage_one_predictions()
    
    results = evaluator.evaluate_comprehensive()
    evaluator.print_results(results)
    
    results_path = os.path.join(args.model_dir, f'{args.dataset}_evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()