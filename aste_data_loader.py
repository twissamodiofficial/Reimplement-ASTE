import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import spacy


class ASTEDatasetOfficial(Dataset):
    """Dataset for official ASTE format - PAPER COMPLIANT"""
    
    def __init__(self, file_path: str, word_to_id: dict, max_len: int = 128):
        self.examples = []
        self.word_to_id = word_to_id
        self.max_len = max_len
        
        # Label mappings for official format
        self.target_to_id = {'O': 0, 'T-POS': 1, 'T-NEG': 2, 'T-NEU': 3, 'TT-POS': 4, 'TT-NEG': 5, 'TT-NEU': 6}
        self.opinion_to_id = {'O': 0, 'S': 1, 'SS': 2, 'SSS': 3, 'SSSS': 4, 'SSSSS': 5, 'SSSSSS': 6}
        
        # Unified label mapping (Paper's 13-class scheme)
        self.unified_to_id = {
            'O': 0,
            'B-POS': 1, 'I-POS': 2, 'E-POS': 3, 'S-POS': 4,
            'B-NEG': 5, 'I-NEG': 6, 'E-NEG': 7, 'S-NEG': 8,
            'B-NEU': 9, 'I-NEU': 10, 'E-NEU': 11, 'S-NEU': 12
        }
        
        # Boundary labels (for multi-task learning)
        self.boundary_to_id = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}
        
        # Load spaCy for dependency parsing
        self.nlp = spacy.load("en_core_web_lg")
        
        self._load_data(file_path)
    
    def _load_ground_truth_triplets(self, file_path: str):
        """
        Load ground truth triplets from AAAI 2020 V1 pickle files
        
        The AAAI 2020 paper uses pickle files to store authoritative triplet annotations
        because the T-/TT- tag format has known ambiguities (as noted in README).
        
        Directory structure:
        - data/14res/train.txt -> data/14res/14rest_pair/train_pair.pkl
        - data/14lap/train.txt -> data/14lap/14lap_pair/train_pair.pkl
        - etc.
        
        Returns:
            List of triplets for each sentence, or None if pickle file not found
            Format per sentence: [([target_indices], [opinion_indices], sentiment_id), ...]
            - sentiment_id: 0=NEU, 1=POS, 2=NEG (AAAI 2020 encoding)
        """
        # Determine pickle file path based on data file path
        base_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        
        # Extract dataset name and split (train/test/dev)
        dataset_name = os.path.basename(base_dir)  # e.g., "14res", "14lap"
        split_name = file_name.replace('.txt', '')  # e.g., "train", "test", "dev"
        
        # AAAI 2020 V1 naming conventions for pair directories
        pair_dirs = [
            os.path.join(base_dir, f"{dataset_name.replace('res', 'rest')}_pair"),  # 14rest_pair for 14res
            os.path.join(base_dir, f"{dataset_name}_pair"),  # 14lap_pair for 14lap
            os.path.join(base_dir, "pair"),  # Generic fallback
        ]
        
        for pair_dir in pair_dirs:
            pickle_path = os.path.join(pair_dir, f"{split_name}_pair.pkl")
            if os.path.exists(pickle_path):
                try:
                    with open(pickle_path, 'rb') as f:
                        triplets = pickle.load(f)
                    return triplets
                except Exception as e:
                    print(f"⚠️  Error loading pickle file {pickle_path}: {e}")
                    return None
        
        return None
    
    def _load_data(self, file_path: str):
        """Load data from ASTE format file"""
        print(f"Loading data from {file_path}")
        
        # Try to load ground truth triplets from pickle file
        gt_triplets = self._load_ground_truth_triplets(file_path)
        if gt_triplets:
            print(f"✅ Loaded ground truth triplets from pickle file ({len(gt_triplets)} sentences)")
        else:
            print("⚠️  No pickle file found - will infer triplets from tags (may have mismatches)")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('####')
                if len(parts) != 3:
                    continue
                
                sentence = parts[0]
                target_part = parts[1]
                opinion_part = parts[2]
                
                tokens, target_labels, opinion_labels = self._parse_labels(
                    sentence, target_part, opinion_part
                )
                
                if len(tokens) > self.max_len or len(tokens) == 0:
                    continue
                
                # Create dependency matrix
                dep_matrix = self._create_dependency_matrix(tokens)
                
                # Convert to boundary labels for multi-task learning
                target_boundary_labels = self._convert_to_boundary_labels_FIXED(target_labels, 'target')
                opinion_boundary_labels = self._convert_to_boundary_labels_FIXED(opinion_labels, 'opinion')
                
                # Extract triplets - use pickle ground truth if available, else infer from tags
                if gt_triplets and line_idx < len(gt_triplets):
                    # Convert from pickle format to our internal format
                    triplets = self._convert_pickle_triplets(gt_triplets[line_idx], tokens)
                else:
                    # Fallback: infer from tags (may have mismatches)
                    triplets = self._extract_triplets_FIXED(target_labels, opinion_labels, tokens)
                
                example = {
                    'id': line_idx,
                    'tokens': tokens,
                    'target_labels': [self.target_to_id.get(label, 0) for label in target_labels],
                    'opinion_labels': [self.opinion_to_id.get(label, 0) for label in opinion_labels],
                    'unified_labels': self._convert_to_unified_bio_labels(target_labels),
                    'target_boundary': [self.boundary_to_id.get(label, 0) for label in target_boundary_labels],
                    'opinion_boundary': [self.boundary_to_id.get(label, 0) for label in opinion_boundary_labels],
                    'dep_matrix': dep_matrix,
                    'triplets': triplets,
                    'length': len(tokens)
                }
                
                self.examples.append(example)
        
        print(f"Loaded {len(self.examples)} examples from {file_path}")
    
    def _parse_labels(self, sentence: str, target_part: str, opinion_part: str):
        """Parse tokens and labels from the format"""
        target_items = target_part.split()
        tokens = []
        target_labels = []
        
        for item in target_items:
            if '=' in item:
                token, label = item.rsplit('=', 1)
                tokens.append(token)
                target_labels.append(label)
        
        opinion_items = opinion_part.split()
        opinion_labels = []
        
        for item in opinion_items:
            if '=' in item:
                _, label = item.rsplit('=', 1)
                opinion_labels.append(label)
        
        return tokens, target_labels, opinion_labels
    
    def _convert_to_boundary_labels_FIXED(self, labels: List[str], label_type: str) -> List[str]:
        """
        FIXED: Convert ASTE labels to BIES boundary labels
        Paper-compliant handling of multi-token spans
        """
        boundary_labels = ['O'] * len(labels)
        
        if label_type == 'target':
            i = 0
            while i < len(labels):
                label = labels[i]
                
                if label.startswith('T-') and not label.startswith('TT'):
                    # Start of target span
                    span_start = i
                    span_length = 1
                    sentiment = label.split('-')[1] if '-' in label else 'NEU'
                    
                    # Count continuation tokens (TT-XXX)
                    j = i + 1
                    while j < len(labels) and labels[j].startswith('TT'):
                        span_length += 1
                        j += 1
                    
                    # Assign BIES tags
                    if span_length == 1:
                        boundary_labels[i] = 'S'
                    else:
                        boundary_labels[span_start] = 'B'
                        for k in range(span_start + 1, span_start + span_length - 1):
                            boundary_labels[k] = 'I'
                        boundary_labels[span_start + span_length - 1] = 'E'
                    
                    i = span_start + span_length
                else:
                    boundary_labels[i] = 'O'
                    i += 1
        
        elif label_type == 'opinion':
            i = 0
            while i < len(labels):
                label = labels[i]
                
                if label == 'S':
                    # Start of opinion span
                    span_start = i
                    span_length = 1
                    
                    # Count continuation (SS, SSS, etc.)
                    j = i + 1
                    while j < len(labels) and len(labels[j]) > 0 and all(c == 'S' for c in labels[j]):
                        span_length += 1
                        j += 1
                    
                    # Assign BIES tags
                    if span_length == 1:
                        boundary_labels[i] = 'S'
                    else:
                        boundary_labels[span_start] = 'B'
                        for k in range(span_start + 1, span_start + span_length - 1):
                            boundary_labels[k] = 'I'
                        boundary_labels[span_start + span_length - 1] = 'E'
                    
                    i = span_start + span_length
                else:
                    boundary_labels[i] = 'O'
                    i += 1
        
        return boundary_labels
    
    def _convert_to_unified_bio_labels(self, target_labels: List[str]) -> List[int]:
        """Convert target labels to paper's unified BIO format with sentiment"""
        unified_labels = []
        
        # Identify aspect spans
        spans = []
        current_span = None
        
        for i, label in enumerate(target_labels):
            if label.startswith('T-') or label.startswith('TT-'):
                sentiment = label.split('-')[1] if '-' in label else 'NEU'
                
                if current_span is None:
                    current_span = {'start': i, 'end': i, 'sentiment': sentiment}
                else:
                    current_span['end'] = i
            else:
                if current_span is not None:
                    spans.append(current_span)
                    current_span = None
        
        if current_span is not None:
            spans.append(current_span)
        
        # Assign BIO labels
        for i in range(len(target_labels)):
            in_span = None
            for span in spans:
                if span['start'] <= i <= span['end']:
                    in_span = span
                    break
            
            if in_span is None:
                unified_labels.append(self.unified_to_id['O'])
            else:
                sentiment = in_span['sentiment']
                if in_span['start'] == in_span['end']:
                    unified_tag = f"S-{sentiment}"
                elif i == in_span['start']:
                    unified_tag = f"B-{sentiment}"
                elif i == in_span['end']:
                    unified_tag = f"E-{sentiment}"
                else:
                    unified_tag = f"I-{sentiment}"
                
                unified_labels.append(self.unified_to_id.get(unified_tag, 0))
        
        return unified_labels
    
    def _convert_pickle_triplets(self, pickle_triplets: List, tokens: List[str]) -> List[Tuple[int, int, int, int, str]]:
        """
        Convert AAAI 2020 pickle format triplets to internal format
        
        AAAI 2020 V1 Pickle format: [([target_indices], [opinion_indices], sentiment_id), ...]
        - target_indices: List of token positions for aspect (e.g., [2] or [11])
        - opinion_indices: List of token positions for opinion (e.g., [5] or [15, 16])
        - sentiment_id: Integer encoding: 0=NEU, 1=POS, 2=NEG
        
        Internal format: [(target_start, target_end, opinion_start, opinion_end, sentiment_str), ...]
        - All indices are token positions (0-indexed)
        - sentiment_str: lowercase string ('neu', 'pos', 'neg')
        
        NOTE: This mapping is consistent with codebase usage:
        - Stage One extracts aspects with sentiment from T-POS/T-NEG/T-NEU tags
        - Stage Two uses triplets for pairing validation (sentiment preserved)
        - Evaluation compares using lowercase sentiment strings
        """
        # AAAI 2020 V1 sentiment encoding (verified from pickle files)
        sentiment_map = {0: 'neu', 1: 'pos', 2: 'neg'}
        triplets = []
        
        for target_indices, opinion_indices, sentiment_id in pickle_triplets:
            if not target_indices or not opinion_indices:
                continue
            
            # Convert index lists to span boundaries (inclusive start, inclusive end)
            target_start = min(target_indices)
            target_end = max(target_indices)
            opinion_start = min(opinion_indices)
            opinion_end = max(opinion_indices)
            
            # Map sentiment ID to lowercase string (matches codebase convention)
            sentiment = sentiment_map.get(sentiment_id, 'neu')
            
            triplets.append((target_start, target_end, opinion_start, opinion_end, sentiment))
        
        return triplets
    
    def _extract_triplets_FIXED(self, target_labels: List[str], opinion_labels: List[str], 
                            tokens: List[str]) -> List[Tuple[int, int, int, int, str]]:
        """
        PAPER COMPLIANT: Extract triplets from official ASTE format
        
        Paper Evidence (Section 2.1): "The example consists of two target and opinion pairs, 
        the first pair is 'price' and 'best', the second pair is 'feature' and 'newer'. 
        Note that 'TT-POS' is only used for indicating the pairing relation with 'SS'"
        
        Official Format Rules:
        1. T-XXX marks START of aspect with sentiment XXX
        2. TT-XXX marks CONTINUATION tokens of multi-word aspects
        3. S marks START of opinion
        4. SS, SSS, etc. mark CONTINUATION tokens of multi-word opinions
        5. POSITIONAL CORRESPONDENCE: 1st T- pairs with 1st S, 2nd T-/TT- pairs with 2nd S/SS
        
        Returns:
            List of (aspect_start, aspect_end, opinion_start, opinion_end, sentiment)
        """
        # Extract target spans with sentiments
        target_spans = []
        i = 0
        while i < len(target_labels):
            label = target_labels[i]
            
            if label.startswith('T-') and not label.startswith('TT'):
                # NEW aspect starts here
                start = i
                sentiment = label.split('-')[1] if '-' in label else 'NEU'
                
                # Find continuation tokens (TT- with same sentiment)
                end = start
                j = i + 1
                while j < len(target_labels) and target_labels[j].startswith('TT'):
                    if '-' in target_labels[j]:
                        cont_sentiment = target_labels[j].split('-')[1]
                        if cont_sentiment == sentiment:
                            end = j
                            j += 1
                        else:
                            break
                    else:
                        break
                
                target_spans.append((start, end, sentiment.lower()))
                i = end + 1
            else:
                i += 1
        
        # Extract opinion spans
        opinion_spans = []
        i = 0
        while i < len(opinion_labels):
            label = opinion_labels[i]
            
            if label == 'S':
                # NEW opinion starts here
                start = i
                end = start
                
                # Find continuation tokens (SS, SSS, etc.)
                j = i + 1
                while j < len(opinion_labels):
                    if len(opinion_labels[j]) > 0 and all(c == 'S' for c in opinion_labels[j]):
                        end = j
                        j += 1
                    else:
                        break
                
                opinion_spans.append((start, end))
                i = end + 1
            else:
                i += 1
        
        # PAPER COMPLIANT: Positional correspondence pairing
        triplets = []
        
        if len(target_spans) != len(opinion_spans):
            # Mismatch in AAAI V1 format - use min count pairing
            # (This is only used as fallback when pickle files unavailable)
            min_count = min(len(target_spans), len(opinion_spans))
            target_spans = target_spans[:min_count]
            opinion_spans = opinion_spans[:min_count]
        
        # PAPER METHOD: Position-based pairing (1st with 1st, 2nd with 2nd, ...)
        for (t_start, t_end, sentiment), (o_start, o_end) in zip(target_spans, opinion_spans):
            triplets.append((t_start, t_end, o_start, o_end, sentiment))
        
        return triplets
    
    def _create_dependency_matrix(self, tokens: List[str]) -> np.ndarray:
        """Create dependency adjacency matrix"""
        sentence = " ".join(tokens)
        doc = self.nlp(sentence)
        
        seq_len = len(tokens)
        dep_matrix = np.eye(seq_len, dtype=np.float32)
        
        # Simple alignment
        token_mapping = {i: i for i in range(min(len(tokens), len(doc)))}
        
        # Add dependency edges
        for i, token in enumerate(doc):
            if i < seq_len and token.head.i < seq_len and i != token.head.i:
                dep_matrix[i, token.head.i] = 1.0
                dep_matrix[token.head.i, i] = 1.0
        
        return dep_matrix
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        token_ids = [self.word_to_id.get(token.lower(), self.word_to_id.get('<UNK>', 1)) 
                     for token in example['tokens']]
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'labels': torch.tensor(example['unified_labels'], dtype=torch.long),
            'target_labels': torch.tensor(example['target_boundary'], dtype=torch.long),
            'opinion_labels': torch.tensor(example['opinion_boundary'], dtype=torch.long),
            'attention_mask': torch.ones(len(token_ids), dtype=torch.float),
            'dep_matrix': torch.tensor(example['dep_matrix'], dtype=torch.float),
            'triplets': example['triplets'],
            'tokens': example['tokens']
        }


def collate_fn_official(batch):
    """
    Collate function for official ASTE dataset
    
    Paper uses max_len=128 (no arbitrary filtering)
    Previously had <= 50 filter which caused data loss (4 examples in 14res train)
    """
    # Remove the arbitrary 50-token filter - use all examples from batch
    filtered_batch = batch
    
    if not filtered_batch:
        return {
            'input_ids': torch.empty(0, 1, dtype=torch.long),
            'labels': torch.empty(0, 1, dtype=torch.long),
            'target_labels': torch.empty(0, 1, dtype=torch.long),
            'opinion_labels': torch.empty(0, 1, dtype=torch.long),
            'attention_mask': torch.empty(0, 1, dtype=torch.float),
            'dep_matrix': torch.empty(0, 1, 1, dtype=torch.float),
            'triplets': [],
            'tokens': []
        }
    
    max_len = max(len(item['input_ids']) for item in filtered_batch)
    
    input_ids = []
    labels = []
    target_labels = []
    opinion_labels = []
    attention_masks = []
    dep_matrices = []
    all_triplets = []
    all_tokens = []
    
    for item in filtered_batch:
        seq_len = len(item['input_ids'])
        pad_len = max_len - seq_len
        
        input_ids.append(torch.cat([item['input_ids'], torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(torch.cat([item['labels'], torch.zeros(pad_len, dtype=torch.long)]))
        target_labels.append(torch.cat([item['target_labels'], torch.zeros(pad_len, dtype=torch.long)]))
        opinion_labels.append(torch.cat([item['opinion_labels'], torch.zeros(pad_len, dtype=torch.long)]))
        attention_masks.append(torch.cat([item['attention_mask'], torch.zeros(pad_len, dtype=torch.float)]))
        
        dep_matrix = torch.zeros(max_len, max_len, dtype=torch.float)
        dep_matrix[:seq_len, :seq_len] = item['dep_matrix']
        dep_matrices.append(dep_matrix)
        
        all_triplets.append(item['triplets'])
        all_tokens.append(item['tokens'])
    
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
        'target_labels': torch.stack(target_labels),
        'opinion_labels': torch.stack(opinion_labels),
        'attention_mask': torch.stack(attention_masks),
        'dep_matrix': torch.stack(dep_matrices),
        'triplets': all_triplets,
        'tokens': all_tokens
    }


def build_vocab_from_aste(data_dirs: List[str], min_freq: int = 1) -> Dict[str, int]:
    """Build vocabulary from ASTE dataset files"""
    word_freq = {}
    
    for data_dir in data_dirs:
        for split in ['train.txt', 'dev.txt', 'test.txt']:
            file_path = os.path.join(data_dir, split)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split('####')
                        if len(parts) >= 2:
                            target_part = parts[1]
                            for item in target_part.split():
                                if '=' in item:
                                    token, _ = item.rsplit('=', 1)
                                    token = token.lower()
                                    word_freq[token] = word_freq.get(token, 0) + 1
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab