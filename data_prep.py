import os
import re
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import List, Tuple, Dict
import spacy


class SemEvalParser:
    
    def __init__(self, data_dir="data"):
        self.nlp = spacy.load("en_core_web_lg")
        self.data_dir = data_dir
    
    def parse_xml_file(self, xml_path: str) -> List[Dict]:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        sentences = []
        
        for sentence in root.findall('.//sentence'):
            sent_id = sentence.get('id')
            text = sentence.find('text').text
            
            aspects = []
            aspect_terms = sentence.find('aspectTerms')
            
            if aspect_terms is not None:
                for aspect_term in aspect_terms.findall('aspectTerm'):
                    aspect = {
                        'term': aspect_term.get('term'),
                        'polarity': aspect_term.get('polarity'),
                        'from': int(aspect_term.get('from')),
                        'to': int(aspect_term.get('to'))
                    }
                    aspects.append(aspect)
            
            sentences.append({
                'id': sent_id,
                'text': text,
                'aspects': aspects
            })
        
        return sentences
    
    def extract_opinions_heuristic(self, text: str, aspects: List[Dict]) -> List[Dict]:
        doc = self.nlp(text)
        opinions = []
        
        aspect_spans = [(a['from'], a['to']) for a in aspects]
        aspect_tokens = set()
        
        for aspect in aspects:
            aspect_start, aspect_end = aspect['from'], aspect['to']
            for token in doc:
                if token.idx >= aspect_start and token.idx < aspect_end:
                    aspect_tokens.add(token)
        
        for token in doc:
            token_start = token.idx
            token_end = token.idx + len(token.text)
            
            is_aspect = any(
                start <= token_start < end or start < token_end <= end
                for start, end in aspect_spans
            )
            
            if is_aspect:
                continue
            
            is_opinion = False
            confidence = 0.0
            
            if (token.pos_ in ['ADJ', 'ADV'] and 
                not token.is_stop and len(token.text) > 2):
                is_opinion = True
                confidence = 0.8
            
            elif (token.pos_ == 'VERB' and token.lemma_ in 
                  ['love', 'hate', 'like', 'dislike', 'enjoy', 'prefer', 'recommend', 
                   'praise', 'criticize', 'appreciate', 'disapprove', 'admire']):
                is_opinion = True
                confidence = 0.7
            
            elif self._has_opinion_dependency(token, aspect_tokens, doc):
                is_opinion = True
                confidence = 0.9
            
            elif (token.pos_ == 'NOUN' and token.lemma_ in 
                  ['quality', 'problem', 'issue', 'advantage', 'benefit', 'drawback',
                   'strength', 'weakness', 'excellence', 'perfection', 'disaster']):
                is_opinion = True
                confidence = 0.6
            
            if is_opinion:
                opinions.append({
                    'term': token.text,
                    'from': token_start,
                    'to': token_end,
                    'pos': token.pos_,
                    'confidence': confidence,
                    'dependency_rel': token.dep_,
                    'head': token.head.text if token.head != token else None
                })
        
        opinions = [op for op in opinions if op['confidence'] >= 0.5]
        opinions = self._remove_overlapping_opinions(opinions)
        
        return opinions
    
    def _has_opinion_dependency(self, token, aspect_tokens, doc) -> bool:
        opinion_deps = {
            'amod',
            'advmod',
            'nsubj',
            'acomp',
            'xcomp',
            'ccomp',
            'dobj',
        }
        
        for aspect_token in aspect_tokens:
            if (token.head == aspect_token and token.dep_ in opinion_deps):
                return True
                
            if (aspect_token.head == token and aspect_token.dep_ in opinion_deps):
                return True
                
            if (token.head == aspect_token.head and 
                token.dep_ in opinion_deps and aspect_token.dep_ in opinion_deps):
                return True
        
        if token.dep_ == 'attr' and token.head.lemma_ == 'be':
            for child in token.head.children:
                if child in aspect_tokens:
                    return True
        
        return False
    
    def _remove_overlapping_opinions(self, opinions: List[Dict]) -> List[Dict]:
        opinions = sorted(opinions, key=lambda x: x['confidence'], reverse=True)
        filtered = []
        
        for opinion in opinions:
            overlaps = any(
                (opinion['from'] < existing['to'] and opinion['to'] > existing['from'])
                for existing in filtered
            )
            
            if not overlaps:
                filtered.append(opinion)
        
        return filtered
    
    def get_train_examples(self):
        train_file = os.path.join(self.data_dir, 'train_improved.json')
        if os.path.exists(train_file):
            with open(train_file, 'r') as f:
                return json.load(f)
        else:
            train_file = os.path.join(self.data_dir, 'train.json')
            if os.path.exists(train_file):
                with open(train_file, 'r') as f:
                    return json.load(f)
            else:
                raise FileNotFoundError(f"Training file not found: {train_file}")
    
    def get_test_examples(self):
        test_file = os.path.join(self.data_dir, 'test_improved.json')
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                return json.load(f)
        else:
            test_file = os.path.join(self.data_dir, 'test.json')
            if os.path.exists(test_file):
                with open(test_file, 'r') as f:
                    return json.load(f)
            else:
                raise FileNotFoundError(f"Test file not found: {test_file}")
    
    def build_vocab(self, examples):
        vocab = {'<PAD>': 0, '<UNK>': 1}
        vocab_idx = 2
        
        for example in examples:
            for token in example['tokens']:
                if token.lower() not in vocab:
                    vocab[token.lower()] = vocab_idx
                    vocab_idx += 1
        
        id_to_word = {v: k for k, v in vocab.items()}
        
        return vocab, id_to_word


class ASTEDataConverter:
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        
        self.unified_tags = {
            'B-POS': 0, 'I-POS': 1, 'E-POS': 2, 'S-POS': 3,
            'B-NEG': 4, 'I-NEG': 5, 'E-NEG': 6, 'S-NEG': 7, 
            'B-NEU': 8, 'I-NEU': 9, 'E-NEU': 10, 'S-NEU': 11,
            'O': 12
        }
        
        self.target_tags = {
            'B': 0, 'I': 1, 'E': 2, 'S': 3, 'O': 4
        }
        
        self.opinion_tags = {
            'B': 0, 'I': 1, 'E': 2, 'S': 3, 'O': 4
        }
        
        self.sentiment_map = {
            'positive': 'POS',
            'negative': 'NEG', 
            'neutral': 'NEU',
            'conflict': 'NEU'
        }
    
    def tokenize_sentence(self, text: str) -> Tuple[List[str], List[int]]:
        doc = self.nlp(text)
        tokens = []
        offsets = []
        
        for token in doc:
            tokens.append(token.text)
            offsets.append((token.idx, token.idx + len(token.text)))
        
        return tokens, offsets
    
    def align_spans_to_tokens(self, span_start: int, span_end: int, 
                              offsets: List[Tuple[int, int]]) -> Tuple[int, int]:
        token_start = -1
        token_end = -1
        
        for i, (offset_start, offset_end) in enumerate(offsets):
            if token_start == -1 and offset_start >= span_start:
                token_start = i
            
            if offset_end <= span_end:
                token_end = i
            elif token_start != -1:
                break
        
        if token_start == -1:
            token_start = 0
        if token_end == -1 or token_end < token_start:
            token_end = token_start
        
        return token_start, token_end
    
    def create_bio_tags(self, tokens: List[str], spans: List[Tuple[int, int, str]], 
                        tag_type: str) -> List[str]:
        tags = ['O'] * len(tokens)
        
        for start_idx, end_idx, label in spans:
            if start_idx > end_idx or start_idx >= len(tokens):
                continue
                
            end_idx = min(end_idx, len(tokens) - 1)
            
            if tag_type == 'unified':
                if start_idx == end_idx:
                    tags[start_idx] = f'S-{label}'
                else:
                    tags[start_idx] = f'B-{label}'
                    for i in range(start_idx + 1, end_idx):
                        tags[i] = f'I-{label}'
                    tags[end_idx] = f'E-{label}'
            
            elif tag_type in ['target', 'opinion']:
                if start_idx == end_idx:
                    tags[start_idx] = 'S'
                else:
                    tags[start_idx] = 'B'
                    for i in range(start_idx + 1, end_idx):
                        tags[i] = 'I'
                    tags[end_idx] = 'E'
        
        return tags
    
    def convert_to_aste_format(self, sentences: List[Dict], 
                               extract_opinions: bool = True) -> List[Dict]:
        parser = SemEvalParser()
        aste_data = []
        
        for sent_data in sentences:
            text = sent_data['text']
            aspects = sent_data['aspects']
            
            tokens, offsets = self.tokenize_sentence(text)
            
            aspect_spans = []
            target_spans = []
            
            for aspect in aspects:
                start_char = aspect['from']
                end_char = aspect['to']
                sentiment = self.sentiment_map.get(aspect['polarity'].lower(), 'NEU')
                
                start_tok, end_tok = self.align_spans_to_tokens(
                    start_char, end_char, offsets)
                
                aspect_spans.append((start_tok, end_tok, sentiment))
                target_spans.append((start_tok, end_tok, 'TARGET'))
            
            opinion_spans = []
            if extract_opinions:
                opinions = parser.extract_opinions_heuristic(text, aspects)
                
                for opinion in opinions:
                    start_char = opinion['from']
                    end_char = opinion['to']
                    
                    start_tok, end_tok = self.align_spans_to_tokens(
                        start_char, end_char, offsets)
                    
                    opinion_spans.append((start_tok, end_tok, 'OPINION'))
            
            unified_tags = self.create_bio_tags(tokens, aspect_spans, 'unified')
            target_tags = self.create_bio_tags(tokens, target_spans, 'target')
            opinion_tags = self.create_bio_tags(tokens, opinion_spans, 'opinion')
            
            unified_indices = [self.unified_tags.get(tag, self.unified_tags['O']) 
                             for tag in unified_tags]
            target_indices = [self.target_tags.get(tag, self.target_tags['O']) 
                            for tag in target_tags]
            opinion_indices = [self.opinion_tags.get(tag, self.opinion_tags['O']) 
                             for tag in opinion_tags]
            
            aste_data.append({
                'id': sent_data['id'],
                'text': text,
                'tokens': tokens,
                'unified_tags': unified_indices,
                'target_tags': target_indices, 
                'opinion_tags': opinion_indices,
                'tg_tags': target_indices.copy(),
                'aspects': aspects,
                'opinions': opinions if extract_opinions else [],
                'length': len(tokens)
            })
        
        return aste_data


def create_sample_data(output_dir: str = 'data/'):
    os.makedirs(output_dir, exist_ok=True)
    
    samples = [
        {
            'text': "The food was delicious but the service was terrible.",
            'aspects': [
                {'term': 'food', 'polarity': 'positive', 'from': 4, 'to': 8},
                {'term': 'service', 'polarity': 'negative', 'from': 33, 'to': 40}
            ]
        },
        {
            'text': "Great atmosphere and friendly staff.",
            'aspects': [
                {'term': 'atmosphere', 'polarity': 'positive', 'from': 6, 'to': 16},
                {'term': 'staff', 'polarity': 'positive', 'from': 30, 'to': 35}
            ]
        },
        {
            'text': "The pasta is average.",
            'aspects': [
                {'term': 'pasta', 'polarity': 'neutral', 'from': 4, 'to': 9}
            ]
        }
    ]
    
    converter = ASTEDataConverter()
    aste_data = []
    
    for i, sample in enumerate(samples):
        sample['id'] = f'sample_{i}'
        
    aste_data = converter.convert_to_aste_format(samples)
    
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(aste_data, f, indent=2)
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    vocab_idx = 2
    
    for sample in aste_data:
        for token in sample['tokens']:
            if token.lower() not in vocab:
                vocab[token.lower()] = vocab_idx
                vocab_idx += 1
    
    with open(os.path.join(output_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab, f, indent=2)
    
    print(f"Created sample data with {len(aste_data)} sentences")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Data saved to {output_dir}")


def convert_semeval_data(xml_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    parser = SemEvalParser()
    sentences = parser.parse_xml_file(xml_path)
    
    converter = ASTEDataConverter()
    aste_data = converter.convert_to_aste_format(sentences)
    
    output_file = os.path.join(output_dir, f"{os.path.basename(xml_path).split('.')[0]}.json")
    
    with open(output_file, 'w') as f:
        json.dump(aste_data, f, indent=2)
    
    print(f"Converted {len(aste_data)} sentences from {xml_path}")
    print(f"Saved to {output_file}")
    
    return aste_data


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 2:
        xml_path = sys.argv[1]
        output_dir = sys.argv[2]
        convert_semeval_data(xml_path, output_dir)
    else:
        create_sample_data()