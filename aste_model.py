import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from typing import List, Tuple, Dict
import spacy


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, num_layers=2):
        super(GCNLayer, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(in_features, out_features))
            else:
                self.layers.append(nn.Linear(out_features, out_features))
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, adj):
        batch_size, seq_len, _ = x.size()
        
        if adj is None:
            adj = torch.eye(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len, seq_len)
        
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
        elif adj.dim() == 1:
            adj = torch.eye(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len, seq_len)
        
        if adj.size(0) != batch_size or adj.size(1) != seq_len or adj.size(2) != seq_len:
            adj = torch.eye(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len, seq_len)
        
        rowsum = adj.sum(2)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        
        adj_norm = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        
        for i, layer in enumerate(self.layers):
            x = torch.bmm(adj_norm, x)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        return x


class SentimentConsistency(nn.Module):
    def __init__(self, hidden_size):
        super(SentimentConsistency, self).__init__()
        self.hidden_size = hidden_size
        self.gate = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, h_s, target_mask=None):
        batch_size, seq_len, hidden_size = h_s.size()
        device = h_s.device
        
        g = torch.sigmoid(self.gate(h_s.view(-1, hidden_size))).view(batch_size, seq_len, hidden_size)
        
        h_s_tilde_list = []
        
        h_s_tilde_list.append(g[:, 0, :] * h_s[:, 0, :])
        
        for t in range(1, seq_len):
            gated_input = g[:, t, :] * h_s[:, t, :]
            
            prev_contrib = (1 - g[:, t, :]) * h_s_tilde_list[-1]
            
            h_s_tilde_list.append(gated_input + prev_contrib)
        
        h_s_tilde = torch.stack(h_s_tilde_list, dim=1)
        
        return h_s_tilde


class BoundaryGuidance(nn.Module):
    def __init__(self, target_tag_size, unified_tag_size):
        super(BoundaryGuidance, self).__init__()
        self.register_buffer('W_tr', self._init_transformation_matrix(target_tag_size, unified_tag_size))
    
    def _init_transformation_matrix(self, target_tag_size, unified_tag_size):
        W_tr = torch.zeros(target_tag_size, unified_tag_size)
        
        tag_map = {
            1: [1, 5, 9],
            2: [2, 6, 10],
            3: [3, 7, 11],
            4: [4, 8, 12],
            0: [0]
        }
        
        for src, tgt_list in tag_map.items():
            for tgt in tgt_list:
                W_tr[src, tgt] = 1.0 / len(tgt_list)
        
        return W_tr
    
    def forward(self, z_t, z_s, alpha):
        z_s_prime = torch.matmul(z_t, self.W_tr)
        
        alpha = alpha.unsqueeze(-1)
        z_ts = alpha * z_s_prime + (1 - alpha) * z_s
        
        return z_ts


class StageOneModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1, dropout=0.5):
        super(StageOneModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.embedding.weight.data[0].fill_(0)
        
        self.lstm_t = nn.LSTM(embed_dim, hidden_size, num_layers, 
                              batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        
        self.lstm_s = nn.LSTM(hidden_size * 2, hidden_size, num_layers,
                              batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        
        self.lstm_opt = nn.LSTM(embed_dim + hidden_size * 4, hidden_size, num_layers,
                                batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        
        self.gcn = GCNLayer(embed_dim, hidden_size * 2, num_layers=2)
        
        self.sc = SentimentConsistency(hidden_size * 2)
        
        self.bg = BoundaryGuidance(target_tag_size=5, unified_tag_size=13)
        
        self.target_classifier = nn.Linear(hidden_size * 2, 5)
        self.unified_classifier = nn.Linear(hidden_size * 4, 13)
        self.tg_classifier = nn.Linear(hidden_size * 4, 5)
        self.opinion_classifier = nn.Linear(hidden_size * 2, 5)
        
        self.dropout = nn.Dropout(dropout)
        
        self.epsilon = 0.5
        
        self.layer_norm_t = nn.LayerNorm(hidden_size * 2)
        self.layer_norm_s = nn.LayerNorm(hidden_size * 2)
        self.layer_norm_o = nn.LayerNorm(hidden_size * 2)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name.lower():
                    nn.init.orthogonal_(param)
                elif len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                if 'lstm' in name.lower():
                    nn.init.zeros_(param)
                    if 'bias_hh' in name:
                        hidden_size = param.shape[0] // 4
                        param.data[hidden_size:2*hidden_size].fill_(1.0)
                else:
                    nn.init.zeros_(param)
    
    def create_target_mask(self, target_labels):
        return (target_labels != 4).float()
    
    def forward(self, x, adj_matrix, lengths, target_labels=None):
        batch_size, seq_len = x.size()
        
        embed = self.embedding(x)
        embed = self.dropout(embed)
        
        device = embed.device
        
        if batch_size == 1 or lengths is None:
            h_t, _ = self.lstm_t(embed)
        else:
            if isinstance(lengths, torch.Tensor):
                lengths_int = lengths.long().flatten().cpu()
            else:
                lengths_int = torch.tensor(lengths).long().flatten().cpu()
            
            packed_embed = pack_padded_sequence(embed, lengths_int, batch_first=True, enforce_sorted=False)
            packed_h_t, _ = self.lstm_t(packed_embed)
            h_t, _ = pad_packed_sequence(packed_h_t, batch_first=True, total_length=seq_len)

        h_t = self.layer_norm_t(h_t)
        h_t = self.dropout(h_t)
        
        z_t = self.target_classifier(h_t)
        z_t_prob = F.softmax(z_t, dim=-1)
        
        if batch_size == 1 or lengths is None:
            h_s, _ = self.lstm_s(h_t)
        else:
            packed_h_t = pack_padded_sequence(h_t, lengths_int, batch_first=True, enforce_sorted=False)
            packed_h_s, _ = self.lstm_s(packed_h_t)
            h_s, _ = pad_packed_sequence(packed_h_s, batch_first=True, total_length=seq_len)
        
        h_s = self.layer_norm_s(h_s)
        h_s = self.dropout(h_s)
        
        target_mask = None
        if target_labels is not None:
            target_mask = self.create_target_mask(target_labels)
        else:
            pred_target_labels = torch.argmax(z_t_prob, dim=-1)
            target_mask = self.create_target_mask(pred_target_labels)
        
        h_s_tilde = self.sc(h_s, target_mask)
        
        h_o_gcn = self.gcn(embed, adj_matrix)
        h_o_gcn = self.dropout(h_o_gcn)
        
        h_tg = torch.cat([h_t, h_o_gcn], dim=-1)
        z_tg = self.tg_classifier(h_tg)
        
        h_tg_input = torch.cat([embed, h_tg], dim=-1)
        
        if batch_size == 1 or lengths is None:
            h_opt, _ = self.lstm_opt(h_tg_input)
        else:
            packed_h_tg = pack_padded_sequence(h_tg_input, lengths_int, batch_first=True, enforce_sorted=False)
            packed_h_opt, _ = self.lstm_opt(packed_h_tg)
            h_opt, _ = pad_packed_sequence(packed_h_opt, batch_first=True, total_length=seq_len)
        
        h_opt = self.layer_norm_o(h_opt)
        h_opt = self.dropout(h_opt)
        
        z_opt = self.opinion_classifier(h_opt)
        
        h_u = torch.cat([h_s_tilde, h_opt], dim=-1)
        z_s = self.unified_classifier(h_u)
        z_s_prob = F.softmax(z_s, dim=-1)
        
        c_t = torch.sum(z_t_prob * z_t_prob, dim=-1)
        
        alpha_final = self.epsilon * c_t
        
        z_ts = self.bg(z_t_prob, z_s_prob, alpha_final)
        
        z_ts_logit = torch.log(z_ts + 1e-8)
        
        return {
            'target_logits': z_t,
            'unified_logits': z_ts_logit,
            'tg_logits': z_tg,
            'opinion_logits': z_opt,
            'h_t': h_t,
            'h_opt': h_opt,
            'embeddings': embed
        }


class StageTwoModel(nn.Module):
    def __init__(self, embed_dim=300, hidden_size=300, max_distance=100, num_layers=1, dropout=0.5):
        super(StageTwoModel, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.max_distance = max_distance
        
        position_dim = 50
        self.position_embedding = nn.Embedding(max_distance + 1, position_dim)
        nn.init.xavier_uniform_(self.position_embedding.weight)
        
        self.lstm = nn.LSTM(
            embed_dim + position_dim,
            hidden_size, 
            num_layers,
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, 2)
        )
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    @staticmethod
    def calculate_paper_position_indices(seq_len, aspect_span, opinion_span):
        aspect_start, aspect_end = aspect_span
        opinion_start, opinion_end = opinion_span
        
        aspect_start = int(aspect_start)
        aspect_end = int(aspect_end)
        opinion_start = int(opinion_start)
        opinion_end = int(opinion_end)
        seq_len = int(seq_len)
        
        aspect_start = max(0, min(aspect_start, seq_len - 1))
        aspect_end = max(aspect_start, min(aspect_end, seq_len - 1))
        opinion_start = max(0, min(opinion_start, seq_len - 1))
        opinion_end = max(opinion_start, min(opinion_end, seq_len - 1))
        
        aspect_center = (aspect_start + aspect_end) / 2.0
        opinion_center = (opinion_start + opinion_end) / 2.0
        distance = int(abs(aspect_center - opinion_center))
        
        distance = min(distance, 100)
        
        position_indices = torch.zeros(seq_len, dtype=torch.long)
        
        for i in range(aspect_start, aspect_end + 1):
            if i < seq_len:
                position_indices[i] = distance
        
        for i in range(opinion_start, opinion_end + 1):
            if i < seq_len:
                position_indices[i] = distance
        
        return position_indices

    def forward(self, word_embeds, aspect_spans, opinion_spans, lengths):
        batch_size, seq_len, embed_dim = word_embeds.size()
        device = word_embeds.device
        
        position_indices = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        
        for i in range(batch_size):
            asp_start = int(aspect_spans[i, 0].item())
            asp_end = int(aspect_spans[i, 1].item())
            opi_start = int(opinion_spans[i, 0].item())
            opi_end = int(opinion_spans[i, 1].item())
            
            seq_len_curr = min(lengths[i].item(), seq_len) if lengths is not None else seq_len
            asp_start = max(0, min(asp_start, seq_len_curr - 1))
            asp_end = max(asp_start, min(asp_end, seq_len_curr - 1))
            opi_start = max(0, min(opi_start, seq_len_curr - 1))
            opi_end = max(opi_start, min(opi_end, seq_len_curr - 1))
            
            pos_indices = self.calculate_paper_position_indices(
                seq_len_curr, (asp_start, asp_end), (opi_start, opi_end)
            )
            
            position_indices[i, :len(pos_indices)] = pos_indices[:seq_len]
        
        pos_embeddings = self.position_embedding(position_indices)
        
        combined = torch.cat([word_embeds, pos_embeddings], dim=-1)
        combined = self.dropout(combined)
        
        if lengths is not None:
            packed_input = pack_padded_sequence(combined, lengths.long().cpu(), batch_first=True, enforce_sorted=False)
            packed_output, _ = self.lstm(packed_input)
            output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=seq_len)
        else:
            output, _ = self.lstm(combined)
        
        output = self.dropout(output)
        
        aspect_reprs = []
        opinion_reprs = []
        
        for i in range(batch_size):
            asp_start = int(aspect_spans[i, 0].item())
            asp_end = int(aspect_spans[i, 1].item())
            opi_start = int(opinion_spans[i, 0].item())
            opi_end = int(opinion_spans[i, 1].item())
            
            seq_len_curr = min(int(lengths[i].item()), seq_len) if lengths is not None else seq_len
            asp_start = max(0, min(asp_start, seq_len_curr - 1))
            asp_end = max(asp_start, min(asp_end, seq_len_curr - 1))
            opi_start = max(0, min(opi_start, seq_len_curr - 1))
            opi_end = max(opi_start, min(opi_end, seq_len_curr - 1))
            
            aspect_output = output[i, asp_start:asp_end + 1, :]
            opinion_output = output[i, opi_start:opi_end + 1, :]
            
            aspect_repr = torch.mean(aspect_output, dim=0)
            opinion_repr = torch.mean(opinion_output, dim=0)
            
            aspect_reprs.append(aspect_repr)
            opinion_reprs.append(opinion_repr)
        
        aspect_repr = torch.stack(aspect_reprs, dim=0)
        opinion_repr = torch.stack(opinion_reprs, dim=0)
        
        pair_repr = torch.cat([aspect_repr, opinion_repr], dim=-1)
        pair_logits = self.classifier(pair_repr)
        
        return pair_logits


class ASTEModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_size=300, 
                 num_layers=1, dropout=0.5, max_distance=100):
        super(ASTEModel, self).__init__()
        
        self.stage_one = StageOneModel(vocab_size, embed_dim, hidden_size, num_layers, dropout)
        self.stage_two = StageTwoModel(embed_dim, hidden_size, max_distance, num_layers, dropout)
    
    def forward_stage_one(self, x, adj_matrix, lengths):
        return self.stage_one(x, adj_matrix, lengths)
    
    def forward_stage_two(self, word_embeds, aspect_positions, opinion_positions, lengths):
        return self.stage_two(word_embeds, aspect_positions, opinion_positions, lengths)


def compute_stage_one_loss(outputs, target_labels, unified_labels, tg_labels, opinion_labels, loss_weights=None):
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    if loss_weights is None:
        loss_weights = {
            'target': 1.0,
            'unified': 1.0, 
            'tg': 1.0,
            'opinion': 1.0
        }
    
    target_logits = outputs['target_logits'].view(-1, 5)
    unified_logits = outputs['unified_logits'].view(-1, 13)
    tg_logits = outputs['tg_logits'].view(-1, 5)
    opinion_logits = outputs['opinion_logits'].view(-1, 5)
    
    target_labels = target_labels.view(-1)
    unified_labels = unified_labels.view(-1)
    tg_labels = tg_labels.view(-1)
    opinion_labels = opinion_labels.view(-1)
    
    loss_t = criterion(target_logits, target_labels)
    loss_ts = criterion(unified_logits, unified_labels)
    loss_tg = criterion(tg_logits, tg_labels)
    loss_opt = criterion(opinion_logits, opinion_labels)
    
    total_loss = (loss_weights['target'] * loss_t + 
                  loss_weights['unified'] * loss_ts + 
                  loss_weights['tg'] * loss_tg + 
                  loss_weights['opinion'] * loss_opt)
    
    return total_loss, {
        'loss_t': loss_t.item(),
        'loss_ts': loss_ts.item(),
        'loss_tg': loss_tg.item(),
        'loss_opt': loss_opt.item(),
        'total_loss': total_loss.item()
    }


def compute_stage_two_loss(pair_logits, pair_labels):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(pair_logits, pair_labels)
    return loss


def extract_spans(labels, tag_schema='BIO'):
    spans = []
    start = -1
    
    if tag_schema == 'unified':
        tag_names = ['B-POS', 'I-POS', 'E-POS', 'S-POS', 
                     'B-NEG', 'I-NEG', 'E-NEG', 'S-NEG',
                     'B-NEU', 'I-NEU', 'E-NEU', 'S-NEU', 'O']
    else:
        tag_names = ['B', 'I', 'E', 'S', 'O']
    
    for i, label in enumerate(labels):
        tag = tag_names[label] if label < len(tag_names) else 'O'
        
        if tag.startswith('B-') or tag == 'B':
            if start != -1:
                spans.append((start, i-1, prev_sentiment if tag_schema == 'unified' else None))
            start = i
            prev_sentiment = tag.split('-')[1] if '-' in tag else None
        elif tag.startswith('S-') or tag == 'S':
            sentiment = tag.split('-')[1] if '-' in tag else None
            spans.append((i, i, sentiment))
            start = -1
        elif tag.startswith('E-') or tag == 'E':
            if start != -1:
                sentiment = tag.split('-')[1] if '-' in tag else None
                spans.append((start, i, sentiment))
            start = -1
        elif tag == 'O' or tag.startswith('O'):
            if start != -1 and tag_schema != 'unified':
                spans.append((start, i-1, None))
            start = -1
    
    if start != -1:
        spans.append((start, len(labels)-1, prev_sentiment if tag_schema == 'unified' else None))
    
    return spans


def create_dependency_matrix(sentences, nlp=None):
    if nlp is None:
        nlp = spacy.load("en_core_web_lg")
    
    adj_matrices = []
    
    for sentence in sentences:
        if isinstance(sentence, list):
            text = " ".join(sentence)
        else:
            text = sentence
            
        doc = nlp(text)
        tokens = [token.text for token in doc]
        n_tokens = len(tokens)
        
        adj_matrix = np.zeros((n_tokens, n_tokens), dtype=np.float32)
        
        for token in doc:
            i = token.i
            adj_matrix[i, i] = 1.0
            
            if token.head != token:
                j = token.head.i
                adj_matrix[i, j] = 1.0
                adj_matrix[j, i] = 1.0
            
            for child in token.children:
                j = child.i
                adj_matrix[i, j] = 1.0
                adj_matrix[j, i] = 1.0
        
        adj_matrices.append(adj_matrix)
    
    return adj_matrices