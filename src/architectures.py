import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn import MultiheadAttention, TransformerEncoder, TransformerEncoderLayer

class HyPepToxFuse_NLP(nn.Module):
    def __init__(self, input_dim_1,
                        input_dim_2,
                        input_dim_3,
                        gated_dim,
                        num_heads_attn=8,
                        num_heads_transformer=8,
                        num_layers_transformers=4,
                        n_classes=2,
                        drop=0.0):
        
        super(HyPepToxFuse_NLP, self).__init__()
        
        self.fc_1 = nn.Linear(input_dim_1, gated_dim)
        self.fc_2 = nn.Linear(input_dim_2, gated_dim)
        self.fc_3 = nn.Linear(input_dim_3, gated_dim)
        
        self.drop_1 = nn.Dropout(p=drop) 
        self.drop_2 = nn.Dropout(p=drop)
        self.drop_3 = nn.Dropout(p=drop)
        
        self.attn_12 = MultiheadAttention(embed_dim=gated_dim, num_heads=num_heads_attn, batch_first=True)
        self.attn_23 = MultiheadAttention(embed_dim=gated_dim, num_heads=num_heads_attn, batch_first=True)
        self.attn_31 = MultiheadAttention(embed_dim=gated_dim, num_heads=num_heads_attn, batch_first=True)
        

        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=gated_dim, 
                                    nhead=num_heads_transformer, 
                                    batch_first=True,
                                    activation='gelu'),
            num_layers=num_layers_transformers
        )
        
        self.cls_embeddings = nn.Parameter(torch.randn(1, 1, gated_dim))
        init.normal_(self.cls_embeddings, std=0.02)
        
        self.mlp_head = nn.Sequential(
            nn.Linear(gated_dim, gated_dim),
            nn.LayerNorm(gated_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(gated_dim, n_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        
    def forward(self, f1, f2, f3,
                mask_f1=None, mask_f2=None, mask_f3=None):
        fc_1 = self.drop_1(self.fc_1(f1))
        fc_2 = self.drop_2(self.fc_2(f2))
        fc_3 = self.drop_3(self.fc_3(f3))
        
        attn_12_out, _ = self.attn_12(fc_1, fc_2, fc_2, key_padding_mask=mask_f2)
        attn_23_out, _ = self.attn_23(fc_2, fc_3, fc_3, key_padding_mask=mask_f3)
        attn_31_out, _ = self.attn_31(fc_3, fc_1, fc_1, key_padding_mask=mask_f1)
        
        h_123 = torch.cat([attn_12_out, attn_23_out, attn_31_out], dim=1)
        
        # Update the padding mask
        if mask_f1 is not None and mask_f2 is not None and mask_f3 is not None:
            combined_mask = torch.cat([mask_f1, mask_f2, mask_f3], dim=1)
            cls_mask = torch.zeros((mask_f1.size(0), 1), dtype=torch.bool, device=mask_f1.device)
            combined_mask = torch.cat([cls_mask, combined_mask], dim=1)
        else:
            combined_mask = None
            
        batch_size = h_123.size(0)
        cls_embeddings = self.cls_embeddings.expand(batch_size, -1, -1)
        h_123 = torch.cat([cls_embeddings, h_123], dim=1)
        
        h_123 = self.transformer_encoder(h_123, src_key_padding_mask=combined_mask)
        
        cls_output = h_123[:, 0, :]
        
        final_h = self.mlp_head(cls_output)
        
        prob = self.classifier(final_h)
        
        return prob, final_h

    
class HyPepToxFuse_Hybrid(nn.Module):
    def __init__(self, input_dim_1,
                        input_dim_2,
                        input_dim_3,
                        handcraft_dim,
                        gated_dim,
                        num_heads_attn=8,
                        num_heads_transformer=8,
                        num_layers_transformers=4,
                        num_mlp_layers=4,
                        n_classes=2,
                        drop=0.0):
        
        super(HyPepToxFuse_Hybrid, self).__init__()
        
        self.fc_1 = nn.Linear(input_dim_1, gated_dim)
        self.fc_2 = nn.Linear(input_dim_2, gated_dim)
        self.fc_3 = nn.Linear(input_dim_3, gated_dim)
        
        self.drop_1 = nn.Dropout(p=drop) 
        self.drop_2 = nn.Dropout(p=drop)
        self.drop_3 = nn.Dropout(p=drop)
        
        self.attn_12 = MultiheadAttention(embed_dim=gated_dim, num_heads=num_heads_attn, batch_first=True)
        self.attn_23 = MultiheadAttention(embed_dim=gated_dim, num_heads=num_heads_attn, batch_first=True)
        self.attn_31 = MultiheadAttention(embed_dim=gated_dim, num_heads=num_heads_attn, batch_first=True)
        

        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=gated_dim, 
                                    nhead=num_heads_transformer, 
                                    batch_first=True,
                                    activation='gelu'),
            num_layers=num_layers_transformers
        )
        
        self.cls_embeddings = nn.Parameter(torch.randn(1, 1, gated_dim))
        init.normal_(self.cls_embeddings, std=0.02)
        
        self.mlp_head = []
        current_dim = handcraft_dim + gated_dim
        for _ in range(num_mlp_layers):
            self.mlp_head.append(nn.Linear(current_dim, current_dim // 2))
            self.mlp_head.append(nn.LayerNorm(current_dim // 2))
            self.mlp_head.append(nn.GELU())
            # self.mlp_head.append(nn.Dropout(p=drop))
            current_dim = current_dim // 2
        self.mlp_head = nn.Sequential(*self.mlp_head)
        self.classifier = nn.Linear(current_dim, n_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        
    def forward(self, f1, f2, f3, hc_f,
                mask_f1=None, mask_f2=None, mask_f3=None):
        fc_1 = self.drop_1(self.fc_1(f1))
        fc_2 = self.drop_2(self.fc_2(f2))
        fc_3 = self.drop_3(self.fc_3(f3))
        
        attn_12_out, _ = self.attn_12(fc_1, fc_2, fc_2, key_padding_mask=mask_f2)
        attn_23_out, _ = self.attn_23(fc_2, fc_3, fc_3, key_padding_mask=mask_f3)
        attn_31_out, _ = self.attn_31(fc_3, fc_1, fc_1, key_padding_mask=mask_f1)
        
        
        h_123 = torch.cat([attn_12_out, attn_23_out, attn_31_out], dim=1)
        
        # Update the padding mask
        if mask_f1 is not None and mask_f2 is not None and mask_f3 is not None:
            combined_mask = torch.cat([mask_f1, mask_f2, mask_f3], dim=1)
            cls_mask = torch.zeros((mask_f1.size(0), 1), dtype=torch.bool, device=mask_f1.device)
            combined_mask = torch.cat([cls_mask, combined_mask], dim=1)
        else:
            combined_mask = None
            
        batch_size = h_123.size(0)
        cls_embeddings = self.cls_embeddings.expand(batch_size, -1, -1)
        h_123 = torch.cat([cls_embeddings, h_123], dim=1)
        
        h_123 = self.transformer_encoder(h_123, src_key_padding_mask=combined_mask)
        
        cls_output = h_123[:, 0, :]
        
        combined_output = torch.cat([cls_output, hc_f], dim=-1)
        final_h = self.mlp_head(combined_output)
        
        prob = self.classifier(final_h)
        
        return prob, final_h