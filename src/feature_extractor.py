import sys
import os

import torch
import esm
from transformers import T5EncoderModel, T5Tokenizer

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "iFeatureOmega-CLI"))
import iFeatureOmegaCLI as ifo

from .ccd_feature_order import CCD_INFO


class ESM:
    def __init__(self, model_name="esm1_t34_670M_UR50S", device="cpu"):
        self.device = torch.device(device)

        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model.eval()
        self.model.to(self.device)

        self.last_repr_layer = self.model.num_layers
        
        self.is_esm2 = model_name.startswith("esm2")

    def _batch(self, L, batch_size):
        seq_lens = [len(x) for x in L]
        max_len = max(seq_lens)
        for i in range(0, len(L), batch_size):
            for j in range(i, min(len(L), i + batch_size)):
                L[j] = L[j] + [self.alphabet.padding_idx] * (max_len - len(L[j]))
            yield L[i : i + batch_size], seq_lens[i : i + batch_size]

    def get_features_batch(self, seqs, batch_size=4):
        seqs = [self.alphabet.get_tok(self.alphabet.cls_idx) + seq + (self.alphabet.get_tok(self.alphabet.eos_idx) if self.is_esm2 else "") for seq in seqs]
        all_toks = [self.alphabet.encode(seq) for seq in seqs]

        for idx, batch in enumerate(self._batch(all_toks, batch_size)):
            all_features = []
            batch_toks, batch_seq_lens = batch
            features = self.model(
                torch.tensor(batch_toks).to(self.device),
                repr_layers=[self.last_repr_layer],
                return_contacts=False,
            )

            for i, seq_len in enumerate(batch_seq_lens):
                all_features.append(
                    features["representations"][self.last_repr_layer][
                        i, 1:seq_len-int(self.is_esm2), :
                    ].unsqueeze(0)
                )

            yield all_features

    def get_features_all(self, seqs, batch_size=4):
        seqs = [self.alphabet.get_tok(self.alphabet.cls_idx) + seq + (self.alphabet.get_tok(self.alphabet.eos_idx) if self.is_esm2 else "") for seq in seqs]
        all_toks = [self.alphabet.encode(seq) for seq in seqs]

        all_features = []
        for idx, batch in enumerate(self._batch(all_toks, batch_size)):
            batch_toks, batch_seq_lens = batch
            features = self.model(
                torch.tensor(batch_toks).to(self.device),
                repr_layers=[self.last_repr_layer],
                return_contacts=False,
            )

            for i, seq_len in enumerate(batch_seq_lens):
                all_features.append(
                    features["representations"][self.last_repr_layer][
                        i, 1:seq_len-int(self.is_esm2), :
                    ].unsqueeze(0)
                )

        return all_features


class ProtTrans:
    def __init__(self, model_name="Rostlab/prot_t5_xl_uniref50", device="cpu"):
        self.device = torch.device(device)

        self.model = T5EncoderModel.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    def _batch(self, L, batch_size):
        max_len = max(self.seq_lens)
        for i in range(0, len(L), batch_size):
            yield L[i : i + batch_size], self.seq_lens[i : i + batch_size]

    def get_features_batch(self, seqs, batch_size=4):
        self.seq_lens = [len(seq) for seq in seqs]
        seqs = [
            " ".join(list(seq.replace("U", "X").replace("Z", "X").replace("O", "X")))
            for seq in seqs
        ]

        for batch in self._batch(seqs, batch_size):
            all_features = []
            batch_seqs, batch_seq_lens = batch
            batch_toks = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                add_special_tokens=True,
                padding="longest",
            ).to(self.device)
            features = self.model(**batch_toks, output_hidden_states=True)

            for i, seq_len in enumerate(batch_seq_lens):
                all_features.append(
                    features.hidden_states[-1][i, :seq_len, :].unsqueeze(0)
                )

            yield all_features

    def get_features_all(self, seqs, batch_size=4):
        self.seq_lens = [len(seq) for seq in seqs]
        seqs = [
            " ".join(list(seq.replace("U", "X").replace("Z", "X").replace("O", "X")))
            for seq in seqs
        ]

        all_features = []
        for batch in self._batch(seqs, batch_size):
            batch_seqs, batch_seq_lens = batch
            batch_toks = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                add_special_tokens=True,
                padding="longest",
            ).to(self.device)
            features = self.model(**batch_toks, output_hidden_states=True)

            for i, seq_len in enumerate(batch_seq_lens):
                all_features.append(
                    features.hidden_states[-1][i, :seq_len, :].unsqueeze(0)
                )

        return all_features


class CCD:
    def __init__(self, data_dict=None, file_path=None):
        self.file_path = file_path
        self.ifo = ifo.iProtein(data_dict=data_dict, file=file_path)
        self.init_feature()

    def init_feature(self):
        feature_name_list = list(CCD_INFO.keys())

        CCD_FEATURES = torch.tensor(
            [[-1 for _ in range(887)] for _ in range(self.ifo.sequence_number)],
            dtype=torch.float
        )

        for feature_name in feature_name_list:
            self.ifo.get_descriptor(feature_name)
            encodings = self.ifo.encodings
            indices = CCD_INFO[feature_name]["indices"]
            order = CCD_INFO[feature_name]["order"]

            CCD_FEATURES[:, order] = torch.tensor(encodings.iloc[:, indices].values, dtype=torch.float)
  
        self.CCD_FEATURES = CCD_FEATURES
    
    def get_features_batch(self, batch_size=4):
        for i in range(0, len(self.CCD_FEATURES), batch_size):
            batch = [self.CCD_FEATURES[i].unsqueeze(0) for i in range(i, min(i + batch_size, len(self.CCD_FEATURES)))]
            yield batch

    def get_features_all(self):
        return [self.CCD_FEATURES[i].unsqueeze(0) for i in range(len(self.CCD_FEATURES))]
