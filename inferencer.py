from Bio import SeqIO
from src.feature_extractor import ESM, ProtTrans, CCD
from predict import HyPepToxFuse_Predictor
import yaml
import torch
from tqdm import tqdm

class Inferencer:
    def __init__(self, predictor, device='cpu'):
        self.predictor = predictor
        self.esm_1 = ESM(model_name='esm1_t34_670M_UR50S', device=device)
        self.esm_2 = ESM(model_name='esm2_t36_3B_UR50D', device=device)
        self.prot_t5 = ProtTrans(model_name='Rostlab/prot_t5_xl_uniref50', device=device)
    
    def _read_fasta_file(self, fasta_file):
        keys = []
        seqs = []
        for record in SeqIO.parse(fasta_file, 'fasta'):
            keys.append(record.id)
            seqs.append(str(record.seq))
        return keys, seqs
    
    def predict_fasta_file(self, fasta_file, threshold=0.5, batch_size=4):
        keys, seqs = self._read_fasta_file(fasta_file)
        total_batch_len = (len(seqs) // batch_size) + int(len(seqs) % batch_size == 0)
        esm_1_generator = self.esm_1.get_features_batch(seqs, batch_size=batch_size)
        esm_2_generator = self.esm_2.get_features_batch(seqs, batch_size=batch_size)
        prot_t5_generator = self.prot_t5.get_features_batch(seqs, batch_size=batch_size)
        ccd_generator = CCD(fasta_file).get_features_batch(batch_size=batch_size)
        
        ALL_LABELS = []
        ALL_PROBS = []
        for esm_1_features, esm_2_features, prot_t5_features, ccd_features in tqdm(zip(esm_1_generator, esm_2_generator, prot_t5_generator, ccd_generator), total=total_batch_len):
            labels, probs = self.predictor(
                esm_2_features, esm_1_features, prot_t5_features, ccd_features, threshold=threshold
            )
            ALL_LABELS.extend(labels)
            ALL_PROBS.extend(probs)
        
        return {key: [label, prob] for key, label, prob in zip(keys, ALL_LABELS, ALL_PROBS)}

    def save_csv_file(self, outputs, output_path):
        with open(output_path, 'w') as f:
            f.write('ID,Toxicity,Probability 1,Probability 2,Probability 3,Probability 4,Probability 5\n')
            for key, value in outputs.items():
                f.write(f'{key},{value[0]},{",".join(map(str, value[1]))}\n')