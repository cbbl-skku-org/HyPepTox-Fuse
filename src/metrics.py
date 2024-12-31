from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, roc_curve
import math
import numpy as np

def calculate_metrics(probs, predictions, targets):
    # Calculate true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN)
    tp = ((predictions == 1) & (targets == 1)).sum()
    tn = ((predictions == 0) & (targets == 0)).sum()
    fp = ((predictions == 1) & (targets == 0)).sum()
    fn = ((predictions == 0) & (targets == 1)).sum()
    
    sn = sensitivity(tp, fn)
    sp = specificity(tn, fp)
    fdr = false_discovery_rate(fp, tp)
    acc = accuracy(tp, tn, fp, fn)
    f1 = f1_score(tp, fp, fn)
    mcc = matthews_corr_coef(tp, tn, fp, fn)
    auc_roc, fpr, tpr = calculate_auc_roc_auc_prc(targets, probs, predictions)
    
    return {
        'sn': sn,
        'sp': sp,
        'fdr': fdr,
        'acc': acc,
        'f1': f1,
        'mcc': mcc,
        'auc_roc': auc_roc,
        # 'fpr': fpr,
        # 'tpr': tpr,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }
    
def calculate_auc_roc_auc_prc(y_true, probs, y_pred):
    # Calculate area under the receiver operating characteristic curve (auROC)
    try:
        auc_roc = roc_auc_score(y_true, probs)
        fpr, tpr, _ = roc_curve(y_true, probs)
        return auc_roc, fpr, tpr
    except:
        return 0, 0, 0

def sensitivity(tp, fn):
    return tp / (tp+fn)

def specificity(tn, fp):
    return tn / (tn+fp)

def false_discovery_rate(fp, tp):
    return fp / (fp+tp)

def accuracy(tp, tn, fp, fn):
    return (tp+tn) / (tp+tn+fp+fn)

def f1_score(tp, fp, fn):
    prec = tp / (tp+fp)
    rec = tp / (tp+fn)
    return (2*prec*rec)/(prec+rec)

def matthews_corr_coef(tp, tn, fp, fn):
    return (tp*tn-fp*fn)/math.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))

