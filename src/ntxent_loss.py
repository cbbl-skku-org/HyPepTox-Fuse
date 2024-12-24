import torch
import torch.nn.functional as F



def nt_bxent_loss(x, pos_indices, temperature):
    assert len(x.size()) == 2

    device = x.device  # Get the device of the input tensor
    # Add indexes of the principal diagonal elements to pos_indices
    pos_indices = torch.cat([
        pos_indices,
        torch.arange(x.size(0), device=device).reshape(x.size(0), 1).expand(-1, 2),
    ], dim=0)
    
    # Ground truth labels
    target = torch.zeros(x.size(0), x.size(0), device=device)
    target[pos_indices[:,0], pos_indices[:,1]] = 1.0

    # Cosine similarity
    xcs = F.cosine_similarity(x[None,:,:], x[:,None,:], dim=-1)
    # Set logit of diagonal element to "inf" signifying complete
    # correlation. sigmoid(inf) = 1.0 so this will work out nicely
    # when computing the Binary cross-entropy Loss.
    xcs[torch.eye(x.size(0), device=device).bool()] = float("inf")

    # Standard binary cross-entropy loss. We use binary_cross_entropy() here and not
    # binary_cross_entropy_with_logits() because of
    # https://github.com/pytorch/pytorch/issues/102894
    # The method *_with_logits() uses the log-sum-exp-trick, which causes inf and -inf values
    # to result in a NaN result.
    loss = F.binary_cross_entropy((xcs / temperature).sigmoid(), target, reduction="none")
    
    target_pos = target.bool()
    target_neg = ~target_pos
    
    loss_pos = torch.zeros(x.size(0), x.size(0), device=device).masked_scatter(target_pos, loss[target_pos])
    loss_neg = torch.zeros(x.size(0), x.size(0), device=device).masked_scatter(target_neg, loss[target_neg])
    loss_pos = loss_pos.sum(dim=1)
    loss_neg = loss_neg.sum(dim=1)
    num_pos = target.sum(dim=1)
    num_neg = x.size(0) - num_pos

    return ((loss_pos / num_pos) + (loss_neg / num_neg)).mean()