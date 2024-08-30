import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.functional import cosine_similarity


def similarity_by_editdistance(term, tokens):
    """
    按编辑距离从 tokens 中找到与 term 最相似的词
    """
    edit_dis = []
    for token in tokens: 
        edit_dis.append(editdistance.eval(term, token))
    rank = sorted(
        range(len(edit_dis)), 
        key=lambda k: edit_dis[k], 
        reverse=False
        )
    return [tokens[idx] for idx in rank]

    words, new_words = term.split(' '), []
    for word in words:
        edit_dis = []
        for token in tokens: 
            edit_dis.append(editdistance.eval(word, token))
        idx = edit_dis.index(min(edit_dis))
        new_words.append(tokens[idx])
    
    return ' '.join(new_words) 

def scl_old(embedding, label, temp):
    """calculate the contrastive loss
    """
    # cosine similarity between embeddings
    cosine_sim = cosine_similarity(embedding.unsqueeze(1), embedding.unsqueeze(0), dim=-1)
    # remove diagonal elements from matrix
    dis = cosine_sim[~torch.eye(cosine_sim.shape[0], dtype=bool)].reshape(cosine_sim.shape[0], -1)
    # apply temprature to elements
    dis = dis / temp
    cosine_sim = cosine_sim / temp
    # apply exp to elements
    dis = torch.exp(dis)
    cosine_sim = torch.exp(cosine_sim)

    # calculate row sum
    row_sum = []
    for i in range(len(embedding)):
        row_sum.append(sum(dis[i]))
    # calculate outer sum
    contrastive_loss = 0
    for i in range(len(embedding)):
        n_i = label.tolist().count(label[i]) - 1
        inner_sum = 0
        # calculate inner sum
        for j in range(len(embedding)):
            if label[i] == label[j] and i != j:
                inner_sum = inner_sum + torch.log(cosine_sim[i][j] / row_sum[i])
        if n_i != 0:
            contrastive_loss += (inner_sum / (-n_i))
        else:
            contrastive_loss += 0
    return contrastive_loss/len(embedding)

def scl(embeddings, labels, temp=0.3):
    """
    calculate the contrastive loss (optimized)
    embedding: [bz, dim]
    label: [bz] multi-class label
    """
    # cosine similarity between embeddings
    cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1) / temp
    # remove diagonal elements from matrix
    mask = ~torch.eye(cosine_sim.shape[0], dtype=bool,device=cosine_sim.device)
    dis = cosine_sim[mask].reshape(cosine_sim.shape[0], -1)
    # apply exp to elements
    dis_exp = torch.exp(dis)
    cosine_sim_exp = torch.exp(cosine_sim)
    row_sum = dis_exp.sum(dim=1) # calculate row sum
    # Pre-compute label counts
    unique_labels, counts = labels.unique(return_counts=True)
    label_count = dict(zip(unique_labels.tolist(), counts.tolist()))

    # calculate contrastive loss
    contrastive_loss = 0
    for i in range(len(embeddings)):
        n_i = label_count[labels[i].item()] - 1
        mask = (labels == labels[i]) & (torch.arange(len(embeddings),device=embeddings.device) != i)
        inner_sum = torch.log(cosine_sim_exp[i][mask] / row_sum[i]).sum()
        contrastive_loss += inner_sum / (-n_i) if n_i != 0 else 0

    return contrastive_loss / len(embeddings)


class DotProductSimilarity(nn.Module):
    def __init__(self, scale_output=False) -> None:
        super(DotProductSimilarity, self).__init__()
        self.scale_output = scale_output
    
    def forward(self, tensor_1, tensor_2):
        result = torch.matmul(tensor_1, tensor_2.T)
        # result = (tensor_1*tensor_2).sum(dim=-1)
        if self.scale_output:
            result /= math.sqrt(tensor_1.size(-1))
        return result


class CosineSimilarity(nn.Module):
    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super(CosineSimilarity, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return F.cosine_similarity(x1, x2, self.dim, self.eps)


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """
    def __init__(self, temp, method):
        super().__init__()
        self.temp = temp
        self.method = method
        self.cos = CosineSimilarity(dim=-1)
        self.dot = DotProductSimilarity()

    def forward(self, x, y):
        """
        x, y: [bz, dim]
        """
        sim = []
        if len(x.shape) == 1: x.unsqueeze(dim=0)
        if len(y.shape) == 1: y.unsqueeze(dim=0)
        for t in x:
            if self.method == 'cos': sim.append(self.cos(t, y) / self.temp)
            if self.method == 'dot': sim.append(self.dot(t, y) / self.temp)
        return torch.stack(sim)


class Contrast_Single(nn.Module):
    def __init__(self, method, temp=1) -> None:
        super().__init__()
        self.sim = Similarity(temp=temp, method=method)
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, z1, z2, labels=None, loss=0):
        """
        仅有一个样本是目标样本的正样本

        z1_z2_sim -> [bz, samples_num] 
        labels -> [bz]
        
        labels确定哪个sample是正样本, 其他的均为负样本
        """
        
        z1_z2_sim = self.sim(z1.unsqueeze(dim=1), z2.unsqueeze(dim=0))
        # z1_z2_sim = self.sim(z1, z2)
        if labels is not None:
            loss = self.loss_ce(z1_z2_sim, labels)

        return z1_z2_sim

        return {
            'sim': z1_z2_sim,
            'loss': loss,
        }


class Contrast_Multi(Contrast_Single):
    def __init__(self, args, method) -> None:
        super().__init__(args, method)
        self.eps = 1e-12
        self.sim = Similarity(temp=5, method='dot')
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, z1, z2, labels):
        """
        多个样本是目标样本的正样本
        z1: batch 样本表示 -> [bz, dim]
        z2: 对比样本表示 -> [num, dim] (batch样本表示拼接其他的/单独其他的)
        labels -> [bz, num] (1表示正样本, 0表示负样本) 
        
        labels 值为1的是对应的正样本, 其他的均为负样本
        """
        assert labels.shape == torch.Size([z1.shape[0], z2.shape[0]])
        z1_z2_sim = self.sim(z1, z2)
        z1_z2_sim -= z1_z2_sim.max(dim=-1)[0].detach() # 减去最大值？
        logits = torch.exp(z1_z2_sim)
        logits -= torch.eye(logits.shape[0]).type_as(logits)*logits.diag() # 去除自身
        log_prob = z1_z2_sim - torch.log(logits.sum(dim=1)+self.eps)
        log_prob_pos_mean = (labels*log_prob).sum(dim=1) / (labels.sum(dim=1)+self.eps)
        loss = (-log_prob_pos_mean).mean()

        return loss
        return {
            'loss': loss,
            'logits': log_prob,
        }