import torch
from torch import nn


class Sampler(nn.Module):

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, top_ps: torch.Tensor):
        # C2 契约：temperature=0 是显式 greedy 请求。基座 logits.div_(0) 得 NaN，
        # 故先短路出 greedy 位；argmax 对正缩放不变，除以 safe_t 不改变 greedy 侧结果。
        greedy = temperatures == 0
        safe_t = torch.where(greedy, torch.ones_like(temperatures), temperatures)
        logits = logits.float().div_(safe_t.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        if bool((top_ps < 1.0).any()):
            # nucleus（top-p）截断：保留累计质量首次到达 top_p 的最小前缀（vLLM 语义）。
            # 严格大于使 top_p=0 恰好保留单个最高概率 token。
            sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
            cumulative = sorted_probs.cumsum(dim=-1)
            mask = (cumulative - sorted_probs) > top_ps.unsqueeze(dim=1)
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min_(1e-10)
            gumbel = sorted_probs.div_(torch.empty_like(sorted_probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
            sample_tokens = sorted_idx.gather(-1, gumbel.unsqueeze(-1)).squeeze(-1)
        else:
            sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return torch.where(greedy, logits.argmax(dim=-1), sample_tokens)
