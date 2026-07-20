from typing import Callable, Iterable

import torch
import torch.nn as nn

from flowmodels.basis import ObjectiveSupports, PredictionSupports, SamplingSupports


class DriftingModel(nn.Module, ObjectiveSupports, PredictionSupports, SamplingSupports):
    """
    Generative Modeling via Drifting, Deng et al., 2026.
    """

    def __init__(
        self,
        module: nn.Module,
        prior_dim: int = 32,
        temps: float | list[float] = 0.05,
    ):
        super().__init__()
        self.generator = module
        self.prior_dim = prior_dim
        self.temps = temps
        if isinstance(temps, float):
            self.temps = [temps]

    def forward(
        self, z: torch.Tensor, label: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict the sample points from the given prior.
        Args:
            z: [FloatLike; [B, ...]], the prior samples.
        Returns:
            the predicted sample points.
        """
        kwargs = {}
        if label is not None:
            kwargs["label"] = label
        return self.generator(z, **kwargs)

    def predict(
        self, x_t: torch.Tensor, t: torch.Tensor, label: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict the sample points from the given prior samples.
        Args:
            x_t: [FloatLike; [B, ...]], the given prior samples.
        Returns:
            the predicted sample points.
        """
        return self.forward(x_t, label)

    def sample(
        self,
        prior: torch.Tensor,
        label: torch.Tensor | None = None,
        steps: int | None = 1,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        with torch.inference_mode():
            result = self.forward(prior, label=label)
        return result, [result]

    def compute_V(
        self, gen: torch.Tensor, pos: torch.Tensor, temp: float
    ) -> torch.Tensor:
        batch_size, *_ = gen.shape
        # [2B, ...]
        targets = torch.cat([gen, pos], dim=0)
        # [B, 2B]
        dist = torch.cdist(gen, targets)
        dist[:, :batch_size].fill_diagonal_(1e6)
        kernel = (-dist / temp).exp()
        # [B, 2B]
        denom = kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2, keepdim=True)
        denom = denom.clamp_min(1e-12).sqrt()
        # [B, B], [B, B]
        gen_kernel, pos_kernel = (kernel / denom).split(
            [batch_size, batch_size], dim=-1
        )
        # [B, ...]
        pos_V = (pos_kernel * gen_kernel.sum(dim=-1, keepdim=True)) @ pos
        gen_V = (gen_kernel * pos_kernel.sum(dim=-1, keepdim=True)) @ gen
        return pos_V - gen_V

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss from the sample.
        Args:
            sample: [FloatLike; [B, ...]], training data.
            prior: [FloatLike; [B, ...]], sample from the source distribution.
        Returns:
            [FloatLike; []], loss value.
        """
        batch_size, *_ = sample.shape
        if prior is None:
            # TODO: support for arbitrarily given tensor shapes.
            prior = torch.randn(batch_size, self.prior_dim).to(sample)
        # [B, ...]
        # TODO: intra-class population support
        x = self.forward(prior, label)

        loss = 0.0
        for temp in self.temps:
            with torch.no_grad():
                target = x + self.compute_V(x, sample, temp)
            loss = loss + (x - target).square().mean()
        return loss


class WFlowSinkhornDrift(DriftingModel):
    """
    One-Step Generative Modeling via Wasserstein Gradient Flows, Han et al., 2026.
    """

    def __init__(
        self,
        module: nn.Module,
        prior_dim: int = 32,
        temps: float | list[float] = 0.05,
        resample_neg: bool = True,
        num_iter: int = 1,
    ):
        super().__init__(module, prior_dim, temps)
        self.resample_neg = resample_neg
        self.num_iter = num_iter

    def _sinkhorn_projection(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        temp: float,
        mask_diagonal: bool = False,
        num_iter: int = 1,
    ) -> torch.Tensor:
        bsize, *_ = src.shape
        # [B, B], quadratic cost
        C = 0.5 * torch.cdist(src, tgt).square()
        if mask_diagonal:
            C.fill_diagonal_(1e6)
        # [B, B]
        log_K = -C / temp
        # [B]
        # TODO: -logN and -logM initialization for `a` and `b`, respectively.
        # NOTE: since we do not assume N- and M-axis currently, we set N = M = 1.
        log_a = torch.zeros(bsize).to(C)
        log_b = torch.zeros(bsize).to(C)
        # [B]
        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)
        for _ in range(num_iter):
            log_u = log_a - (log_K + log_v).logsumexp(dim=-1)
            log_v = log_b - (log_K.T + log_u).logsumexp(dim=-1)
        # [B, B]
        Pi = torch.exp(log_u[:, None] + log_K + log_v)
        return (Pi @ tgt) / Pi.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def compute_V(
        self,
        gen: torch.Tensor,
        pos: torch.Tensor,
        temp: float,
        neg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask_diagonal = False
        if neg is None:
            mask_diagonal = True
            neg = gen

        T_pq = self._sinkhorn_projection(gen, pos, temp, False, self.num_iter)
        T_qq = self._sinkhorn_projection(gen, neg, temp, mask_diagonal, self.num_iter)
        return T_pq - T_qq

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss from the sample.
        Args:
            sample: [FloatLike; [B, ...]], training data.
            prior: [FloatLike; [B, ...]], sample from the source distribution.
        Returns:
            [FloatLike; []], loss value.
        """
        batch_size, *_ = sample.shape
        if prior is None:
            # TODO: support for arbitrarily given tensor shapes.
            prior = torch.randn(batch_size, self.prior_dim).to(sample)
        # [B, ...]
        # TODO: intra-class population support
        x = self.forward(prior, label)

        neg = None
        if self.resample_neg:
            with torch.no_grad():
                neg = self.forward(torch.randn_like(prior), label)

        loss = 0.0
        for temp in self.temps:
            with torch.no_grad():
                target = x + self.compute_V(x, sample, temp, neg=neg)
            loss = loss + (x - target).square().mean()
        return loss
