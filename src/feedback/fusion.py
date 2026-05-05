"""Adaptive fusion module for multi-source feedback."""

import numpy as np


class AdaptiveFusion:
    """Fuse multiple reward signals with variance-aware adaptive weighting.

    Args:
        sources: dict of {name: (score_fn, reliability_prior)}
        method: 'adaptive' (variance-aware) or 'fixed' (static weights)
    """

    def __init__(self, sources, method="adaptive"):
        self.sources = sources
        self.method = method
        self.names = list(sources.keys())

    def compute_rewards(self, completions, **kwargs):
        """Compute fused rewards for a batch of completions.

        Returns:
            tuple: (fused_rewards, component_rewards, weights)
        """
        # Compute each source's scores
        component_rewards = {}
        for name, (score_fn, _) in self.sources.items():
            component_rewards[name] = np.array(score_fn(completions, **kwargs))

        if self.method == "fixed":
            weights = {name: rho for name, (_, rho) in self.sources.items()}
            total_rho = sum(weights.values())
            weights = {k: v / total_rho for k, v in weights.items()}
        else:
            weights = self._adaptive_weights(component_rewards)

        # Fused reward
        fused = np.zeros(len(completions))
        for name in self.names:
            fused += weights[name] * component_rewards[name]

        return fused.tolist(), component_rewards, weights

    def _adaptive_weights(self, component_rewards):
        """Variance-aware adaptive weighting.

        w_i = (rho_i / sigma_i^2) / sum(rho_j / sigma_j^2)
        """
        weights = {}
        for name in self.names:
            scores = component_rewards[name]
            rho = self.sources[name][1]  # reliability prior
            var = max(np.var(scores), 1e-6)  # avoid division by zero
            weights[name] = rho / var

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            # Fallback to equal weights
            weights = {name: 1.0 / len(self.names) for name in self.names}

        return weights
