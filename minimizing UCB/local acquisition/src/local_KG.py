
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import torch
from botorch import settings
from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
)
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.cost_aware import CostAwareUtility
from botorch.acquisition.monte_carlo import MCAcquisitionFunction, qSimpleRegret
from botorch.acquisition.objective import (
    AcquisitionObjective,
    MCAcquisitionObjective,
    ScalarizedObjective,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)
from torch import Tensor


class Local_KG(MCAcquisitionFunction, OneShotAcquisitionFunction):
    r"""Batch Knowledge Gradient using one-shot optimization.

    This computes the batch Knowledge Gradient using fantasies for the outer
    expectation and either the model posterior mean or MC-sampling for the inner
    expectation.

    In addition to the design variables, the input `X` also includes variables
    for the optimal designs for each of the fantasy models. For a fixed number
    of fantasies, all parts of `X` can be optimized in a "one-shot" fashion.
    """

    def __init__(
        self,
        model: Model,
        num_fantasies: Optional[int] = 64,
        sampler: Optional[MCSampler] = None,
        objective: Optional[AcquisitionObjective] = None,
        inner_sampler: Optional[MCSampler] = None,
        X_pending: Optional[Tensor] = None,
        current_value: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        r"""q-Knowledge Gradient (one-shot optimization).

        Args:
            model: A fitted model. Must support fantasizing.
            num_fantasies: The number of fantasy points to use. More fantasy
                points result in a better approximation, at the expense of
                memory and wall time. Unused if `sampler` is specified.
            sampler: The sampler used to sample fantasy observations. Optional
                if `num_fantasies` is specified.
            objective: The objective under which the samples are evaluated. If
                `None` or a ScalarizedObjective, then the analytic posterior mean
                is used, otherwise the objective is MC-evaluated (using
                inner_sampler).
            inner_sampler: The sampler used for inner sampling. Ignored if the
                objective is `None` or a ScalarizedObjective.
            X_pending: A `m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation
                but have not yet been evaluated.
            current_value: The current value, i.e. the expected best objective
                given the observed points `D`. If omitted, forward will not
                return the actual KG value, but the expected best objective
                given the data set `D u X`.
        """
        if sampler is None:
            if num_fantasies is None:
                raise ValueError(
                    "Must specify `num_fantasies` if no `sampler` is provided."
                )
            # base samples should be fixed for joint optimization over X, X_fantasies
            sampler = SobolQMCNormalSampler(
                num_samples=num_fantasies, resample=False, collapse_batch_dims=True
            )
        elif num_fantasies is not None:
            if sampler.sample_shape != torch.Size([num_fantasies]):
                raise ValueError(
                    f"The sampler shape must match num_fantasies={num_fantasies}."
                )
        else:
            num_fantasies = sampler.sample_shape[0]
        super(MCAcquisitionFunction, self).__init__(model=model)
        # if not explicitly specified, we use the posterior mean for linear objs
        if isinstance(objective, MCAcquisitionObjective) and inner_sampler is None:
            inner_sampler = SobolQMCNormalSampler(
                num_samples=128, resample=False, collapse_batch_dims=True
            )
        if objective is None and model.num_outputs != 1:
            raise UnsupportedError(
                "Must specify an objective when using a multi-output model."
            )
        self.sampler = sampler
        self.objective = objective
        self.set_X_pending(X_pending)
        self.inner_sampler = inner_sampler
        self.num_fantasies = num_fantasies
        self.current_value = current_value

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qKnowledgeGradient on the candidate set `X`.

        Args:
            X: A `b x (q + num_fantasies) x d` Tensor with `b` t-batches of
                `q + num_fantasies` design points each. We split this X tensor
                into two parts in the `q` dimension (`dim=-2`). The first `q`
                are the q-batch of design points and the last num_fantasies are
                the current solutions of the inner optimization problem.

                `X_fantasies = X[..., -num_fantasies:, :]`
                `X_fantasies.shape = b x num_fantasies x d`

                `X_actual = X[..., :-num_fantasies, :]`
                `X_actual.shape = b x q x d`

        Returns:
            A Tensor of shape `b`. For t-batch b, the q-KG value of the design
                `X_actual[b]` is averaged across the fantasy models, where
                `X_fantasies[b, i]` is chosen as the final selection for the
                `i`-th fantasy model.
                NOTE: If `current_value` is not provided, then this is not the
                true KG value of `X_actual[b]`, and `X_fantasies[b, : ]` must be
                maximized at fixed `X_actual[b]`.
        """
        X_actual, X_fantasies = _split_fantasy_points(X=X, n_f=self.num_fantasies)

        # We only concatenate X_pending into the X part after splitting
        if self.X_pending is not None:
            X_actual = torch.cat(
                [X_actual, match_batch_shape(self.X_pending, X_actual)], dim=-2
            )

        # construct the fantasy model of shape `num_fantasies x b`
        fantasy_model = self.model.fantasize(
            X=X_actual, sampler=self.sampler, observation_noise=True
        )

        # get the value function
        value_function = _get_value_function(
            model=fantasy_model, objective=self.objective, sampler=self.inner_sampler
        )

        # make sure to propagate gradients to the fantasy model train inputs
        with settings.propagate_grads(True):
            values = value_function(X=X_fantasies)  # num_fantasies x b

        if self.current_value is not None:
            values = values - self.current_value

        # return average over the fantasy samples
        return values.mean(dim=0)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def evaluate(self, X: Tensor, bounds: Tensor, **kwargs: Any) -> Tensor:
        r"""Evaluate qKnowledgeGradient on the candidate set `X_actual` by
        solving the inner optimization problem.

        Args:
            X: A `b x q x d` Tensor with `b` t-batches of `q` design points
                each. Unlike `forward()`, this does not include solutions of the
                inner optimization problem.
            bounds: A `2 x d` tensor of lower and upper bounds for each column of
                the solutions to the inner problem.
            kwargs: Additional keyword arguments. This includes the options for
                optimization of the inner problem, i.e. `num_restarts`, `raw_samples`,
                an `options` dictionary to be passed on to the optimization helpers, and
                a `scipy_options` dictionary to be passed to `scipy.minimize`.

        Returns:
            A Tensor of shape `b`. For t-batch b, the q-KG value of the design
                `X[b]` is averaged across the fantasy models.
                NOTE: If `current_value` is not provided, then this is not the
                true KG value of `X[b]`.
        """
        if hasattr(self, "expand"):
            X = self.expand(X)

        # construct the fantasy model of shape `num_fantasies x b`
        fantasy_model = self.model.fantasize(
            X=X, sampler=self.sampler, observation_noise=True
        )

        # get the value function
        value_function = _get_value_function(
            model=fantasy_model,
            objective=self.objective,
            sampler=self.inner_sampler,
            project=getattr(self, "project", None),
        )

        from botorch.generation.gen import gen_candidates_scipy

        # optimize the inner problem
        from botorch.optim.initializers import gen_value_function_initial_conditions

        initial_conditions = gen_value_function_initial_conditions(
            acq_function=value_function,
            bounds=bounds,
            num_restarts=kwargs.get("num_restarts", 20),
            raw_samples=kwargs.get("raw_samples", 1024),
            current_model=self.model,
            options={**kwargs.get("options", {}), **kwargs.get("scipy_options", {})},
        )

        _, values = gen_candidates_scipy(
            initial_conditions=initial_conditions,
            acquisition_function=value_function,
            lower_bounds=bounds[0],
            upper_bounds=bounds[1],
            options=kwargs.get("scipy_options"),
        )
        # get the maximizer for each batch
        values, _ = torch.max(values, dim=0)
        if self.current_value is not None:
            values = values - self.current_value

        if hasattr(self, "cost_aware_utility"):
            values = self.cost_aware_utility(
                X=X, deltas=values, sampler=self.cost_sampler
            )
        # return average over the fantasy samples
        return values.mean(dim=0)

    def get_augmented_q_batch_size(self, q: int) -> int:
        r"""Get augmented q batch size for one-shot optimization.

        Args:
            q: The number of candidates to consider jointly.

        Returns:
            The augmented size for one-shot optimization (including variables
            parameterizing the fantasy solutions).
        """
        return q + self.num_fantasies

    def extract_candidates(self, X_full: Tensor) -> Tensor:
        r"""We only return X as the set of candidates post-optimization.

        Args:
            X_full: A `b x (q + num_fantasies) x d`-dim Tensor with `b`
                t-batches of `q + num_fantasies` design points each.

        Returns:
            A `b x q x d`-dim Tensor with `b` t-batches of `q` design points each.
        """
        return X_full[..., : -self.num_fantasies, :]
    






class ProjectedAcquisitionFunction(AcquisitionFunction):
    r"""
    Defines a wrapper around  an `AcquisitionFunction` that incorporates the project
    operator. Typically used to handle value functions in look-ahead methods.
    """

    def __init__(
        self,
        base_value_function: AcquisitionFunction,
        project: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__(base_value_function.model)
        self.base_value_function = base_value_function
        self.project = project
        self.objective = base_value_function.objective
        self.sampler = getattr(base_value_function, "sampler", None)

    def forward(self, X: Tensor) -> Tensor:
        return self.base_value_function(self.project(X))


def _get_value_function(
    model: Model,
    objective: Optional[Union[MCAcquisitionObjective, ScalarizedObjective]] = None,
    sampler: Optional[MCSampler] = None,
    project: Optional[Callable[[Tensor], Tensor]] = None,
    valfunc_cls: Optional[Type[AcquisitionFunction]] = None,
    valfunc_argfac: Optional[Callable[[Model, Dict[str, Any]]]] = None,
) -> AcquisitionFunction:
    r"""Construct value function (i.e. inner acquisition function)."""
    if valfunc_cls is not None:
        common_kwargs: Dict[str, Any] = {"model": model, "objective": objective}
        if issubclass(valfunc_cls, MCAcquisitionFunction):
            common_kwargs["sampler"] = sampler
        kwargs = valfunc_argfac(model=model) if valfunc_argfac is not None else {}
        base_value_function = valfunc_cls(**common_kwargs, **kwargs)
    else:
        if isinstance(objective, MCAcquisitionObjective):
            base_value_function = qSimpleRegret(
                model=model, sampler=sampler, objective=objective
            )
        else:
            base_value_function = Local_acquisition(model=model, objective=objective)

    if project is None:
        return base_value_function
    else:
        return ProjectedAcquisitionFunction(
            base_value_function=base_value_function,
            project=project,
        )


def _split_fantasy_points(X: Tensor, n_f: int) -> Tuple[Tensor, Tensor]:
    r"""Split a one-shot optimization input into actual and fantasy points

    Args:
        X: A `batch_shape x (q + n_f) x d`-dim tensor of actual and fantasy
            points

    Returns:
        2-element tuple containing

        - A `batch_shape x q x d`-dim tensor `X_actual` of input candidates.
        - A `n_f x batch_shape x 1 x d`-dim tensor `X_fantasies` of fantasy
            points, where `X_fantasies[i, batch_idx]` is the i-th fantasy point
            associated with the batch indexed by `batch_idx`.
    """
    if n_f > X.size(-2):
        raise ValueError(
            f"n_f ({n_f}) must be less than the q-batch dimension of X ({X.size(-2)})"
        )
    split_sizes = [X.size(-2) - n_f, n_f]
    X_actual, X_fantasies = torch.split(X, split_sizes, dim=-2)
    # X_fantasies is b x num_fantasies x d, needs to be num_fantasies x b x 1 x d
    # for batch mode evaluation with batch shape num_fantasies x b.
    # b x num_fantasies x d --> num_fantasies x b x d
    X_fantasies = X_fantasies.permute(-2, *range(X_fantasies.dim() - 2), -1)
    # num_fantasies x b x 1 x d
    X_fantasies = X_fantasies.unsqueeze(dim=-2)
    return X_actual, X_fantasies









class Local_acquisition(AnalyticAcquisitionFunction):
    r"""Single-outcome Posterior Mean.

    Only supports the case of q=1. Requires the model's posterior to have a
    `mean` property. The model must be single-outcome.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> PM = PosteriorMean(model)
        >>> pm = PM(test_X)
    """

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the posterior mean on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Posterior Mean values at the given design
            points `X`.
        """
        posterior = self._get_posterior(X=X)
        return posterior.mean.view(X.shape[:-2])-3*posterior.variance.sqrt().view(X.shape[:-2])
    



    
class Local_acquisition_1(AnalyticAcquisitionFunction):
    r"""Single-outcome Posterior Mean.

    Only supports the case of q=1. Requires the model's posterior to have a
    `mean` property. The model must be single-outcome.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> PM = PosteriorMean(model)
        >>> pm = PM(test_X)
    """

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the posterior mean on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Posterior Mean values at the given design
            points `X`.
        """
        posterior = self._get_posterior(X=X)
        return posterior.mean.view(X.shape[:-2])-3*posterior.variance.sqrt().view(X.shape[:-2])