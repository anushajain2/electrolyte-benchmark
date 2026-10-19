"""Model adapters with one interface.

``fit(X, y_mS_cm, meta)`` and ``predict(X, meta)`` take the canonical feature
matrix (indexed by measurement_id) and the canonical measurement rows. Predict
returns a frame with measurement_id, prediction_mS_cm, status and
failure_reason. Unless stated otherwise models learn log10(conductivity in
mS/cm) and predictions are transformed back with 10**y.

Status values: ok | invalid_nonpositive | unsupported_input | failed
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _frame(ids, pred, status=None, reason=None) -> pd.DataFrame:
    n = len(ids)
    return pd.DataFrame(
        {
            "measurement_id": np.asarray(ids),
            "prediction_mS_cm": np.asarray(pred, dtype=float),
            "status": status if status is not None else np.full(n, "ok", dtype=object),
            "failure_reason": reason if reason is not None else np.full(n, "", dtype=object),
        }
    )


class Adapter:
    name = "base"
    target = "log10"  # or "raw"
    stochastic = False

    def __init__(self, seed: int = 0, **params):
        self.seed = seed
        self.params = params

    def fit(self, X: pd.DataFrame, y: np.ndarray, meta: pd.DataFrame) -> "Adapter":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class _SklearnAdapter(Adapter):
    def _make(self):
        raise NotImplementedError

    def fit(self, X, y, meta):
        self.model = self._make()
        yt = np.log10(y) if self.target == "log10" else np.asarray(y, dtype=float)
        # canonical row order (by measurement_id): neighbour ties and tree
        # randomness then cannot depend on how the caller ordered the rows
        order = np.argsort(X.index.to_numpy().astype(str), kind="stable")
        self.model.fit(X.to_numpy()[order], yt[order])
        return self

    def predict(self, X, meta):
        p = self.model.predict(X.to_numpy())
        if self.target == "log10":
            return _frame(X.index, 10.0**p)
        status = np.where(p > 0, "ok", "invalid_nonpositive").astype(object)
        reason = np.where(p > 0, "", "raw-scale prediction <= 0 cannot be scored on log scale").astype(object)
        return _frame(X.index, p, status, reason)


class MedianModel(Adapter):
    name = "median"

    def fit(self, X, y, meta):
        self.value = float(10 ** np.median(np.log10(y)))
        return self

    def predict(self, X, meta):
        return _frame(X.index, np.full(len(X), self.value))


class KNNModel(_SklearnAdapter):
    name = "knn"

    def _make(self):
        n = self.params.get("n_neighbors", 5)
        return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=n, weights=self.params.get("weights", "uniform")))

    def fit(self, X, y, meta):
        # never ask for more neighbours than training rows
        self.params = {**self.params, "n_neighbors": min(self.params.get("n_neighbors", 5), len(X))}
        return super().fit(X, y, meta)


class RidgeModel(_SklearnAdapter):
    name = "ridge"

    def _make(self):
        return make_pipeline(StandardScaler(), Ridge(alpha=self.params.get("alpha", 1.0)))


class RidgeRawModel(RidgeModel):
    name = "ridge_raw"
    target = "raw"


class ExtraTreesModel(_SklearnAdapter):
    name = "extra_trees"
    stochastic = True

    def _make(self):
        return ExtraTreesRegressor(random_state=self.seed, n_jobs=1, **self.params)


class HistGBModel(_SklearnAdapter):
    name = "hist_gb"
    stochastic = True

    def _make(self):
        return HistGradientBoostingRegressor(random_state=self.seed, early_stopping=False, **self.params)


class ArrheniusLocal(Adapter):
    """Per-formulation log10(k) = a + b * (1000/T) fitted on that formulation's
    training temperatures only. Inputs without enough training temperatures
    are returned as unsupported rather than silently imputed."""

    name = "arrhenius_local"

    def __init__(self, seed: int = 0, min_points: int = 3, **params):
        super().__init__(seed, **params)
        self.min_points = min_points

    def fit(self, X, y, meta):
        d = pd.DataFrame({"f": meta.formulation_id.to_numpy(), "x": 1000.0 / meta.temperature_K.to_numpy(), "y": np.log10(y)})
        self.coef = {}
        for f, g in d.groupby("f"):
            if g.x.round(4).nunique() >= self.min_points:
                b, a = np.polyfit(g.x, g.y, 1)
                self.coef[f] = (a, b)
        return self

    def predict(self, X, meta):
        f = meta.formulation_id.to_numpy()
        x = 1000.0 / meta.temperature_K.to_numpy()
        pred = np.full(len(f), np.nan)
        status = np.full(len(f), "unsupported_input", dtype=object)
        reason = np.full(len(f), "formulation has fewer than 3 training temperatures", dtype=object)
        for i, fi in enumerate(f):
            if fi in self.coef:
                a, b = self.coef[fi]
                pred[i] = 10 ** (a + b * x[i])
                status[i] = "ok"
                reason[i] = ""
        return _frame(X.index, pred, status, reason)


REGISTRY: dict[str, type[Adapter]] = {
    m.name: m for m in [MedianModel, KNNModel, RidgeModel, RidgeRawModel, ExtraTreesModel, HistGBModel, ArrheniusLocal]
}


def config_grid(family_cfg: dict, max_configs: int) -> list[dict]:
    grid = family_cfg.get("grid") or {}
    fixed = family_cfg.get("fixed") or {}
    keys = sorted(grid)
    configs = [{**fixed, **dict(zip(keys, vals))} for vals in itertools.product(*(grid[k] for k in keys))] or [dict(fixed)]
    if len(configs) > max_configs:
        raise ValueError(f"{len(configs)} configurations exceed the declared budget of {max_configs}")
    return configs


def make_model(name: str, seed: int, params: dict) -> Adapter:
    return REGISTRY[name](seed=seed, **params)
