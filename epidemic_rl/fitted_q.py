"""Original England V18 / Belgian bootstrapped Double-DQN fitted-Q procedure.

The input dimension is now the PF median/SD feature count. The ReLU network,
bootstraps, target construction, FQI/epoch schedule, AdamW and member seeds are
those of the supplied sources; independent members run in separate processes.
"""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import pickle

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .parallel import (atomic_pickle, digest_arrays, guard_checkpoint, ordered_map,
                       single_thread_worker, worker_pool)


@dataclass
class Dataset:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    done: np.ndarray
    next_mask: np.ndarray
    episode: np.ndarray
    step: np.ndarray

    def __len__(self):
        return len(self.actions)


class QNetwork(nn.Module):
    def __init__(self, input_dim=83, n_actions=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_actions))

    def forward(self, values):
        return self.net(values)


class FittedPolicy:
    def __init__(self, nets, scaler):
        self.nets, self.scaler = nets, scaler

    def values(self, state, return_std=False):
        z = ((np.asarray(state, np.float32) - self.scaler["mean"]) / self.scaler["sd"])[None, :]
        with torch.no_grad():
            values = np.stack([net(torch.from_numpy(z)).numpy()[0] for net in self.nets])
        return (values.mean(0), values.std(0)) if return_std else values.mean(0)

    def payload(self):
        return {"states": [{k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
                           for net in self.nets], "scaler": self.scaler,
                "input_dim": len(self.scaler["mean"])}

    @classmethod
    def from_payload(cls, payload):
        nets = []
        for state in payload["states"]:
            net = QNetwork(payload["input_dim"])
            net.load_state_dict({key: torch.as_tensor(value) for key, value in state.items()})
            nets.append(net.eval())
        return cls(nets, payload["scaler"])


_FIT = None


def _fit_init(context):
    global _FIT
    single_thread_worker()
    _FIT = context


def _fit_member(member):
    c = _FIT
    X, XN, A, D, R = (c[key] for key in ("X", "XN", "A", "D", "R"))
    cfg, seed = c["config"], c["seed"]
    torch.manual_seed(seed + 1009 * member)
    online = QNetwork(X.shape[1]); target = QNetwork(X.shape[1])
    target.load_state_dict(online.state_dict())
    rng = np.random.default_rng(seed + member * 7919)
    boot = rng.choice(np.arange(len(A)), size=len(A), replace=True)
    tensor_next = torch.from_numpy(XN)
    masks = torch.from_numpy(c["masks"])
    logs, updates = [], 0
    print(f"FIT member {member+1}/{cfg['ensemble_size']} START pid={os.getpid()}", flush=True)
    for iteration in range(1, cfg["fqi_iterations"] + 1):
        online.eval(); target.eval()
        estimated = []
        with torch.no_grad():
            for lo in range(0, len(A), 32768):
                values = tensor_next[lo:lo+32768]
                choice = online(values).masked_fill(~masks[lo:lo+32768], -1e12).argmax(1)
                estimated.append(target(values).gather(1, choice[:, None]).squeeze(1).numpy())
        Y = (R + (1-D)*np.concatenate(estimated)).astype(np.float32)
        generator = torch.Generator(); generator.manual_seed(seed + member*100003 + iteration)
        loader = DataLoader(TensorDataset(torch.from_numpy(X[boot]), torch.from_numpy(A[boot]),
                                         torch.from_numpy(Y[boot])), batch_size=cfg["batch_size"],
                            shuffle=True, generator=generator)
        # The source creates AdamW afresh at each fitted-Q iteration.
        optimizer = torch.optim.AdamW(online.parameters(), lr=cfg["learning_rate"],
                                     weight_decay=cfg["weight_decay"])
        online.train()
        for _ in range(cfg["epochs_per_iteration"]):
            for xb, ab, yb in loader:
                prediction = online(xb).gather(1, ab.long()[:, None]).squeeze(1)
                loss = nn.functional.smooth_l1_loss(prediction, yb)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), cfg["grad_clip_norm"])
                optimizer.step(); updates += 1
        if iteration % cfg["target_sync_every"] == 0:
            target.load_state_dict(online.state_dict())
        row = {"member": member, "iteration": iteration}
        if c["country"] == "england":
            # Preserve even the source diagnostic RNG draws.
            with torch.no_grad():
                idx = rng.choice(len(A), size=min(len(A), 10000), replace=False)
                error = online(torch.from_numpy(X[idx])).gather(
                    1, torch.from_numpy(A[idx]).long()[:, None]).squeeze(1).numpy() - Y[idx]
                row["held_training_subset_rmse"] = float(np.sqrt(np.mean(error*error)))
        logs.append(row)
    result = {"member": member, "state": {k: v.detach().numpy().copy()
                                         for k, v in online.state_dict().items()},
              "logs": logs, "updates": updates, "pid": os.getpid()}
    if c["directory"] is not None:
        path = c["directory"] / f"member_{member}.pkl"
        atomic_pickle(path, result)
        path.with_suffix(".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    return result


def fit_ddqn(dataset, config, seed, country, workers=3, checkpoint=None, method="auto"):
    """Same member arithmetic for workers=1 and workers>1; ordered outputs."""
    single_thread_worker()
    mean = dataset.states.mean(0).astype(np.float32)
    sd = np.maximum(dataset.states.std(0), 1e-6).astype(np.float32)
    X, XN = (dataset.states-mean)/sd, (dataset.next_states-mean)/sd
    arrays = [dataset.states, dataset.actions, dataset.rewards, dataset.next_states,
              dataset.done, dataset.next_mask]
    digest = digest_arrays(arrays, {"config": config, "seed": seed, "country": country,
                                    "source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    directory = guard_checkpoint(checkpoint, digest) if checkpoint is not None else None
    results = {}
    if directory is not None:
        for member in range(config["ensemble_size"]):
            path = directory / f"member_{member}.pkl"
            if path.exists():
                if path.with_suffix(".sha256").read_text() != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise RuntimeError(f"Corrupt fitted-Q member checkpoint: {path}")
                results[member] = pickle.loads(path.read_bytes())
    missing = [i for i in range(config["ensemble_size"]) if i not in results]
    context = {"X": X, "XN": XN, "A": dataset.actions, "D": dataset.done,
               "R": dataset.rewards, "masks": dataset.next_mask, "config": config,
               "seed": int(seed), "directory": directory, "country": country}
    if missing:
        with worker_pool(min(workers, len(missing)), _fit_init, context, method) as pool:
            for result in ordered_map(pool, _fit_member, missing):
                results[result["member"]] = result
    payload = {"states": [results[i]["state"] for i in range(config["ensemble_size"])],
               "input_dim": X.shape[1], "scaler": {"mean": mean, "sd": sd}}
    audit = {"worker_pids": sorted({result["pid"] for result in results.values()}),
             "workers_requested": workers, "updates": sum(r["updates"] for r in results.values()),
             "member_count": len(results), "fqi_iterations": config["fqi_iterations"],
             "epochs_per_iteration": config["epochs_per_iteration"]}
    logs = [row for i in range(config["ensemble_size"]) for row in results[i]["logs"]]
    return FittedPolicy.from_payload(payload), logs, audit
