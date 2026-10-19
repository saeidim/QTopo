"""
main.py — QTopo-Meta Experiment Runner
=======================================
Run from the project root:

    python main.py                          # default: UCI-HAR raw + Indian Pines, 5-shot
    python main.py --datasets uci_har       # single dataset
    python main.py --datasets uci_har_raw indian_pines --shots 1 5
    python main.py --quick                  # fast smoke-test (5 epochs)
    python main.py --cross_domain           # also run cross-domain experiments
    python main.py --methods ProtoNet MAML QTopo_Meta   # subset of methods

Repository: https://github.com/saeidim/QTopo
Paper: QTopo-Meta: Quantum-Inspired Topological Meta-Learning for
       Cross-Domain Few-Shot Pattern Recognition
Authors: Saeid Iranmanesh, Raad Raad — University of Wollongong
"""

import argparse
import hashlib
import os
import pickle
import sys
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Tuple

# -- Make sub-packages importable from project root --------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as CFG
from datasets.loaders     import load_dataset, make_few_shot_splits
from datasets.tda_pipeline import TDAPipeline
from models.all_models    import build_model, EpisodeSampler
from experiments.trainer  import MetaTrainer, final_test
from utils.plotting       import save_all_plots


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="QTopo-Meta experiment runner")
    p.add_argument("--datasets", nargs="+",
                   default=CFG.EXPERIMENT_CFG["datasets"],
                   choices=["uci_har", "uci_har_raw", "covtype", "indian_pines"],
                   help="Datasets to run")
    p.add_argument("--methods", nargs="+",
                   default=CFG.EXPERIMENT_CFG["methods"],
                   choices=["ProtoNet","MAML","RelationNet",
                             "FWT","TIN_FSL","QTopo_Meta"],
                   help="Methods to compare")
    p.add_argument("--shots", nargs="+", type=int, default=[5],
                   help="K-shot values to evaluate")
    p.add_argument("--cross_domain", action="store_true",
                   help="Also run cross-domain experiments")
    p.add_argument("--hetero_train", action="store_true",
                   help="Use heterogeneous episodic training (mixes modalities "
                        "within each episode; requires >=2 datasets)")
    p.add_argument("--quick", action="store_true",
                   help="Smoke-test mode: 5 epochs, 20 ep/epoch")
    p.add_argument("--no_plots", action="store_true",
                   help="Skip plot generation")
    p.add_argument("--device", default=CFG.DEVICE,
                   choices=["cpu","cuda"], help="Compute device")
    p.add_argument("--seed", type=int, default=CFG.TRAIN_CFG["seed"])
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# Dataset preparation (load → split → TDA pipeline)
# ════════════════════════════════════════════════════════════════════════════

def _tda_cache_key(dataset_name: str, seed: int, pipeline_cfg: dict,
                   n_train_cls: int, n_val_cls: int, n_test_cls: int,
                   min_ways: int) -> str:
    payload = {
        "dataset": dataset_name, "seed": seed, "pipeline": pipeline_cfg,
        "n_train_cls": n_train_cls, "n_val_cls": n_val_cls,
        "n_test_cls": n_test_cls, "min_ways": min_ways,
    }
    blob = repr(sorted(payload.items())).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def prepare_domain(
    dataset_name : str,
    k_shot       : int,
    pipeline_cfg : dict,
    data_dir     : str,
    seed         : int,
) -> dict:
    """
    Full Stage 1–3 pipeline for one dataset.

    Returns
    -------
    dict with keys "train", "test", "meta", "pipeline".
    "train" and "test" each contain {"X": array, "y": array, "dgms": list}.
    """
    print(f"\n{'-'*55}")
    print(f"[Dataset] {dataset_name.upper()}")
    print(f"{'-'*55}")

    ds_cfg_pre = CFG.DATASET_CFG[dataset_name]
    cache_key  = _tda_cache_key(
        dataset_name, seed, pipeline_cfg,
        ds_cfg_pre["n_train_cls"], ds_cfg_pre["n_val_cls"],
        ds_cfg_pre["n_test_cls"], CFG.EPISODE_CFG["N"],
    )
    cache_path = Path(data_dir) / f"{dataset_name}_tda_{cache_key}.pkl"
    if cache_path.exists():
        print(f"  [TDA cache] Loading {cache_path.name}")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        return cached

    # -- Load raw data -------------------------------------------------------
    X_tr_raw, y_tr_raw, X_te_raw, y_te_raw, meta = load_dataset(
        dataset_name, data_dir=data_dir
    )
    print(f"  Loaded: train={X_tr_raw.shape}  test={X_te_raw.shape}")
    print(f"  Classes: {meta['n_classes']}  |  Features: {meta['n_features']}")
    print(f"  Modality: {meta['modality']}")

    # -- Class splits --------------------------------------------------------
    ds_cfg = CFG.DATASET_CFG[dataset_name]
    splits = make_few_shot_splits(
        X=np.vstack([X_tr_raw, X_te_raw]),
        y=np.concatenate([y_tr_raw, y_te_raw]),
        n_train=ds_cfg["n_train_cls"],
        n_val  =ds_cfg["n_val_cls"],
        n_test =ds_cfg["n_test_cls"],
        seed   =seed,
        min_ways=CFG.EPISODE_CFG["N"],
    )
    X_tr, y_tr = splits["train"]
    X_te, y_te = splits["test"]
    print(f"  Meta-train: {X_tr.shape}  classes={np.unique(y_tr)}")
    print(f"  Meta-test:  {X_te.shape}  classes={np.unique(y_te)}")

    # -- TDA pipeline (Stages 1–3) -------------------------------------------
    n_fit = min(pipeline_cfg.get("n_fit_cap", 1000), len(X_tr))
    print(f"\n  Running TDA pipeline (Stages 1–3)  n_fit={n_fit}...")
    pipe = TDAPipeline(pipeline_cfg)
    pipe.fit(X_tr, modality=meta["modality"], n_fit=n_fit)

    print(f"  Transforming train set ({len(X_tr)} samples)...")
    V_tr, D_tr = pipe.transform(X_tr, desc="train")

    print(f"  Transforming test set ({len(X_te)} samples)...")
    V_te, D_te = pipe.transform(X_te, desc="test")

    print(f"  TDA feature dim: d = {V_tr.shape[1]}")

    out = {
        "train"   : {"X": V_tr, "y": y_tr, "dgms": D_tr},
        "test"    : {"X": V_te, "y": y_te, "dgms": D_te},
        "meta"    : meta,
        "pipeline": pipe,
    }

    try:
        with open(cache_path, "wb") as f:
            pickle.dump(out, f)
        print(f"  [TDA cache] Saved {cache_path.name}")
    except Exception as e:
        print(f"  [TDA cache] Save failed ({e}) — continuing")

    return out


# ════════════════════════════════════════════════════════════════════════════
# Single experiment: one dataset, one K-shot value
# ════════════════════════════════════════════════════════════════════════════

def run_experiment(
    domain_data  : dict,
    method_names : list,
    k_shot       : int,
    train_cfg    : dict,
    model_cfg    : dict,
    device       : str,
    plots_dir    : str,
    results_dir  : str,
    dataset_name : str,
    no_plots     : bool = False,
    hetero_domains: dict = None,  # {name: domain_data} for heterogeneous training
) -> Tuple[dict, dict]:
    """
    Train all methods on one dataset and produce tables + plots.

    Parameters
    ----------
    hetero_domains : dict or None
        If provided, QTopo_Meta is trained using a heterogeneous episodic
        sampler that draws support and query sets from different domains
        within each episode.  Other methods still train homogeneously.
    """
    d   = domain_data["train"]["X"].shape[1]
    ep_cfg = {
        "N": CFG.EPISODE_CFG["N"],
        "K": k_shot,
        "Q": CFG.EPISODE_CFG["Q"],
    }
    seed = train_cfg.get("seed", 42)

    train_dom = {"main": domain_data["train"]}
    test_dom  = {"main": domain_data["test"]}

    # Heterogeneous training domain set: include all paired domains so
    # EpisodeSampler can draw support from one modality and query from another.
    if hetero_domains:
        # Merge: keep the primary domain plus all paired domains.
        hetero_train_dom = dict(hetero_domains)
        hetero_train_dom["main"] = domain_data["train"]
    else:
        hetero_train_dom = train_dom

    # Within-domain val/test samplers (homogeneous, always)
    vl_sampler = EpisodeSampler(test_dom,  N=ep_cfg["N"], K=k_shot,
                                 Q=ep_cfg["Q"], seed=seed+1, hetero=False)
    te_sampler = EpisodeSampler(test_dom,  N=ep_cfg["N"], K=k_shot,
                                 Q=ep_cfg["Q"], seed=999, hetero=False)

    # -- Train all methods ---------------------------------------------------
    print(f"\n{'='*55}")
    print(f"Training {len(method_names)} methods — "
          f"{dataset_name} — {k_shot}-shot")
    print(f"{'='*55}")

    hist_within  = {}
    models_within = {}
    for name in method_names:
        # QTopo_Meta uses heterogeneous episodes when --hetero_train is set
        # and multiple domains are available.  All other baselines always
        # train homogeneously so comparisons are fair.
        use_hetero = (name == "QTopo_Meta" and hetero_domains is not None
                      and len(hetero_train_dom) > 1)
        tr_sampler = EpisodeSampler(
            hetero_train_dom if use_hetero else train_dom,
            N=ep_cfg["N"], K=k_shot, Q=ep_cfg["Q"],
            seed=seed, hetero=use_hetero,
        )
        if use_hetero:
            print(f"  [{name}] heterogeneous episodic training "
                  f"({len(hetero_train_dom)} domains)")

        model   = build_model(name, d=d, cfg=model_cfg, device=device)
        trainer = MetaTrainer(
            model         = model,
            train_sampler = tr_sampler,
            val_sampler   = vl_sampler,
            train_cfg     = train_cfg,
            ckpt_dir      = str(Path(results_dir)/dataset_name/f"K{k_shot}"/"ckpt"),
        )
        hist_within[name]   = trainer.train(verbose=True)
        models_within[name] = model

    # -- Within-domain final test --------------------------------------------
    print(f"\nFinal test — Within-domain [{dataset_name}] {k_shot}-shot:")
    test_within = {}
    for name, model in models_within.items():
        acc, ci, _ = model.evaluate(te_sampler, train_cfg.get("test_episodes", 1000))
        test_within[name] = {"acc": round(acc*100, 2), "ci": round(ci*100, 2)}
        print(f"  {name:20s}: {acc*100:.2f}% ± {ci*100:.2f}%")

    # Cross-domain results populated later in main() via post-hoc evaluation.
    hist_cross : dict = {}
    test_cross : dict = {}

    # -- Save results CSV ----------------------------------------------------
    out_dir = Path(results_dir) / dataset_name / f"K{k_shot}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in method_names:
        row = {"dataset": dataset_name, "model": name, "K": k_shot,
               "within_acc" : test_within.get(name, {}).get("acc"),
               "within_ci"  : test_within.get(name, {}).get("ci"),
               "cross_acc"  : test_cross.get(name,  {}).get("acc"),
               "cross_ci"   : test_cross.get(name,  {}).get("ci")}
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test_results.csv", index=False)
    print(f"\n  Saved: {out_dir / 'test_results.csv'}")

    # Per-epoch history
    hist_rows = []
    for name, hist in hist_within.items():
        for ep_i, (ta, va, tl, vl) in enumerate(zip(
            hist.get("train_acc",  []),
            hist.get("val_acc",    []),
            hist.get("train_loss", []),
            hist.get("val_loss",   []),
        )):
            hist_rows.append({"model":name,"epoch":ep_i+1,
                               "train_acc":ta,"val_acc":va,
                               "train_loss":tl,"val_loss":vl})
    pd.DataFrame(hist_rows).to_csv(out_dir/"epoch_history.csv", index=False)

    # -- Plots ---------------------------------------------------------------
    if not no_plots:
        save_all_plots(
            histories_within = hist_within,
            histories_cross  = hist_cross,
            test_within      = test_within,
            test_cross       = test_cross if test_cross else None,
            plots_dir        = str(Path(plots_dir)/dataset_name/f"K{k_shot}"),
            dataset_name     = f"{dataset_name} {k_shot}-shot",
            k_shots          = k_shot,
        )

    # -- Print LaTeX table rows ----------------------------------------------
    print(f"\n{'-'*55}")
    print(f"LaTeX table rows -- {dataset_name} {k_shot}-shot:")
    print(f"{'-'*55}")
    for name in method_names:
        w = test_within.get(name, {})
        c = test_cross.get(name,  {})
        bold = "QTopo" in name
        wa   = (f"${'\\mathbf{' if bold else ''}"
                f"{w.get('acc','--')}{{\\pm}}{w.get('ci','--')}\\%"
                f"{'}'if bold else ''}$")
        ca   = (f"${'\\mathbf{' if bold else ''}"
                f"{c.get('acc','--')}{{\\pm}}{c.get('ci','--')}\\%"
                f"{'}'if bold else ''}$") if c else "---"
        print(f"  {name:20s} & {wa} & {ca} \\\\")

    return {
        "hist_within" : hist_within,
        "hist_cross"  : hist_cross,
        "test_within" : test_within,
        "test_cross"  : test_cross,
    }, models_within


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Apply quick mode overrides
    train_cfg = dict(CFG.TRAIN_CFG)
    if args.quick:
        print("\n[QUICK MODE] Overriding to 5 epochs for smoke-test.")
        train_cfg.update({
            "n_epochs"           : 5,
            "episodes_per_epoch" : 20,
            "val_episodes"       : 30,
            "test_episodes"      : 100,
            "patience"           : 5,
        })

    model_cfg    = dict(CFG.MODEL_CFG)
    pipeline_cfg = dict(CFG.PIPELINE_CFG)

    print(f"\n{'='*60}")
    print(f"QTopo-Meta -- Experiments")
    print(f"  Code: https://github.com/saeidim/QTopo")
    print(f"{'='*60}")
    print(f"  Datasets      : {args.datasets}")
    print(f"  Methods       : {args.methods}")
    print(f"  Shots         : {args.shots}")
    print(f"  Device        : {args.device}")
    print(f"  Epochs        : {train_cfg['n_epochs']}")
    print(f"  Ep/epoch      : {train_cfg['episodes_per_epoch']}")
    print(f"  Hetero train  : {args.hetero_train}")

    all_results : dict = {}
    # FIX: store one domain_data per (ds_name, k) pair so different K values
    # get their own cached objects with the correct episode configuration.
    all_domains : dict = {}   # (ds_name, k) -> domain_data
    all_models  : dict = {}   # ds_name -> k -> {method_name: model}

    for ds_name in args.datasets:
        all_results[ds_name] = {}

        for k in args.shots:
            print(f"\n{'#'*60}")
            print(f"# Dataset: {ds_name}  K={k}")
            print(f"{'#'*60}")

            try:
                domain_data = prepare_domain(
                    dataset_name = ds_name,
                    k_shot       = k,
                    pipeline_cfg = pipeline_cfg,
                    data_dir     = CFG.DATA_DIR,
                    seed         = args.seed,
                )
            except FileNotFoundError as e:
                print(f"\n[SKIP] {ds_name}: {e}")
                continue
            except Exception as e:
                print(f"\n[ERROR] {ds_name}: {e}")
                import traceback; traceback.print_exc()
                continue

            # FIX: use (ds_name, k) key to avoid overwriting with first K only
            all_domains[(ds_name, k)] = domain_data
            all_models.setdefault(ds_name, {})[k] = {}

            # Build heterogeneous domain dict for this K value if requested.
            # Only include OTHER already-processed datasets with matching d.
            hetero_domains = None
            if args.hetero_train:
                d_main = domain_data["train"]["X"].shape[1]
                hetero_domains = {}
                for (other_ds, other_k), other_data in all_domains.items():
                    if other_ds == ds_name or other_k != k:
                        continue
                    if other_data["train"]["X"].shape[1] != d_main:
                        print(f"  [hetero skip] {other_ds}: dim mismatch "
                              f"({other_data['train']['X'].shape[1]} vs {d_main})")
                        continue
                    hetero_domains[other_ds] = other_data["train"]

            results, models = run_experiment(
                domain_data   = domain_data,
                method_names  = args.methods,
                k_shot        = k,
                train_cfg     = train_cfg,
                model_cfg     = model_cfg,
                device        = args.device,
                plots_dir     = CFG.PLOTS_DIR,
                results_dir   = CFG.RESULTS_DIR,
                dataset_name  = ds_name,
                no_plots      = args.no_plots,
                hetero_domains = hetero_domains if hetero_domains else None,
            )
            all_results[ds_name][k] = results
            all_models[ds_name][k]  = models

    # -- Cross-domain phase: evaluate each source-trained model on every
    #    other dataset's test set (no retraining).
    if args.cross_domain and len(all_domains) >= 2:
        print(f"\n{'='*60}")
        print("CROSS-DOMAIN — train on source, test on target")
        print(f"{'='*60}")

        cross_rows = []
        for src in args.datasets:
            if src not in all_models:
                continue
            for k, models in all_models[src].items():
                for tgt in args.datasets:
                    # FIX: use (tgt, k) key; skip same-domain pairs explicitly
                    if tgt == src:
                        continue
                    tgt_data = all_domains.get((tgt, k))
                    if tgt_data is None:
                        print(f"  [skip] {src}->{tgt} K={k}: target not loaded")
                        continue

                    src_d = all_domains[(src, k)]["train"]["X"].shape[1]
                    tgt_d = tgt_data["test"]["X"].shape[1]
                    if src_d != tgt_d:
                        print(f"  [skip] {src}->{tgt}: feature dim mismatch "
                              f"({src_d} vs {tgt_d})")
                        continue

                    te_x = EpisodeSampler(
                        {"main": tgt_data["test"]},
                        N=CFG.EPISODE_CFG["N"], K=k, Q=CFG.EPISODE_CFG["Q"],
                        seed=999, hetero=False,
                    )
                    print(f"\n  [{src} -> {tgt}]  K={k}")
                    pair_results = {}
                    for name, model in models.items():
                        acc, ci, _ = model.evaluate(
                            te_x, train_cfg.get("test_episodes", 1000)
                        )
                        pair_results[name] = {
                            "acc": round(acc * 100, 2),
                            "ci" : round(ci  * 100, 2),
                        }
                        print(f"    {name:20s}: {acc*100:.2f}% ± {ci*100:.2f}%")
                        cross_rows.append({
                            "source": src, "target": tgt, "model": name, "K": k,
                            "cross_acc": round(acc*100, 2),
                            "cross_ci" : round(ci*100, 2),
                        })

                    # Stash cross-domain numbers for the source dataset's results.
                    res = all_results.get(src, {}).get(k, {})
                    res.setdefault("test_cross_by_target", {})[tgt] = pair_results

        if cross_rows:
            cross_df   = pd.DataFrame(cross_rows)
            cross_path = Path(CFG.RESULTS_DIR) / "cross_domain.csv"
            cross_df.to_csv(cross_path, index=False)
            print(f"\n  Cross-domain summary saved: {cross_path}")

    # -- Grand summary -------------------------------------------------------
    print(f"\n{'='*60}")
    print("GRAND SUMMARY — All datasets, all methods")
    print(f"{'='*60}")
    summary_rows = []
    for ds, ks in all_results.items():
        for k, res in ks.items():
            tw = res.get("test_within", {})
            tc = res.get("test_cross",  {})
            for name in args.methods:
                w = tw.get(name, {})
                c = tc.get(name, {})
                summary_rows.append({
                    "dataset"    : ds,
                    "model"      : name,
                    "K"          : k,
                    "within_acc" : w.get("acc"),
                    "within_ci"  : w.get("ci"),
                    "cross_acc"  : c.get("acc"),
                    "cross_ci"   : c.get("ci"),
                })
                best = max((tw.get(n, {}).get("acc") or 0) for n in args.methods)
                flag = " <-- BEST" if (w.get("acc") or 0) == best else ""
                print(f"  {ds:12s} K={k} {name:20s}: "
                      f"within={w.get('acc','--')}%  "
                      f"cross={c.get('acc','--')}%{flag}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = Path(CFG.RESULTS_DIR) / "grand_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  Grand summary saved: {summary_path}")

    print(f"\nAll results saved to: {CFG.RESULTS_DIR}")
    print(f"All plots saved to:   {CFG.PLOTS_DIR}")


if __name__ == "__main__":
    main()
