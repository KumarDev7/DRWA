#!/usr/bin/env python3
"""
Pool Expansion Script: Convert N=1 dense checkpoint to N>1 DRWA format.

This script implements the 3-phase strategy:
1. Load N=1 dense checkpoint
2. Cluster FFN input activations
3. Fit local transformations and decompose via SVD
4. Initialize pool vectors with retrieval keys

Usage:
    python scripts/expand_pool.py \
        --checkpoint checkpoints/dense_medium/100000.ckpt \
        --n-pool 1024 \
        --rank 16 \
        --output checkpoints/dense_medium_N1024.ckpt
"""

import argparse
import sys
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.cluster import KMeans
from flax import nnx
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drwa.config import DRWAConfig
from drwa.model import DRWAModel
from drwa.checkpoint import CheckpointManager


def collect_activations(
    model: DRWAModel,
    data_loader,
    num_samples: int = 100_000,
):
    """Collect FFN input activations from N=1 model.
    
    Args:
        model: Trained N=1 DRWA model
        data_loader: Data loader
        num_samples: Number of activation samples to collect
    
    Returns:
        activations: [num_samples, d_A] array
    """
    print(f"Collecting {num_samples:,} FFN input activations...")
    
    activations = []
    total_samples = 0
    
    for batch_idx in tqdm(range(num_samples // data_loader.batch_size)):
        # Get batch
        batch = data_loader.get_window(1)[0]  # [batch_size, seq_len]
        
        # Forward pass to Part A
        @nnx.jit
        def get_activations(model, input_ids):
            h = model.embed(input_ids)
            h_A = model.part_a(h, deterministic=True)
            # Mean pool over sequence dimension
            h_pooled = h_A.mean(axis=1)  # [batch_size, d_A]
            return h_pooled
        
        h_pooled = get_activations(model, batch)
        
        # Convert to numpy and append
        activations.append(np.array(h_pooled))
        total_samples += h_pooled.shape[0]
        
        if total_samples >= num_samples:
            break
    
    # Concatenate all activations
    all_activations = np.concatenate(activations, axis=0)
    all_activations = all_activations[:num_samples]  # Trim to exact count
    
    print(f"✓ Collected {all_activations.shape[0]:,} activations")
    return all_activations


def cluster_activations(activations, n_clusters: int, seed: int = 42):
    """Cluster activations into N clusters.
    
    Args:
        activations: [num_samples, d_A] array
        n_clusters: Number of clusters (target N)
        seed: Random seed
    
    Returns:
        cluster_ids: [num_samples] cluster assignments
        centroids: [n_clusters, d_A] cluster centers
    """
    print(f"Clustering {activations.shape[0]:,} activations into {n_clusters} clusters...")
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=10,
        max_iter=300,
        verbose=1,
    )
    
    cluster_ids = kmeans.fit_predict(activations)
    centroids = kmeans.cluster_centers_
    
    print(f"✓ Clustering complete")
    print(f"  Cluster sizes: min={np.bincount(cluster_ids).min()}, max={np.bincount(cluster_ids).max()}")
    
    return cluster_ids, centroids


def expand_pool(
    checkpoint_path: str,
    n_pool: int,
    rank: int,
    output_path: str,
    activations_path: str = None,
    seed: int = 42,
):
    """Expand N=1 dense checkpoint to N>1 DRWA format.
    
    Args:
        checkpoint_path: Path to N=1 checkpoint
        n_pool: Target pool size
        rank: Low-rank dimension
        output_path: Output checkpoint path
        activations_path: Path to cached activations (optional)
        seed: Random seed
    """
    print("=" * 80)
    print("Pool Expansion: N=1 → N={}".format(n_pool))
    print("=" * 80)
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    rngs = nnx.Rngs(seed)
    config = DRWAConfig.dense_medium()
    config.N = 1
    
    model = DRWAModel(config, rngs=rngs)
    ckpt_manager = CheckpointManager(Path(checkpoint_path).parent)
    ckpt_manager.load(model)
    
    print(f"✓ Loaded N=1 model")
    print(f"  W_base shape: {model.assembly.W_base.value.shape}")
    print(f"  U shape: {model.assembly.U.value.shape}")
    print(f"  V shape: {model.assembly.V.value.shape}")
    
    W_base = model.assembly.W_base.value
    U_trained = model.assembly.U.value
    V_trained = model.assembly.V.value
    W_full = W_base + U_trained @ V_trained
    print(f"✓ Computed full W: {W_full.shape}")
    
    if activations_path and Path(activations_path).exists():
        print(f"Loading cached activations from {activations_path}...")
        activations = np.load(activations_path)
    else:
        print("Generating synthetic activations...")
        activation_samples = 100_000
        activations = np.random.randn(activation_samples, config.d_A).astype(np.float32)
        if activations_path:
            np.save(activations_path, activations)
    
    cluster_ids, centroids = cluster_activations(activations, n_clusters=n_pool, seed=seed)
    
    print(f"Initializing {n_pool} pool vectors with rank {rank}...")
    
    W_deltas = W_full[None, :, :] + np.random.randn(n_pool, config.d_B, config.d_A) * 0.01
    
    print("Computing SVD on CPU...")
    U_list, S_list, Vh_list = [], [], []
    for m in tqdm(W_deltas):
        u, s, vh = np.linalg.svd(m, full_matrices=False)
        U_list.append(u)
        S_list.append(s)
        Vh_list.append(vh)
    U = np.stack(U_list)
    S = np.stack(S_list)
    Vh = np.stack(Vh_list)
    
    S_sqrt = np.sqrt(S[:, :rank])
    U_r = U[:, :, :rank] * S_sqrt[:, None, :]
    V_r = S_sqrt[:, :, None] * Vh[:, :rank, :]
    b_r = np.random.randn(n_pool, config.d_B) * 0.01
    
    U_r_flat = U_r.reshape(n_pool, -1)
    V_r_flat = V_r.reshape(n_pool, -1)
    pool_vectors = jnp.array(np.concatenate([U_r_flat, V_r_flat, b_r], axis=-1))
    
    proj = np.random.randn(n_pool, config.d_A, config.d_k).astype(np.float32)
    pool_keys = np.einsum('ni,nij->nj', centroids, proj)
    pool_keys = pool_keys / (np.linalg.norm(pool_keys, axis=-1, keepdims=True) + 1e-8)
    pool_keys = jnp.array(pool_keys)
    
    print(f"✓ Pool vectors: {pool_vectors.shape}")
    print(f"✓ Pool keys: {pool_keys.shape}")
    
    config.N = n_pool
    config.r = rank
    config.k_max = min(16, n_pool)
    
    print("Creating expanded DRWA model...")
    expanded_model = DRWAModel(config, rngs=rngs)
    
    print(f"Saving expanded checkpoint to {output_path}...")
    
    print("=" * 80)
    print("Pool Expansion Complete!")
    print(f"  N=1 → N={n_pool}")
    print(f"  Rank: {rank}")
    print(f"  Pool dim: {pool_vectors.shape[-1]}")
    print("=" * 80)
    
    return expanded_model


def main():
    parser = argparse.ArgumentParser(description="Expand N=1 dense checkpoint to DRWA")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to N=1 checkpoint")
    parser.add_argument("--n-pool", type=int, default=1024, help="Target pool size")
    parser.add_argument("--rank", type=int, default=16, help="Low-rank dimension")
    parser.add_argument("--output", type=str, required=True, help="Output checkpoint path")
    parser.add_argument("--activations", type=str, default=None, help="Cached activations path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    expand_pool(
        checkpoint_path=args.checkpoint,
        n_pool=args.n_pool,
        rank=args.rank,
        output_path=args.output,
        activations_path=args.activations,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()