"""
2D Mesh Sharding for DRWA Model.

Megatron-LM style tensor parallelism on a 2D device mesh (data x model
parallel) for the DRWA architecture.  Extends the existing 1D data-parallel
sharding to support model-parallel weight distribution.

Mesh axes
---------
When ``n_model > 1`` the mesh has two named axes: ``('data', 'model')``.
Data batches are sharded along ``data`` while weights are sharded along
``model`` using column-parallel / row-parallel alternation (Megatron-LM
convention).  When ``n_model == 1`` the module falls back to the existing
1D ``('data',)`` mesh — behaviour is unchanged from the current ``train.py``.

Usage
-----

.. code-block:: python

    from drwa.sharding import (
        create_mesh,
        get_param_sharding,
        shard_model,
        get_data_shardings,
        shard_data,
    )
    from drwa.run_config import ShardingConfig

    # 2D mesh on v5e-8 (8 devices)
    config = ShardingConfig(n_data=2, n_model=4)
    mesh, info = create_mesh(config)

    # Get PartitionSpec rules and shard model
    model = shard_model(model, mesh)

    # Shard data
    data_sharding, window_sharding = get_data_shardings(mesh)
    batch = shard_data(batch_np, mesh)
    window = shard_data(window_np, mesh, windowed=True)

    # 1D fallback (current train.py behaviour)
    config = ShardingConfig(n_data=1, n_model=1)
    mesh, info = create_mesh(config)  # uses all devices as data replicas
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from flax import nnx

logger = logging.getLogger(__name__)

__all__ = [
    "create_mesh",
    "get_param_sharding",
    "shard_model",
    "get_data_shardings",
    "shard_data",
]

def _path_str(path: Tuple) -> str:
    """``('part_a','layers',0,'attn','wq','kernel')`` → ``"part_a/layers/0/attn/wq/kernel"``."""
    return "/".join(str(p) for p in path)


# Megatron-LM tensor-parallel convention:
#   Column-parallel: P(None, 'model')  — shard output dim across model axis
#   Row-parallel:    P('model', None)  — shard input dim across model axis
#   Replicated:      P(None) / P()     — no sharding

_SHARDING_RULES: List[Tuple[str, P]] = [
    ("embed/embedding", P(None, "model")),

    ("attn/wq/kernel", P(None, "model")),
    ("attn/wq/bias", P("model",)),
    ("attn/wk/kernel", P(None, "model")),
    ("attn/wk/bias", P("model",)),
    ("attn/wv/kernel", P(None, "model")),
    ("attn/wv/bias", P("model",)),
    ("attn/wo/kernel", P("model", None)),
    ("attn/wo/bias", P(None)),
    ("attn/cos", P(None)),
    ("attn/sin", P(None)),

    ("ffn/w1/kernel", P(None, "model")),
    ("ffn/w1/bias", P("model",)),
    ("ffn/w2/kernel", P("model", None)),
    ("ffn/w2/bias", P(None)),

    ("norm1/scale", P(None)),
    ("norm1/bias", P(None)),
    ("norm2/scale", P(None)),
    ("norm2/bias", P(None)),

    ("assembly/W_base", P("model", None)),
    ("assembly/b_base", P("model",)),
    ("assembly/gamma", P(None)),
    ("assembly/U", P("model", None)),
    ("assembly/V", P(None, None)),
    ("assembly/b", P("model",)),
    ("assembly/norm/scale", P(None)),
    ("assembly/norm/bias", P(None)),
    ("assembly/proj/kernel", P(None, "model")),
    ("assembly/proj/bias", P("model",)),

    ("part_a/norm/scale", P(None)),
    ("part_a/norm/bias", P(None)),
    ("part_b/norm/scale", P(None)),
    ("part_b/norm/bias", P(None)),

    ("lm_head/kernel", P(None, "model")),
    ("lm_head/bias", P("model",)),
]


def _match_rule(path: Tuple, rule_str: str) -> bool:
    """Does *path* match *rule_str*?

    Two matching modes (disambiguated by trailing ``/``):

    *Component match* (rule ends with ``/``)
       The stripped rule text must appear as **any** component of *path*.
       Example: ``"norm1/"`` matches ``("part_a","layers",2,"norm1","scale")``.

    *Suffix match* (no trailing ``/``)
       The dot-separated components of *rule_str* must equal the **last N**
       components of *path*.
       Example: ``"attn/wq/kernel"`` matches
       ``("part_b","layers",3,"attn","wq","kernel")``.
    """
    parts = _path_str(path).split("/")

    if rule_str.endswith("/"):
        needle = rule_str.rstrip("/")
        return needle in parts

    rule_parts = rule_str.split("/")
    return parts[-len(rule_parts):] == rule_parts if len(rule_parts) <= len(parts) else False


def create_mesh(
    sharding_config,  # : ShardingConfig
    devices: Optional[List[jax.Device]] = None,
) -> Tuple[Mesh, Dict[str, Any]]:
    """Create a 1D or 2D device mesh.

    Parameters
    ----------
    sharding_config : ShardingConfig
        Must contain ``n_data`` and ``n_model``.
    devices : list of jax.Device, optional
        Defaults to ``jax.devices()``.

    Returns
    -------
    mesh : jax.sharding.Mesh
    info : dict
        Keys: ``n_data``, ``n_model``, ``n_devices``, ``is_2d``,
        ``mesh_shape``, ``axis_names``.

    Raises
    ------
    ValueError
        If ``n_data * n_model != len(devices)``.

    Examples
    --------
    >>> from drwa.run_config import ShardingConfig
    >>> mesh, info = create_mesh(ShardingConfig(n_data=2, n_model=4))
    >>> info["is_2d"]
    True
    >>> mesh, info = create_mesh(ShardingConfig(n_data=1, n_model=1))
    >>> info["is_2d"]
    False
    """
    if devices is None:
        devices = jax.devices()

    n_data = sharding_config.n_data
    n_model = sharding_config.n_model
    n_devices = len(devices)

    if n_data == 1 and n_model == 1:
        n_data = n_devices

    total = n_data * n_model
    if total != n_devices:
        raise ValueError(
            f"n_data ({n_data}) * n_model ({n_model}) = {total} "
            f"must equal number of devices ({n_devices})"
        )

    if n_model == 1:
        device_array = mesh_utils.create_device_mesh((n_data,))
        mesh = Mesh(device_array, axis_names=("data",))
        is_2d = False
    else:
        device_array = mesh_utils.create_device_mesh((n_data, n_model))
        mesh = Mesh(device_array, axis_names=("data", "model"))
        is_2d = True

    info: Dict[str, Any] = {
        "n_data": n_data,
        "n_model": n_model,
        "n_devices": n_devices,
        "is_2d": is_2d,
        "mesh_shape": tuple(mesh.shape),
        "axis_names": mesh.axis_names,
    }

    logger.info(
        "Created %s mesh shape=%s axes=%s (n_data=%d n_model=%d)",
        "2D" if is_2d else "1D",
        mesh.shape,
        mesh.axis_names,
        n_data,
        n_model,
    )

    return mesh, info


def get_param_sharding(
    model: nnx.Module,
    mesh: Mesh,
) -> Dict[Tuple, P]:
    """Map every model parameter to a ``PartitionSpec``.

    Walks the model's nnx state tree and applies the canonical Megatron-LM
    sharding rules.  Unmatched parameters receive ``P()`` (fully replicated).

    Parameters
    ----------
    model : nnx.Module
        DRWAModel instance.  Used only to discover parameter names/shapes.
    mesh : jax.sharding.Mesh
        Mesh from :func:`create_mesh`.

    Returns
    -------
    dict
        ``{nnx_state_path_tuple: PartitionSpec}``.  Example keys:
        ``("part_a", "layers", 0, "attn", "wq", "kernel")`` → ``P(None, 'model')``.

    Notes
    -----
    For a 1D mesh (``'model'`` not in axis_names) every parameter maps to
    ``P()`` — effectively replicating the entire model.
    """
    is_2d = len(mesh.shape) > 1 and "model" in mesh.axis_names

    _, state = nnx.split(model)
    flat_state = nnx.traversals.flatten_mapping(state)

    if not is_2d:
        return {key: P() for key in flat_state}

    sharding_map: Dict[Tuple, P] = {}
    unmatched: List[str] = []

    for path_key in flat_state:
        for rule_str, spec in _SHARDING_RULES:
            if _match_rule(path_key, rule_str):
                sharding_map[path_key] = spec
                break
        else:
            sharding_map[path_key] = P()
            unmatched.append(_path_str(path_key))

    if unmatched:
        logger.warning(
            "Unmatched parameters (replicating %d): %s",
            len(unmatched),
            unmatched[:25],
        )

    return sharding_map


def shard_model(
    model: nnx.Module,
    mesh: Mesh,
) -> nnx.Module:
    """Physically distribute model parameters across the mesh.

    Splits the model into graphdef + state, calls ``jax.device_put`` with
    the appropriate ``NamedSharding`` on each parameter tensor, then merges
    back.  JAX/XLA handles the necessary all-reduce / reduce-scatter
    communication during the forward and backward passes — no manual
    communication primitives are needed.

    Parameters
    ----------
    model : nnx.Module
        DRWAModel instance to shard.
    mesh : jax.sharding.Mesh
        Mesh from :func:`create_mesh`.

    Returns
    -------
    nnx.Module
        A new model instance with parameters placed on the mesh.
        The original model is unmodified.

    Examples
    --------
    >>> from drwa.run_config import ShardingConfig
    >>> mesh, _ = create_mesh(ShardingConfig(n_data=2, n_model=4))
    >>> sharded = shard_model(model, mesh)
    """
    if len(mesh.shape) <= 1 and "model" not in mesh.axis_names:
        return model

    param_specs = get_param_sharding(model, mesh)
    graphdef, state = nnx.split(model)

    flat_state = nnx.traversals.flatten_mapping(state)
    sharded_flat: Dict[Tuple, Any] = {}

    for key, variable in flat_state.items():
        spec = param_specs.get(key, P())
        if spec == P():
            sharded_flat[key] = variable
            continue

        ns = NamedSharding(mesh, spec)
        sharded_value = jax.device_put(
            jnp.asarray(variable.value), ns
        )
        sharded_flat[key] = variable.replace(value=sharded_value)

    sharded_state = nnx.traversals.unflatten_mapping(sharded_flat)
    return nnx.merge(graphdef, sharded_state)


def get_data_shardings(
    mesh: Mesh,
) -> Tuple[NamedSharding, NamedSharding]:
    """Return ``NamedSharding`` objects for input data tensors.

    Parameters
    ----------
    mesh : jax.sharding.Mesh
        Mesh from :func:`create_mesh`.

    Returns
    -------
    data_sharding : NamedSharding
        For ``[B, T]`` single-batch tensors.
    window_sharding : NamedSharding
        For ``[steps, B, T]`` windowed-data tensors.

    Examples
    --------
    >>> from drwa.run_config import ShardingConfig
    >>> mesh, _ = create_mesh(ShardingConfig(n_data=8, n_model=1))
    >>> ds, ws = get_data_shardings(mesh)
    >>> # ds  → NamedSharding(mesh, P('data', None))
    >>> # ws  → NamedSharding(mesh, P(None, 'data', None))

    2D mesh (data=2, model=4):
    >>> mesh, _ = create_mesh(ShardingConfig(n_data=2, n_model=4))
    >>> ds, ws = get_data_shardings(mesh)
    >>> # ds  → NamedSharding(mesh, P('data', None, None))
    >>> # ws  → NamedSharding(mesh, P(None, 'data', None, None))
    """
    if "model" in mesh.axis_names:
        return (
            NamedSharding(mesh, P("data", None, None)),
            NamedSharding(mesh, P(None, "data", None, None)),
        )
    return (
        NamedSharding(mesh, P("data", None)),
        NamedSharding(mesh, P(None, "data", None)),
    )


def shard_data(
    data: jnp.ndarray,
    mesh: Mesh,
    windowed: bool = False,
) -> jax.Array:
    """Place a data tensor on the mesh.

    Parameters
    ----------
    data : jnp.ndarray
        Input with shape ``[B, T]`` (single batch) or ``[steps, B, T]``
        (windowed).
    mesh : jax.sharding.Mesh
        Mesh from :func:`create_mesh`.
    windowed : bool
        When True, *data* is a windowed batch ``[steps, B, T]`` and is
        sharded with ``P(None, 'data', ...)``.

    Returns
    -------
    jax.Array
        Data array distributed on the mesh devices.
    """
    data_sharding, window_sharding = get_data_shardings(mesh)
    target = window_sharding if windowed else data_sharding
    return jax.device_put(data, target)
