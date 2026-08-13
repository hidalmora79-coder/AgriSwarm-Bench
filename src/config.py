"""
src/config.py

Sistema de configuración centralizada para algoritmos de enjambre.
Carga defaults desde YAML y permite sobrescritura vía kwargs/CLI.
"""

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "algorithms.yaml"

_PROFILE = None


def _load_yaml(path: str | Path = None) -> dict:
    if yaml is None:
        raise ImportError("PyYAML is required. Install: uv pip install pyyaml")
    path = Path(path or _DEFAULT_CONFIG_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Aplana dict anidado para sobrescritura CLI."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) and v:
            items.update(_flatten(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def _unflatten(items: dict, sep: str = ".") -> dict:
    """Reconstruye dict anidado desde claves planas."""
    result = {}
    for flat_key, value in items.items():
        parts = flat_key.split(sep)
        current = result
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return result


def get_profile(overrides: dict = None, config_path: str | Path = None) -> dict:
    """
    Retorna la configuración completa con sobrescrituras aplicadas.

    Args:
        overrides: Dict plano de sobrescrituras (e.g. {'hybrid.alpha': 2.0})
        config_path: Ruta alternativa al YAML

    Returns:
        Dict anidado: {pso: {...}, aco: {...}, hybrid: {...}, experiment: {...}}
    """
    global _PROFILE
    if _PROFILE is None:
        _PROFILE = _load_yaml(config_path)

    if not overrides:
        return _PROFILE

    merged = _flatten(_PROFILE)
    merged.update(overrides)
    return _unflatten(merged)


def reset_profile():
    """Limpia caché (útil en tests)."""
    global _PROFILE
    _PROFILE = None


def pso_params(config: dict = None) -> dict:
    c = config or get_profile()
    return dict(c["pso"])


def aco_params(config: dict = None) -> dict:
    c = config or get_profile()
    return dict(c["aco"])


def hybrid_params(config: dict = None) -> dict:
    c = config or get_profile()
    return dict(c["hybrid"])


def boustrophedon_params(config: dict = None) -> dict:
    c = config or get_profile()
    return dict(c.get("boustrophedon", {
        "stress_threshold": 0.4, "sweep_step": 3.0,
        "descent_step": 3.0, "micro_steps": 5,
    }))


def experiment_params(config: dict = None) -> dict:
    c = config or get_profile()
    return dict(c["experiment"])


def build_overrides_from_args(args: list[str]) -> dict:
    """
    Construye dict de sobrescrituras desde lista de strings 'key=value'.
    Ej: ['hybrid.alpha=2.0', 'pso.w=0.8']
    """
    overrides = {}
    for arg in args:
        if "=" not in arg:
            continue
        key, value = arg.split("=", 1)
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        overrides[key.strip()] = value
    return overrides
