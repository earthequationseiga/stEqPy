"""Run configuration: preset < original control file < TOML file < call arguments.

A TOML file (see configs/jma.toml) has these tables; every key is optional except
what is needed to find the data:

    preset = "default"            # or "original" (bit-exact reproduction of the Fortran)
    [input]   control | catalog, polygon, t_min, t_start, t_end, mag_min
    [params]  initial ETAS parameters mu, A, c, alpha, p, D, q, gamma
    [grid]    lon = [min, max], lat = [min, max], n = [nlon, nlat]   (rate map)
    [model] [convergence] [run] [advanced]    algorithm settings (Settings fields)

`control` points at an original-format control file (e.g. jap.in); it supplies the
input/params/grid and npp/bwm/optimise, and anything else in the TOML overrides it.
Relative paths are relative to the TOML file. Unknown keys are errors (catches typos).
"""
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:     # Python 3.10
    import tomli as tomllib

from .readers import read_control

# Settings fields by TOML table (the tables are only for organisation: names are unique)
SECTIONS = {
    "model": ("projection", "npp", "bwm", "optimise"),
    "convergence": ("converge", "schedule", "outer_metric", "outer_tol", "max_outer",
                    "inner_metric", "inner_tol", "max_inner"),
    "run": ("threads",),
    "advanced": ("quadrature", "sgl_nodes", "sgl_nodes_background", "xlamb", "ndiv", "ndiv_check",
                 "ndiv_check_tol", "fast_xint", "tol", "max_sweeps", "ls_step", "ls_shrink",
                 "ls_expand", "frint_eps", "frint_eps_r", "log_floor", "log_penalty",
                 "chunksize"),
}
INPUT_KEYS = ("control", "catalog", "polygon", "t_min", "t_start", "t_end", "mag_min")
PARAM_KEYS = ("mu", "A", "c", "alpha", "p", "D", "q", "gamma")
GRID_KEYS = ("lon", "lat", "n")

PRESETS = {
    "default": {},                                  # = the Settings defaults
    "original": dict(converge=False, quadrature="ndiv", fast_xint=False, ndiv_check=False,
                     projection="flat"),
}


def _check_keys(table, allowed, where):
    bad = set(table) - set(allowed)
    if bad:
        raise ValueError(f"unknown key(s) in {where}: {', '.join(sorted(bad))} "
                         f"(allowed: {', '.join(allowed)})")


def load(source=None, **overrides):
    """source: TOML path, an already-parsed dict, or None. overrides: Settings fields
    and/or preset=. Returns dict(settings=, input=, params=, grid=), all plain values."""
    if source is None:
        raw, base = {}, Path(".")
    elif isinstance(source, dict):
        raw, base = source, Path(".")
    else:
        source = Path(source)
        with open(source, "rb") as fh:
            raw = tomllib.load(fh)
        base = source.parent
    _check_keys(raw, ("preset", "input", "params", "grid", *SECTIONS), "config")
    for sec, keys in SECTIONS.items():
        _check_keys(raw.get(sec, {}), keys, f"[{sec}]")
    _check_keys(raw.get("input", {}), INPUT_KEYS, "[input]")
    _check_keys(raw.get("params", {}), PARAM_KEYS, "[params]")
    _check_keys(raw.get("grid", {}), GRID_KEYS, "[grid]")
    all_fields = [k for keys in SECTIONS.values() for k in keys]
    _check_keys({k: 0 for k in overrides if k != "preset"}, all_fields, "call arguments")

    preset = overrides.pop("preset", raw.get("preset", "default"))
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {', '.join(PRESETS)}, not {preset!r}")
    settings = dict(PRESETS[preset])
    inp, params, grid = {}, {}, {}

    inp_raw = dict(raw.get("input", {}))
    for k in ("control", "catalog", "polygon"):
        if isinstance(inp_raw.get(k), str):
            inp_raw[k] = str(base / inp_raw[k])
    if "control" in inp_raw:
        c = read_control(inp_raw["control"])
        settings.update(npp=c["npp"], bwm=f"{c['bwm']!r} deg", optimise=c["ioptimise"] == 1)
        inp.update(control=inp_raw["control"], catalog=c["catalog"],
                   polygon=[list(v) for v in zip(c["poly_lon"], c["poly_lat"])],
                   t_min=c["zmin"], t_start=c["tstart"], t_end=c["tz"] + c["zmin"],
                   mag_min=c["xmg0"], _t_span=c["tz"])
        params.update(zip(PARAM_KEYS, map(float, c["b"])))
        params["D"] = f"{params['D']!r} deg2"       # repr round-trips the float exactly
        mx, my, lo1, lo2, la1, la2 = c["grid"]
        grid.update(lon=[lo1, lo2], lat=[la1, la2], n=[mx, my])
    if "t_end" in inp_raw or "t_min" in inp_raw:
        inp.pop("_t_span", None)                    # span recomputed from the new ends
    inp.update(inp_raw)
    params.update(raw.get("params", {}))
    grid.update(raw.get("grid", {}))
    for sec in SECTIONS:
        settings.update(raw.get(sec, {}))
    settings.update(overrides)
    if "schedule" in settings:
        settings["schedule"] = tuple(int(k) for k in settings["schedule"])

    missing = [k for k in ("catalog", "polygon", "t_min", "t_start", "t_end", "mag_min")
               if k not in inp]
    if missing:
        raise ValueError(f"[input] needs {', '.join(missing)} (or a control file)")
    missing = [k for k in PARAM_KEYS if k not in params]
    if missing:
        raise ValueError(f"[params] needs initial values for {', '.join(missing)}")
    if grid and set(grid) != set(GRID_KEYS):
        raise ValueError("[grid] needs lon = [min, max], lat = [min, max], n = [nlon, nlat]")
    return dict(preset=preset, settings=settings, input=inp, params=params, grid=grid)


def _toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(float(v)) if isinstance(v, float) else repr(int(v))
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"cannot write {type(v).__name__} to TOML")


def dump(settings, inp, params, grid, preset):
    """Fully resolved configuration as TOML text (re-loadable: reproduces the run)."""
    lines = [f"preset = {_toml_value(preset)}", "", "[input]"]
    lines += [f"{k} = {_toml_value(v)}" for k, v in inp.items()
              if k not in ("control", "_t_span")]
    lines += ["", "[params]"] + [f"{k} = {_toml_value(params[k])}" for k in PARAM_KEYS]
    if grid:
        lines += ["", "[grid]"] + [f"{k} = {_toml_value(grid[k])}" for k in GRID_KEYS]
    for sec, keys in SECTIONS.items():
        lines += ["", f"[{sec}]"] + [f"{k} = {_toml_value(settings[k])}" for k in keys]
    return "\n".join(lines) + "\n"
