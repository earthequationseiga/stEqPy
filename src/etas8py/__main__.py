"""python -m etas8py CONFIG.toml --out DIR [--threads N] [--preset original]

Fits the model described by CONFIG (see configs/jma.toml) and writes probs.dat,
rates.dat, params.txt, config.toml (resolved) and run.log to DIR. Command-line options
override the file.
"""
import argparse
import time
import tomllib
from pathlib import Path

from .declustering import StochasticDeclustering


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m etas8py")
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--preset", choices=("default", "original"))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a setting, TOML syntax: --set projection='\"laea\"' "
                         "--set inner_tol=1e-5")
    a = ap.parse_args(argv)
    over = {k: v for k, v in (("threads", a.threads), ("preset", a.preset)) if v is not None}
    for kv in a.set:
        over.update(tomllib.loads(kv))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "run.log", "w", encoding="utf-8") as logf:
        def log(msg):
            logf.write(msg + "\n")
            logf.flush()
        t0 = time.perf_counter()
        sd = StochasticDeclustering.from_config(a.config, log, **over)
        sd.fit()
        t1 = time.perf_counter()
        sd.write_outputs(out)
        t2 = time.perf_counter()
        log(f" fit {t1 - t0:.1f} s, outputs {t2 - t1:.1f} s, likelihood calls "
            f"{sd.n_likelihood_calls}, converged {sd.converged}")
    print(f"fit {t1 - t0:.1f} s  outputs {t2 - t1:.1f} s  -logL {sd.neg_loglik:.6f}  "
          f"converged {sd.converged}  -> {out}")


if __name__ == "__main__":
    main()
