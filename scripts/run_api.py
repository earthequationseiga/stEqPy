"""Full fit through the OOP API; writes para/probs.dat/rates.dat in run.py's formats so the
files can be byte-compared with a run.py run.
Usage: python -I scripts/run_api.py data/jma/jap.in OUTDIR THREADS
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402

from etas8py import StochasticDeclustering  # noqa: E402

if __name__ == "__main__":
    infile, out, threads = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "w")
    t0 = time.perf_counter()
    sd = StochasticDeclustering.from_input_file(infile, log=lambda s: logf.write(s + "\n"),
                                                threads=threads)
    sd.fit()
    t1 = time.perf_counter()
    mx, my, tx1, tx2, ty1, ty2 = sd.grid
    gx = np.array([tx1 + (tx2 - tx1) / (mx - 1) * i for i in range(mx)])
    gy = np.array([ty1 + (ty2 - ty1) / (my - 1) * j for j in range(my)])
    rates = sd.rates(gx, gy)
    t2 = time.perf_counter()

    (out / "para").write_text(" " + "".join(f"{v:20.9f}" for v in sd.params.to_array()) + "\n")
    with open(out / "probs.dat", "w") as fh:
        for i in range(len(sd.catalog)):
            fh.write(f" {i + 1:8d}{sd.background_prob[i]:20.10f}{sd.bandwidth[i]:20.10f}"
                     f"{sd.background_rate[i]:20.10f}\n")
    with open(out / "rates.dat", "w") as fh:
        fh.write(f"{mx:12d}\n{my:12d}\n")
        np.savetxt(fh, rates, fmt="%20.10f")
    print(f"fit {t1 - t0:.1f} s  rates {t2 - t1:.1f} s  likelihood calls "
          f"{sd.n_likelihood_calls}")
    logf.close()
