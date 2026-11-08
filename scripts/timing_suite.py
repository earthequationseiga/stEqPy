"""Timing suite: this package vs the original Fortran (MS-MPI) on JMA, back to back.

Nothing else should use the CPU meanwhile. Results go to build/timing/<date-time>/, and summary.csv
is rewritten after every case, so an interrupted run keeps what finished.
Peak memory is a working-set poll every 5 s (all ranks summed for the Fortran), so it is approximate.
Before each case: CPU load (above 15% is flagged) and a fixed speed probe. This PC has been seen
running 3-6x slow with the CPU load reading 2%, so load alone does not catch it; a probe more than
1.5x the run's fastest probe flags that case's timing as suspect.
Cases: the defaults (BFGS, K form), the old default (davidn, A form) for comparison, flat
projection, preset "original" (bit-exact Fortran path), thread scaling, then the Fortran (JMA only).
Cases on other catalogues (LOCAL_CASES) run only where their config exists: their data is not in
the repo.

Run from anywhere, about 2 h in total (of which the Fortran about 66 min):
    python -I scripts/timing_suite.py [--dry-run] [--only NAME ...] [--no-fortran]
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/jma.toml"
FORTRAN = ROOT / "build/fortran_mpi/etas8p.exe"
BOOT = "import sys; sys.path.insert(0, 'src'); from steqpy.__main__ import main; main()"
BUSY_LOAD = 15
SLOW_PROBE = 1.5
FIELDS = ["case", "threads", "wall_s", "fit_s", "outputs_s", "calls", "neg_loglik", "converged",
          "peak_mb", "load_before_pct", "probe_s", "exit"]

# name, threads, extra CLI arguments. Numba compiles on first use, so "warmup" is not a timing.
OLD_DEFAULT = ["--set", 'optimiser="davidn"', "--set", 'form="A"']
PYTHON_CASES = [("warmup", 10, []), ("warmup_old", 10, OLD_DEFAULT),
                ("default_1", 10, []), ("default_2", 10, []), ("default_3", 10, []),
                ("davidn_A", 10, OLD_DEFAULT),
                ("default_flat", 10, ["--set", 'projection="flat"']),
                ("original_t10", 10, ["--preset", "original"]),
                ("default_t1", 1, []), ("default_t2", 2, []), ("default_t4", 4, []),
                ("default_t8", 8, [])]
# Other catalogues, local only (configs gitignored, data not shareable): name -> config.
LOCAL_CASES = {"kma": "configs/kma.toml", "kma15": "configs/kma_m15.toml"}
for _name, _cfg in LOCAL_CASES.items():
    if (ROOT / _cfg).exists():
        PYTHON_CASES += [(f"{_name}_default", 10, [], ROOT / _cfg),
                         (f"{_name}_original", 10, ["--preset", "original"], ROOT / _cfg)]
FORTRAN_CASE = "fortran_np10"


def cpu_load():
    r = subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average"],
                       capture_output=True, text=True)
    return r.stdout.strip()


def speed_probe():
    """Seconds for a fixed single-threaded numpy job (best of 3): a yardstick for the machine
    itself, independent of this package."""
    code = ("import time, numpy as np; a = np.random.default_rng(0).random((600, 600)); b = []\n"
            "for _ in range(3):\n t = time.perf_counter(); [np.linalg.qr(a) for _ in range(10)]; "
            "b.append(time.perf_counter() - t)\nprint(min(b))")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env)
    return float(r.stdout.strip()) if r.returncode == 0 else float("nan")


class PeakMemory(threading.Thread):
    """Largest working set seen, summed over the processes matching a tasklist filter."""

    def __init__(self, tasklist_filter):
        super().__init__(daemon=True)
        self.filter, self.peak_kb, self.stop = tasklist_filter, 0, threading.Event()

    def run(self):
        while not self.stop.wait(5.0):
            out = subprocess.run(["tasklist", "/FI", self.filter, "/FO", "CSV", "/NH"],
                                 capture_output=True, text=True).stdout
            rows = list(csv.reader(out.splitlines()))
            total = sum(int(re.sub(r"\D", "", r[4])) for r in rows if len(r) >= 5)
            self.peak_kb = max(self.peak_kb, total)


def run_python(name, threads, extra, out_root, config=CONFIG):
    out = out_root / name
    cmd = [sys.executable, "-I", "-c", BOOT, str(config), "--out", str(out),
           "--threads", str(threads), *extra]
    load, probe = cpu_load(), speed_probe()
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    mem = PeakMemory(f"PID eq {p.pid}")
    mem.start()
    text = p.communicate()[0]
    wall = time.perf_counter() - t0
    mem.stop.set()
    (out_root / f"{name}.stdout.txt").write_text(text)
    m = re.search(r"fit ([\d.]+) s\s+outputs ([\d.]+) s\s+-logL (\S+)\s+converged (\S+)", text)
    log = out / "run.log"
    c = re.search(r"likelihood calls (\d+)", log.read_text()) if log.exists() else None
    return dict(case=name, threads=threads, wall_s=f"{wall:.1f}", fit_s=m and m[1], outputs_s=m and m[2],
                calls=c and c[1], neg_loglik=m and m[3], converged=m and m[4],
                peak_mb=f"{mem.peak_kb / 1024:.0f}", load_before_pct=load, probe_s=f"{probe:.3f}",
                exit=p.returncode)


def run_fortran(out_root):
    d = out_root / FORTRAN_CASE
    shutil.copytree(ROOT / "data/jma", d)
    load, probe = cpu_load(), speed_probe()
    mem = PeakMemory(f"IMAGENAME eq {FORTRAN.name}")
    mem.start()
    t0 = time.perf_counter()
    with open(d / "jap.in") as fin, open(d / "jap.out", "w") as fout, open(d / "stderr.txt", "w") as ferr:
        code = subprocess.run(["mpiexec", "-n", "10", str(FORTRAN)], cwd=d, stdin=fin, stdout=fout,
                              stderr=ferr).returncode
    wall = time.perf_counter() - t0
    mem.stop.set()
    nll = re.findall(r"- log likelihood =\s*(\S+)", (d / "jap.out").read_text())
    return dict(case=FORTRAN_CASE, threads=10, wall_s=f"{wall:.1f}", fit_s=None, outputs_s=None, calls=None,
                neg_loglik=nll[-1].replace("D", "E") if nll else None, converged=None,
                peak_mb=f"{mem.peak_kb / 1024:.0f}", load_before_pct=load, probe_s=f"{probe:.3f}",
                exit=code)


def write_summary(rows, out_root):
    with open(out_root / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)


def compare_outputs(out_root):
    """Timings only mean something if the runs agree: repeats must match, and differences from
    the Fortran are reported rather than assumed."""
    lines = []
    probs = lambda name: out_root / name / "probs.dat"
    for a, b in (("default_1", "default_2"), ("default_1", "default_3"), ("default_1", "default_t1")):
        if probs(a).exists() and probs(b).exists():
            same = probs(a).read_bytes() == probs(b).read_bytes()
            lines.append(f"probs.dat {a} vs {b}: {'byte-identical' if same else 'DIFFERENT'}")
    if probs("default_1").exists() and probs("davidn_A").exists():
        x, y = np.loadtxt(probs("default_1")), np.loadtxt(probs("davidn_A"))
        lines.append(f"phi default_1 (BFGS, K) vs davidn_A: max |diff| {np.abs(x[:, 1] - y[:, 1]).max():.3e}")
    if probs(FORTRAN_CASE).exists():
        ref = np.loadtxt(probs(FORTRAN_CASE))
        for a in ("original_t10", "default_flat", "default_1"):
            if probs(a).exists():
                x = np.loadtxt(probs(a))
                lines.append(f"phi {a} vs Fortran: max |diff| {np.abs(x[:, 1] - ref[:, 1]).max():.3e}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="check the setup and print the plan only")
    ap.add_argument("--only", nargs="+", metavar="NAME", help="run only these cases")
    ap.add_argument("--no-fortran", action="store_true")
    a = ap.parse_args()

    # The conda DLLs and mpiexec live here; -I does not activate the environment.
    os.environ["PATH"] = str(Path(sys.prefix) / "Library/bin") + os.pathsep + os.environ["PATH"]
    os.environ["NUMBA_NUM_THREADS"] = "10"

    names = [c[0] for c in PYTHON_CASES] + [FORTRAN_CASE]
    wanted = a.only or names
    unknown = sorted(set(wanted) - set(names))
    if unknown:
        ap.error(f"unknown cases {unknown}; choose from {names}")
    with_fortran = FORTRAN_CASE in wanted and not a.no_fortran
    missing = [str(p) for p in (CONFIG, ROOT / "data/jma/jap.in") if not p.exists()]
    if with_fortran:
        missing += [] if FORTRAN.exists() else [str(FORTRAN) + " (bash scripts/build_fortran.sh)"]
        missing += [] if shutil.which("mpiexec") else ["mpiexec"]
    if missing:
        sys.exit("missing: " + ", ".join(missing))

    cases = [c for c in PYTHON_CASES if c[0] in wanted]
    print("cases:", [c[0] for c in cases] + ([FORTRAN_CASE] if with_fortran else []))
    if a.dry_run:
        print("mpiexec:", shutil.which("mpiexec"), "| CPU load now:", cpu_load(), "% | speed probe:",
              f"{speed_probe():.3f} s")
        return

    out_root = ROOT / "build/timing" / time.strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True)
    rows = []
    # The Fortran goes last: the longest case should not cost the others if something stops it.
    jobs = [(c[0], lambda c=c: run_python(*c[:3], out_root, *c[3:])) for c in cases]
    if with_fortran:
        jobs.append((FORTRAN_CASE, lambda: run_fortran(out_root)))
    for name, job in jobs:
        print(f"{time.strftime('%H:%M:%S')} {name} ...", flush=True)
        row = job()
        rows.append(row)
        write_summary(rows, out_root)
        flag = "  LOAD HIGH BEFORE START" if row["load_before_pct"].isdigit() and int(row["load_before_pct"]) > BUSY_LOAD else ""
        print(f"  wall {row['wall_s']} s, fit {row['fit_s']} s, peak {row['peak_mb']} MB, "
              f"-logL {row['neg_loglik']}, exit {row['exit']}{flag}", flush=True)
    probes = [float(r["probe_s"]) for r in rows if r["probe_s"] != "nan"]
    if probes:
        best = min(probes)
        slow = [r["case"] for r in rows if r["probe_s"] != "nan" and float(r["probe_s"]) > SLOW_PROBE * best]
        print(f"speed probe: fastest {best:.3f} s; suspect (> {SLOW_PROBE}x): {slow or 'none'}")
    report = compare_outputs(out_root)
    (out_root / "compare.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print("results:", out_root)


if __name__ == "__main__":
    main()
