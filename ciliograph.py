#!/usr/bin/env python3
"""
Ciliograph -- convert videos of beating cilia to digital ciliary models.

A single-file, cross-platform (Windows / Linux / macOS) Python program:
from a folder of video frames to a fitted beat pattern and its node
positions.  Nothing else is needed: drop this file anywhere and run it.

    python ciliograph.py fit        interactive fitter
    python ciliograph.py shapes     cilium shape at each time step
    python ciliograph.py info       summarise a specification file
    python ciliograph.py track      follow one cilium through every frame

Everything about a recording's analysis lives in one specification file,
`ciliograph.json`, beside the images: the scale and frame range, the base
and its orientation, the main cycle and cbf, the cilium length, the
enhancement settings, and the fitted beat -- the Fourier coefficients of the
pattern.  Ciliograph never writes a `wapa.pqd` (the file of the earlier
versions, the AXWA format): one left in a folder is read once, with its
`fourier_coeffs.txt`, and the settings then live in `ciliograph.json`.  The
z*.data node exports keep the AXWA column layout.

On Linux/macOS use `python3`.

There is no default dataset: every run asks which folder of images to use, so
opening the program can never silently reuse the last one.  Pass `--images
<folder>` to skip the dialog, or `--run-dir <folder>` for a run laid out the
old way with `data1/` and `input_images/`.  The fitter writes nothing on
opening or on importing: a folder with no specification starts from the
default parameters in memory, and Save creates the file beside the images.
(The command-line tools, which take a file as input, still write a fresh one.)

Requirements:
    pip install numpy matplotlib pillow          (scipy optional)

--------------------------------------------------------------------------
THE MODEL

The beat is a truncated Fourier series in time with a polynomial in s along
the cilium (Blake's form, with the published table of coefficients as the
starting point):

    R(s, t) = L * sum_n sum_m (A_nm cos(2 pi n t/T) + B_nm sin(2 pi n t/T)) s^m

with n = 0..7 and m = 1..3, s = 0..1 along the cilium, x along the cell
surface and z its normal.  The raw series is resampled to equal arc length
and rescaled so the cilium has the same length L at every phase; the cycle
starts at the end of the effective stroke (the cilium's sideways extreme).
The 48 coefficients are fitted to the frames by the joint image fit on the
params tab (Auto fit) and stored in the specification.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import glob
import copy
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np


__version__ = "1.0.0"


# =========================================================================
# Compute backend: a CUDA GPU through CuPy when there is one, otherwise
# every CPU core
# =========================================================================
#
# Three steps carry nearly all of the run time, and each splits into pieces
# that do not depend on one another: the minimal-path tracer (every frame is
# its own dynamic programme, in both passes), the period search in Find
# cycle (every candidate lag is its own comparison of the stack with
# itself), and the joint image fit (every start of the beam is its own
# ascent).  Auto enhance scores each of its settings independently, so the
# settings run in separate worker processes, as many as the cores and the
# free memory allow; and frames are read, enhanced and filtered on threads
# ahead of the steps that walk through them in order.  The pieces are
# computed exactly as before, only several at a time, so the CPU result is
# the single-core result bit for bit.
#
# With a CUDA GPU and CuPy installed, the tracer's dynamic programme and the
# period search run on the GPU.  The tracer runs there as one kernel per
# layer doing the CPU's own additions and comparisons, each rounded on its
# own, so it is exact too -- and it is checked against the CPU on a small
# problem the first time it is used, and left on the CPU if the two differ.
# The period search's averages are summed in a different order on the GPU
# and agree with the CPU's to float32 rounding.  The joint fit's arrays are
# a few thousand numbers each -- too small for a GPU to pay for its launch
# overhead -- so it always uses the CPU cores.
#
# Choose with --device auto|cpu|gpu and --workers N on any command, or the
# environment variables CILIOGRAPH_DEVICE and CILIOGRAPH_WORKERS.  CuPy is
# optional and never required: `pip install cupy-cuda12x` (or the build for
# your CUDA version) enables the GPU.

_COMPUTE = {"device": None, "workers": None, "resolved": None, "warned": set()}
_POOLS = {"process": None, "process_n": 0, "thread": None, "thread_n": 0,
          "broken": False}

# Temporaries the period search may hold at once, across all threads.  Each
# lag allocates about three copies of the frame stack; with many cores and a
# long recording an uncapped thread count would ask for gigabytes.
PARALLEL_MEMORY_BUDGET = 1.5e9


def _compute_warn(key: str, message: str) -> None:
    if key not in _COMPUTE["warned"]:
        _COMPUTE["warned"].add(key)
        print("ciliograph: " + message, file=sys.stderr)


def _app_dir() -> str:
    """The folder the program lives in: this file's, or -- in the
    standalone build (PyInstaller) -- the .exe's.  A one-file .exe unpacks
    itself to a temporary folder that is deleted on exit, so nothing may be
    written next to __file__ there."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _in_worker_process() -> bool:
    """True inside one of our own pool workers: they never nest pools."""
    try:
        import multiprocessing as _mp
        return _mp.parent_process() is not None
    except Exception:
        return False


def _available_cpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)


def available_memory() -> float:
    """Physical memory free for use now, in bytes (a guess of 4 GB if the
    system will not say)."""
    try:
        if sys.platform == "win32":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = _MemStatus()
            st.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return float(st.ullAvailPhys)
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", encoding="ascii") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return float(line.split()[1]) * 1024.0
        pages = os.sysconf("SC_AVPHYS_PAGES")
        return float(pages) * float(os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        return 4e9


# Threads a pool worker may use for its own per-frame work.  A worker is one
# of several running side by side, so it gets its share of the cores, which
# the parent hands over with the job.
_WORKER_THREADS = {"n": 1}


def frame_threads() -> int:
    """Threads for reading, enhancing and filtering frames in this process."""
    if _in_worker_process():
        return max(1, int(_WORKER_THREADS["n"]))
    return compute_info()["workers"]


def _prefetch(video, frames=None) -> None:
    """video.prefetch(frames) for videos that have it (see VideoFrames)."""
    fetch = getattr(video, "prefetch", None)
    if fetch is not None:
        fetch(frames)


def lookahead_map(fn, items, threads: int | None = None):
    """Yield ``fn(x)`` for each item, in order, computed a few items ahead in
    background threads.  For a chain that must take its inputs one at a
    time -- the walk tracer, whose every frame starts from the frame before
    -- while the inputs themselves (filtered frames) are independent: they
    are prepared on the other cores while the chain works.  At most twice
    ``threads`` results are held at once."""
    items = list(items)
    n = int(threads or frame_threads())
    if n <= 1 or len(items) <= 1:
        for it in items:
            yield fn(it)
        return
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque
    ahead = 2 * n
    with ThreadPoolExecutor(max_workers=n) as ex:
        queue = deque()
        source = iter(items)

        def refill():
            for x in source:
                queue.append(ex.submit(fn, x))
                if len(queue) >= ahead:
                    return

        refill()
        while queue:
            fut = queue.popleft()
            refill()
            yield fut.result()


def configure_compute(device: str | None = None, workers: int | None = None):
    """Choose the compute device ("auto", "cpu", "gpu") and CPU worker count.

    None leaves a setting as it is (initially: the environment variables
    CILIOGRAPH_DEVICE / CILIOGRAPH_WORKERS, else "auto" and every core).
    """
    if device is not None:
        device = str(device).lower()
        if device not in ("auto", "cpu", "gpu"):
            raise ValueError("device must be auto, cpu or gpu")
        _COMPUTE["device"] = device
    if workers is not None:
        _COMPUTE["workers"] = max(1, int(workers))
    _COMPUTE["resolved"] = None
    return compute_info()


def _probe_gpu():
    """(cupy, name) when a usable CUDA device is present, else (None, why)."""
    try:
        import cupy as cp
    except Exception as exc:
        return None, "CuPy is not installed ({})".format(
            type(exc).__name__)
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return None, "no CUDA device"
        # Compile and run one kernel: an installed CuPy with a missing or
        # mismatched CUDA toolkit fails here rather than mid-analysis.
        a = cp.arange(16, dtype=cp.float32)
        if float((a * a).sum()) != 1240.0:
            return None, "the GPU returned a wrong test result"
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = props.get("name", b"CUDA GPU")
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        return cp, str(name)
    except Exception as exc:
        return None, "CuPy could not use the GPU ({}: {})".format(
            type(exc).__name__, exc)


def compute_info() -> dict:
    """The resolved backend: device, CuPy module, GPU name, CPU workers."""
    res = _COMPUTE["resolved"]
    if res is not None:
        return res
    device = _COMPUTE["device"] or os.environ.get("CILIOGRAPH_DEVICE", "auto")
    device = str(device).lower()
    if device not in ("auto", "cpu", "gpu"):
        device = "auto"
    workers = _COMPUTE["workers"]
    if workers is None:
        try:
            workers = int(os.environ.get("CILIOGRAPH_WORKERS", "0"))
        except ValueError:
            workers = 0
    if workers <= 0:
        workers = _available_cpus()
    if _in_worker_process():
        device, workers = "cpu", 1
    cp, note = (None, "CPU only, as asked")
    if device in ("auto", "gpu"):
        cp, note = _probe_gpu()
        if cp is None and device == "gpu":
            _compute_warn("nogpu", "GPU asked for but not available: {}; "
                          "using the CPU".format(note))
    res = {"device": "gpu" if cp is not None else "cpu", "xp": cp,
           "gpu_name": note if cp is not None else None,
           "note": None if cp is not None else note,
           "workers": int(workers), "requested": device}
    _COMPUTE["resolved"] = res
    return res


def describe_compute() -> str:
    """One line for the status readout and `info`."""
    c = compute_info()
    cores = "{} CPU worker{}".format(c["workers"], "s" if c["workers"] != 1 else "")
    if c["device"] == "gpu":
        return "GPU ({}) + {}".format(c["gpu_name"], cores)
    return "{} (no GPU: {})".format(cores, c["note"])


def _gpu():
    """The CuPy module when the GPU is in use, else None."""
    return compute_info()["xp"]


def _shutdown_pools() -> None:
    for kind in ("process", "thread"):
        ex = _POOLS.get(kind)
        if ex is not None:
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except TypeError:                     # Python < 3.9
                ex.shutdown(wait=False)
            except Exception:
                pass
            _POOLS[kind] = None


def _processes_allowed() -> bool:
    """May worker processes be started from this program?

    A spawned worker re-imports the main script.  Ciliograph's own entry
    point is guarded, and an interactive session or notebook has no script
    to re-import, so both are safe.  Someone else's script may not be: if
    its analysis is not under ``if __name__ == "__main__":``, every worker
    would run the whole analysis again.  From such a script processes are
    used only when asked for explicitly (configure_compute(workers=N) or
    CILIOGRAPH_WORKERS); otherwise the work runs in threads.
    """
    if getattr(sys, "frozen", False):
        return True                    # the standalone .exe: our own entry point
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    if not path:
        return True
    try:
        if os.path.samefile(path, __file__):
            return True
    except (OSError, NameError, TypeError):
        pass
    return (_COMPUTE["workers"] is not None
            or bool(os.environ.get("CILIOGRAPH_WORKERS")))


def _executor(kind: str, n: int):
    """A persistent pool of ``n`` workers; processes are spawned, never
    forked, so a worker starts the same way on Windows and Linux and never
    inherits a half-copied GUI or thread state."""
    if _POOLS[kind] is not None and _POOLS[kind + "_n"] == n:
        return _POOLS[kind]
    if _POOLS[kind] is not None:
        try:
            _POOLS[kind].shutdown(wait=False)
        except Exception:
            pass
    if kind == "thread":
        from concurrent.futures import ThreadPoolExecutor
        ex = ThreadPoolExecutor(max_workers=n)
    else:
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor
        # One BLAS thread per worker: the workers are the parallelism, and
        # n workers each starting n BLAS threads only fight over the cores.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        ex = ProcessPoolExecutor(max_workers=n,
                                 mp_context=_mp.get_context("spawn"))
    if not _POOLS.get("_atexit"):
        import atexit
        atexit.register(_shutdown_pools)
        _POOLS["_atexit"] = True
    _POOLS[kind] = ex
    _POOLS[kind + "_n"] = n
    return ex


def parallel_map(fn, items, progress=None, threads: bool = False,
                 workers: int | None = None):
    """``[fn(x) for x in items]``, spread over the CPU cores.

    Processes by default (``fn`` and the items must pickle: module-level
    functions and numpy arrays), threads with ``threads=True`` for work that
    numpy already runs outside the GIL.  Results come back in order, and an
    exception is raised for the first failing item, as a plain loop would.
    ``progress(fraction)`` is called in the caller's thread as pieces
    finish.  One worker, one item, or a failure to start the pool runs the
    plain loop instead -- the answer is the same either way.
    """
    items = list(items)
    n = int(workers or compute_info()["workers"])
    if not threads and sys.platform == "win32":
        n = min(n, 61)              # the most a process pool takes on Windows
    n = max(1, min(n, len(items)))

    def serial():
        out = []
        for i, it in enumerate(items):
            out.append(fn(it))
            if progress is not None:
                progress((i + 1) / float(len(items)))
        return out

    if n <= 1 or (not threads and (_in_worker_process() or _POOLS["broken"])):
        return serial()
    if not threads and not _processes_allowed():
        _compute_warn("guard", "using threads, not processes, because the "
                      "program was not started as ciliograph.py; to use "
                      "every core fully, put your script's code under "
                      "`if __name__ == \"__main__\":` and call "
                      "configure_compute(workers=N) first")
        threads = True
    from concurrent.futures import FIRST_COMPLETED, wait
    import pickle
    try:
        from concurrent.futures.process import BrokenProcessPool
    except Exception:                                     # pragma: no cover
        BrokenProcessPool = RuntimeError
    # The pool is kept at its full size and reused; a call that wants fewer
    # at once (to stay within memory) keeps only ``n`` pieces in flight
    # rather than restarting the pool with fewer workers.
    kind = "thread" if threads else "process"
    futures = [None] * len(items)
    pending = set()
    nxt = 0
    try:
        ex = _executor(kind, max(n, _POOLS[kind + "_n"]
                                 if _POOLS[kind] is not None else n))
        while nxt < len(items) and len(pending) < n:
            futures[nxt] = ex.submit(fn, items[nxt])
            pending.add(futures[nxt])
            nxt += 1
    except (BrokenProcessPool, pickle.PicklingError, OSError,
            RuntimeError) as exc:
        if threads:
            raise
        _POOLS["broken"] = True
        _compute_warn("pool", "parallel workers unavailable ({}); running on "
                      "one core".format(exc))
        return serial()
    done = 0
    while pending:
        finished, pending = wait(pending, return_when=FIRST_COMPLETED)
        done += len(finished)
        if progress is not None:
            progress(done / float(len(items)))
        if any(f.exception() is not None for f in finished):
            nxt = len(items)               # stop feeding; report below
        try:
            while nxt < len(items) and len(pending) < n:
                futures[nxt] = ex.submit(fn, items[nxt])
                pending.add(futures[nxt])
                nxt += 1
        except (BrokenProcessPool, RuntimeError) as exc:
            if threads:
                raise
            for f in pending:
                f.cancel()
            _POOLS["broken"] = True
            _shutdown_pools()
            _compute_warn("pool", "parallel workers failed ({}); running "
                          "on one core".format(exc))
            return serial()
    if any(f is None for f in futures):
        # A piece failed and the rest were never sent: a plain loop would
        # also have stopped at the first failure, so raise it.
        for fut in futures:
            if fut is not None and fut.exception() is not None:
                exc = fut.exception()
                if not threads and isinstance(
                        exc, (BrokenProcessPool, pickle.PicklingError)):
                    _POOLS["broken"] = True
                    _shutdown_pools()
                    _compute_warn("pool", "parallel workers failed ({}); "
                                  "running on one core".format(exc))
                    return serial()
                raise exc
    out = []
    for fut in futures:
        exc = fut.exception()
        if exc is not None:
            if not threads and isinstance(
                    exc, (BrokenProcessPool, pickle.PicklingError)):
                _POOLS["broken"] = True
                _shutdown_pools()
                _compute_warn("pool", "parallel workers failed ({}); running "
                              "on one core".format(exc))
                return serial()
            raise exc
        out.append(fut.result())
    return out


# =========================================================================
# Constants
# =========================================================================

# Slider range for the Fourier coefficients.  The largest value in the default
# table is about 1.9, so this leaves room either side.
FOURIER_RANGE = (-3.0, 3.0)

# Step taken by the up/down arrows on each contrast setting, and the range
# each one is allowed to take.
# Largest per-frame jump allowed when following a particle, in pixels.  This
# only gates the linking; it is not a physical setting, so it is not exposed.
# Image scale used when a new image set has no specification yet, in pixels per
# micron.  Equivalently 0.085 um/px, the scale of the recordings this was
# built against.  Always check it against your own objective.
DEFAULT_PPUM = 11.76

MCC_MAX_DISP_PX = 25.0

# Most a traced centreline may turn in total, in degrees.  The hand-fitted
# reference waveform turns about 95 degrees from base to tip and the published
# Blake pattern about 90, so a cilium that has turned much beyond 100 is not a
# cilium: it is a hairpin, or a path that has bent over and set off along a
# transverse structure -- the cell surface, the ciliary band, or the envelope
# joining the tips of a row of cilia.  Allowing 140 let all three happen; on a
# drawn metachronal wave the traces turned 100 to 133 degrees as they ran off
# sideways along the tips, and at 110 they follow single cilia and turn in a
# smooth progression through the cycle.
MAX_TRACE_TURN = 110.0

# Below this, the frames barely repeat with the beat and a phase-resolved
# waveform is poorly constrained however it is fitted.  The scale is the height
# of the self-correlation peak above its own baseline: a clean side-on view of
# one cilium measures about 1.1, a dense carpet 0.4 to 0.6.
BEAT_WEAK_CORR = 0.6

CONTRAST_STEP = {"gamma": 0.2, "sharpness": 0.5, "brightness": 0.1}
CONTRAST_LIMIT = {"gamma": (0.05, 10.0), "sharpness": (0.0, 20.0),
                  "brightness": (0.0, 10.0)}

# Spatial degree: the powers of s used, i.e. s^1 .. s^N_FOURIER_M.  Blake's
# formulation uses a cubic (3), which is the default.  It can be reduced, but a
# lower degree cannot represent an S-shaped cilium, and the fit error then hits
# a floor that no number of time modes will lower -- see the `convert` command.
N_FOURIER_M = 3


def fourier_slots(n_space: int = N_FOURIER_M):
    """The coefficients of one Fourier mode n, laid out in two columns.

    x components on the left, z on the right; each entry is
    ``(block, m_index, column)``.  There are ``4 * n_space`` of them.
    """
    slots = []
    for block, col in (("Ax", 0), ("Bx", 0), ("Az", 1), ("Bz", 1)):
        for mi in range(n_space):
            slots.append((block, mi, col))
    return slots


FOURIER_SLOTS = fourier_slots()

COLOR_FIRST = "#090"
COLOR_RECOVERY = "#888"
COLOR_SPLIT = "#00a"
COLOR_EFFECTIVE = "#900"

# The in-app help: one structure, rendered as plain text for the terminal
# (the h key) and as a page for the browser (the Help button).  Each
# section is a title and items; an item is a paragraph, or ("table", rows)
# with (name, text) rows.
HELP_SECTIONS = [
    ("The window", [
        ("table", [
            ("image", "the current frame, with the model cilium drawn in green "
                      "from its base (the red dot).  Drag a rectangle after "
                      "pressing Crop to keep only that part; z resets the zoom."),
            ("timeline", "under the image: every frame as a tick, the current "
                         "frame in yellow, the main cycle in magenta.  Click to "
                         "jump."),
            ("tab strip", "the seven tabs in the order of a run; the panel "
                          "below it holds only the controls that tab changes."),
            ("shapes", "top right: the beat over one cycle, sixteen phases in "
                       "grey and the current frame's phase in blue, in microns, "
                       "x along the cell surface and z up its normal."),
            ("readout", "under the shapes: tab, frame and phase; the Fourier "
                        "mode on the sliders and its twelve coefficients (> "
                        "marks the selected one); ppum, cbf, length, the beat's "
                        "amplitude (tip sweep, fan angle, tip height) and the "
                        "base; last the status line -- what the last action did."),
            ("buttons", "Play/Pause, < Frame, Frame >, Save, Help; Export nodes "
                        "below them."),
            ("terminal", "the window Ciliograph was started from gets the long "
                         "readouts: Auto fit's energies and ranking, the sweep "
                         "and lean lines, timings."),
        ]),
    ]),
    ("A typical run, left to right along the tab strip", [
        ("table", [
            ("1 video", "Import images..., then type ppum and fps.  Level rotates "
                        "the frames until the line of cilia bases is horizontal.  "
                        "Crop, then drag a rectangle on the image, to keep only "
                        "the part worth analysing; every later step works inside "
                        "it.  Clear undoes it."),
            ("2 enhance", "optional: clean the frames for the measurements -- "
                          "remove the static background, flatten the lighting, "
                          "invert, deflicker (frame-to-frame lamp jitter), JPEG "
                          "deblock, denoise, line denoise (along the cilia only), "
                          "deblur (Wiener deconvolution), sharpen, local contrast "
                          "(CLAHE), upsample (bilinear or Lanczos); Fiji's tools "
                          "for thin structures -- line filter (Tubeness: draws the "
                          "dark lines of a cilium's width, nothing else), top-hat "
                          "(keeps what is thinner than the disc), bandpass (FFT "
                          "bandpass), median background (Z Project median) -- set "
                          "the base and length on the raw frames before switching "
                          "a drawing on; or an AI model of your own from an .onnx "
                          "file (needs onnxruntime; check it against a rendered "
                          "lawn with validate --model first; train_cilia_net.py "
                          "makes one that draws the cilia alone).  Not just the "
                          "view: every later step reads the enhanced frames.  Auto "
                          "sets everything here for you and touches nothing else: "
                          "with the base and the cycle set it tries the settings "
                          "on the tracker and fits beats under the leaders in the "
                          "background, keeping the setting whose pattern agrees "
                          "best with the others' (a minute or two); without them "
                          "it derives the settings from the frames alone (a "
                          "second).  Then Auto fit on the params tab."),
            ("3 position", "Find base puts the green cilium on the one you want to "
                           "measure; the sliders nudge it.  \"frame:\" chooses the "
                           "frame the pattern is drawn and exported in: upright "
                           "(turned to the fitted fan's own axis, so the fan stands "
                           "symmetric about the vertical; the surface is then "
                           "tilted in the pattern) or surface (x along the cell "
                           "surface, the lean left in)."),
            ("4 cycle", "Find cycle picks out one beat in these frames and reports "
                        "the cbf over that stretch, measured around the base "
                        "(cells beat at different rates).  The \"effective "
                        "stroke\" button says which way the effective stroke "
                        "sweeps (right, left or auto); the main cycle starts "
                        "where it ends, at the cilium's sideways extreme (the "
                        "rightmost state), in the frame the pattern is drawn in.  "
                        "With a beat fitted, Find cycle takes the start from the "
                        "beat (the traced tip is worse on a dense lawn)."),
            ("5 length", "Find length estimates how long the cilium really is "
                         "here; Click tip sets it by hand instead."),
            ("6 params", "Auto fit fits the whole waveform to that cilium: a beat "
                         "through the frame-by-frame traces, one through the "
                         "phase-averaged frames, and one fitted to the frames "
                         "jointly; the image energy picks the winner.  The "
                         "polarity (dark, bright or either) is measured, and on a "
                         "lawn beating in step the fit is held to the height the "
                         "band reaches at each phase.  The readout says how far "
                         "the runners-up lie from the winner -- how well the "
                         "frames determine the beat.  On a dense carpet the "
                         "readout says so -- the pattern then follows the "
                         "bundle's texture at that spot.  The fit holds the beat "
                         "to one stroke each way -- no back-and-forth within a "
                         "stroke -- and to the effective stroke being the faster, "
                         "sweeping the way the cycle tab's \"effective stroke\" "
                         "button says (right: left to right, the recovery right "
                         "to left; left; or auto, whichever sweep is faster).  "
                         "Auto fit again polishes the beat that is there (it is "
                         "kept unless a new start beats it by 0.03); Fresh fit "
                         "starts over.  Export nodes (below Play/Pause) writes "
                         "the node positions at every time step to output/, named "
                         "after the recording; x00/y00 is the cycle start."),
            ("7 MCC", "optional: if the frames contain particles, follow one and "
                      "get its transport velocity."),
        ]),
    ]),
    ("The tabs -- each holds only the controls it changes", [
        ("table", [
            ("video", "frame, first/last frame, ppum, fps, rotation, Level, image "
                      "import, Crop / Clear.  ppum has no tab of its own -- it is "
                      "a text box here."),
            ("enhance", "background removal, illumination flattening, invert, "
                        "deflicker, JPEG deblock, denoise, line denoise, deblur, "
                        "sharpen, contrast limit (CLAHE), upsample and its method, "
                        "AI model (.onnx path, Browse, pixel range) -- applied to "
                        "the frames the measurements read, and saved in the "
                        "specification."),
            ("position", "base x0 / y0 on the image and its rotation, Find base, "
                         "and the pattern's frame.  j/l slides sideways, i/k "
                         "drives height or rotation, p switches which."),
            ("cycle", "main beat cycle: first and last frame, which set cbf; cbf "
                      "is shown read-only.  Find cycle, the effective-stroke "
                      "setting, and playback of the main cycle or of all frames."),
            ("length", "cilium length, node count, time steps, Find length, "
                       "Click tip."),
            ("params", "Auto fit, Fresh fit, and the beat's coefficients: Fourier "
                       "mode n + its 12 coefficients A_nm / B_nm (x components "
                       "left, z components right).  t steps the mode n, p selects "
                       "a coefficient, i/k changes it, j/l steps n."),
            ("MCC", "particle transport: Pick particle, Track, Export.  The "
                    "calibration and frame range come from the video tab."),
        ]),
    ]),
    ("Buttons", [
        ("table", [
            ("Play/Pause", "play the frames (the main cycle or all of them, as the "
                           "cycle tab says) at the playback rate."),
            ("< Frame / Frame >", "step one frame; , and . do the same from any tab."),
            ("Save", "write the specification file ciliograph.json beside the "
                     "images -- every setting and the fitted beat -- keeping a "
                     "timestamped .bak of the one it replaces.  Nothing is "
                     "written otherwise."),
            ("Help", "open this page in the browser (and print it in the terminal)."),
            ("Export nodes", "write the beat's node positions at every time step "
                             "to output/ beside the images: <name>_z01a.data (the "
                             "AXWA column layout x00,y00 .. x13,y13), "
                             "<name>_cilium_shapes.csv and .npz; x00/y00 is the "
                             "cycle start, the rightmost state."),
        ]),
    ]),
    ("Keys (off while a text box is being typed into)", [
        "Click into a box and type: the first character replaces the old value; "
        "Enter applies, Escape puts the old value back.",
        ("table", [
            ("space", "play / pause"),
            ("a / s", "next / previous tab"),
            ("i / k", "increase / decrease this tab's value  (I / K = 10x)"),
            ("j / l", "left / right  (J / L = 10x)"),
            (", / .", "previous / next frame  (from any tab)"),
            ("t", "next Fourier mode n (params tab)"),
            ("p", "next coefficient (params tab); on the position tab switches "
                  "i/k between height and rotation"),
            ("o", "toggle the model overlay"),
            ("z", "reset the zoom to the whole frame"),
            ("w", "save the specification (ciliograph.json)"),
            ("r", "reload the specification from disk (undoes what was not saved)"),
            ("h", "print this help in the terminal"),
            ("q", "quit"),
        ]),
    ]),
    ("Files", [
        ("table", [
            ("ciliograph.json", "the specification, beside the frames: scale and "
                                "frame range, base and orientation, main cycle and "
                                "cbf, length, enhancement settings, the pattern's "
                                "frame, and the fitted beat's Fourier coefficients.  "
                                "Save writes it, Reload (r) reads it back; "
                                "importing a folder never loads one by itself."),
            ("wapa.pqd", "a file earlier versions left in some folders (with "
                         "fourier_coeffs.txt beside it).  Ciliograph never writes "
                         "one: if it finds one it reads it once, and from then on "
                         "everything is in ciliograph.json."),
            ("output/", "Export nodes' files, named after the recording."),
        ]),
    ]),
    ("The beat model", [
        "R(s, t) = L * sum_n sum_m (A_nm cos(2 pi n t/T) + B_nm sin(2 pi n t/T)) "
        "s^m, with 8 time modes (n = 0..7) and s^1..s^3: the form of the published "
        "pattern, which is the template every fit starts from.  The cilium has the "
        "same length L at every phase, and the cycle starts at the end of the "
        "effective stroke -- the cilium's sideways extreme, the rightmost state when "
        "the effective stroke sweeps right.",
    ]),
    ("Speed", [
        "Find cycle, Auto fit, Auto enhance and the tracer use every CPU core, and "
        "a CUDA GPU as well when CuPy is installed (pip install cupy-cuda12x).  Auto "
        "enhance tries its settings side by side, as many at once as the cores and "
        "free memory allow.  The answers are the same as on one core.  Start with "
        "--device cpu or --workers N to change this; `ciliograph info` shows what is "
        "in use.",
    ]),
    ("Command line", [
        "python ciliograph.py opens the fitter (--images <folder> skips the folder "
        "dialog; --spec <file> names the specification).  The other commands take "
        "the same options: shapes (the node export), track, beatfit, cycle, locate, "
        "length, level, beat, synth and validate (rendered carpets with a known "
        "beat, scored), info, check, selftest.  --help on any of them lists its "
        "options.",
    ]),
]


def _help_wrap(text: str, width: int, indent: int) -> list:
    """Wrap a paragraph to ``width`` columns, every line indented."""
    import textwrap
    return textwrap.wrap(text, width=width, initial_indent=" " * indent,
                         subsequent_indent=" " * indent) or [" " * indent]


def help_text(width: int = 76) -> str:
    """The help as plain text, for the terminal."""
    out = ["Ciliograph -- help", ""]
    for title, items in HELP_SECTIONS:
        out.append(title)
        for item in items:
            if isinstance(item, str):
                out += _help_wrap(item, width, 2)
            else:
                rows = item[1]
                col = min(14, max(len(name) for name, _ in rows) + 2)
                for name, text in rows:
                    lines = _help_wrap(text, width, col + 2)
                    first = lines[0].lstrip()
                    out.append("  {:<{w}}{}".format(name, first, w=col))
                    out += lines[1:]
            out.append("")
    return "\n".join(out)


def help_html() -> str:
    """The help as a page, for the Help button."""
    import html as _html
    e = _html.escape
    parts = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
             "<title>Ciliograph help</title><style>",
             "body{font-family:sans-serif;max-width:60em;margin:2em auto;"
             "padding:0 1em;line-height:1.45;color:#222}",
             "h1{font-size:1.6em}h2{font-size:1.2em;margin-top:1.6em;"
             "border-bottom:1px solid #ccc;padding-bottom:.2em}",
             "table{border-collapse:collapse;margin:.6em 0}",
             "td{vertical-align:top;padding:.3em .8em .3em 0}",
             "td.k{font-family:monospace;white-space:nowrap;color:#004}",
             "p{margin:.5em 0}</style></head><body>",
             "<h1>Ciliograph</h1>",
             "<p>From a folder of video frames to a fitted beat pattern and its "
             "node positions.  What the parts of the window do, in the order of "
             "a run.</p>"]
    for title, items in HELP_SECTIONS:
        parts.append("<h2>{}</h2>".format(e(title)))
        for item in items:
            if isinstance(item, str):
                parts.append("<p>{}</p>".format(e(item)))
            else:
                parts.append("<table>")
                for name, text in item[1]:
                    parts.append("<tr><td class='k'>{}</td><td>{}</td></tr>".format(
                        e(name), e(text)))
                parts.append("</table>")
    parts.append("</body></html>\n")
    return "\n".join(parts)


HELP_TEXT = help_text()


# Matplotlib is imported lazily so that `shapes` and `info` work on machines
# with no display, and so importing this file is cheap.
plt = None
Slider = Button = TextBox = CheckButtons = None


def _load_pyplot(interactive: bool = True):
    """Import pyplot on first use.  Returns the matplotlib module."""
    global plt, Slider, Button, TextBox, CheckButtons
    import matplotlib

    if plt is None:
        if not interactive:
            # Only force the non-GUI backend if nothing has been set up yet.
            try:
                matplotlib.use("Agg")
            except Exception:
                pass
        from matplotlib import pyplot as _plt
        from matplotlib.widgets import (Slider as _S, Button as _B,
                                        TextBox as _T, CheckButtons as _C)
        plt = _plt

        # A matplotlib widget on a hidden axes still answers to clicks: its
        # hit test asks the axes patch whether the point is inside, and
        # never whether the axes is visible.  The tabs hide their controls
        # by visibility, and two of them overlap in figure coordinates --
        # Find base on the position tab sits over Import images... on the
        # video tab -- so pressing Find base opened the folder dialog.
        # Every widget here therefore also ignores events while its axes
        # is hidden.
        class _HiddenIgnores:
            def ignore(self, event):
                return super().ignore(event) or not self.ax.get_visible()

        class Slider(_HiddenIgnores, _S):
            pass

        class Button(_HiddenIgnores, _B):
            pass

        class TextBox(_HiddenIgnores, _T):
            """matplotlib's text box behaves like an entry field: click
            into it and type, and the first character *replaces* the old
            text (typing "12.5" into a box showing 11.76 used to give
            "12.511.76", the keys landing wherever the click put the
            cursor); backspace, delete or an arrow key edits the old text
            instead; Escape puts it back; Enter submits and leaves the box
            (clicking elsewhere submits too -- matplotlib's own behaviour)."""
            _fresh = False
            _old = ""

            def begin_typing(self):
                super().begin_typing()
                self._fresh = True
                self._old = self.text

            def _click(self, event):
                was_typing = self.capturekeystrokes
                super()._click(event)
                if was_typing and self.capturekeystrokes:
                    self._fresh = False      # a click to place the cursor

            def _keypress(self, event):
                if self.ignore(event) or not self.capturekeystrokes:
                    return super()._keypress(event)
                key = event.key
                if key == "escape":
                    self.text_disp.set_text(self._old)
                    self.cursor_index = len(self._old)
                    self._fresh = False
                    self._old = self.text
                    self.stop_typing()
                    self.ax.figure.canvas.draw_idle()
                    return
                if self._fresh:
                    self._fresh = False
                    if key is not None and len(key) == 1:
                        self.text_disp.set_text("")
                        self.cursor_index = 0
                super()._keypress(event)
                if key in ("enter", "return"):
                    # submitted; done with the box (matplotlib keeps the
                    # focus, and the shortcut keys stayed dead)
                    self._old = self.text
                    self.eventson = False
                    try:
                        self.stop_typing()
                    finally:
                        self.eventson = True

        class CheckButtons(_HiddenIgnores, _C):
            pass
    return matplotlib


# =========================================================================
# The specification file  --  ciliograph.json
# =========================================================================

SPEC_NAME = "ciliograph.json"
SPEC_FORMAT = "ciliograph-spec-1"
# md5 of the z01a.data the node export writes for the published template
# beat at L = 6 um, 14 time steps, 201 nodes (selftest check 6): the same
# file since the Fourier model came in.
Z01A_TEMPLATE_MD5 = "59d73bae3a1addee91794d601cd430d4"

# The parameter keys and their meaning, as written to the specification
# file.  The names are the ones the program has always used internally.
SPEC_DOC = {
    "vid_image_filename": "prefix of the frame files",
    "vid_image_w": "frame width, px", "vid_image_h": "frame height, px",
    "vid_first_frame_to_load": "first frame used",
    "vid_last_frame_to_load": "last frame used",
    "vid_frame_main_cycle_start": "frame at which the main cycle starts (phase 0, the start of the recovery stroke)",
    "vid_ppum": "pixels per micron", "vid_fps": "frames per second",
    "vid_cil_x0": "base x, px", "vid_cil_y0_invy": "base y, px from the bottom",
    "vid_cil_orientation": "orientation of the pattern frame, deg (+ turns its normal to the left)",
    "vid_image_rotation": "rotation applied to the frames, deg",
    "length": "cilium length, um", "dl": "node spacing of the export, um",
    "cbf": "ciliary beat frequency, Hz",
    "vid_effective_stroke": "which way the effective stroke sweeps: right, left or auto",
    "vid_pattern_frame": "the frame the pattern is drawn in: upright or surface",
    "vid_pattern_turn": "how far that frame is turned from the surface's, deg",
    "vid_enh_*": "the enhance tab's settings",
    "vid_crop_*": "the crop, px of the rotated frame",
    "fourier": "the fitted beat: R(s,t) = L sum_n sum_m (A_nm cos 2 pi n t + B_nm sin 2 pi n t) s^m; "
               "rows m = 1..n_space, columns n = 0..n_modes-1; absent = the published template",
}


def _spec_value(v):
    """A parameter value as it goes into the file: numbers as numbers, to
    12 significant digits (far beyond any measurement here; it only drops
    the binary noise of a computed value, so that 300 / 30 reads 10)."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float("{:.12g}".format(float(v)))
        return int(f) if f.is_integer() and abs(f) < 1e15 else f
    if isinstance(v, str):
        # a number that arrived as text (the old files, or a typed box)
        try:
            return _spec_value(float(v))
        except ValueError:
            return v
    return v


class Spec:
    """A recording's specification: every parameter of the analysis and the
    fitted beat's Fourier coefficients, in one JSON file (ciliograph.json)
    beside the frames.

    Attributes
    ----------
    scalars : dict
        The parameters, by key (SPEC_DOC lists them).
    coeffs : dict or None
        The Fourier coefficient blocks Ax, Az, Bx, Bz (lists of rows), or
        None when no beat has been fitted yet (the template stands in).

    A ``wapa.pqd`` of the earlier versions (the AXWA format) is read too:
    its ``!key`` entries become the parameters and a ``fourier_coeffs.txt``
    beside it the coefficients.  Nothing here ever writes one: ``save``
    writes ciliograph.json whatever the file was loaded from.
    """

    def __init__(self, path: str | None = None):
        self.path = None
        self.scalars: dict = {}
        self.coeffs = None
        self.last_backup = None      # the .bak the last save made, if any
        if path is not None:
            self.load(path)

    # ------------------------------------------------------------ defaults

    @classmethod
    def defaults(cls, basename: str = "im", width: int = 320, height: int = 240,
                 first: int = 0, last: int = 0, cycle_start: float = 0,
                 ppum: float = DEFAULT_PPUM, fps: float = 300.0,
                 length: float = 6.0, cbf: float = 15.0,
                 x0: float | None = None, y0: float | None = None) -> "Spec":
        """The neutral parameter set the program opens a recording with."""
        obj = cls()
        obj.scalars = {
            "vid_image_filename": basename or "im",
            "vid_image_w": int(width), "vid_image_h": int(height),
            "vid_first_frame_to_load": int(first),
            "vid_last_frame_to_load": int(last),
            "vid_frame_main_cycle_start": cycle_start,
            "vid_ppum": float(ppum), "vid_fps": float(fps),
            "vid_cil_x0": float(width / 2.0 if x0 is None else x0),
            "vid_cil_y0_invy": float(height / 2.0 if y0 is None else y0),
            "vid_cil_orientation": 0.0,
            "vid_suimvi_first": int(first), "vid_suimvi_last": int(last),
            "vid_suimvi_on": 0,
            "length": float(length), "dl": 0.01, "cbf": float(cbf),
            "phasef_ert": 0.5, "phasef_eet": 0.9, "phasef_eft": 0.0,
        }
        return obj

    @classmethod
    def from_text(cls, text: str, path: str | None = None) -> "Spec":
        obj = cls()
        obj._parse(text, path)
        obj.path = path
        return obj

    # ---------------------------------------------------------------- load

    def load(self, path: str) -> "Spec":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        self._parse(text, path)
        self.path = path
        return self

    def _parse(self, text: str, path: str | None) -> None:
        self.scalars = {}
        self.coeffs = None
        body = text.lstrip("\ufeff \t\r\n")
        if body.startswith("{"):
            data = json.loads(body)
            params = data.get("parameters", {})
            if not isinstance(params, dict):
                raise ValueError("{}: 'parameters' is not a table".format(path))
            for k, v in params.items():
                self.scalars[str(k)] = v
            fourier = data.get("fourier")
            if isinstance(fourier, dict) and all(k in fourier for k in FOURIER_KEYS):
                self.set_coeffs({k: fourier[k] for k in FOURIER_KEYS})
            return
        # the AXWA wapa.pqd of the earlier versions
        for raw in text.split("\n"):
            line = raw.strip()
            if line.startswith("!") and "\t" in line:
                key, _, value = line.partition("\t")
                key = key[1:].strip()
                value = value.strip()
                if key in self.scalars or key in ("all_continuous", "constant_n",
                                                  "param_series", "end"):
                    continue
                try:
                    self.scalars[key] = float(value)
                except ValueError:
                    self.scalars[key] = value
        if path:
            beside = os.path.join(os.path.dirname(os.path.abspath(path)),
                                  "fourier_coeffs.txt")
            if os.path.exists(beside):
                try:
                    self.set_coeffs(load_fourier_coeffs(beside))
                except (ValueError, OSError):
                    pass

    # ----------------------------------------------------------- convenience

    def get(self, key: str, default=None):
        return self.scalars.get(key, default)

    def getf(self, key: str, default: float = 0.0) -> float:
        v = self.scalars.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def set(self, key: str, value) -> None:
        self.scalars[key] = value

    def set_coeffs(self, coeffs) -> None:
        """Keep a Fourier coefficient set (blocks Ax, Az, Bx, Bz); None
        forgets it, and the template stands in again."""
        if coeffs is None:
            self.coeffs = None
            return
        blocks = {}
        shape = None
        for k in FOURIER_KEYS:
            a = np.asarray(coeffs[k], dtype=float)
            if a.ndim != 2:
                raise ValueError("coefficient block {} is not a table".format(k))
            if shape is None:
                shape = a.shape
            elif a.shape != shape:
                raise ValueError("coefficient blocks differ in size")
            blocks[k] = [[float(v) for v in row] for row in a]
        self.coeffs = blocks

    # ----------------------------------------------------------------- save

    def _document(self) -> dict:
        params = {k: _spec_value(self.scalars[k]) for k in sorted(self.scalars)}
        doc = {"format": SPEC_FORMAT, "parameters": params}
        if self.coeffs is not None:
            n_space = len(self.coeffs["Ax"])
            n_modes = len(self.coeffs["Ax"][0]) if n_space else 0
            doc["fourier"] = {"n_modes": n_modes, "n_space": n_space}
            for k in FOURIER_KEYS:
                doc["fourier"][k] = self.coeffs[k]
        return doc

    def dumps(self) -> str:
        return json.dumps(self._document(), indent=2) + "\n"

    def save(self, path: str | None = None, backup: bool = True) -> str:
        """Write the file, keeping a timestamped .bak of the one it replaces.

        A ``wapa.pqd`` of the earlier versions is never overwritten: a spec
        loaded from one is written as ciliograph.json beside it, and the
        path written is returned.
        """
        path = path or self.path
        if path is None:
            raise ValueError("no path given and this spec was not loaded from a file")
        if path.lower().endswith(".pqd"):
            path = os.path.join(os.path.dirname(os.path.abspath(path)), SPEC_NAME)
        self.last_backup = None
        if backup and os.path.exists(path):
            stamp = datetime.datetime.now().strftime("d%Y%m%d-%H%M%S")
            self.last_backup = "{}-{}.bak".format(path, stamp)
            shutil.copy2(path, self.last_backup)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(self.dumps())
        self.path = path
        return path


# the old name, for scripts written against earlier versions
Pqd = Spec


# =========================================================================
# Periodic spline over the beat cycle
# =========================================================================

# =========================================================================
# The ciliary shape model
# =========================================================================

# =========================================================================
# Fourier-series beat pattern  (Blake-style analytic waveform)
# =========================================================================
#
#   R_c(s,t) = L_c * sum_{n=0}^{N-1} sum_{m=1}^{3}
#              ( A_nm cos(2 pi n t / T_c) + B_nm sin(2 pi n t / T_c) ) s^m
#
# evaluated separately for the x and z components.  Because m starts at 1 the
# base is pinned at the origin for all t.
#
# IMPORTANT: s is NOT the arc-length fraction.  It coincides with arc length
# only at the base (s=0) and the tip (s=1); in between it must be solved for
# numerically if you want nodes equally spaced along the cilium.  This is
# stated explicitly in Poon et al.'s supplemental material, and it is also why
# the cilium's length is not exactly constant over the beat -- the coefficients
# are scaled so the length averages L_c over one period.

# Default coefficients: rows are m = 1, 2, 3; columns are n = 0 .. 7.
FOURIER_DEFAULT = {
    "Ax": [[0.0596, 0.0109, 0.0021, -0.0297, -0.0121, 0.0112, -0.0039, 0.0019],
           [-0.0167, 1.9045, -0.1661, 0.0360, 0.0562, -0.0678, 0.0207, -0.0038],
           [0.1597, -1.2507, -0.0128, 0.0449, -0.0409, 0.0449, -0.0123, 0.00197]],
    "Az": [[1.0811, 0.0226, 0.1453, -0.0382, 0.0387, -0.0026, -0.0014, 0.0060],
           [-0.7417, -0.2192, -0.9769, 0.2147, -0.1518, 0.0117, 0.0156, -0.0289],
           [0.4318, 0.1192, 0.6941, -0.1402, 0.0771, -0.0022, -0.0134, 0.01997]],
    "Bx": [[0.0, 0.0032, 0.0122, -0.0331, -0.0097, -0.0003, -0.0016, 0.0],
           [0.0, -0.4997, 0.1381, 0.1000, 0.0022, -0.0195, 0.0032, 0.0],
           [0.0, 0.4722, -0.2068, -0.0216, -0.0003, 0.0087, 0.0016, 0.0]],
    "Bz": [[0.0, -0.0633, 0.0130, -0.0049, 0.0059, -0.0026, -0.0016, 0.0],
           [0.0, 0.2792, 0.1092, -0.0492, 0.0232, 0.0125, 0.0156, 0.0],
           [0.0, -0.2116, -0.0837, 0.0692, -0.0535, 0.0072, -0.0145, 0.0]],
}

FOURIER_KEYS = ("Ax", "Az", "Bx", "Bz")

# Number of Fourier modes used by default: all eight the table carries.  The
# high harmonics contribute little on their own -- truncating to four costs
# about 0.09 um RMS on a 6 um cilium -- but they are what carries the sharply
# curved part of the recovery stroke, and that is the part of the beat the
# automatic fit was losing.  There is no cost to keeping them: the table is
# published data, not something being searched for.
FOURIER_DEFAULT_MODES = 8


def truncate_space(coeffs: dict, n_space: int) -> dict:
    """Keep the first ``n_space`` powers of s (rows), zero-padding if needed.

    WARNING: unlike dropping high Fourier modes, dropping a power of s is *not*
    a harmless truncation.  The s^m terms of a fitted set largely cancel one
    another, so deleting the cubic row leaves the rest unbalanced -- truncating
    the built-in table to s^1..s^2 stretches the cilium from 6 um to over 10 um.
    To work at a lower spatial degree, re-fit at that degree instead:

        ciliograph.py convert --space 2

    which solves for coefficients that are correct for the reduced basis.
    """
    dropped = len(coeffs["Ax"]) - n_space
    if dropped > 0:
        print("warning: dropping {} power(s) of s from an existing coefficient "
              "set is not a valid truncation; re-fit with "
              "`convert --space {}` instead".format(dropped, n_space),
              file=sys.stderr)
    out = {}
    for key in FOURIER_KEYS:
        rows = [list(r) for r in coeffs[key]]
        width = len(rows[0]) if rows else 0
        if n_space <= len(rows):
            out[key] = rows[:n_space]
        else:
            out[key] = rows + [[0.0] * width
                               for _ in range(n_space - len(rows))]
    return out


def truncate_coeffs(coeffs: dict, n_modes: int) -> dict:
    """Keep the first ``n_modes`` Fourier modes, zero-padding if asked for more.

    Dropping high-n terms is a plain truncation of the series -- the remaining
    coefficients are unchanged, so the low-order shape is preserved.
    """
    out = {}
    for key in FOURIER_KEYS:
        rows = []
        for row in coeffs[key]:
            row = list(row)
            if n_modes <= len(row):
                rows.append(row[:n_modes])
            else:
                rows.append(row + [0.0] * (n_modes - len(row)))
        out[key] = rows
    return out


def load_fourier_coeffs(path: str) -> dict:
    """Read a Fourier coefficient file.

    Format -- blank lines and lines starting with # are ignored:

        Ax
        <m=1 row: one value per n, whitespace separated>
        <m=2 row>
        <m=3 row>
        Az
        ...
        Bx
        ...
        Bz
        ...

    All four blocks must have the same number of columns (Fourier modes N).
    """
    blocks: dict = {}
    current = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            token = line.replace(":", "").strip()
            if token in FOURIER_KEYS:
                current = token
                blocks[current] = []
                continue
            if current is None:
                raise ValueError(
                    "{}: data before any of {} was named".format(
                        path, ", ".join(FOURIER_KEYS)))
            blocks[current].append([float(v) for v in line.split()])

    missing = [k for k in FOURIER_KEYS if k not in blocks]
    if missing:
        raise ValueError("{}: missing block(s) {}".format(path, ", ".join(missing)))

    heights = {len(blocks[key]) for key in FOURIER_KEYS}
    if len(heights) != 1:
        raise ValueError(
            "{}: all blocks must have the same number of s powers, saw {}".format(
                path, sorted(heights)))
    if heights == {0}:
        raise ValueError("{}: blocks are empty".format(path))

    widths = set()
    for key in FOURIER_KEYS:
        for row in blocks[key]:
            widths.add(len(row))
    if len(widths) != 1:
        raise ValueError(
            "{}: all rows must have the same number of modes, saw {}".format(
                path, sorted(widths)))

    return blocks


def save_fourier_coeffs(path: str, coeffs: dict) -> str:
    """Write coefficients in the format :func:`load_fourier_coeffs` reads."""
    n_modes = len(coeffs["Ax"][0])
    out = [
        "# Fourier coefficients for the ciliary beat pattern.",
        "#   R_c(s,t) = L * sum_n sum_m (A_nm cos(2 pi n t/T) "
        "+ B_nm sin(2 pi n t/T)) s^m",
        "# Rows are m = 1, 2, 3.  Columns are n = 0 .. {}.".format(n_modes - 1),
        "",
    ]
    for key in FOURIER_KEYS:
        out.append(key)
        for row in coeffs[key]:
            out.append("  " + "  ".join("{: .6g}".format(v) for v in row))
        out.append("")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out))
    return path


def fit_fourier_to_shapes(xs, ys, phases, length: float, n_modes: int = 4,
                          n_s: int = 20, n_space: int = N_FOURIER_M):
    """Least-squares fit of Fourier coefficients to a set of cilium shapes.

    This is how these coefficients are meant to be obtained -- Blake's method,
    as used by Poon et al. -- rather than by adjusting them by hand.  Because
    the model is linear in A_nm and B_nm, the fit is a single exact solve with
    no iteration or initial guess.

    Parameters
    ----------
    xs, ys : sequence of arrays
        Centreline coordinates in microns, one array per time step.  Each shape
        is assumed to be sampled at equal arc length along the cilium.
    phases : sequence of float
        Cycle phase in [0, 1) for each shape.
    length : float
        Cilium length L_c, in microns; coefficients come out dimensionless.
    n_modes : int
        Number of Fourier modes N; coefficients are produced for n = 0..N-1.
    n_s : int
        Points along the cilium used in the fit.

    Returns
    -------
    (coeffs, report) where coeffs is the usual dict of blocks and report holds
    the residual statistics.
    """
    phases = np.asarray(phases, dtype=float)
    s = np.linspace(0.0, 1.0, n_s)

    # Resample every shape onto the same equal-arc-length grid.
    X = np.zeros((len(phases), n_s))
    Z = np.zeros((len(phases), n_s))
    for k in range(len(phases)):
        x = np.asarray(xs[k], dtype=float)
        y = np.asarray(ys[k], dtype=float)
        seg = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
        frac = seg / seg[-1] if seg[-1] > 0 else np.linspace(0, 1, len(x))
        X[k] = np.interp(s, frac, x)
        Z[k] = np.interp(s, frac, y)

    # Design matrix: (cos 2 pi n t, sin 2 pi n t) x s^m -- linear in the
    # unknown coefficients, so one lstsq solve gives the exact optimum.
    columns, index = [], []
    for n in range(n_modes):
        for m in range(1, n_space + 1):
            columns.append(np.outer(np.cos(2 * np.pi * n * phases), s ** m).ravel())
            index.append(("A", n, m))
            columns.append(np.outer(np.sin(2 * np.pi * n * phases), s ** m).ravel())
            index.append(("B", n, m))
    G = np.column_stack(columns)

    cx, *_ = np.linalg.lstsq(G, X.ravel() / length, rcond=None)
    cz, *_ = np.linalg.lstsq(G, Z.ravel() / length, rcond=None)

    coeffs = {k: [[0.0] * n_modes for _ in range(n_space)]
              for k in FOURIER_KEYS}
    for value_x, value_z, (ab, n, m) in zip(cx, cz, index):
        coeffs["{}x".format(ab)][m - 1][n] = float(value_x)
        coeffs["{}z".format(ab)][m - 1][n] = float(value_z)

    rx = G @ cx * length - X.ravel()
    rz = G @ cz * length - Z.ravel()
    dist = np.hypot(rx, rz)
    report = {
        "n_modes": n_modes,
        "n_coeffs": 2 * len(cx),
        "rms_um": float(np.sqrt(np.mean(rx ** 2 + rz ** 2))),
        "max_um": float(dist.max()),
        "length_um": float(length),
        "n_shapes": len(phases),
        "n_s": n_s,
    }
    return coeffs, report


class FourierWave:
    """Analytic Fourier-series ciliary beat pattern.

    Parameters
    ----------
    coeffs : dict, optional
        Blocks ``Ax``, ``Az``, ``Bx``, ``Bz``, each 3 rows (m=1,2,3) by N
        columns (n=0..N-1).  Defaults to the built-in table.
    length : float
        Cilium length L_c, in microns.
    cbf : float
        Ciliary beat frequency in Hz; the period T_c is its inverse.
    fixed_length : bool
        Hold the arc length at L_c at every phase (the default).  ``s`` is a
        parameter, not the arc-length fraction, so the cubic in ``s`` has
        whatever length it happens to have at each phase -- on one recording
        it ran from 4.2 to 6.3 microns over a beat for a 5.5 micron cilium,
        and the drawn cilium visibly grew and shrank.  A cilium does not.
        Each phase's shape is therefore scaled about the base so its arc
        length is exactly L_c: the shape the fit found is kept, only its
        size is corrected.  Off, the raw polynomial is returned, which the
        fit needs when it measures its own mean length.
    """

    def __init__(self, coeffs: dict | None = None, length: float = 6.0,
                 cbf: float = 1.0, phase_shift: float = 0.0,
                 n_modes: int | None = None, n_space: int | None = None,
                 fixed_length: bool = True):
        coeffs = coeffs or FOURIER_DEFAULT
        if n_modes is not None:
            coeffs = truncate_coeffs(coeffs, int(n_modes))
        if n_space is not None:
            coeffs = truncate_space(coeffs, int(n_space))
        self.set_coeffs(coeffs)
        self.length = float(length)
        self.cbf = float(cbf)
        self.phase_shift = float(phase_shift)
        self.fixed_length = bool(fixed_length)

    def set_coeffs(self, coeffs: dict) -> None:
        self.coeffs = {k: np.array(coeffs[k], dtype=float) for k in FOURIER_KEYS}
        self.n_space, self.n_modes = self.coeffs["Ax"].shape
        for key in FOURIER_KEYS:
            if self.coeffs[key].shape != (self.n_space, self.n_modes):
                raise ValueError(
                    "coefficient block {} has shape {}, expected {}".format(
                        key, self.coeffs[key].shape,
                        (self.n_space, self.n_modes)))
        # Powers s^1 .. s^n_space, and their derivatives for arc length.
        self._powers = np.arange(1, self.n_space + 1)

    # -------------------------------------------------------- evaluation

    def _sm_coeffs(self, t: float):
        """The three s^m coefficients for x and z at cycle phase t."""
        n = np.arange(self.n_modes)
        ang = 2.0 * np.pi * n * (float(t) + self.phase_shift)
        c, s = np.cos(ang), np.sin(ang)
        cx = (self.coeffs["Ax"] * c + self.coeffs["Bx"] * s).sum(axis=1)
        cz = (self.coeffs["Az"] * c + self.coeffs["Bz"] * s).sum(axis=1)
        return cx, cz

    def _raw_position(self, s, t: float):
        """The polynomial itself, for L_c = 1 and no length correction."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        cx, cz = self._sm_coeffs(t)
        powers = np.stack([s ** m for m in self._powers])
        return ((cx[:, None] * powers).sum(axis=0),
                (cz[:, None] * powers).sum(axis=0))

    def _raw_tangent(self, s, t: float):
        s = np.atleast_1d(np.asarray(s, dtype=float))
        cx, cz = self._sm_coeffs(t)
        d = np.stack([m * s ** (m - 1) for m in self._powers])
        return ((cx[:, None] * d).sum(axis=0), (cz[:, None] * d).sum(axis=0))

    def raw_arclength(self, t: float, n_quad: int = 2001) -> float:
        """Arc length of the unit polynomial at phase t, before correction.

        One for a perfect fit; how far it strays from one at each phase is
        how much the fit had to be rescaled there, and a fair measure of the
        fit's own quality.
        """
        return float(self._arclength_table(t, n_quad=n_quad)[1][-1])

    def _scale(self, t: float) -> float:
        """Microns per unit of the raw polynomial at phase t."""
        if not self.fixed_length:
            return self.length
        raw = self.raw_arclength(t)
        return self.length / raw if raw > 0 else self.length

    def position(self, s, t: float):
        """(x, z) in microns at parameter s and cycle phase t in [0, 1)."""
        x, z = self._raw_position(s, t)
        k = self._scale(t)
        return k * x, k * z

    def tangent(self, s, t: float):
        """d(x,z)/ds -- used for arc length."""
        dx, dz = self._raw_tangent(s, t)
        k = self._scale(t)
        return k * dx, k * dz

    # -------------------------------------------------- arc-length spacing

    def _arclength_table(self, t: float, n_quad: int = 2001):
        """Cumulative arc length of the *raw* polynomial against s.

        Raw on purpose: the length correction is a constant factor at a
        given phase, so it cannot change where the equal-arc-length nodes
        fall, and using the corrected tangent here would need the correction
        that this table is used to compute.
        """
        s = np.linspace(0.0, 1.0, n_quad)
        dx, dz = self._raw_tangent(s, t)
        speed = np.hypot(dx, dz)
        # Cumulative trapezoidal integral of |dR/ds|.
        cum = np.concatenate(([0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1])
                                              * np.diff(s))))
        return s, cum

    def arclength(self, t: float) -> float:
        """Total arc length of the cilium at phase t, in microns.

        With ``fixed_length`` this is L_c at every phase, by construction.
        """
        if self.fixed_length:
            return self.length
        return float(self._arclength_table(t)[1][-1] * self._scale(t))

    def s_for_equal_arclength(self, n_pts: int, t: float):
        """The s values putting n_pts nodes at equal fractions of arc length.

        s is not the arc-length fraction, so this inverts the relationship
        numerically -- the step Poon et al. describe for placing nodes.
        """
        s_grid, cum = self._arclength_table(t)
        total = cum[-1]
        if total <= 0:
            return np.linspace(0.0, 1.0, n_pts)
        targets = np.linspace(0.0, total, n_pts)
        return np.interp(targets, cum, s_grid)

    def shape_at(self, t: float, n_pts: int = 201, equal_arclength: bool = True):
        """(x, z) centreline at cycle phase t, in microns.

        With ``equal_arclength`` the nodes are equally spaced along the cilium
        (what the paper does); otherwise they are equally spaced in s.
        """
        if equal_arclength:
            s = self.s_for_equal_arclength(n_pts, t)
        else:
            s = np.linspace(0.0, 1.0, n_pts)
        return self.position(s, t)

    def shapes(self, n_cs: int, n_pts: int = 201, equal_arclength: bool = True):
        """Shapes at n_cs phases evenly spaced over one beat cycle."""
        phases = [i / n_cs for i in range(n_cs)]
        xs, ys = [], []
        for t in phases:
            x, z = self.shape_at(t, n_pts=n_pts, equal_arclength=equal_arclength)
            xs.append(x)
            ys.append(z)
        return phases, xs, ys

    def length_over_cycle(self, n: int = 200):
        """(phases, arc lengths in microns) over one beat.

        Constant at L_c with ``fixed_length``; the raw polynomial's own
        variation is what stretch_over_cycle reports.
        """
        ph = np.linspace(0.0, 1.0, n, endpoint=False)
        return ph, np.array([self.arclength(p) for p in ph])

    def stretch_over_cycle(self, n: int = 200):
        """(phases, raw polynomial length / its mean) over one beat.

        How much each phase had to be rescaled to hold the length.  Close to
        one everywhere for a good fit; 0.76..1.14 was the recording on which
        the growing-and-shrinking cilium was first noticed.
        """
        ph = np.linspace(0.0, 1.0, n, endpoint=False)
        raw = np.array([self.raw_arclength(p) for p in ph])
        mean = raw.mean()
        return ph, raw / mean if mean > 0 else raw

    @classmethod
    def from_spec(cls, spec, coeffs: dict | None = None,
                  n_modes: int | None = None,
                  n_space: int | None = None) -> "FourierWave":
        """The beat of a specification: its coefficients (or ``coeffs``
        given here, or the published template), its length and cbf."""
        if coeffs is None:
            coeffs = getattr(spec, "coeffs", None)
        return cls(coeffs=coeffs,
                   length=spec.getf("length", 6.0),
                   cbf=spec.getf("cbf", 1.0),
                   n_modes=n_modes, n_space=n_space)

    from_pqd = from_spec


class WaveModel:
    """The geometry of a recording's analysis, built from a :class:`Spec`:
    scale, frame size, base, orientation, beat period, length -- and the
    fitted beat (``wave``, a :class:`FourierWave`: the spec's coefficients,
    or the published template until a beat is fitted)."""

    def __init__(self, pqd: Spec, wave=None):
        self.pqd = pqd
        self.wave = wave
        self.rebuild()

    # ------------------------------------------------------------- geometry

    def rebuild(self) -> None:
        """Re-read the length and node spacing; keep the beat in step."""
        self.length = self.pqd.getf("length", 6.0)
        self.dl = self.pqd.getf("dl", 0.01)
        self.n_xy = int(self.length / self.dl) + 1
        if self.wave is None:
            self.wave = FourierWave.from_spec(self.pqd)
        self.wave.length = self.length
        self.wave.cbf = self.cbf

    def shape_at(self, t: float):
        """(x, y) centreline at cycle phase t, in microns, n_xy nodes at equal
        arc length -- the fitted beat, or the template before a fit."""
        return self.wave.shape_at(t, n_pts=self.n_xy)

    def shapes(self, n_cs: int):
        """Shapes at n_cs phases evenly spaced over one beat cycle."""
        phases = [i / n_cs for i in range(n_cs)]
        xs, ys = [], []
        for t in phases:
            x, y = self.shape_at(t)
            xs.append(x)
            ys.append(y)
        return phases, xs, ys

    # ------------------------------------------------------- video / timing

    @property
    def fps(self) -> float:
        return self.pqd.getf("vid_fps", 300.0)

    @property
    def cbf(self) -> float:
        """Ciliary beat frequency, Hz."""
        return self.pqd.getf("cbf", 1.0)

    @property
    def period_frames(self) -> float:
        """Length of one beat cycle, in video frames."""
        cbf = self.cbf
        return self.fps / cbf if cbf > 0 else 1.0

    @property
    def macysta(self) -> float:
        """Frame number at which the main beat cycle starts."""
        return self.pqd.getf("vid_frame_main_cycle_start", 0.0)

    def phase_of_frame(self, frame: float) -> float:
        return ((frame - self.macysta) / self.period_frames) % 1.0

    # --------------------------------------------------- placement on image

    @property
    def upsample(self) -> int:
        """Enhancement upsampling factor: frames are this many times larger."""
        return int(np.clip(round(self.pqd.getf("vid_enh_upsample", 1.0)), 1, 4))

    @property
    def ppum(self) -> float:
        """Pixels per micron of the frames as delivered -- after upsampling."""
        return self.pqd.getf("vid_ppum", 1.0) * self.upsample

    @property
    def image_h(self) -> float:
        """Height of the frame the model is placed in -- crop and upsampling."""
        crop = crop_from_pqd(self.pqd)
        h = (float(crop[3] - crop[1]) if crop is not None
             else self.pqd.getf("vid_image_h", 0.0))
        return h * self.upsample

    @property
    def image_w(self) -> float:
        crop = crop_from_pqd(self.pqd)
        w = (float(crop[2] - crop[0]) if crop is not None
             else self.pqd.getf("vid_image_w", 0.0))
        return w * self.upsample

    def to_pixels(self, x, y, image_h: float | None = None):
        """Map model coordinates (microns) to image pixel coordinates.

        The base sits at (vid_cil_x0, vid_cil_y0_invy), where y is measured up
        from the bottom of the image -- that is what the `_invy` suffix means.
        vid_cil_orientation rotates the cilium about its base, in degrees.
        """
        if image_h is None:
            image_h = self.image_h

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        x0 = self.pqd.getf("vid_cil_x0", 0.0)
        y0_invy = self.pqd.getf("vid_cil_y0_invy", 0.0)
        ori = np.deg2rad(self.pqd.getf("vid_cil_orientation", 0.0))
        ppum = self.ppum

        c, s = np.cos(ori), np.sin(ori)
        px = x0 + (x * c - y * s) * ppum
        py = (image_h - y0_invy) - (x * s + y * c) * ppum
        return px, py


# =========================================================================
# Video frames
# =========================================================================

def _is_image(path: str) -> bool:
    """Is this a real image file we should treat as a frame?

    Hidden files are excluded, which matters more than it sounds: copying a
    folder from a Mac leaves an AppleDouble sidecar ``._im0000.jpg`` beside
    every ``im0000.jpg``.  They carry an image extension but hold 4 KB of
    metadata, and because "._" sorts before letters they were the first thing
    the loader tried -- so a folder of perfectly good images looked unreadable.
    """
    name = os.path.basename(str(path))
    if name.startswith("."):
        return False
    return os.path.splitext(name)[1].lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".pgm", ".ppm",
    }


def _is_tiff(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in {".tif", ".tiff"}


def natural_key(name: str):
    """Sort key that orders embedded numbers numerically.

    Frame files are often numbered without padding, e.g. ``...-9``, ``...-10``,
    ``...-100``.  A plain string sort then reads 0, 1, 10, 100, 11, 2 -- the
    frames load out of time order, which silently corrupts anything that
    depends on the sequence: the beat period, particle tracking, the fits.
    """
    import re
    parts = re.split(r"(\d+)", os.path.basename(str(name)))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def image_pages(path) -> int:
    """Number of pages in an image file.  Multi-page TIFF stacks are common."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(getattr(im, "n_frames", 1))
    except Exception:
        return 1


def check_images(directory: str, basename: str = ""):
    """Try to read every frame, and report the ones that fail.

    Worth running when a folder will not load: it names the offending files
    instead of stopping at the first exception.
    """
    video = VideoFrames(directory, basename=basename)
    bad = []
    for f in range(video.first, video.last + 1):
        path, page = video.index[f]
        try:
            _load_image(path, page)
        except Exception as exc:
            bad.append((f, path, page, str(exc).splitlines()[0]))
    return {"n_frames": len(video), "bad": bad, "video": video}


def _pil_read(path, page: int = 0):
    """Read one page with Pillow, working around its TIFF decoder failures.

    Pillow raises ``OSError: -2`` from ``_load_libtiff`` on some compressed
    TIFFs -- the libtiff path is fussy about how the file is presented and
    about a few tag combinations.  Each fallback below fixes a different one of
    those, so they are tried in turn rather than letting the first failure
    abort the load.
    """
    from PIL import Image, ImageFile, TiffImagePlugin
    import io

    attempts = []

    def straight():
        with Image.open(path) as im:
            if page:
                im.seek(page)
            return np.asarray(im)

    def in_memory():
        # libtiff can fail on the file handle but succeed on the same bytes.
        with open(path, "rb") as fh:
            data = fh.read()
        with Image.open(io.BytesIO(data)) as im:
            if page:
                im.seek(page)
            return np.asarray(im)

    def tolerant():
        old = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            return in_memory()
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = old

    def without_libtiff():
        old = TiffImagePlugin.READ_LIBTIFF
        TiffImagePlugin.READ_LIBTIFF = False
        try:
            return in_memory()
        finally:
            TiffImagePlugin.READ_LIBTIFF = old

    for name, fn in (("direct", straight), ("in memory", in_memory),
                     ("tolerant", tolerant),
                     ("without libtiff", without_libtiff)):
        try:
            return fn()
        except Exception as exc:
            attempts.append("{}: {}".format(name, exc))

    raise OSError("could not read {}{}\n  {}".format(
        os.path.basename(str(path)),
        " page {}".format(page) if page else "",
        "\n  ".join(attempts)))


def _tiff_read_raw(path, page: int = 0):
    """Decode a TIFF without Pillow.

    Pillow's libtiff binding raises ``OSError: -2`` on some builds -- notably
    Anaconda's, on deflate-compressed files -- and none of its own fallbacks
    help because they all end up in the same codec.  This reads the tags and
    inflates the strips directly, which needs nothing beyond the standard
    library.  Covers the strip layouts microscopes actually write: no
    compression, deflate, and PackBits, 8/16/32-bit, single plane.
    """
    import struct
    import zlib

    with open(path, "rb") as fh:
        data = fh.read()

    if len(data) < 8:
        raise OSError("{}: too short to be a TIFF".format(path))
    order = data[:2]
    if order == b"II":
        end = "<"
    elif order == b"MM":
        end = ">"
    else:
        raise OSError("{}: not a TIFF (bad byte order mark)".format(path))
    magic, = struct.unpack(end + "H", data[2:4])
    if magic != 42:
        raise OSError("{}: not a classic TIFF (magic {})".format(path, magic))

    offset, = struct.unpack(end + "I", data[4:8])
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8,
             11: 4, 12: 8}
    fmts = {1: "B", 2: "c", 3: "H", 4: "I", 6: "b", 8: "h", 9: "i", 11: "f",
            12: "d"}

    for _ in range(page + 1):
        if offset == 0 or offset >= len(data):
            raise OSError("{}: no page {}".format(path, page))
        count, = struct.unpack(end + "H", data[offset:offset + 2])
        tags = {}
        for i in range(count):
            p = offset + 2 + i * 12
            tag, typ, n = struct.unpack(end + "HHI", data[p:p + 8])
            size = sizes.get(typ, 1) * n
            if size <= 4:
                raw = data[p + 8:p + 8 + size]
            else:
                at, = struct.unpack(end + "I", data[p + 8:p + 12])
                raw = data[at:at + size]
            if typ in fmts and size:
                try:
                    tags[tag] = struct.unpack(
                        end + fmts[typ] * n, raw[:sizes[typ] * n])
                except struct.error:
                    tags[tag] = ()
            else:
                tags[tag] = raw
        nxt, = struct.unpack(end + "I", data[offset + 2 + count * 12:
                                             offset + 6 + count * 12])
        offset = nxt

    def one(tag, default=None):
        v = tags.get(tag)
        return v[0] if v else default

    width = one(256)
    height = one(257)
    bits = one(258, 8)
    compression = one(259, 1)
    samples = one(277, 1)
    if width is None or height is None:
        raise OSError("{}: missing image dimensions".format(path))

    starts = tags.get(273) or ()
    counts = tags.get(279) or ()
    if not starts:
        raise OSError("{}: not a strip-based TIFF".format(path))

    chunks = []
    for start, length in zip(starts, counts):
        blob = data[start:start + length]
        if compression == 1:
            chunks.append(blob)
        elif compression in (8, 32946):
            chunks.append(zlib.decompress(blob))
        elif compression == 32773:                       # PackBits
            out = bytearray()
            i = 0
            while i < len(blob):
                n = blob[i]
                i += 1
                if n < 128:
                    out += blob[i:i + n + 1]
                    i += n + 1
                elif n > 128:
                    out += bytes([blob[i]]) * (257 - n)
                    i += 1
            chunks.append(bytes(out))
        else:
            raise OSError("{}: unsupported TIFF compression {}".format(
                path, compression))

    raw = b"".join(chunks)
    dtype = {8: np.uint8, 16: np.uint16, 32: np.uint32}.get(bits)
    if dtype is None:
        raise OSError("{}: unsupported bit depth {}".format(path, bits))
    arr = np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder(end))
    need = width * height * samples
    if arr.size < need:
        raise OSError("{}: short image data ({} of {} values)".format(
            path, arr.size, need))
    arr = arr[:need].reshape(height, width, samples) if samples > 1 \
        else arr[:need].reshape(height, width)
    return np.asarray(arr)


def _to_byte_scale(arr):
    """Put an image on the 0..255 scale whichever reader produced it.

    matplotlib returns PNG as floats in 0..1 while Pillow returns JPEG and
    TIFF as bytes in 0..255, so the same carpet rendered as PNG or TIFF
    reached the code at scales 255 apart, and anything written against one
    of them was wrong for the other.
    """
    a = np.asarray(arr)
    if a.dtype.kind == "f" and a.size and float(np.nanmax(a)) <= 1.0 + 1e-6:
        return (a * 255.0).astype(np.float32)
    return a


def _load_image(path, page: int = 0):
    """Read one page of an image file as an array.

    TIFF goes through a chain of readers because Pillow cannot be relied on for
    it: whichever of Pillow, tifffile, imageio, OpenCV or the built-in reader
    works first is used.
    """
    path = str(path)
    if not (page or _is_tiff(path)):
        try:
            import matplotlib.image as mpimg
            arr = mpimg.imread(path)
        except Exception:
            arr = _pil_read(path, page)
        return _to_byte_scale(arr)

    errors = []
    degenerate = []

    def via_pil():
        return _pil_read(path, page)

    def via_tifffile():
        import tifffile
        arr = tifffile.imread(path)
        return arr[page] if (page and getattr(arr, "ndim", 2) > 2) else arr

    def via_imageio():
        import imageio.v2 as imageio
        if page:
            return imageio.mimread(path)[page]
        return imageio.imread(path)

    def via_cv2():
        import cv2
        if page:
            raise RuntimeError("OpenCV cannot address TIFF pages")
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise RuntimeError("OpenCV returned nothing")
        return arr

    def via_builtin():
        return _tiff_read_raw(path, page)

    # The built-in reader goes first for TIFF.  libtiff on some builds does not
    # fail cleanly on deflate data -- it prints "ZIPDecode: Decoding error at
    # scanline 0" and hands back an all-black frame, which no exception
    # handler can catch.  Decoding the strips ourselves avoids that entirely.
    for name, fn in (("built-in", via_builtin), ("tifffile", via_tifffile),
                     ("pillow", via_pil), ("imageio", via_imageio),
                     ("opencv", via_cv2)):
        try:
            arr = np.asarray(fn())
        except Exception as exc:
            errors.append("{}: {}".format(name, str(exc).splitlines()[0]))
            continue
        if arr.size == 0:
            errors.append("{}: empty image".format(name))
            continue
        if arr.min() == arr.max():
            # A uniform frame means the decoder gave up part way through.
            degenerate.append((name, arr))
            errors.append("{}: decoded to a blank frame".format(name))
            continue
        return _to_byte_scale(arr)

    if degenerate:
        # Everything agrees it is blank, so it probably really is.
        return _to_byte_scale(degenerate[0][1])

    raise OSError("could not read {}{}\n  {}".format(
        os.path.basename(path), " page {}".format(page) if page else "",
        "\n  ".join(errors)))


def display_scale(value_max: float) -> float:
    """Full-scale value for a pixel range, so display works at any bit depth.

    Assuming 8 bits turns a 16-bit TIFF into a white rectangle, which is what
    used to happen.  The ladder covers float 0-1, 8-bit, 12-bit sensors stored
    in 16-bit files, and true 16-bit.

    The tolerance matters: interpolating during a rotation can leave the peak
    at 255.00000000000003, and without it the scale jumped to 4095 and the
    picture went black.
    """
    for full in (1.0, 255.0, 4095.0, 65535.0):
        if value_max <= full * (1.0 + 1e-6):
            return full
    return float(value_max)


# =========================================================================
# MCC -- particle tracking and transport velocity
# =========================================================================
#
# A port of particle_velocity_tracking.py to numpy alone.  The original used
# OpenCV for the bandpass and connected components and pandas for the table;
# this file keeps its "numpy + matplotlib, scipy optional" footprint, so both
# were reimplemented.
#
# The connected-component stage is dropped on purpose: it labelled every blob
# in the frame only to pick the one nearest the seed.  Searching a window
# around the predicted position finds the same particle directly, and is both
# faster and less likely to jump to a neighbour.

def _box_blur(a, radius: int, axis: int):
    """Mean filter along one axis, via a cumulative sum."""
    if radius < 1:
        return a
    k = 2 * radius + 1
    pad = [(0, 0), (0, 0)]
    pad[axis] = (radius, radius)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis)
    # Window sums c[i+k-1] - c[i-1], with c[-1] = 0.  Written into one array
    # by slices; this used to prepend a zero row with np.concatenate and
    # gather with np.take over a range, which copied the cumulative sum
    # twice and indexed it element by element -- a sixth of Auto enhance.
    # The first window is c[k-1] itself, exactly what subtracting the zero
    # row gave.
    n = c.shape[axis]

    def sl(start, stop):
        idx = [slice(None), slice(None)]
        idx[axis] = slice(start, stop)
        return tuple(idx)

    out = np.empty_like(c[sl(k - 1, n)])
    out[sl(0, 1)] = c[sl(k - 1, k)]
    np.subtract(c[sl(k, n)], c[sl(0, n - k)], out=out[sl(1, None)])
    out /= k
    return out


def gaussian_blur(a, sigma: float):
    """Gaussian blur approximated by three box passes (no scipy needed)."""
    a = np.asarray(a, dtype=float)
    if sigma <= 0:
        return a
    radius = max(1, int(round(sigma * 0.797)))
    out = a
    for _ in range(3):
        out = _box_blur(_box_blur(out, radius, 0), radius, 1)
    return out


def particle_bandpass(gray, bg_sigma: float = 15.0, smooth_sigma: float = 1.0):
    """Flatten the illumination so dark particles become positive peaks."""
    g = np.asarray(gray, dtype=float)
    if g.ndim == 3:
        g = g[..., :3].mean(axis=2)
    return np.clip(gaussian_blur(gaussian_blur(g, bg_sigma) - g, smooth_sigma),
                   0.0, None)


def track_particle(video, seed_xy, max_disp: float = 25.0,
                   thresh_frac: float = 0.35, bg_sigma: float = 15.0,
                   smooth_sigma: float = 1.0, refine: int = 4, progress=None):
    """Follow one dark particle through the frames from a seed position.

    Returns an (n, 4) array of ``frame, x, y, peak``; x and y are NaN on frames
    where the particle was not found -- which is the normal way a track ends
    when the particle leaves the field of view.
    """
    seed = np.array(seed_xy, dtype=float)
    cur = seed.copy()
    rows = []
    r = int(round(max_disp))

    total = max(1, video.last - video.first + 1)
    for f in range(video.first, video.last + 1):
        if progress is not None:
            progress((f - video.first) / total, "tracking")
        img = np.asarray(video.get(f), dtype=float)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)
        h, w = img.shape
        bp = particle_bandpass(img, bg_sigma, smooth_sigma)
        thr = bp.max() * thresh_frac

        x0, x1 = max(0, int(cur[0]) - r), min(w, int(cur[0]) + r + 1)
        y0, y1 = max(0, int(cur[1]) - r), min(h, int(cur[1]) + r + 1)
        sub = bp[y0:y1, x0:x1]
        if sub.size == 0:
            rows.append((f, np.nan, np.nan, 0.0))
            continue

        iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
        peak = float(sub[iy, ix])
        if peak < thr:
            rows.append((f, np.nan, np.nan, peak))
            continue

        px, py = x0 + ix, y0 + iy
        a0, a1 = max(0, px - refine), min(w, px + refine + 1)
        b0, b1 = max(0, py - refine), min(h, py + refine + 1)
        patch = bp[b0:b1, a0:a1]
        total = patch.sum()
        if total <= 0:
            rows.append((f, np.nan, np.nan, peak))
            continue
        yy, xx = np.mgrid[b0:b1, a0:a1]
        cx = float((xx * patch).sum() / total)
        cy = float((yy * patch).sum() / total)

        if np.hypot(cx - cur[0], cy - cur[1]) > max_disp:
            rows.append((f, np.nan, np.nan, peak))
            continue

        rows.append((f, cx, cy, peak))
        cur = np.array([cx, cy])

    return np.array(rows, dtype=float)


def particle_velocity(traj, pixel_size: float, fps: float):
    """Velocities from a pixel trajectory.

    ``pixel_size`` is microns per pixel and ``fps`` frames per second, matching
    the reference script's PIXEL_SIZE / FPS.
    """
    ok = ~np.isnan(traj[:, 1])
    if ok.sum() < 2:
        raise ValueError("the particle was found in fewer than 2 frames")

    dt = 1.0 / float(fps)
    frame = traj[ok, 0]
    x_um = traj[ok, 1] * pixel_size
    y_um = traj[ok, 2] * pixel_size
    time_s = frame * dt

    step = np.hypot(np.diff(x_um), np.diff(y_um))
    inst = step / dt
    total_time = (len(x_um) - 1) * dt
    path = float(step.sum())
    net = float(np.hypot(x_um[-1] - x_um[0], y_um[-1] - y_um[0]))

    return {
        "frame": frame, "time_s": time_s, "x_um": x_um, "y_um": y_um,
        "x_px": traj[ok, 1], "y_px": traj[ok, 2],
        "step_um": step, "inst_umps": inst,
        "n_tracked": int(ok.sum()), "n_frames": int(len(traj)),
        "first_frame": int(frame[0]), "last_frame": int(frame[-1]),
        "total_time_s": total_time,
        "total_path_um": path,
        "net_disp_um": net,
        "mean_speed_umps": path / total_time if total_time else float("nan"),
        "net_velocity_umps": net / total_time if total_time else float("nan"),
        "mean_inst_umps": float(inst.mean()),
        "std_inst_umps": float(inst.std()),
        "max_inst_umps": float(inst.max()),
        "min_inst_umps": float(inst.min()),
        "pixel_size": pixel_size, "fps": fps,
    }


def save_particle_csv(path: str, vel: dict) -> str:
    """Trajectory and speeds, columns matching the reference script."""
    lines = ["frame,time_s,x_px,y_px,x_um,y_um,step_dist_um,inst_speed_umps\n"]
    for i in range(len(vel["frame"])):
        step = "" if i == 0 else "{:.6f}".format(vel["step_um"][i - 1])
        inst = "" if i == 0 else "{:.6f}".format(vel["inst_umps"][i - 1])
        lines.append("{:.0f},{:.6f},{:.4f},{:.4f},{:.6f},{:.6f},{},{}\n".format(
            vel["frame"][i], vel["time_s"][i], vel["x_px"][i], vel["y_px"][i],
            vel["x_um"][i], vel["y_um"][i], step, inst))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)
    return path


# Burnt-in graphics -- a logo, a time stamp, a scale bar -- are pixels at
# the black or white end of the scale that are the same in every frame, in
# blobs no bigger than this fraction of the frame (a tissue body clipped
# to black is larger and stays image data).
MARKER_MAX_FRAC = 0.02


def small_blobs(mask, max_frac: float = MARKER_MAX_FRAC):
    """The pixels of ``mask`` lying in 4-connected blobs of at most
    ``max_frac`` of the image's area."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    limit = max(1, int(max_frac * h * w))
    try:                                   # scipy when there is one
        from scipy import ndimage
        labels, n = ndimage.label(mask)
        if n == 0:
            return np.zeros_like(mask)
        areas = np.bincount(labels.ravel())
        areas[0] = 0
        return (areas[labels] > 0) & (areas[labels] <= limit)
    except Exception:
        pass
    out = np.zeros_like(mask)
    seen = np.zeros_like(mask)
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        blob = [(int(y0), int(x0))]
        seen[y0, x0] = True
        i = 0
        while i < len(blob):
            y, x = blob[i]
            i += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    blob.append((ny, nx))
        if len(blob) <= limit:
            for y, x in blob:
                out[y, x] = True
    return out


def rotation_valid(shape, degrees: float):
    """Which pixels of a rotated frame came from inside the original.

    Rotating keeps the frame size, so the corners are filled in.  That filling
    has to be tracked, not just made to look like background: its boundary is a
    long straight edge, and both surface detection and cilium tracing latch
    onto long straight edges.
    """
    h, w = int(shape[0]), int(shape[1])
    if abs(float(degrees)) < 1e-9:
        return np.ones((h, w), dtype=bool)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    t = np.deg2rad(float(degrees))
    ct, st = np.cos(t), np.sin(t)
    sx = ct * (xx - cx) + st * (yy - cy) + cx
    sy = -st * (xx - cx) + ct * (yy - cy) + cy
    return ((sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1))


def rotate_image(img, degrees: float, fill=None):
    """Rotate a frame about its centre, bilinearly, keeping the same size.

    Unlike the contrast settings this is *not* cosmetic: the rotated frames are
    what everything downstream sees.  That is deliberate -- surface detection
    assumes the cell surface runs roughly left-to-right, so straightening a
    tilted recording has to happen before any measurement, not after.

    Corners that fall outside the original are filled with the median level so
    they read as background rather than as a hard edge.
    """
    a = np.asarray(img, dtype=float)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    if abs(float(degrees)) < 1e-9:
        return a

    h, w = a.shape
    if fill is None:
        fill = float(np.median(a))
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]

    t = np.deg2rad(float(degrees))
    ct, st = np.cos(t), np.sin(t)
    # Inverse map: for each output pixel, where did it come from?
    sx = ct * (xx - cx) + st * (yy - cy) + cx
    sy = -st * (xx - cx) + ct * (yy - cy) + cy

    x0 = np.floor(sx).astype(int)
    y0 = np.floor(sy).astype(int)
    fx = sx - x0
    fy = sy - y0
    inside = (x0 >= 0) & (x0 < w - 1) & (y0 >= 0) & (y0 < h - 1)
    x0c = np.clip(x0, 0, w - 2)
    y0c = np.clip(y0, 0, h - 2)

    out = (a[y0c, x0c] * (1 - fx) * (1 - fy)
           + a[y0c, x0c + 1] * fx * (1 - fy)
           + a[y0c + 1, x0c] * (1 - fx) * fy
           + a[y0c + 1, x0c + 1] * fx * fy)
    # Interpolation cannot exceed the input range mathematically, but rounding
    # can, and downstream code keys off the maximum.
    out = np.clip(out, a.min(), a.max())
    return np.where(inside, out, fill)


def rotate_point(point, degrees: float, width: int, height: int):
    """Where a point in the unrotated frame ends up after ``rotate_image``.

    Needed when a fit is restricted to the neighbourhood of a fixed feature:
    the window has to follow the anatomy as the frame turns, not stay at the
    same pixel, or each iteration looks at a different piece of the image.
    """
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    t = np.deg2rad(float(degrees))
    ct, st = np.cos(t), np.sin(t)
    dx, dy = float(point[0]) - cx, float(point[1]) - cy
    return (cx + ct * dx - st * dy, cy + st * dx + ct * dy)


def _blur5(a):
    """Separable 5-tap binomial blur -- no scipy needed."""
    k = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
    k /= k.sum()
    pad = 2
    b = np.pad(a, ((0, 0), (pad, pad)), mode="edge")
    out = sum(k[i] * b[:, i:i + a.shape[1]] for i in range(5))
    b = np.pad(out, ((pad, pad), (0, 0)), mode="edge")
    return sum(k[i] * b[i:i + a.shape[0], :] for i in range(5))


def apply_contrast(img, bytescale: bool = False, gamma: float = 1.0,
                   sharpness: float = 1.0, brightness: float = 1.0):
    """Contrast adjustments for *displaying* a frame.

    Purely cosmetic: every measurement in this file reads the raw frames, so
    changing these never alters a fit, an estimate or an export.

    bytescale   stretch this frame's own levels to the full range -- its
                0.5th to 99.5th percentiles, so that a few extreme pixels
                (the rim of a background-removed frame) do not set the range
    gamma       output = input ** (1/gamma); above 1 lifts the mid-tones
    sharpness   unsharp mask; 1.0 is off
    brightness  simple multiplier
    """
    a = np.asarray(img, dtype=float)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)

    lo, hi = float(a.min()), float(a.max())
    if bytescale:
        p_lo, p_hi = np.percentile(a, [0.5, 99.5])
        if p_hi > p_lo:
            lo, hi = float(p_lo), float(p_hi)
        a = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
    else:
        # Keep the raw levels, mapped into 0..1 only for display, at whatever
        # bit depth the file actually uses.
        a = a / display_scale(hi)

    if brightness != 1.0:
        a = a * float(brightness)
    if gamma != 1.0 and gamma > 0:
        a = np.clip(a, 0.0, None) ** (1.0 / float(gamma))
    if sharpness != 1.0:
        a = a + (float(sharpness) - 1.0) * (a - _blur5(a))

    return np.clip(a, 0.0, 1.0)


REQUIRED_PACKAGES = (
    # import name, pip name, what it is for
    ("numpy", "numpy", "everything"),
    ("matplotlib", "matplotlib", "the window and PNG frames"),
    ("PIL", "pillow", "JPEG and TIFF frames"),
)


def missing_packages():
    """The required packages that will not import, as (pip name, purpose)."""
    out = []
    for module, pip_name, purpose in REQUIRED_PACKAGES:
        try:
            __import__(module)
        except ImportError:
            out.append((pip_name, purpose))
    return out


def dependency_hint(errors=None) -> str:
    """An install line for whatever is missing, or '' if nothing is.

    ``errors`` are messages already collected; a 'No module named PIL' among
    them is the tell for a missing Pillow even when the import itself was
    swallowed on the way up.
    """
    missing = missing_packages()
    if not missing and errors and any("PIL" in str(e) for e in errors):
        missing = [("pillow", "JPEG and TIFF frames")]
    if not missing:
        return ""
    return ("\n  {} not installed ({}); run:\n"
            "  python3 -m pip install {}".format(
                " and ".join(p for p, _ in missing),
                "; ".join("{} for {}".format(p, why) for p, why in missing),
                " ".join(p for p, _ in missing)))


def image_size(path):
    """(width, height, dpi_or_None) of an image, without decoding it fully."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
            dpi = im.info.get("dpi")
            xdpi = float(dpi[0]) if dpi else None
            # A zero or 1 dpi tag means "not set" rather than a real scale.
            if xdpi is not None and xdpi <= 1.0:
                xdpi = None
            return w, h, xdpi
    except Exception:
        pass
    arr = _load_image(path)
    return arr.shape[1], arr.shape[0], None


def detect_basename(names) -> str:
    """The common non-numeric filename prefix, e.g. 'im' for im0000.jpg."""
    import re
    from collections import Counter
    prefixes = [re.match(r"^([^0-9]*)", n).group(1) for n in names]
    if not prefixes:
        return ""
    return Counter(prefixes).most_common(1)[0][0]


def sequence_name(directory: str, names=None) -> str:
    """A name for the recording, for the exported files: the frame files'
    common stem without the frame counter (CO-121-01_a_07_300fps_37C_40xBFw
    from CO-121-01_a_07_300fps_37C_40xBFw-12082025164822-150.jpeg), or the
    folder's name when the files are called im0000.jpg and the like."""
    try:
        if names is None:
            names = sorted(n for n in os.listdir(directory)
                           if _is_image(os.path.join(directory, n)))
        stems = [os.path.splitext(n)[0] for n in names]
        stem = os.path.commonprefix(stems) if stems else ""
        stem = stem.rstrip("0123456789").rstrip("-_ .")
    except Exception:
        stem = ""
    if len(stem) < 4:
        stem = os.path.basename(os.path.normpath(os.path.abspath(directory)))
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in stem) or "cilium"


def scan_image_folder(directory: str):
    """Inspect a folder of frames.

    Returns a dict describing the sequence, or None if it holds no images.
    Note that pixels-per-micron cannot be recovered from the pixels alone -- it
    needs a scale bar or a known calibration.  If the files carry a physical
    resolution tag we convert it and offer it as a suggestion only.
    """
    if not os.path.isdir(directory):
        return None

    names = sorted((n for n in os.listdir(directory)
                    if _is_image(os.path.join(directory, n))), key=natural_key)
    if not names:
        return None

    # A multi-page TIFF contributes one frame per page, not one per file.
    pages = [image_pages(os.path.join(directory, n)) if _is_tiff(n) else 1
             for n in names]
    n_frames = int(sum(pages))

    # Check a sample of the frames for a consistent size, skipping any that
    # cannot be read -- one bad file should not stop the folder being scanned.
    sizes = set()
    unreadable = []
    w = h = dpi = None
    # Sample across the whole sequence, not just the start: a bad patch at one
    # end should not decide what the folder looks like.
    step = max(1, len(names) // 60)
    for n in names[::step][:60]:
        try:
            sw, sh, sdpi = image_size(os.path.join(directory, n))
        except Exception as exc:
            unreadable.append((n, str(exc).splitlines()[0]))
            continue
        sizes.add((sw, sh))
        if w is None:
            w, h, dpi = sw, sh, sdpi
    if w is None:
        # Say why, not just that.  The commonest reason is a machine
        # without Pillow: matplotlib decodes PNG on its own but needs Pillow
        # for JPEG and TIFF, so on a fresh Linux install every .jpg frame
        # "could not be read" and the ImportError underneath never showed.
        reasons = [r for _, r in unreadable]
        hint = dependency_hint(reasons)
        raise ValueError(
            "none of the images in {} could be read{}\n  first error: {}\n"
            "  try:  python ciliograph.py check --images {}".format(
                directory, hint, reasons[0] if reasons else "?", directory))

    return {
        "directory": directory,
        "names": names,
        "count": n_frames,
        "n_files": len(names),
        "pages": pages,
        "multipage": n_frames != len(names),
        "basename": detect_basename(names),
        "width": w,
        "height": h,
        "dpi": dpi,
        # 1 inch = 25400 um, so pixels/inch -> pixels/um.
        "ppum_from_dpi": (dpi / 25400.0) if dpi else None,
        "uniform": len(sizes) == 1,
        "sizes": sorted(sizes),
        "unreadable": unreadable,
    }


def motion_and_mean(video, first: int | None = None, last: int | None = None,
                    progress=None):
    """(temporal std, temporal mean) of the frames, as greyscale float images.

    A beating cilium sweeps through a region of the image, so the per-pixel
    standard deviation over time lights it up strongly against a static
    background.  That "motion map" is a far cleaner signal to locate the
    cilium with than any single frame.
    """
    first = video.first if first is None else max(first, video.first)
    last = video.last if last is None else min(last, video.last)

    acc = None
    acc2 = None
    n = 0
    total = max(1, int(last) - int(first) + 1)
    _prefetch(video, range(int(first), int(last) + 1))
    for f in range(int(first), int(last) + 1):
        if progress is not None:
            progress((f - int(first)) / total, "reading frames")
        img = np.asarray(video.get(f), dtype=float)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)
        acc = img if acc is None else acc + img
        acc2 = img ** 2 if acc2 is None else acc2 + img ** 2
        n += 1
    mean = acc / n
    var = np.maximum(acc2 / n - mean ** 2, 0.0)
    return np.sqrt(var), mean


def static_mean(video, mean_img):
    """The image to find the surface in: the raw temporal mean.

    The tissue edge is static structure, and that is exactly what
    background removal and illumination flattening take out of the frames
    the measurements read.  With them on, the mean of the enhanced frames is
    a featureless grey, and the surface -- hence the base, the mask above
    it, and the length measured from it -- went astray: one recording's
    length fell from 5.7 to 2.3 microns the moment the background was
    removed.  So wherever a surface is wanted, the mean of the frames before
    enhancement is used, brought to the same size as the frames after it.
    """
    enh = getattr(video, "enhance", None) or {}
    if not (enh.get("background") or enh.get("flatten") or enh.get("invert")
            or enh.get("contrast") or enh.get("deblur") or enh.get("bandpass")
            or enh.get("tophat") or enh.get("lines")
            or clean_model_path(enh.get("model"))):
        # (A network that draws the cilia alone -- train_cilia_net.py --
        # leaves no tissue to find the surface in: its dense band of bases
        # was taken for the tissue and half of every cilium discarded.  The
        # line filter and the top-hat draw the same way.)
        return mean_img
    try:
        raw = video.background()
    except Exception:
        return mean_img
    if enh.get("invert"):
        raw = 255.0 - raw
    up = int(enh.get("upsample") or 1)
    if up > 1:
        raw = resample_frame(raw, float(up))
    if raw.shape != np.asarray(mean_img).shape:
        return mean_img
    return raw


def detect_surface(mean_img, frac: float = 0.45, smooth: int = 15,
                   valid=None):
    """Row of the cell surface in each column: the top of the dark tissue.

    Columns with no tissue at all are reported as the image height, which is
    outside the frame, so masks built from this exclude them: a column of open
    medium has no cell surface and therefore no cilium bases.  They are also
    kept out of the smoothing, because letting a bogus 480 bleed into its
    neighbours pulled the surface down across the edge of a tissue that filled
    only half the field.

    ``valid`` excludes pixels that are not real image data -- the corners that
    rotating a frame leaves empty.  Without it the padding boundary reads as a
    long straight dark edge, which is exactly what surface detection is looking
    for, and after levelling a strongly tilted recording the "surface" found
    was the edge of the padding rather than the cell.
    """
    h, w = mean_img.shape
    if valid is None:
        usable = np.ones((h, w), dtype=bool)
    else:
        usable = np.asarray(valid, dtype=bool)
    inside = mean_img[usable]
    if inside.size == 0:
        return np.full(w, float(h))
    lo, hi = float(inside.min()), float(inside.max())
    dark = (mean_img < (lo + frac * (hi - lo))) & usable

    rows = np.full(w, np.nan)
    for c in range(w):
        if dark[:, c].any():
            rows[c] = float(np.argmax(dark[:, c]))

    # "Top of the dark tissue" presumes there is tissue.  A drawing or a
    # simulation may show cilia alone on a plain background, and then the top
    # of the dark is the cilium *tips*: the mask built from it keeps only the
    # empty space beyond them and nothing can be traced at all.  A real
    # recording has a solid dark body under the line -- 21% of the pixels below
    # it are dark on the recordings here -- while cilia alone leave 6%.  Where
    # there is no body, the cilia are anchored at the *bottom* of the drawn
    # structure instead.
    have = ~np.isnan(rows)
    if have.any():
        below = [dark[int(rows[c]) + 1:, c].mean() for c in np.nonzero(have)[0]
                 if rows[c] < h - 2]
        if below and float(np.mean(below)) < 0.12:
            for c in np.nonzero(have)[0]:
                rows[c] = float(h - 1 - np.argmax(dark[::-1, c]))

    # Bridge short gaps.  A sparse structure -- a row of separate cilia rather
    # than a continuous tissue edge -- leaves the columns between them with no
    # dark pixel at all, and reporting those as "outside the frame" turned the
    # anchor line into a comb oscillating by 40 pixels between one cilium and
    # the next.  Columns inside a short gap are interpolated from their
    # neighbours; a long run with nothing in it is genuinely empty and stays
    # unknown.
    have = ~np.isnan(rows)
    if have.any() and not have.all():
        gap_limit = max(8, int(0.05 * w))
        idx = np.arange(w)
        filled = np.interp(idx, idx[have], rows[have])
        missing = np.concatenate(([False], ~have, [False])).astype(np.int8)
        change = np.diff(missing)
        for a, b in zip(np.nonzero(change == 1)[0], np.nonzero(change == -1)[0]):
            if 0 < a and b < w and (b - a) <= gap_limit:
                rows[a:b] = filled[a:b]

    if smooth > 1:
        k = np.ones(int(smooth))
        vals = np.where(np.isnan(rows), 0.0, rows)
        wts = (~np.isnan(rows)).astype(float)
        pad = int(smooth) // 2
        num = np.convolve(np.pad(vals, pad, mode="edge"), k, "valid")[:w]
        den = np.convolve(np.pad(wts, pad, mode="edge"), k, "valid")[:w]
        smoothed = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
        rows = np.where(np.isnan(rows), np.nan, smoothed)

    return np.where(np.isnan(rows), float(h), rows)


def surface_slope_at(surface, x0: float, h: float, span: int = 40):
    """Local slope of the cell surface at a column, in rows per column.

    Fitted over a window and only from columns that actually have tissue, so a
    field the cell fills only half of does not drag the answer.
    """
    surface = np.asarray(surface, dtype=float)
    w = len(surface)
    lo = int(max(0, round(x0) - span))
    hi = int(min(w, round(x0) + span + 1))
    xs = np.arange(lo, hi, dtype=float)
    ys = surface[lo:hi]
    good = (ys < h - 1) & (ys > 0)
    if good.sum() < 5:
        return 0.0, 0.0
    slope, intercept = np.polyfit(xs[good], ys[good], 1)
    resid = ys[good] - (slope * xs[good] + intercept)
    return float(slope), float(np.sqrt(np.mean(resid ** 2)))


def base_tangent_angle(model, k: int = 1) -> float:
    """Direction the model's base axis -- the z axis of the beat's frame,
    the surface normal the cilium is anchored along -- points on the image,
    in radians."""
    px, py = model.to_pixels(np.array([0.0, 0.0]), np.array([0.0, 1.0]))
    return float(np.arctan2(py[1] - py[0], px[1] - px[0]))


def orientation_for_direction(model, target: float) -> float:
    """The vid_cil_orientation, in degrees, that aims the base along ``target``.

    Solved by measuring rather than by reasoning about the sign convention:
    the mapping from orientation to base direction is a rotation, so two
    evaluations pin it down exactly.
    """
    pqd = model.pqd
    keep = pqd.getf("vid_cil_orientation", 0.0)
    try:
        pqd.set("vid_cil_orientation", 0.0)
        model.rebuild()
        a0 = base_tangent_angle(model)
        pqd.set("vid_cil_orientation", 10.0)
        model.rebuild()
        a1 = base_tangent_angle(model)
    finally:
        pqd.set("vid_cil_orientation", keep)
        model.rebuild()
    per_deg = np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0)) / 10.0
    if abs(per_deg) < 1e-6:
        return float(keep)
    delta = np.arctan2(np.sin(target - a0), np.cos(target - a0))
    return float(((delta / per_deg) + 180.0) % 360.0 - 180.0)


def base_orientation(video, model, span: int = 40, progress=None):
    """The orientation that points the cilium out of the surface at its base.

    A cilium is anchored perpendicular to the epithelium it grows from, so on a
    curved or obliquely viewed carpet its base direction is not "up" -- on one
    recording here the tissue edge runs diagonally across the frame at about
    50 degrees, and a model that insists on leaving the base vertically cannot
    represent any of it.  Levelling the whole image is one answer, but it fills
    the corners with padding and moves the problem rather than removing it;
    setting the orientation is exact and costs nothing.
    """
    _, mean_img = motion_and_mean(video, progress=progress)
    h, w = mean_img.shape
    # The same surface the tracer uses: on a drawn or segmented carpet the
    # wall line.  detect_surface on a dense drawn lawn found the top of the
    # layer of lying cilia on one side of the base and the frame's bottom
    # on the other, and the slope between them tilted the beat by 24 deg.
    if frames_drawn_source(video):
        surface = wall_line(static_mean(video, mean_img), valid=video.valid)
    else:
        surface = detect_surface(static_mean(video, mean_img), valid=video.valid)
    x0 = model.pqd.getf("vid_cil_x0", w / 2.0)
    slope, scatter = surface_slope_at(surface, x0, h, span=span)

    # Outward normal of a surface running (1, slope) is (slope, -1): away from
    # the tissue, which lies at larger row numbers.
    target = float(np.arctan2(-1.0, slope))
    orientation = orientation_for_direction(model, target)
    return {
        "orientation": orientation,
        "current": model.pqd.getf("vid_cil_orientation", 0.0),
        "slope": slope,
        "surface_tilt_deg": float(np.rad2deg(np.arctan(slope))),
        "normal_deg": float(np.rad2deg(target)),
        "scatter_px": scatter,
        "surface_row": float(surface[int(np.clip(round(x0), 0, w - 1))]),
        "has_surface": bool(surface[int(np.clip(round(x0), 0, w - 1))] < h - 1),
    }


def measure_surface_tilt(video, margin: int = 20, trim: float = 0.0):
    """Tilt of the cell surface the cilia grow from, in degrees.

    Positive means the surface runs downhill to the right in image
    coordinates.  Columns where no surface was found are dropped.

    ``trim`` discards the worst-fitting fraction before refitting.  It defaults
    to off: on a real epithelium the surface is genuinely slightly wavy, and
    trimming threw away good data -- it moved the estimate from -4.17 to -3.57
    degrees and left 0.59 degrees behind after correcting, against 0.02 with
    the untrimmed fit.
    """
    _, mean_img = motion_and_mean(video)
    h, w = mean_img.shape
    surface = detect_surface(static_mean(video, mean_img), valid=video.valid)

    x = np.arange(w, dtype=float)
    good = (surface < h - 1) & (surface > 0)
    if margin > 0:
        good[:margin] = False
        good[w - margin:] = False
    if good.sum() < 10:
        raise ValueError("could not find the cell surface in these images")

    xs, ys = x[good], surface[good]
    slope, intercept = np.polyfit(xs, ys, 1)
    if 0.0 < trim < 0.5:                       # drop the worst residuals once
        resid = np.abs(ys - (slope * xs + intercept))
        keep = resid <= np.quantile(resid, 1.0 - trim)
        if keep.sum() >= 10:
            slope, intercept = np.polyfit(xs[keep], ys[keep], 1)

    # How line-like the detected surface actually is.  On a flat epithelium
    # this is a couple of pixels; on a rounded cell, where there is no
    # horizontal surface to find, it is tens of pixels and the fitted "tilt"
    # is meaningless.
    fit_rms = float(np.std(ys - (slope * xs + intercept)))

    return {
        "tilt_deg": float(np.degrees(np.arctan(slope))),
        "slope": float(slope),
        "n_columns": int(good.sum()),
        "fit_rms_px": fit_rms,
        "reliable": bool(fit_rms <= max(8.0, 0.04 * h)),
        "surface": surface,
        "mean": mean_img,
    }


def measure_band_angle(video, percentile: float = 98.0, progress=None,
                       around=None, radius_px: float | None = None,
                       iters: int = 3, keep: float = 0.8):
    """Orientation of the band the cilia sweep out, in degrees.

    Used where there is no flat surface to fit -- a cell seen side-on, say.
    The beating cilia light up a stripe in the motion map and its principal
    axis gives the direction the bases run along.

    Two refinements matter in practice.  Stray debris drifting elsewhere in the
    frame drags a plain principal axis badly off (it read -25 deg against a
    band clearly lying at -42), so points far from the current line are
    dropped and the fit repeated.  And because the band is usually an arc
    rather than a line, ``around`` restricts the fit to the neighbourhood of a
    point -- the cilium base -- so the answer is the direction *there* rather
    than an average over the whole curve.
    """
    motion, mean_img = motion_and_mean(video, progress=progress)
    h, w = motion.shape
    threshold = np.percentile(motion, percentile)
    ys, xs = np.where(motion >= threshold)
    if len(xs) < 20:
        raise ValueError("not enough movement to find a direction")
    weight = motion[ys, xs]

    if around is not None:
        radius = radius_px or 0.25 * max(h, w)
        near = np.hypot(xs - around[0], ys - around[1]) <= radius
        if near.sum() >= 40:
            xs, ys, weight = xs[near], ys[near], weight[near]

    major = np.array([1.0, 0.0])
    values = np.array([1.0, 1.0])
    cx = cy = 0.0
    spread = float("nan")
    for _ in range(max(1, iters)):
        cx = float((xs * weight).sum() / weight.sum())
        cy = float((ys * weight).sum() / weight.sum())
        centred = np.stack([xs - cx, ys - cy])
        cov = (centred * weight) @ centred.T / weight.sum()
        values, vectors = np.linalg.eigh(cov)
        major = vectors[:, int(np.argmax(values))]
        normal = np.array([-major[1], major[0]])
        offset = np.abs(normal @ centred)
        spread = float(np.sqrt((offset ** 2).mean()))
        cut = np.quantile(offset, keep)
        mask = offset <= cut
        if mask.sum() < 60:
            break
        xs, ys, weight = xs[mask], ys[mask], weight[mask]

    angle = np.degrees(np.arctan2(major[1], major[0]))
    angle = ((angle + 90.0) % 180.0) - 90.0
    elongation = float(np.sqrt(max(values) / max(min(values), 1e-9)))
    return {
        "tilt_deg": float(angle),
        "elongation": elongation,
        "reliable": bool(elongation >= 1.6),
        "centre": (cx, cy),
        "n_columns": int(len(xs)),
        "band_spread_px": spread,
        "fit_rms_px": spread,
        "surface": None,
        "mean": mean_img,
    }


def level_rotation(video, pqd, iterations: int = 4, tol: float = 0.05,
                   progress=None, **kwargs):
    """The image rotation that makes the cilia base line horizontal.

    Rotating by theta adds roughly theta to the measured tilt, but only
    roughly: rotating moves the surface within the frame and fills the corners,
    so a single correction leaves over a degree behind when starting from a
    large angle.  This measures, applies, and re-measures until it settles.
    """
    start = float(video.rotation)
    current = float(pqd.getf("vid_image_rotation", 0.0))
    angle = current
    first = None
    steps = 0
    # Prefer the cell surface; where there is none, fall back to the direction
    # of the band the cilia sweep out.
    method = "surface"
    # Level the band where the cilium actually is: the arc's direction varies
    # along it, so a global fit answers a question nobody asked.
    h_img = int(pqd.getf("vid_image_h", 0) or 0)
    around = None
    if h_img:
        around = (pqd.getf("vid_cil_x0", 0.0),
                  h_img - pqd.getf("vid_cil_y0_invy", 0.0))

    probe = measure_surface_tilt(video, **kwargs)
    if not probe["reliable"]:
        try:
            band = measure_band_angle(video, around=around)
        except ValueError:
            band = None
        if band is not None and band["reliable"]:
            method = "cilia band"
        else:
            out = dict(probe)
            out.update(current_rotation=current, rotation=current,
                       residual_deg=probe["tilt_deg"], iterations=1,
                       method="none")
            return out

    shape = video.shape          # (h, w)

    def tissue_sides(angle: float):
        """Texture well above and well below the beating band, at a rotation.

        Levelling only fixes the *line*; which side of it the cell lies on is
        left over, and the rest of the program assumes fluid above and tissue
        below -- surface detection looks for the top of the dark tissue.  Get
        that backwards and the "surface" found is the edge of the rotation
        padding, the masks erase almost the whole frame, and nothing can be
        traced at all.  One recording here levels to put the cell on top, so
        the choice has to be made rather than assumed.

        The cell is the textured side: it is full of structure, while the
        medium is smooth.  Brightness will not do -- these recordings disagree
        about whether the cell is darker or brighter than the medium -- and
        neither will motion, because the beating band sits near the middle of
        the frame rather than at one end.
        """
        keep = video.rotation
        try:
            video.set_rotation(angle)
            motion, mean_img = motion_and_mean(video)
            usable = video.valid
        finally:
            video.set_rotation(keep)
        h = mean_img.shape[0]
        rows = np.where(usable, motion, 0.0).sum(axis=1)
        band = int(np.argmax(np.convolve(rows, np.ones(5) / 5.0, "same")))
        rough = np.abs(mean_img - gaussian_blur(mean_img, 3.0))
        pad = max(4, h // 12)
        out = []
        for sl in (slice(0, max(1, band - pad)), slice(min(h - 1, band + pad), h)):
            mask = usable[sl]
            out.append(float(rough[sl][mask].mean()) if mask.any() else 0.0)
        return out

    def orient_tissue_down(angle: float) -> float:
        above, below = tissue_sides(angle)
        # Only a decisive difference may flip the picture; on a low-texture
        # recording both sides read the same and the measured angle stands.
        if above > 1.25 * max(below, 1e-9):
            flipped = ((angle + 180.0) + 180.0) % 360.0 - 180.0
            f_above, f_below = tissue_sides(flipped)
            if f_below > 1.25 * max(f_above, 1e-9):
                return flipped
        return angle

    if method == "cilia band":
        # A direction transforms exactly under rotation, so one measurement is
        # enough -- no need to re-read and re-rotate every frame per iteration.
        angle = ((current - band["tilt_deg"] + 180.0) % 360.0) - 180.0
        angle = orient_tissue_down(angle)
        try:
            video.set_rotation(angle)
            here = rotate_point(around, angle - start, shape[1], shape[0]) \
                if around is not None else None
            final = measure_band_angle(video, around=here)
        finally:
            video.set_rotation(start)
        out = dict(band)
        out.update(current_rotation=current, rotation=float(angle),
                   residual_deg=float(final["tilt_deg"]), iterations=1,
                   method=method)
        return out

    def measure():
        return measure_surface_tilt(video, **kwargs)

    try:
        for it in range(max(1, iterations)):
            if progress is not None:
                progress(it / max(1, iterations),
                         "levelling, pass {}".format(it + 1))
            info = measure()
            if first is None:
                first = info
            steps += 1
            if abs(info["tilt_deg"]) <= tol:
                break
            angle = ((angle - info["tilt_deg"] + 180.0) % 360.0) - 180.0
            video.set_rotation(angle)
        angle = orient_tissue_down(angle)
        video.set_rotation(angle)
        final = measure()
    finally:
        video.set_rotation(start)

    out = dict(first)
    out["current_rotation"] = current
    out["rotation"] = float(angle)
    out["residual_deg"] = float(final["tilt_deg"])
    out["iterations"] = steps
    out["method"] = method
    return out


def locate_base(video, model, n_cs: int = 20, step: int = 2,
                mask_pad: int = 8, margin: int = 25, sub: int = 8,
                beat_weight: float = 0.8, snap_tol: int = 3, progress=None):
    """Find where the model cilium best explains the observed motion.

    The model's swept region -- the union of its shapes over one beat -- is
    correlated against the motion map for every candidate base position.  A
    plain "how much motion does the model pass through" score is far too flat
    (the model can slide anywhere inside the bright blob); correlating the
    swept *shape* penalises both sticking out into still areas and leaving
    moving areas uncovered, which pins the position down much better.

    Returns a dict with the estimate, the score map and a confidence measure.
    """
    motion, mean_img = motion_and_mean(video, progress=progress)
    h, w = motion.shape
    # On a drawn carpet there is no cell body: detect_surface finds the tip
    # envelope, and the base was snapped to it seven pixels above the wall.
    surface = (wall_line(static_mean(video, mean_img)) if frames_drawn_source(video)
               else detect_surface(static_mean(video, mean_img), valid=video.valid))

    # Ignore the tissue and its edge: that boundary shimmers slightly and
    # would otherwise drag the estimate downwards.
    M = motion.copy()
    rows = np.arange(h)[:, None]
    M[rows > (surface[None, :] - mask_pad)] = 0.0
    Mc = M - M.mean()

    # The swept region, as pixel offsets from the base.
    ppum = model.ppum
    offs = set()
    for i in range(n_cs):
        x, y = model.shape_at(i / n_cs)
        px = np.round(x[::sub] * ppum).astype(int)
        py = np.round(-y[::sub] * ppum).astype(int)
        offs.update(zip(px.tolist(), py.tolist()))
    off = np.array(sorted(offs), dtype=np.int32)
    dx, dy = off[:, 0], off[:, 1]

    # The search compares every candidate position against every pixel of the
    # swept region at once, so the work is the product of the two.  On a finely
    # sampled recording that product reached 2.7e9 entries and the process was
    # killed for memory, so the grid is coarsened until it fits.  Coarsening
    # costs only the precision of the starting guess, which the fit refines
    # anyway.
    budget = 2e8
    while (len(np.arange(margin, w - margin, step))
           * len(np.arange(margin, h - 1, step)) * len(off) > budget):
        step += 1

    xs = np.arange(margin, w - margin, step, dtype=np.int32)
    # Rows run to the bottom edge: on a 91-row rendering the wall was at
    # row 85, the grid stopped at 81, and no candidate could be snapped to
    # the surface.
    ys = np.arange(margin, h - 1, step, dtype=np.int32)
    gx, gy = np.meshgrid(xs, ys)
    cand = np.column_stack([gx.ravel(), gy.ravel()])

    cols = cand[:, 0][:, None] + dx[None, :]
    rws = cand[:, 1][:, None] + dy[None, :]
    ok = (cols >= 0) & (cols < w) & (rws >= 0) & (rws < h)
    np.clip(cols, 0, w - 1, out=cols)
    np.clip(rws, 0, h - 1, out=rws)

    # The whole beat has to be in the picture.  Rotating a frame to level
    # the surface leaves empty corners, and a base near that boundary --
    # or near the frame edge -- has part of its sweep cut off; on one
    # recording the best-scoring base sat right at the padding edge, the
    # cilium's rightward sweep left the frame, and the traces there were
    # unusable.  A candidate whose swept region is more than a tenth
    # outside the real image data is dropped.
    try:
        valid_mask = np.asarray(video.valid, dtype=bool)
        if valid_mask.shape != (h, w):
            valid_mask = None
    except Exception:
        valid_mask = None
    # Tested with the reach of the cilium, not the current template's sweep:
    # the template is whatever the last fit left and may lean one way; a
    # cilium can swing anywhere within its length of the base.
    reach = max(3.0, 0.9 * model.length * ppum)
    ang = np.linspace(0.0, np.pi, 25)
    rad = np.linspace(0.3, 1.0, 6)
    disc = np.array([(r * reach * np.cos(a), -r * reach * np.sin(a))
                     for r in rad for a in ang])
    dcols = cand[:, 0][:, None] + np.round(disc[:, 0]).astype(int)[None, :]
    drows = cand[:, 1][:, None] + np.round(disc[:, 1]).astype(int)[None, :]
    din = (dcols >= 0) & (dcols < w) & (drows >= 0) & (drows < h)
    if valid_mask is not None:
        din &= valid_mask[np.clip(drows, 0, h - 1), np.clip(dcols, 0, w - 1)]
    coverage = din.mean(axis=1)

    vals = Mc[rws, cols] * ok
    counts = ok.sum(axis=1)
    # Normalised by sqrt(count): the swept mask has constant norm apart from
    # clipping at the image edge, so this keeps edge positions from winning.
    scores = np.where(counts > 0, vals.sum(axis=1) / np.sqrt(counts), -np.inf)
    scores = np.where(coverage >= 0.9, scores, -np.inf)
    if not np.isfinite(scores).any():
        # A frame smaller than the sweep: fall back to the plain scores.
        scores = np.where(counts > 0, vals.sum(axis=1) / np.sqrt(counts),
                          -np.inf)

    S = scores.reshape(len(ys), len(xs))

    # A cilium is anchored in the cell, so its base is on the cell surface.
    # Searching a free two-dimensional grid ignored that, and on a thick
    # beating band the best match to the swept region put the base in the
    # middle of the moving fluid -- 34 pixels above the surface on one
    # recording, 58% of a cilium length out into open water.  Every trace then
    # starts in mid-fluid and no waveform fitted from them can be right.
    allowed = np.zeros_like(S, dtype=bool)
    tol = max(snap_tol, step)
    for j, x in enumerate(xs):
        row = surface[int(np.clip(x, 0, w - 1))]
        if row < h - 1:
            allowed[np.abs(ys - row) <= tol, j] = True
    if allowed.any():
        S = np.where(allowed, S, -np.inf)

    # Prefer a column where the beat actually repeats.  Matching the swept
    # region to the motion map says only "something moves here"; a waveform
    # also needs the motion to come back to the same place every beat, and
    # those are not the same column.  On one recording the plain match chose a
    # base that repeats 0.29 while 0.59 was available a couple of hundred
    # pixels away, and a waveform fitted at the weaker spot is poorly
    # constrained however well it is fitted.  Measured across six candidate
    # columns of one recording, the repeat predicted the fit residual well:
    # 0.60 gave 0.13 um and 0.33 gave 0.59 um.  It is weighted heavily for a
    # second reason too -- the swept-region match uses the *current* waveform
    # as its template, so it moves when the waveform is refitted and pressing
    # Find base twice gave two different answers; the repeat does not depend on
    # the model at all, so leaning on it makes the result repeatable.
    beat = None
    if beat_weight > 0:
        try:
            beat = beat_strength(video)
        except Exception:
            beat = None
    if beat is not None and np.any(beat["corr"] > 0):
        prior = np.interp(xs, beat["columns"], beat["corr"])
        top = float(np.max(prior))
        if top > 0:
            prior = np.clip(prior / top, 0.0, 1.0)
            offset = S.max() - S[np.isfinite(S)].min()
            S = S - beat_weight * offset * (1.0 - prior)[None, :]

    i, j = np.unravel_index(np.argmax(S), S.shape)
    bx, by = int(xs[j]), int(ys[i])
    # The grid is coarse, so the winning row can sit a few pixels off the
    # surface; the base itself belongs exactly on it.  On a drawn carpet the
    # grid left the base three to six pixels above the wall, which on a 40
    # px/um rendering is a tenth of a micron every trace starts from.
    surf_row = surface[int(np.clip(bx, 0, w - 1))]
    if np.isfinite(surf_row) and 0 <= surf_row < h - 1:
        by = int(round(surf_row))

    finite = S[np.isfinite(S)]
    near = S >= (S.max() - 0.1 * (S.max() - finite.min()))
    return {
        "x0": bx,
        "y0_row": by,
        "y0_invy": int(h - by),
        "score": float(S.max()),
        "score_map": S,
        "xs": xs, "ys": ys,
        "motion": motion, "mean": mean_img, "surface": surface,
        "surface_row": float(surface[min(max(bx, 0), w - 1)]),
        "spread_x": int(near.any(axis=0).sum() * step),
        "spread_y": int(near.any(axis=1).sum() * step),
        "image_h": h, "image_w": w,
    }


# Mismatch of the best lag, on a scale where 0 is a perfect repeat and 2 is
# unrelated frames (correlation 1 - mismatch/2).  Above this the window
# around the base is not repeating clearly enough to be trusted over the
# whole field.
LOCAL_PERIOD_MAX_MISMATCH = 1.7


def _lag_dissimilarity(F, norm: float):
    """The mismatch of a frame stack with itself shifted by each lag tau.

    Returns ``dissimilarity(taus) -> array``: for each (fractional) lag, the
    mean squared difference between every frame and the frame tau later,
    the later one interpolated linearly between neighbours, over ``norm``.
    Every lag is a separate pass over the whole stack, so the lags are
    computed several at a time -- on the GPU when one is in use, otherwise
    in threads, as many as the memory budget allows (numpy does this
    arithmetic outside the interpreter lock, and each lag is computed
    exactly as it would be alone).
    """
    n = len(F)

    def one(t, xp=np, G=F):
        f0 = int(np.floor(t))
        if f0 + 1 >= n:
            return np.inf
        frac = t - f0
        if frac == 0:
            a, b = G[:n - f0], G[f0:]
        else:
            b = (1 - frac) * G[f0:n - 1] + frac * G[f0 + 1:n]
            a = G[:len(b)]
        return float(xp.mean((a - b) ** 2)) / norm if len(a) else np.inf

    state = {"gpu": None}

    def dissimilarity(taus):
        taus = list(taus)
        cp = _gpu()
        if cp is not None and state["gpu"] is not False:
            try:
                if state["gpu"] is None:
                    state["gpu"] = cp.asarray(F)
                G = state["gpu"]
                return np.array([one(t, cp, G) for t in taus])
            except Exception as exc:          # too big for the GPU's memory
                _compute_warn("gpu_lag", "period search on the GPU failed "
                              "({}); using the CPU".format(exc))
                state["gpu"] = False
        n_thr = int(max(1, min(compute_info()["workers"],
                               PARALLEL_MEMORY_BUDGET // max(3 * F.nbytes, 1))))
        return np.array(parallel_map(one, taus, threads=True, workers=n_thr))
    return dissimilarity


def detect_cycle(video, mask_pad: int = 4, sub: int = 2,
                 tau_min: float = 6.0, tau_max: float | None = None,
                 model=None, progress=None):
    """Find the beat period, and frames at a repeatable point in the cycle.

    The period comes from self-similarity: the lag at which the video best
    repeats itself, measured over every pixel above the cell surface and
    averaged over all the cycles in the record.  That is far steadier than
    reading peak spacings off a derived scalar, which on a short clip of only
    a few beats is dominated by noise.

    The cycle start is reported at the **end-effective position**: the instant
    the effective stroke finishes and the recovery stroke begins, which is the
    convention axwa uses (a main cycle runs recovery -> effective, so both end
    points land at the same ee position one beat apart).

    With a ``model`` the landmark comes from the cilium itself: it is traced
    at the model's base in every frame, its tip folded over the period, and
    the end of the faster sweep taken as the end-effective position (see
    stroke_landmarks).  Without one -- or if the trace fails -- the fallback
    is the horizontal centroid of darkness over the whole field, at its
    maximum.  That fallback assumes the effective stroke runs to the right
    and, worse, describes the carpet rather than the cilium: on a
    metachronal field sixty cilia at different phases average to a centroid
    that barely moves, and on one recording its peak fell mid-way through
    the recovery stroke of the cilium being fitted.  It is kept only because
    it needs no base.
    """
    binf = choose_bin_factor(video)
    stack = []
    total = max(1, video.last - video.first + 1)
    _prefetch(video)
    for f in range(video.first, video.last + 1):
        if progress is not None:
            progress((f - video.first) / total, "reading frames")
        img = np.asarray(video.get(f), dtype=np.float32)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)
        stack.append(bin_frame(img, binf))
    frames = np.asarray(stack, dtype=np.float32)
    n, h, w = frames.shape
    if n < 8:
        raise ValueError("need at least 8 frames to find the beat cycle")

    mean = frames.mean(axis=0)
    usable = video.valid
    if binf > 1:
        usable = bin_frame(usable.astype(np.float32), binf) > 0.5
    surface = detect_surface(static_mean(video, mean), valid=usable)
    rows = np.arange(h)[:, None]
    above = (rows < (surface[None, :] - mask_pad)).astype(float)
    if above.sum() < 16 or float(np.std(frames * above)) < 1e-6:
        raise ValueError(
            "nothing beats in this region -- if the frame is cropped, the crop "
            "should hold the cilia and a little of the surface below them")

    # The period is measured where the cilium is, not over the whole field.
    # Different cells beat at different rates: on one recording the left of
    # the band repeated every 29 frames and the right every 35, and the
    # field-wide answer, 33.5, was right for neither -- by the third beat
    # the phases assigned to the frames were a third of a cycle off and the
    # fitted beat averaged unrelated shapes.  With a base the self-similarity
    # is taken inside a window of one and a half cilium lengths around it;
    # the whole field is kept as the fallback and for the report.
    local = None
    if model is not None:
        try:
            ppum = model.ppum / float(binf)
            reach = max(8.0, 1.5 * model.length * ppum)
            cx = model.pqd.getf("vid_cil_x0", w * binf / 2.0) / float(binf)
            cy = (model.image_h - model.pqd.getf("vid_cil_y0_invy",
                                                 model.image_h / 2.0)) / float(binf)
            c0, c1 = int(max(0, cx - reach)), int(min(w, cx + reach + 1))
            r0 = int(max(0, cy - reach))
            r1 = int(min(h, cy + 0.2 * reach + 1))
            window = np.zeros((h, w), dtype=float)
            window[r0:r1, c0:c1] = 1.0
            local = above * window
            if local.sum() < 16 or float(np.std(frames * local)) < 1e-6:
                local = None
        except Exception:
            local = None

    def dissimilarity_of(mask):
        F = ((frames - mean) * mask)[:, ::sub, ::sub]
        norm = float(np.mean(F ** 2)) or 1.0
        return _lag_dissimilarity(F, norm)

    hi = tau_max if tau_max is not None else max(tau_min + 2, (n - 2) / 2.0)
    if progress is not None:
        progress(0.6, "finding the period")

    def best_period(dissimilarity):
        coarse = np.arange(tau_min, hi + 1e-9, 1.0)
        d = dissimilarity(coarse)
        # A record that repeats every T frames repeats every 2T as well, and
        # which of the two mismatches comes out lower is noise: on a lawn
        # with known truth the lag of two beats won by 0.001.  So when the
        # best lag's half or third matches nearly as well -- within a tenth
        # -- the shorter one is the period.  Only those fractions are
        # considered: taking any short lag that came close let one noisy
        # recording's period fall from 41.5 frames to 6.45.
        k_min = int(np.argmin(d))
        d_min = float(d[k_min])
        t0 = float(coarse[k_min])
        for divisor in (3, 2):
            near = np.abs(coarse - t0 / divisor) <= 1.0
            if near.any() and float(np.min(d[near])) <= d_min + max(0.1 * d_min, 0.02):
                t0 = float(coarse[near][int(np.argmin(d[near]))])
                break
        fine = np.arange(max(tau_min, t0 - 2.0), t0 + 2.0 + 1e-9, 0.05)
        df = dissimilarity(fine)
        k = int(np.argmin(df))
        # How clearly the minimum stands out: the mismatch at the best lag
        # against the mismatch of unrelated frames (one for uncorrelated).
        return float(fine[k]), (fine, df), float(df[k])

    period_global, (fine, df), depth_global = best_period(
        dissimilarity_of(above))
    period, depth = period_global, depth_global
    period_source = "whole field"
    if local is not None:
        period_local, (fine_l, df_l), depth_local = best_period(
            dissimilarity_of(local))
        # A local minimum that is at least as clear as the field-wide one is
        # the one to trust; a window with no repeating motion (mismatch near
        # one at every lag) falls back to the field.
        if depth_local < LOCAL_PERIOD_MAX_MISMATCH:
            period, depth, period_source = period_local, depth_local, "around the base"
            fine, df = fine_l, df_l
        local_report = (period_local, depth_local)
    else:
        local_report = None

    # Track the cilium itself: it is dark against the bright fluid, so weight
    # each pixel by how far below the fluid level it sits.  The horizontal
    # centroid then peaks when the cilium is furthest through the effective
    # stroke -- the end-effective position.
    fluid = float(np.percentile(frames[:, above.astype(bool)], 75))
    dark = np.clip(fluid - frames, 0.0, None) * above
    weight = dark.sum(axis=(1, 2))
    cx = (dark * np.arange(w)[None, None, :]).sum(axis=(1, 2)) / np.maximum(
        weight, 1e-9)

    # Fold over the period so every cycle contributes to the landmark.
    nb = 24
    phase = (np.arange(n) % period) / period
    prof = np.full(nb, np.nan)
    for b in range(nb):
        sel = (phase >= b / nb) & (phase < (b + 1) / nb)
        if sel.any():
            prof[b] = cx[sel].mean()
    best_phase = (float(np.nanargmax(prof)) + 0.5) / nb

    tol = 0.5 / nb

    def frames_at(target):
        return [int(video.first + f) for f in range(n)
                if min(abs(phase[f] - target), 1 - abs(phase[f] - target)) < tol]

    out = {
        "period": period,
        "period_global": period_global,
        "period_source": period_source,
        "mismatch": depth,
        "mismatch_global": depth_global,
        "local": local_report,
        "n_frames": n,
        "n_cycles": n / period,
        "candidates": frames_at(best_phase),
        "best_phase": best_phase,
        "centroid": cx,
        "profile": prof,
        "dissimilarity": (fine, df),
        "source": "carpet centroid",
    }

    if model is not None:
        try:
            if progress is not None:
                progress(0.7, "tracing the cilium for the stroke landmark")
            # Two beats are plenty for a landmark, and tracing every frame
            # of a long record here took two minutes before anything was
            # fitted.
            n_land = int(min(video.last - video.first + 1,
                             max(8, np.ceil(2.0 * period)) + 1))
            land_frames = list(range(video.first, video.first + n_land))
            if len(land_frames) > 48:
                # Forty-odd traces over two beats place the landmark as well
                # as two hundred; on a 100-frame beat the difference is two
                # minutes.
                step_f = int(np.ceil(len(land_frames) / 48.0))
                land_frames = land_frames[::step_f]
            _, curves, times_ms = auto_trace(video, model, frames=land_frames,
                                             progress=None)
            fps = model.fps or 1.0
            tips = np.array([[c[-1, 0] - c[0, 0], c[0, 1] - c[-1, 1]]
                             for c in curves], dtype=float)
            frames_tr = np.array([t * fps / 1000.0 for t in times_ms])
            # The greedy walk can fail wholesale on a crowded recording --
            # on one it put the tip below the base in a third of the frames
            # and flung it a full length left or right in most others -- and
            # a landmark read off that is noise.  Insist that the tip sits
            # above the base in most frames before believing it; otherwise
            # the carpet centroid stands, and Auto fit moves the start later
            # from the fitted beat, which is trustworthy.
            length_px = max(6.0, model.length * model.ppum)
            plausible = float(np.mean(tips[:, 1] > 0.2 * length_px))
            if plausible < 0.7:
                raise ValueError(
                    "the tip sat above the base in only {:.0f}% of frames"
                    .format(100 * plausible))
            marks = stroke_landmarks(tips, (frames_tr - video.first) / period,
                                     prefer=effective_direction(model.pqd))
            out.update({
                "candidates": frames_at(marks["phase_end_effective"]),
                "best_phase": marks["phase_end_effective"],
                "other_candidates": frames_at(marks["phase_other"]),
                "direction": marks["direction"],
                "decided_by": marks["decided_by"],
                "speed_ratio": marks["speed_ratio"],
                "duration_effective": marks["duration_effective"],
                "after_fast": marks.get("after_fast", 0.0),
                "source": "traced tip",
            })
        except Exception as exc:          # fall back to the carpet centroid
            out["trace_error"] = str(exc)
    return out


def rotate_fourier_frame(coeffs: dict, delta_deg: float) -> dict:
    """The same beat in a model frame turned by ``delta_deg`` -- what the
    coefficients become when the base orientation is increased by that
    much, so that the shapes on the frames do not move.  Exact: the
    shape is linear in the x and z coefficients, and a rotation mixes
    each (x, z) pair."""
    a = np.deg2rad(float(delta_deg))
    c, s_ = np.cos(a), np.sin(a)
    out = {}
    for xk, zk in (("Ax", "Az"), ("Bx", "Bz")):
        X = np.asarray(coeffs[xk], dtype=float)
        Z = np.asarray(coeffs[zk], dtype=float)
        out[xk] = X * c + Z * s_
        out[zk] = -X * s_ + Z * c
    return out


def shift_fourier_phase(coeffs: dict, delta: float) -> dict:
    """The same beat with its phase origin moved to old phase ``delta``.

    cos(2 pi n (t + d)) and sin(2 pi n (t + d)) expand into the original pair,
    so each mode's (A_n, B_n) simply rotates by 2 pi n d.  Exact: no refit,
    no resampling, and the shapes at every instant are unchanged -- only
    which instant is called phase zero.
    """
    out = {}
    n = np.arange(np.asarray(coeffs["Ax"]).shape[1])
    c = np.cos(2.0 * np.pi * n * delta)[None, :]
    sn = np.sin(2.0 * np.pi * n * delta)[None, :]
    for a_key, b_key in (("Ax", "Bx"), ("Az", "Bz")):
        a = np.asarray(coeffs[a_key], dtype=float)
        b = np.asarray(coeffs[b_key], dtype=float)
        out[a_key] = a * c + b * sn
        out[b_key] = b * c - a * sn
    return out


def recovery_start_phase(wave, n_eval: int = 96, prefer=None):
    """The phase at which a fitted beat's recovery stroke begins.

    The fitted cilium's own tip, over one cycle, put through the same
    stroke test as the traced tips (stroke_landmarks).  This is the
    trustworthy version: the fit has already averaged every cycle in the
    record and thrown out the frames where the trace wandered, where the
    raw traces on a difficult recording put the tip below the base in a
    third of the frames.
    """
    grid = np.arange(n_eval) / n_eval
    tips = []
    for t in grid:
        x, z = wave.shape_at(t, n_pts=120)
        tips.append((x[-1], z[-1]))
    # with as many harmonics as the beat has, the series is the beat's own
    # tip, rest included (four would smooth a rest into motion)
    n_harm = int(max(4, min(12, getattr(wave, "n_modes", 8) - 1)))
    return stroke_landmarks(np.array(tips), grid, n_harm=n_harm, prefer=prefer)


def _decided_by_text(marks) -> str:
    """How the effective stroke was told: by speed, height, or the setting
    -- and, with the setting, whether the frames agree."""
    by = marks["decided_by"]
    if by != "setting":
        return by
    ratio = float(marks["speed_ratio"])
    if ratio > 1.10:
        return "setting, and the frames agree (that sweep is {:.1f}x faster)".format(ratio)
    if ratio < 1.0 / 1.10:
        return ("setting; NOTE the frames say the other sweep is the faster "
                "({:.1f}x) -- check the effective stroke setting".format(1.0 / ratio))
    return "setting (the two sweeps are within 10% in speed)"


def stroke_landmarks(tips_xy, phases, n_harm: int = 4, n_eval: int = 192,
                     prefer=None):
    """Where the effective stroke ends, from a cilium's tip over the beat.

    ``tips_xy`` are tip positions (x to the right, y up, any units) and
    ``phases`` their cycle phases in [0, 1).  A Fourier series of a few
    harmonics is fitted to the tip's x and y against phase, so every cycle in
    the record contributes and the trace noise of single frames is averaged
    out; its two turning points in x are the two ends of the beat.  Which is
    the *end-effective* position -- the axwa cycle start -- is decided by
    speed: the effective stroke is the fast one, so of the two sweeps between
    the turning points the one with the higher peak speed is the effective
    stroke, and its terminal turning point is the landmark.  If the peaks are
    within ten per cent the tip height decides instead, the effective stroke
    being the extended one.  ``prefer`` +1 or -1 settles it instead (the
    effective-stroke setting): the effective stroke is then the sweep to
    the right or to the left, "decided by setting", and ``speed_ratio``
    below 1 says the other sweep was the faster.  ``phase_end_effective``
    is the sideways extreme the effective stroke ends at (the cycle start),
    ``phase_other`` the other extreme, ``phase_fast_end`` where the fast
    part of the effective sweep ends and ``after_fast`` the fraction of the
    beat between that and the extreme.

    The smoothing is what makes the decision repeatable.  Read off binned
    tips, the same recording gave a speed ratio of 1.2 with one trace length
    and 1.04 with another, the second falling to the height tiebreak and
    flipping the answer; the fitted series shows the asymmetry the frames
    actually have, about two to one.
    """
    tips = np.asarray(tips_xy, dtype=float)
    ph = np.asarray(phases, dtype=float) % 1.0
    ok = np.isfinite(tips).all(axis=1) & np.isfinite(ph)
    tips, ph = tips[ok], ph[ok]
    if len(ph) < 2 * (2 * n_harm + 1):
        raise ValueError("tip found in too few frames to place the strokes")
    n_harm = int(max(1, min(n_harm, (len(ph) // 4 - 1) // 2 or 1)))
    k = np.arange(1, n_harm + 1)
    design = np.column_stack(
        [np.ones(len(ph))]
        + [np.cos(2 * np.pi * kk * ph) for kk in k]
        + [np.sin(2 * np.pi * kk * ph) for kk in k])
    coef, _, _, _ = np.linalg.lstsq(design, tips, rcond=None)
    grid = np.arange(n_eval) / n_eval
    basis = np.column_stack(
        [np.ones(n_eval)]
        + [np.cos(2 * np.pi * kk * grid) for kk in k]
        + [np.sin(2 * np.pi * kk * grid) for kk in k])
    dbasis = np.column_stack(
        [np.zeros(n_eval)]
        + [-2 * np.pi * kk * np.sin(2 * np.pi * kk * grid) for kk in k]
        + [2 * np.pi * kk * np.cos(2 * np.pi * kk * grid) for kk in k])
    curve = basis @ coef                     # (n_eval, 2)
    vel = dbasis @ coef
    speed = np.hypot(vel[:, 0], vel[:, 1])   # per period
    i_max = int(np.argmax(curve[:, 0]))
    i_min = int(np.argmin(curve[:, 0]))
    if i_max == i_min:
        raise ValueError("the tip does not move sideways over the beat")

    def between(a, b):
        return [(a + j) % n_eval for j in range(((b - a) % n_eval) or n_eval)]

    to_max = between(i_min, i_max)           # x increasing sweep
    to_min = between(i_max, i_min)           # x decreasing sweep
    peak_right = float(speed[to_max].max())
    peak_left = float(speed[to_min].max())
    height_right = float(curve[to_max, 1].mean())
    height_left = float(curve[to_min, 1].mean())
    ratio = peak_right / max(peak_left, 1e-9)
    if prefer is not None and prefer != 0:
        rightward, decided_by = prefer > 0, "setting"
    elif ratio > 1.10:
        rightward, decided_by = True, "speed"
    elif ratio < 1.0 / 1.10:
        rightward, decided_by = False, "speed"
    else:
        rightward, decided_by = height_right >= height_left, "height"
    end_eff = i_max if rightward else i_min
    other = i_min if rightward else i_max
    # The cycle starts at the sideways extreme the effective stroke ends
    # at -- the rightmost state when it sweeps right -- as the published
    # table's phase zero does.  The fast part of the sweep (the run around
    # its peak speed in which the tip is faster than the midpoint between
    # its slowest and fastest) can end well before the extreme, the cilium
    # drifting on slowly: 22 % of the beat earlier on the published beat,
    # 40 % on input_images.  That is reported, not used: a cycle started
    # there drew a fan whose first shape was not the rightmost, which was
    # read as a start in the middle of the recovery.
    fast = speed > 0.5 * (speed.max() + speed.min())
    eff_sweep = between(other, end_eff)
    pk = eff_sweep[int(np.argmax(speed[eff_sweep]))]
    half = n_eval // 2
    e, k = pk, 0
    while fast[(e + 1) % n_eval] and k < half:
        e = (e + 1) % n_eval
        k += 1
    return {
        "phase_end_effective": grid[end_eff],
        "phase_other": grid[other],
        "phase_fast_end": grid[(e + 1) % n_eval],
        "after_fast": ((end_eff - e - 1) % n_eval) / float(n_eval),
        "direction": 1 if rightward else -1,
        "decided_by": decided_by,
        "speed_ratio": ratio if rightward else 1.0 / ratio,
        "duration_effective": float(fast.mean()),
        "curve": curve,
        "speed": speed,
    }


def bin_frame(img, factor: int):
    """Average a frame down by an integer factor, discarding the remainder."""
    factor = int(factor)
    if factor <= 1:
        return np.asarray(img, dtype=np.float32)
    a = np.asarray(img, dtype=np.float32)
    h = (a.shape[0] // factor) * factor
    w = (a.shape[1] // factor) * factor
    a = a[:h, :w]
    return a.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def resample_frame(img, scale: float):
    """Resize an image by ``scale`` with bilinear interpolation.

    Used to bring a drawn carpet to the working resolution the tracer wants:
    about six to eight pixels between neighbouring cilia.  Finer than that
    costs time for nothing -- the dynamic programme scales with the cube of
    the resolution -- and coarser than that blurs neighbours into one field
    under the ridge filter, which is when the tracer draws staircases.
    """
    scale = float(scale)
    a = np.asarray(img, dtype=np.float32)
    if abs(scale - 1.0) < 1e-9:
        return a
    h, w = a.shape
    nh, nw = max(2, int(round(h * scale))), max(2, int(round(w * scale)))
    ys = np.clip((np.arange(nh) + 0.5) / scale - 0.5, 0, h - 1)
    xs = np.clip((np.arange(nw) + 0.5) / scale - 0.5, 0, w - 1)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    return (a[y0][:, x0] * (1 - fy) * (1 - fx) + a[y1][:, x0] * fy * (1 - fx)
            + a[y0][:, x1] * (1 - fy) * fx + a[y1][:, x1] * fy * fx)


def upsample_frame(img, up: int):
    """Enlarge an image an integer number of times (see resample_frame)."""
    return resample_frame(img, float(int(up)))


def choose_bin_factor(video, target_ppum: float = 14.0,
                      max_pixels: int = 60_000_000) -> int:
    """How much to shrink the frames before analysing them.

    Nothing in the tracing benefits from more than about 14 pixels per micron
    -- a cilium is then some 80 pixels long, which is finer than the traces are
    ever resolved -- and the cost is quadratic in the sampling.  One recording
    here is 57 pixels per micron over 247 frames of 1587x685, which is 2.1 GB
    of stack in double precision before anything is computed, and slow with it.
    """
    ppum = float(video_ppum(video) or 0.0)
    by_scale = max(1, int(np.floor(ppum / target_ppum))) if ppum > 0 else 1
    h, w = video.shape
    n = max(1, video.last - video.first + 1)
    by_memory = 1
    while (h // by_memory) * (w // by_memory) * n > max_pixels:
        by_memory += 1
    return max(by_scale, by_memory)


def video_ppum(video):
    """Pixels per micron for a video, if it can be worked out."""
    return getattr(video, "ppum_hint", None)


def structure_stack(video, mask_pad: int = 3, bg_sigma: float = 6.0,
                    ramp: float = 0.0, motion_weight: float = 0.0,
                    reference: str = "mean", polarity: str = "auto",
                    bin_factor: int = 1, progress=None):
    """Per-frame map of what stands out from the background, whatever its sign.

    ``reference`` chooses the background.  "mean" subtracts the average of all
    the frames, so what is left is whatever moved into or out of each pixel in
    this frame; "blur" subtracts a spatial blur of the same frame, leaving
    local contrast of any kind.

    "mean" is the default because it is the one that actually finds cilia.
    Scored along the trusted hand-fitted centreline of the one dataset that has
    one, against a sideways path of the same length running along the cell
    surface: local contrast scores 0.40, meaning the surface fringe is two and
    a half times brighter than the cilium, which is exactly why traces kept
    setting off sideways.  Subtracting the temporal mean scores 1.73, because
    the fringe is bright but motionless while the cilium is the thing that
    moves.  Combining the two scores 0.91 -- worse than the mean alone.

    ``polarity`` then keeps only the half of that deviation the cilium actually
    produces.  Taking the magnitude lights up both where the cilium is now and
    where it usually is but currently is not, and that ambiguity made the walk
    steer up the middle of the swept fan: every phase traced as a nearly
    straight vertical line and the fitted beat lost its sweep.  Keeping the
    signed half raises the same score from 1.73 to 2.34.  Which half is
    measured from the skew of the deviation inside the beating band -- a dark
    cilium arriving at a pixel leaves a long negative tail.

    ``darkness_stack`` assumes cilia are darker than the fluid.  That holds for
    a sparse side-on view, but in a dense ciliary carpet individual cilia show
    up both brighter and darker than their surroundings, and half of them were
    invisible to the fit.  Taking the magnitude of the deviation from the local
    background finds them either way -- measured on the ratio of on-cilium to
    off-cilium signal, it scores 2.38 against 1.65 for darkness on such a
    carpet, and is no worse on a sparse view.

    ``ramp`` fades the map in over that many pixels above the surface instead of
    switching it on abruptly.  The cell surface is itself a strong edge, so the
    contrast map has a bright ridge running along it; a hard cut leaves that
    ridge as the brightest thing in the picture and a trace started at the
    cilium base simply runs off sideways along it.

    ``motion_weight`` mixes in how much each pixel varies over time, which is
    the real distinction: the surface edge is bright but still, the cilium is
    the thing that moves.  Weighting by motion removes the edge on a sparse
    side-on view without needing a mask tuned to the picture, and changes
    nothing on a carpet where everything moves.
    """
    frames = []
    total = max(1, video.last - video.first + 1)
    _prefetch(video)
    for f in range(video.first, video.last + 1):
        if progress is not None:
            progress((f - video.first) / total, "reading frames")
        img = np.asarray(video.get(f), dtype=np.float32)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)
        frames.append(bin_frame(img, bin_factor))

    # float32 throughout: at full precision a long, finely sampled recording
    # needs gigabytes for the stack alone, and nothing here needs the digits.
    frames = np.asarray(frames, dtype=np.float32)
    mean = frames.mean(axis=0)
    h, w = mean.shape
    usable = video.valid
    if bin_factor > 1:
        usable = bin_frame(usable.astype(np.float32), bin_factor) > 0.5
    surface = detect_surface(static_mean(video, mean), valid=usable)
    rows = np.arange(h)[:, None]
    depth = (surface[None, :] - mask_pad) - rows       # >0 above the surface
    if ramp > 0:
        above = np.clip(depth / float(ramp), 0.0, 1.0)
    else:
        above = (depth > 0).astype(float)

    if motion_weight > 0:
        motion = gaussian_blur(frames.std(axis=0), 1.5)
        ref = float(np.percentile(motion[above > 0], 90)) if (above > 0).any() \
            else 0.0
        if ref > 0:
            m = np.clip(motion / ref, 0.0, 1.0)
            above = above * ((1.0 - motion_weight) + motion_weight * m)

    above = above * usable

    sign = 0.0
    if polarity == "auto":
        band = above > 0.5
        if band.any():
            dev = (frames - mean[None, :, :])[:, band]
            sd = float(np.std(dev))
            if sd > 0:
                skew = float(np.mean((dev / sd) ** 3))
                sign = 0.0 if abs(skew) < 0.05 else np.sign(skew)
    elif polarity == "darker":
        sign = -1.0
    elif polarity == "brighter":
        sign = 1.0

    out = np.empty_like(frames)
    for i, img in enumerate(frames):
        back = mean if reference == "mean" else gaussian_blur(img, bg_sigma)
        d = img - back
        out[i] = (np.abs(d) if sign == 0.0
                  else np.clip(sign * d, 0.0, None)) * above
    return out, frames


def beat_strength(video, half: int = 12, depth: int = 45, step: int = 4,
                  lag_min: int = 4, progress=None):
    """How periodic the beat is, column by column, and at what lag.

    For each vertical strip above the cell surface, the frames are correlated
    with themselves at every lag; a cilium beating steadily gives a clear peak
    at the beat period.  This is worth measuring before fitting anything, and
    worth reporting: everything downstream -- a phase, a control point, a
    waveform -- assumes the frames repeat with the beat, and on a real carpet
    that assumption can be much weaker than it looks.  The number reported is
    the height of the peak above its own baseline: a clean side-on view of one
    cilium measures about 1.1, a dense carpet 0.4 to 0.6, and a base chosen
    where it is near zero cannot be fitted at all.

    Returns the per-column peak correlation and lag, the best column, and the
    period implied by the strongest columns.
    """
    binf = choose_bin_factor(video)
    stack = []
    total = max(1, video.last - video.first + 1)
    _prefetch(video)
    for f in range(video.first, video.last + 1):
        if progress is not None:
            progress(0.5 * (f - video.first) / total, "reading frames")
        img = np.asarray(video.get(f), dtype=np.float32)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)
        stack.append(bin_frame(img, binf))
    frames = np.asarray(stack, dtype=np.float32)
    n, h, w = frames.shape
    usable = video.valid
    if binf > 1:
        usable = bin_frame(usable.astype(np.float32), binf) > 0.5
    surface = detect_surface(static_mean(video, frames.mean(axis=0)), valid=usable)
    half = max(2, int(round(half / binf)))
    depth = max(4, int(round(depth / binf)))

    lags = np.arange(lag_min, max(lag_min + 1, n // 2))
    cols = np.arange(half, max(half + 1, w - half), step)
    corr = np.zeros(len(cols))
    lag = np.zeros(len(cols))
    for ci, c in enumerate(cols):
        if progress is not None:
            progress(0.5 + 0.5 * ci / len(cols), "measuring the beat")
        # No tissue in this column means no cilium bases in it.
        if surface[c] >= h:
            continue
        top = max(0, int(surface[c]) - depth)
        bot = max(top + 1, int(surface[c]))
        win = frames[:, top:bot, c - half:c + half].reshape(n, -1)
        win = win - win.mean(axis=0)
        norm = np.linalg.norm(win, axis=1)
        if np.median(norm) <= 1e-9:
            continue
        norm[norm <= 0] = 1e-9
        win = win / norm[:, None]
        gram = win @ win.T
        # Mean of each off-diagonal band = correlation at that lag.
        cc = np.array([np.mean(np.diagonal(gram, offset=int(L))) for L in lags])
        # The height of the peak *above the baseline*, not the raw correlation.
        # A still or slowly drifting patch of image correlates highly with
        # itself at every lag, and scoring the raw maximum made such a patch
        # look like the most perfectly periodic thing in the field -- 0.98 in a
        # corner with no cilia in it, which then reported a beat frequency three
        # times the real one.  A beat shows up as a peak standing above the
        # baseline; drift has no peak.
        peak = int(np.argmax(cc))
        corr[ci] = float(cc[peak] - np.median(cc))
        lag[ci] = lags[peak]

    if not np.any(corr > 0):
        raise ValueError("no periodic motion found in these frames")
    best = int(np.argmax(corr))
    strong = corr >= max(corr.max() - 0.05, 0.0)
    return {
        "columns": cols * binf, "corr": corr, "lag": lag,
        "best_column": int(cols[best] * binf),
        "best_corr": float(corr[best]),
        "period": float(np.median(lag[strong])),
        "median_corr": float(np.median(corr)),
        "surface": np.repeat(surface, binf)[:int(w * binf)] * binf,
    }


def _pqd_for_images(images_dir: str):
    """The specification file belonging to a folder of images, or None:
    ciliograph.json inside the folder, else a wapa.pqd of the earlier
    versions there (read and converted on load), else ``../data1/wapa.pqd``
    when the folder is the ``input_images`` of a classic run.  Nothing
    looser: a sibling dataset's parameters must never be adopted.
    """
    images_dir = os.path.abspath(images_dir)
    for name in (SPEC_NAME, "wapa.pqd"):
        inside = os.path.join(images_dir, name)
        if os.path.exists(inside):
            return inside
    if os.path.basename(images_dir).lower() == "input_images":
        run = os.path.dirname(images_dir)
        candidate = os.path.join(run, "data1", "wapa.pqd")
        if os.path.exists(candidate):
            return candidate
    return None


def default_pqd_for_images(info: dict, ppum: float = DEFAULT_PPUM,
                           fps: float = 300.0, length: float = 6.0,
                           cbf: float = 15.0) -> Spec:
    """The neutral parameter set for a folder of images, held in memory.

    Nothing is written: the fitter starts from these and writes the
    specification file only when Save is pressed.  Writing one on import
    left a neutral file in every folder ever opened, which was not wanted.
    """
    return Spec.defaults(basename=info["basename"] or "im",
                         width=info["width"], height=info["height"],
                         first=0, last=info["count"] - 1,
                         ppum=ppum, fps=fps, length=length, cbf=cbf)


def create_spec_for_images(images_dir: str, path: str,
                           ppum: float = DEFAULT_PPUM, fps: float = 300.0,
                           length: float = 6.0, cbf: float = 15.0) -> str:
    """Write a fresh specification file for a folder of images -- for the
    command-line tools, which take a file as their input.  The fitter never
    calls this: it keeps the defaults in memory until Save."""
    info = scan_image_folder(images_dir)
    if info is None:
        raise ValueError("no images found in {}".format(images_dir))
    spec = default_pqd_for_images(info, ppum=ppum, fps=fps, length=length, cbf=cbf)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    spec.save(path, backup=False)
    return path


def _tk_parent():
    """The fitter's own Tk window when matplotlib runs on Tk, so a dialog
    opens as its child.  Creating and destroying a second Tk root inside a
    running Tk program is not safe everywhere."""
    try:
        import tkinter as tk
        for num in plt.get_fignums() if plt is not None else []:
            win = getattr(plt.figure(num).canvas.manager, "window", None)
            if isinstance(win, (tk.Tk, tk.Toplevel)):
                return win
    except Exception:
        pass
    return None


def _with_dialog(fn):
    """Run a tkinter dialog function with a parent window, making a hidden
    root only when the program has no Tk window of its own."""
    try:
        import tkinter as tk
    except Exception:
        return None
    parent = _tk_parent()
    root = None
    if parent is None:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        parent = root
    try:
        return fn(parent)
    finally:
        if root is not None:
            root.destroy()


def choose_folder(title: str = "Select the folder of images",
                  initial: str | None = None):
    """Native folder chooser.  Returns None if unavailable or cancelled."""
    try:
        from tkinter import filedialog
    except Exception:
        return None
    chosen = _with_dialog(lambda parent: filedialog.askdirectory(
        parent=parent, title=title, initialdir=initial or os.getcwd()))
    return chosen or None


def choose_file(title: str = "Select a file", initial: str | None = None,
                patterns=(("ONNX models", "*.onnx"), ("all files", "*.*"))):
    """Native file chooser.  Returns None if unavailable or cancelled."""
    try:
        from tkinter import filedialog
    except Exception:
        return None
    chosen = _with_dialog(lambda parent: filedialog.askopenfilename(
        parent=parent, title=title, initialdir=initial or os.getcwd(),
        filetypes=list(patterns)))
    return chosen or None


def load_trace_csv(path: str):
    """Read the traced centrelines written by trace_cilium_colored.py.

    Columns: step, frame_file, time_ms, s_index, x_px, y_px.  Returns
    ``(steps, curves, times_ms)`` with one Nx2 pixel array per step, ordered
    base -> tip as they were clicked.
    """
    import csv as _csv
    data, times = {}, {}
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        for row in _csv.DictReader(f):
            s = int(row["step"])
            data.setdefault(s, []).append(
                (float(row["x_px"]), float(row["y_px"])))
            times[s] = float(row.get("time_ms", 0.0) or 0.0)
    if not data:
        raise ValueError("{}: no rows".format(path))
    steps = sorted(data)
    return steps, [np.array(data[s], dtype=float) for s in steps], \
        [times[s] for s in steps]


def _sample_bilinear(img, x, y):
    h, w = img.shape
    xi = np.clip(x, 0, w - 1)
    yi = np.clip(y, 0, h - 1)
    x0 = np.floor(xi).astype(int)
    y0 = np.floor(yi).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx, fy = xi - x0, yi - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


def trace_cilium(img, base, n_steps: int, step: float = 1.0,
                 max_turn: float = 12.0, look: int = 3,
                 start_angles=None, prior=None, prior_weight: float = 0.0,
                 min_frac: float = 0.6, turn_weight: float = 0.0,
                 max_total_turn: float = MAX_TRACE_TURN, avoid: float = 3.0,
                 tip_frac: float = 0.0, tip_run: int = 3,
                 min_steps: int | None = None, row_limit=None,
                 look_samples: int = 8, corner_weight: float = 1.5):
    """Follow a cilium outward from its base along the ridge of ``img``.

    A ridge walk: at each step it looks a little way ahead in a fan of
    directions and takes the brightest, with the turn per step capped so the
    path cannot double back.  Many starting directions are walked at once and
    the brightest whole path wins, so it works whether the cilium stands up or
    lies over.

    ``prior`` is the path traced in the previous frame; deviating from it costs
    ``prior_weight`` per pixel.  Without it the walk hops onto whichever
    neighbour happens to be brightest, which in a dense carpet makes the traced
    tip jump around from frame to frame and destroys the periodicity the fit
    depends on.  With it the walk stays on one cilium.

    This is what makes a dense carpet tractable at all.  Refining parameters
    against an image score cannot work there -- the score is nearly flat
    because every direction has a cilium in it -- but a traced centreline is
    exactly the input the accurate trace-based fit wants.

    The walk stops at the tip: once the ridge has faded to ``tip_frac`` of its
    strength near the base for ``tip_run`` steps running, the cilium has ended.
    Without that the walk simply took ``n_steps`` steps whatever the picture
    showed, so the "length measured from the traced reach" was the length it
    started from -- feed it 3 microns and it reports 2.97, feed it 10 and it
    reports 10.00.  Any overestimate from the length estimator was then
    rubber-stamped, and the model was drawn far past the end of the cilium.
    """
    h, w = img.shape
    n_steps = int(n_steps)
    if start_angles is None:
        start_angles = np.deg2rad(np.linspace(-165.0, -15.0, 31))
    ang = np.asarray(start_angles, dtype=float).ravel().copy()
    m = len(ang)
    if m == 0:
        raise ValueError("no starting directions")

    if prior is not None and len(prior) >= 2:
        prior = _resample_curve(np.asarray(prior, dtype=float), n_steps + 1)[0]
    else:
        prior, prior_weight = None, 0.0

    turns = np.deg2rad(np.linspace(-max_turn, max_turn, 13))
    x = np.full(m, float(base[0]))
    y = np.full(m, float(base[1]))
    paths = np.empty((m, n_steps + 1, 2))
    paths[:, 0, 0] = x
    paths[:, 0, 1] = y
    total = np.zeros(m)
    alive = np.ones(m, dtype=bool)
    count = np.zeros(m, dtype=int)

    # Ridge strength near the base, used to recognise the tip further out.
    n_ref = max(2, min(n_steps // 4, 8))
    ref_sum = np.zeros(m)
    ref = np.full(m, np.inf)
    low_run = np.zeros(m, dtype=int)
    end = np.full(m, n_steps, dtype=int)

    turned = np.zeros(m)
    last_turn = np.zeros(m)
    cap = np.deg2rad(max_total_turn)
    # Points this recent are close by construction, so they cannot count as
    # self-crossings.
    gap = int(np.ceil(2.0 * max(avoid, 0.0) / max(step, 1e-6))) + 1
    for k in range(1, n_steps + 1):
        a = ang[:, None] + turns[None, :]
        # Score each candidate along the *arc* it would produce if it kept
        # turning at that rate, not along a straight ray.  A straight probe
        # cannot see curvature: a cilium bends with a radius of about 35 px,
        # so over the 3 px the probe used to reach, a curved ridge departs
        # from a straight line by 0.13 px -- far less than the ridge is wide.
        # Every candidate then scored the same and the walk went perfectly
        # straight; 49 of 70 steps turned by exactly zero degrees, and the
        # traced cilia came out with no curvature at all beyond the first
        # third.  Following the arc, and looking far enough ahead for the
        # departure to exceed the ridge width, is what lets it turn.
        s = np.zeros_like(a)
        px = np.repeat(x[:, None], len(turns), axis=1)
        py = np.repeat(y[:, None], len(turns), axis=1)
        heading = a.copy()
        # A fixed number of samples spread along the probe, whatever its
        # length: one sample per pixel would make the cost grow with the
        # magnification, and on a finely sampled recording (337 px per cilium)
        # that alone came to billions of samples per pass.
        probe = float(look) / look_samples
        # The probe points are found first and sampled in one call, then
        # added up in the same order as before: the same numbers, without
        # paying the call overhead eight times a step.
        pxs, pys = [], []
        for _ in range(look_samples):
            px = px + np.cos(heading) * probe
            py = py + np.sin(heading) * probe
            pxs.append(px)
            pys.append(py)
            heading = heading + turns[None, :] * (probe / max(step, 1e-9))
        if pxs:
            vals = _sample_bilinear(img, np.stack(pxs), np.stack(pys))
            for v in vals:
                s += v
        if prior is not None:
            s -= prior_weight * np.hypot(
                x[:, None] + np.cos(a) * step - prior[k, 0],
                y[:, None] + np.sin(a) * step - prior[k, 1])
        cx = x[:, None] + np.cos(a) * step
        cy = y[:, None] + np.sin(a) * step
        if turn_weight:
            s -= turn_weight * np.abs(turns)[None, :]
        if corner_weight:
            # A cilium's curvature varies smoothly along it, so a sudden change
            # of turn -- a corner -- is never the cilium.  It is how the walk
            # leaves a filament and sets off along whatever crosses it: on a
            # drawn metachronal wave the traces ran up a cilium, turned a sharp
            # corner at the top and continued along the envelope joining the
            # tips of the row.  Penalising the *change* in turn rather than the
            # turn itself forbids that without discouraging a steady bend,
            # which is what a cilium actually does.
            s -= corner_weight * np.abs(turns[None, :] - last_turn[:, None])
        # A cilium's centreline turns through a bounded angle over its length:
        # the hand-fitted reference waveform turns about 95 degrees in total.
        # This cap used to be applied only when a turning penalty was switched
        # on, which it is not by default, so nothing limited the walk -- and
        # traces came back as hairpins, or bent over and ran along the ciliary
        # band instead of out along one cilium.  Both are shapes the waveform
        # cannot represent, so the fit answered them with the same gentle arc
        # at every phase of the beat.
        if cap > 0:
            s = np.where((turned[:, None] + np.abs(turns)[None, :]) > cap,
                         -np.inf, s)
        if row_limit is not None:
            # A cilium grows out of the cell; it cannot pass through it.  With
            # the walk free to cross the surface line, traces at some phases
            # ran down into the tissue and the fitted waveform swept below the
            # base -- shapes that dived into the cell.
            cols = np.clip(np.round(cx).astype(int), 0, len(row_limit) - 1)
            s = np.where(cy > row_limit[cols], -np.inf, s)
        if avoid > 0 and k > gap:
            # A cilium never crosses itself.  Left free, a greedy walk in a
            # textured image spirals -- it keeps finding brightness by curling
            # back over ground it has already covered -- and that, not a lack
            # of smoothness, is what wrecked the traces.  Forbidding steps that
            # return to the path stops it without straightening genuine bends,
            # which a curvature penalty cannot avoid doing.
            prev = paths[:, :k - gap, :]
            d2 = ((cx[:, :, None] - prev[:, None, :, 0]) ** 2
                  + (cy[:, :, None] - prev[:, None, :, 1]) ** 2)
            s = np.where(d2.min(axis=2) < avoid * avoid, -np.inf, s)
        blocked = ~np.isfinite(s).any(axis=1)
        if blocked.any():
            alive &= ~blocked
            s[blocked] = 0.0
        pick = np.argmax(s, axis=1)
        turned += np.abs(turns[pick])
        last_turn = turns[pick]
        ang = a[np.arange(m), pick]

        nx = x + np.cos(ang) * step
        ny = y + np.sin(ang) * step
        inside = (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h)
        alive &= inside
        x = np.where(alive, nx, x)
        y = np.where(alive, ny, y)
        paths[:, k, 0] = x
        paths[:, k, 1] = y
        raw = _sample_bilinear(img, x, y)
        value = raw
        if prior is not None:
            # The same penalty has to enter the score that picks between the
            # starting directions, not just the step-by-step steering: a start
            # aimed at a brighter neighbour would otherwise still win.
            value = value - prior_weight * np.hypot(x - prior[k, 0],
                                                    y - prior[k, 1])
        total += np.where(alive, value, 0.0)
        count += alive

        if k <= n_ref:
            ref_sum += np.where(alive, raw, 0.0)
            if k == n_ref:
                ref = ref_sum / float(n_ref)
        elif tip_frac > 0:
            faded = alive & (raw < tip_frac * ref)
            low_run = np.where(faded, low_run + 1, 0)
            done = alive & (low_run >= tip_run)
            if done.any():
                end = np.where(done, np.minimum(end, k - tip_run), end)
                alive &= ~done
        if not alive.any():
            break

    # Stopping at the tip is the right answer, running off the image is not, so
    # eligibility is an absolute minimum length rather than a fraction of the
    # steps allowed: with the walk free to end early, a fraction of n_steps
    # would rule out every honest trace.
    length_steps = np.minimum(count, end)
    floor_steps = (min_frac * n_steps if min_steps is None
                   else max(2, int(min_steps)))
    score = np.where(length_steps >= floor_steps,
                     total / np.maximum(count, 1), -np.inf)
    if not np.isfinite(score).any():
        raise ValueError("could not trace a cilium from {}".format(tuple(base)))
    best = int(np.argmax(score))
    return paths[best, :int(length_steps[best]) + 1]


def tip_from_profiles(paths, values, tip_frac: float = 0.45, tip_run: int = 4,
                      min_frac: float = 0.3):
    """Where the cilium ends, from the ridge strength along many traces.

    Deciding the tip inside each walk does not work: on a faint cilium the
    ridge dips and recovers, and a per-path test stopped at 2 microns on a
    recording whose hand-fitted length is 6.  Every trace in the record follows
    the same cilium, though, so the median strength as a function of distance
    from the base is a far steadier curve than any one of them, and it falls
    off once past the tip.  One decision, taken from all of the data.

    ``tip_frac`` is calibrated against the one recording here with a trusted
    hand fit: starting the walk from 4 microns and from 9, this setting
    measures 6.9 and 5.7 against a hand-fitted 6.0.  So the measured length is
    good to roughly 15%, no better -- set it by hand in the length tab when it
    matters.

    Returns the number of steps to keep.
    """
    n = max(len(v) for v in values)
    grid = np.full((len(values), n), np.nan)
    for i, v in enumerate(values):
        grid[i, :len(v)] = v
    with np.errstate(all="ignore"):
        profile = np.nanmedian(grid, axis=0)
    good = np.isfinite(profile)
    if good.sum() < 4:
        return n

    lo = max(2, int(0.15 * n))
    plateau = float(np.nanmax(profile[:max(lo, 3)]))
    if not np.isfinite(plateau) or plateau <= 0:
        return n

    floor = max(2, int(min_frac * n))
    run = 0
    for k in range(floor, n):
        if not good[k] or profile[k] < tip_frac * plateau:
            run += 1
            if run >= tip_run:
                return max(floor, k - tip_run + 1)
        else:
            run = 0
    return n


def trace_cilium_path(img, base, length_px: float, thresh: float = 0.22,
                      max_gap: float = 3.0, step_out: float = 1.0,
                      row_limit=None):
    """Trace a cilium as the cheapest path along the bright structure.

    A step-by-step walk decides each step from what it can see just ahead, and
    on a densely packed field it cannot help crossing to a neighbour: measured
    on a drawn metachronal wave, only 61 to 87 per cent of each traced path
    lay on a drawn cilium at all, and the turning was pinned against its limit
    because the path kept swinging between filaments.  This instead finds the
    lowest-cost route from the base over the whole picture at once, where cost
    is distance divided by brightness, so leaving the filament is expensive in
    proportion to the gap crossed.  It is a global choice rather than a
    sequence of local ones, which is what following a thin structure needs.

    The graph is built only on pixels bright enough to be structure, which is
    a percent or two of the frame, so the search is small however large the
    image.  Pixels within ``max_gap`` of each other are joined, so a filament
    broken by noise is still followed, but a wider gap is not crossed.
    """
    import heapq

    h, w = img.shape
    bx, by = float(base[0]), float(base[1])
    reach = 1.35 * length_px
    x_lo = max(0, int(bx - reach)); x_hi = min(w, int(bx + reach) + 1)
    y_lo = max(0, int(by - reach)); y_hi = min(h, int(by + reach) + 1)
    win = img[y_lo:y_hi, x_lo:x_hi]

    strong = win >= thresh * float(win.max() if win.size else 1.0)
    if row_limit is not None:
        cols = np.clip(np.arange(x_lo, x_hi), 0, len(row_limit) - 1)
        limit = np.asarray(row_limit)[cols][None, :]
        strong &= (np.arange(y_lo, y_hi)[:, None] <= limit)
    ys, xs = np.nonzero(strong)
    if len(ys) < 8:
        raise ValueError("no structure near the base to follow")

    pts = np.column_stack([xs + x_lo, ys + y_lo]).astype(float)
    vals = win[ys, xs].astype(float)
    index = {(int(a), int(b)): i for i, (a, b) in enumerate(pts)}

    start = int(np.argmin(np.hypot(pts[:, 0] - bx, pts[:, 1] - by)))
    if np.hypot(*(pts[start] - np.array([bx, by]))) > max(4.0, max_gap):
        raise ValueError("no structure at the base")

    gap = int(np.ceil(max_gap))
    offsets = [(dx, dy) for dy in range(-gap, gap + 1)
               for dx in range(-gap, gap + 1)
               if (dx or dy) and (dx * dx + dy * dy) <= max_gap * max_gap]

    n = len(pts)
    cost = np.full(n, np.inf)
    arc = np.zeros(n)
    parent = np.full(n, -1, dtype=int)
    cost[start] = 0.0
    heap = [(0.0, start)]
    eps = 1e-3
    while heap:
        c, i = heapq.heappop(heap)
        if c > cost[i] + 1e-12:
            continue
        xi, yi = int(pts[i, 0]), int(pts[i, 1])
        for dx, dy in offsets:
            j = index.get((xi + dx, yi + dy))
            if j is None:
                continue
            d = float(np.hypot(dx, dy))
            nc = c + d / (0.5 * (vals[i] + vals[j]) + eps)
            if nc < cost[j] - 1e-12:
                cost[j] = nc
                arc[j] = arc[i] + d
                parent[j] = i
                heapq.heappush(heap, (nc, j))

    reached = np.isfinite(cost) & (arc > 0)
    if not reached.any():
        raise ValueError("nothing reachable from the base")
    # The tip is the end of the cheapest route that is about as long as the
    # cilium; among those, the one that cost least per unit length.
    want = np.abs(arc - length_px) <= 0.25 * length_px
    pick_from = np.nonzero(reached & want)[0]
    if len(pick_from) == 0:
        pick_from = np.nonzero(reached)[0]
        end = int(pick_from[np.argmax(arc[pick_from])])
    else:
        end = int(pick_from[np.argmin(cost[pick_from] / np.maximum(arc[pick_from], 1e-9))])

    path = []
    node = end
    while node >= 0:
        path.append(pts[node])
        node = int(parent[node])
    path = np.array(path[::-1], dtype=float)
    if len(path) < 3:
        raise ValueError("path too short")
    n_out = max(3, int(round(arc[end] / max(step_out, 1e-6))))
    return _resample_curve(path, n_out)[0]


# ---------------------------------------------------------------------------
# Following one named cilium: a minimal path over position and direction
# ---------------------------------------------------------------------------
#
# The greedy walk above decides each step from what it can see ahead of it,
# and in a dense carpet that is not enough: where two cilia nearly touch it
# has no way to tell which one it arrived on, and a wrong turn is permanent.
# What follows is the standard remedy from the filament-tracing literature --
# a minimal path whose state is position *and* direction, solved exactly by
# dynamic programming over arc length.  Because direction is part of the
# state, two filaments crossing are two states and the curve cannot change
# filament there; because the solution is global, no early step is
# irreversible.


def lifted_directions(radius: int = 2):
    """The step directions of the lifted graph, and their lengths.

    Every offset up to ``radius`` whose components are coprime, so each
    direction appears once and not also as its own multiple.  A radius of two
    gives sixteen directions, 22.5 degrees apart, which is fine enough that
    the turn limit is a real constraint rather than a rounding of one.
    """
    offs = []
    for d_r in range(-radius, radius + 1):
        for d_c in range(-radius, radius + 1):
            if (d_r or d_c) and np.gcd(abs(d_r), abs(d_c)) == 1:
                offs.append((d_r, d_c))
    offs = sorted(set(offs), key=lambda d: np.arctan2(d[0], d[1]))
    ang = np.array([np.arctan2(d[0], d[1]) for d in offs], dtype=np.float64)
    dlen = np.array([np.hypot(*d) for d in offs], dtype=np.float32)
    adv = np.maximum(1, np.rint(dlen).astype(int))
    return np.array(offs, dtype=np.int32), ang, dlen, adv


def gauss_hess(img, sigma: float):
    """Second derivatives of an image, by analytic Gaussian kernels.

    Worth the kernels rather than np.gradient of a box blur.  Measured on the
    drawn wave, the crude version gave the steerable filter below a contrast
    of only 1.3 between a direction along a cilium and one 45 degrees off it,
    which is no discrimination at all, and the tracer fell back on bending as
    the only informative term and drew straight lines.  With proper Gaussian
    second derivatives the same contrast is 5 to 8.
    """
    r = max(2, int(round(4.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-x * x / (2.0 * sigma * sigma))
    g /= g.sum()
    g1 = -(x / sigma ** 2) * g
    g2 = ((x * x - sigma * sigma) / sigma ** 4) * g

    def sep(a, k_row, k_col):
        out = np.apply_along_axis(
            lambda m: np.convolve(m, k_row, "same"), 0, a)
        return np.apply_along_axis(
            lambda m: np.convolve(m, k_col, "same"), 1, out)

    a = np.asarray(img, dtype=np.float64)
    s2 = sigma ** 2
    return sep(a, g2, g) * s2, sep(a, g1, g1) * s2, sep(a, g, g2) * s2


def ridge_by_orientation(img, angles, sigma: float = 1.4,
                         blend: float = 1.0):
    """How much like a ridge running along each direction each pixel looks.

    The second derivative taken *across* a candidate direction asks the sharp
    question -- is there a line here running that way? -- where plain
    brightness only asks whether anything is here.  Requiring the derivative
    *along* the direction to be small as well takes the response at 45 degrees
    to nearly nothing, so a route cutting diagonally through a row of cilia
    collects almost no support at the crossings.  ``blend`` below one mixes a
    little of the image itself back in, for filaments fatter or fainter than
    the filter expects.
    """
    d_rr, d_rc, d_cc = gauss_hess(img, sigma)
    out = np.empty((len(angles),) + d_rr.shape, dtype=np.float32)
    for i, th in enumerate(angles):
        n_c, n_r = -np.sin(th), np.cos(th)
        t_c, t_r = np.cos(th), np.sin(th)
        across = (n_c * n_c * d_cc + 2.0 * n_c * n_r * d_rc
                  + n_r * n_r * d_rr)
        along = (t_c * t_c * d_cc + 2.0 * t_c * t_r * d_rc
                 + t_r * t_r * d_rr)
        out[i] = np.clip(-across - np.abs(along), 0.0, None)
    peak = float(np.percentile(out.max(axis=0), 99.5)) or 1.0
    out = np.clip(out / peak, 0.0, 1.0)
    if blend >= 1.0:
        return out
    base = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
    return blend * out + (1.0 - blend) * base[None]


def _bend_dtype(bend):
    """The dtype numpy gives a float32 map plus one element of ``bend``.

    float32 under NumPy 1's value-based casting, float64 under NumPy 2's.
    The dynamic programme below reproduces whichever this numpy does, so its
    answer is the one the original per-direction loop gave on this machine.
    """
    return (np.zeros(1, dtype=np.float32) + bend.flat[0]).dtype


def _minpath_dp(cost, offs, adv, dlen, allowed, bend, n_ok, n_len,
                base_rc, seeds, pen_at):
    """The minimal-path dynamic programme, on the CPU.

    The running ``cand < best`` selection over the allowed turns, as it
    always was, but written into preallocated buffers instead of allocating
    four new maps per turn.  Same operations, same order, same dtypes, so
    the same answer.  Returns (choice, floor_at, arg_at).
    """
    n_dir, h, w = cost.shape
    big = np.float32(1e9)
    layers = [np.full((n_dir, h, w), big, dtype=np.float32) for _ in range(3)]
    for k in seeds:
        layers[0][k, base_rc[0], base_rc[1]] = 0.0
    choice = np.zeros((n_len + 1, n_dir, h, w), dtype=np.int8)
    floor_at = np.zeros(n_len + 1, dtype=np.float64)
    arg_at = np.zeros(n_len + 1, dtype=np.int64)
    dt = _bend_dtype(bend)
    prev = [np.nonzero(allowed[k])[0] for k in range(n_dir)]
    cand = np.empty((h, w), dtype=dt)
    best = np.empty((h, w), dtype=dt)
    upd = np.empty((h, w), dtype=bool)
    pick = np.empty((h, w), dtype=np.int8)
    for n in range(1, n_len + 1):
        cur = np.full((n_dir, h, w), big, dtype=np.float32)
        pen = pen_at(n)
        for k in range(n_dir):
            if n - adv[k] < 0 or n < n_ok[k]:
                continue
            src = layers[(n - adv[k]) % 3]
            best.fill(big)
            pick.fill(0)
            for k_prev in prev[k]:
                np.add(src[k_prev], dt.type(bend[k, k_prev]), out=cand)
                np.less(cand, best, out=upd)
                np.copyto(best, cand, where=upd)
                np.copyto(pick, np.int8(k_prev), where=upd)
            d_r, d_c = int(offs[k][0]), int(offs[k][1])
            dst_r = slice(max(0, d_r), h + min(0, d_r))
            src_r = slice(max(0, -d_r), h + min(0, -d_r))
            dst_c = slice(max(0, d_c), w + min(0, d_c))
            src_c = slice(max(0, -d_c), w + min(0, -d_c))
            moved = np.full((h, w), big, dtype=np.float32)
            moved[dst_r, dst_c] = best[src_r, src_c]
            choice[n, k][dst_r, dst_c] = pick[src_r, src_c]
            step_cost = cost[k] if pen is None else cost[k] + pen
            cur[k] = moved + step_cost * dlen[k]
        layers[n % 3] = cur
        j = int(np.argmin(cur))
        floor_at[n] = float(cur.flat[j])
        arg_at[n] = j
    return choice, floor_at, arg_at


# One layer of the minimal-path programme as a single GPU kernel: one thread
# per (direction, row, column) of the new layer.  Launching the numpy
# formulation op by op costs some 350 kernel launches per layer, and on maps
# of this size the GPU would then be no faster than one CPU core.  The
# arithmetic is the original's, operation for operation -- each addition
# and product rounded on its own (the _rn intrinsics are never fused into a
# multiply-add), the turns tried in the same order, a candidate kept only
# when strictly better -- so the layer is bit for bit the CPU's.
_MINPATH_CELL_SRC = r"""
MP_DEVICE void minpath_cell(
    const long long i, const float* src1, const float* src2,
    const float* cost, const float* pen,
    const int* offs_r, const int* offs_c, const int* adv, const int* n_ok,
    const int* prev_start, const int* prev_idx,
    const double* bend_d, const float* bend_f, const float* dlen,
    float* cur, signed char* choice,
    const int n, const int n_dir, const int h, const int w,
    const int has_pen, const int use_double)
{
    const int hw = h * w;
    const int k = (int)(i / hw);
    const int rc = (int)(i - (long long)k * hw);
    const float big = 1e9f;
    if (n - adv[k] < 0 || n < n_ok[k]) {
        cur[i] = big;
        choice[i] = 0;
        return;
    }
    const int r = rc / w;
    const int c = rc - r * w;
    const int sr = r - offs_r[k];
    const int sc = c - offs_c[k];
    float moved = big;
    signed char pick = 0;
    if (sr >= 0 && sr < h && sc >= 0 && sc < w) {
        const float* src = (adv[k] == 1) ? src1 : src2;
        const int s_off = sr * w + sc;
        if (use_double) {
            double best = (double)big;
            for (int q = prev_start[k]; q < prev_start[k + 1]; ++q) {
                const int kp = prev_idx[q];
                const double cand = MP_DADD((double)src[kp * hw + s_off],
                                            bend_d[k * n_dir + kp]);
                if (cand < best) { best = cand; pick = (signed char)kp; }
            }
            moved = MP_D2F(best);
        } else {
            float best = big;
            for (int q = prev_start[k]; q < prev_start[k + 1]; ++q) {
                const int kp = prev_idx[q];
                const float cand = MP_FADD(src[kp * hw + s_off],
                                           bend_f[k * n_dir + kp]);
                if (cand < best) { best = cand; pick = (signed char)kp; }
            }
            moved = best;
        }
    }
    float step = cost[i];
    if (has_pen) step = MP_FADD(step, pen[rc]);
    cur[i] = MP_FADD(moved, MP_FMUL(step, dlen[k]));
    choice[i] = pick;
}
"""

_MINPATH_KERNEL_SRC = r"""
#define MP_DEVICE __device__
#define MP_DADD(a, b) __dadd_rn((a), (b))
#define MP_FADD(a, b) __fadd_rn((a), (b))
#define MP_FMUL(a, b) __fmul_rn((a), (b))
#define MP_D2F(a) __double2float_rn(a)
""" + _MINPATH_CELL_SRC + r"""
extern "C" __global__ void minpath_layer(
    const float* src1, const float* src2, const float* cost, const float* pen,
    const int* offs_r, const int* offs_c, const int* adv, const int* n_ok,
    const int* prev_start, const int* prev_idx,
    const double* bend_d, const float* bend_f, const float* dlen,
    float* cur, signed char* choice,
    const int n, const int n_dir, const int h, const int w,
    const int has_pen, const int use_double)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)n_dir * h * w) return;
    minpath_cell(i, src1, src2, cost, pen, offs_r, offs_c, adv, n_ok,
                 prev_start, prev_idx, bend_d, bend_f, dlen, cur, choice,
                 n, n_dir, h, w, has_pen, use_double);
}
"""

_GPU_KERNELS = {}


def _minpath_kernel(cp):
    kern = _GPU_KERNELS.get("minpath")
    if kern is None:
        kern = cp.RawKernel(_MINPATH_KERNEL_SRC, "minpath_layer",
                            options=("--fmad=false",))
        _GPU_KERNELS["minpath"] = kern
    return kern


def _minpath_dp_gpu(cp, cost, offs, adv, dlen, allowed, bend, n_ok, n_len,
                    base_rc, seeds, pen_at):
    """The same dynamic programme on the GPU: one kernel launch per layer.

    The tube penalty is still computed on the CPU, as before, and copied
    over; the per-layer minimum is located as its first occurrence, as
    numpy's argmin does.  Returns what _minpath_dp returns.
    """
    n_dir, h, w = cost.shape
    big = np.float32(1e9)
    dt = _bend_dtype(bend)
    kern = _minpath_kernel(cp)
    i32 = np.int32
    prev = [np.nonzero(allowed[k])[0] for k in range(n_dir)]
    prev_start = np.zeros(n_dir + 1, dtype=i32)
    prev_start[1:] = np.cumsum([len(p) for p in prev])
    g = dict(
        cost=cp.asarray(np.ascontiguousarray(cost, dtype=np.float32)),
        offs_r=cp.asarray(np.ascontiguousarray(offs[:, 0], dtype=i32)),
        offs_c=cp.asarray(np.ascontiguousarray(offs[:, 1], dtype=i32)),
        adv=cp.asarray(np.ascontiguousarray(adv, dtype=i32)),
        n_ok=cp.asarray(np.ascontiguousarray(n_ok, dtype=i32)),
        prev_start=cp.asarray(prev_start),
        prev_idx=cp.asarray(np.concatenate(prev).astype(i32)
                            if len(prev) else np.zeros(1, i32)),
        bend_d=cp.asarray(np.ascontiguousarray(bend, dtype=np.float64)),
        bend_f=cp.asarray(np.ascontiguousarray(bend.astype(np.float32))),
        dlen=cp.asarray(np.ascontiguousarray(dlen, dtype=np.float32)))
    no_pen = cp.zeros(1, dtype=cp.float32)
    use_double = i32(1 if dt == np.float64 else 0)
    layers = [cp.full((n_dir, h, w), big, dtype=cp.float32) for _ in range(3)]
    for k in seeds:
        layers[0][k, base_rc[0], base_rc[1]] = 0.0
    choice = cp.zeros((n_len + 1, n_dir, h, w), dtype=cp.int8)
    floor_at = np.zeros(n_len + 1, dtype=np.float64)
    arg_at = np.zeros(n_len + 1, dtype=np.int64)
    total = n_dir * h * w
    threads = 256
    blocks = (total + threads - 1) // threads
    for n in range(1, n_len + 1):
        cur = cp.empty((n_dir, h, w), dtype=cp.float32)
        pen = pen_at(n)
        pen_g = no_pen if pen is None else cp.asarray(
            np.ascontiguousarray(pen, dtype=np.float32))
        kern((blocks,), (threads,),
             (layers[(n - 1) % 3], layers[(n - 2) % 3], g["cost"], pen_g,
              g["offs_r"], g["offs_c"], g["adv"], g["n_ok"],
              g["prev_start"], g["prev_idx"], g["bend_d"], g["bend_f"],
              g["dlen"], cur, choice[n],
              i32(n), i32(n_dir), i32(h), i32(w),
              i32(0 if pen is None else 1), use_double))
        layers[n % 3] = cur
        flat = cur.ravel()
        j = int(cp.flatnonzero(flat == flat.min())[0])
        floor_at[n] = float(flat[j])
        arg_at[n] = j
    return cp.asnumpy(choice), floor_at, arg_at


def _gpu_minpath_verified(cp) -> bool:
    """Run a small programme on both devices once; use the GPU only if the
    two agree exactly."""
    ok = _GPU_KERNELS.get("minpath_ok")
    if ok is not None:
        return ok
    try:
        rng = np.random.default_rng(7)
        img = rng.random((26, 30)).astype(np.float32)
        img[:, 14:16] += 2.0
        args = (img, (15.0, 23.0), 18.0, np.deg2rad([-100.0, -80.0]))
        kw = dict(row_limit=np.full(30, 24.0), len_prior=18.0)
        cpu = trace_minimal_path(*args, **kw)
        prev = cpu[0][::-1].copy()
        cpu2 = trace_minimal_path(*args, prev_path=prev, tube=0.35, **kw)
        _GPU_KERNELS["minpath_ok"] = True       # let the GPU path run
        gpu = trace_minimal_path(*args, gpu=True, **kw)
        gpu2 = trace_minimal_path(*args, gpu=True, prev_path=prev, tube=0.35, **kw)
        # A GPU failure inside the tracer falls back to the CPU and clears
        # the flag; agreement then would prove nothing.
        ok = (_GPU_KERNELS["minpath_ok"]
              and np.array_equal(cpu[0], gpu[0]) and cpu[1] == gpu[1]
              and np.array_equal(cpu2[0], gpu2[0]) and cpu2[1] == gpu2[1])
    except Exception as exc:
        _compute_warn("gpu_verify", "GPU tracing unavailable ({}); using the "
                      "CPU cores".format(exc))
        ok = False
    if not ok:
        _compute_warn("gpu_mismatch", "GPU tracing disagreed with the CPU on "
                      "a test problem; using the CPU cores")
    _GPU_KERNELS["minpath_ok"] = bool(ok)
    return bool(ok)


def _minpath_job(job):
    """One frame's trace, for the worker pool: (args, kwargs) -> curve."""
    args, kwargs = job
    return trace_minimal_path(*args, **kwargs)[0]


def trace_many_minimal_paths(jobs, progress=None):
    """trace_minimal_path for many frames at once: the curves, in order.

    On the GPU the frames go through one at a time, each one's programme
    spread over the device; otherwise they are shared out over the CPU
    cores.  Either way each frame gets exactly the answer it would alone.
    """
    jobs = list(jobs)
    cp = _gpu()
    if cp is not None and _gpu_minpath_verified(cp):
        out = []
        for i, (args, kwargs) in enumerate(jobs):
            out.append(trace_minimal_path(*args, gpu=True, **kwargs)[0])
            if progress is not None:
                progress((i + 1) / float(len(jobs)))
        return out
    return parallel_map(_minpath_job, jobs, progress=progress)


def trace_minimal_path(img, base, max_len: float, start_angles,
                       curvature: float = 1.0, max_turn_deg: float = 30.0,
                       row_limit=None, ridge_sigma: float = 1.4,
                       cone_deg: float = 95.0, cone_base_deg: float = 25.0,
                       cone_ramp: float = 0.1,
                       floor: float = 0.02, blend: float = 1.0,
                       tip_support: float = 0.0, gap_factor: float = 12.0,
                       hard_gap_factor: float = 30.0,
                       base_from_image: bool = True, prev_path=None,
                       tube: float = 0.0, tube_cap: float = 6.0,
                       len_prior: float | None = None,
                       len_slack: float = 0.12, arc_cap: bool = True,
                       gpu: bool = False):
    """The cheapest curve from the base along one filament.

    Dynamic programming over (row, column, direction), the layer index being
    arc length rather than step count.  Bending is charged as the integral of
    curvature squared, the elastica energy, which makes a corner far dearer
    than a gradual bend of the same total angle.

    Every step costs ``1/(ridge + floor)`` per unit length and nothing pays.
    A reward for length is the obvious alternative and it is a trap, because a
    layer advances by ``round`` of the step length: a 45-degree step covers
    1.414 pixels but consumes one layer, so with negative on-ridge costs a
    diagonal earns 41 per cent more per layer than a vertical one and the
    optimum is a straight diagonal fan across the carpet.  The reciprocal also
    matters: with a cost linear in the ridge, a diagonal cutting across the
    cilia -- which sits on a filament for 83 per cent of its length -- comes
    within a quarter of the true curve and which wins is down to noise, while
    under the reciprocal a gap costs an order of magnitude more than a ridge.

    The direction is confined to a cone about the base normal, widening with
    arc length, so the curve can neither turn back towards the wall nor set
    off sideways through the strip where all the bases sit.

    The length is not taken on trust either.  A cilium seen side-on
    foreshortens over the beat, and a curve made to run the full nominal
    length borrows a neighbour to make up the difference.  Instead the
    cheapest cost at each arc length is watched as it grows: on a filament it
    grows by about a unit per pixel, and the step where it jumps by an order
    of magnitude is the step that leaves the filament, so the tip is the pixel
    before that.  ``len_prior`` confines the cut to a slack about a length
    known from elsewhere, for the phases where the target is buried in the
    crest of the wave and its ridge response collapses.

    ``prev_path`` charges the curve for straying from another curve *at the
    same arc length* -- a tube around it, not merely a region.  In a carpet
    this dense one frame does not identify a cilium: neighbours run seven
    pixels apart and nearly parallel, so changing cilium costs almost no
    bending and no support.  The frame before does identify it.

    ``gpu`` runs the dynamic programme on the GPU when one is in use (see
    compute_info); the curve is the same.

    Returns the curve as (x, y) pixel pairs and its mean cost.
    """
    offs, ang, dlen, adv = lifted_directions()
    h, w = img.shape
    start = np.asarray(start_angles, dtype=float)
    if base_from_image:
        # Which way does the filament actually leave this base pixel?  A fan
        # of guesses about the normal is too loose: neighbouring bases are a
        # few pixels apart and a twenty-degree error reaches the next cilium
        # within twenty pixels of travel.  The ridge filter already knows.
        r_all = ridge_by_orientation(img, ang, sigma=ridge_sigma, blend=blend)
        b_r, b_c = int(round(base[1])), int(round(base[0]))
        near = r_all[:, max(0, b_r - 4):b_r + 1,
                     max(0, b_c - 2):b_c + 3].mean(axis=(1, 2))
        off = np.abs(np.angle(np.exp(1j * (ang[:, None] - start[None, :]))))
        near = np.where(off.min(axis=1) <= np.deg2rad(60.0), near, -1.0)
        if near.max() > 0:
            start = np.array([ang[int(np.argmax(near))]])

    mid = np.angle(np.exp(1j * start).sum())
    off_mid = np.abs(np.angle(np.exp(1j * (ang - mid))))
    keep = np.nonzero(off_mid <= np.deg2rad(cone_deg))[0]
    offs, ang, dlen, adv = offs[keep], ang[keep], dlen[keep], adv[keep]
    off_mid = off_mid[keep]
    n_dir = len(offs)
    n_len = int(round(max_len))
    # The cone opens over the first ``cone_ramp`` of the length, not the
    # whole of it.  Opening it linearly to the tip meant that half-way up a
    # direction 60 degrees off the normal was still forbidden, and a cilium
    # lying over -- a recovery stroke does by mid-length, and these
    # effective strokes leave the base at 45 degrees -- could not be
    # followed: where it lay flat the path had only "up" available, and up
    # was white space.  The cone's job, keeping the first steps from sliding
    # along the strip of bases, needs only the first tenth or so.
    span = max(np.deg2rad(cone_deg - cone_base_deg), 1e-6)
    n_ok = np.rint(n_len * cone_ramp * np.clip(
        (off_mid - np.deg2rad(cone_base_deg)) / span, 0.0, 1.0)).astype(int)

    d_ang = np.abs(ang[:, None] - ang[None, :])
    d_ang = np.minimum(d_ang, 2 * np.pi - d_ang)
    allowed = d_ang <= np.deg2rad(max_turn_deg) + 1e-9
    bend = curvature * d_ang ** 2 / np.maximum(dlen[:, None], 1e-6)

    ridge = ridge_by_orientation(img, ang, sigma=ridge_sigma, blend=blend)
    cost = 1.0 / (ridge + max(floor, 1e-6))
    if row_limit is not None:
        rows = np.arange(h)[:, None]
        cost = np.where(rows[None] > np.asarray(row_limit)[None, None, :],
                        1e4, cost)

    if prev_path is not None and tube > 0:
        prev = np.asarray(prev_path, dtype=float)
        s_prev = np.concatenate([[0.0], np.cumsum(
            np.hypot(*np.diff(prev, axis=0).T))])
        want = np.linspace(0.0, s_prev[-1], n_len + 1)
        tube_c = np.interp(want, s_prev, prev[:, 0])
        tube_r = np.interp(want, s_prev, prev[:, 1])
    else:
        tube_c = tube_r = None
    rows_g = np.arange(h, dtype=np.float32)[:, None]
    cols_g = np.arange(w, dtype=np.float32)[None, :]

    b_r, b_c = int(round(base[1])), int(round(base[0]))
    b_r = int(np.clip(b_r, 0, h - 1))
    b_c = int(np.clip(b_c, 0, w - 1))
    seeds = [k for k in range(n_dir)
             if np.abs(np.angle(np.exp(1j * (start - ang[k])))).min()
             <= np.deg2rad(30.0)]

    def pen_at(n):
        # Per unit length, not per layer.  A step of length 2.24 advances two
        # layers and one of 1.41 advances one, so a penalty added once per
        # layer is evaded by striding: with a heavy tube weight the curve
        # answered by taking long steps and ended up further from the
        # reference than with no tube at all.
        if tube_r is None:
            return None
        return (tube * np.minimum(
            np.hypot(rows_g - tube_r[n], cols_g - tube_c[n]),
            tube_cap)).astype(np.float32)

    choice, floor_at, arg_at = None, None, None
    if gpu:
        cp = _gpu()
        if cp is not None:
            try:
                choice, floor_at, arg_at = _minpath_dp_gpu(
                    cp, cost, offs, adv, dlen, allowed, bend, n_ok, n_len,
                    (b_r, b_c), seeds, pen_at)
            except Exception as exc:      # out of GPU memory, driver trouble
                _compute_warn("gpu_dp", "GPU tracing failed ({}); using the "
                              "CPU".format(exc))
                _GPU_KERNELS["minpath_ok"] = False
                choice = None
    if choice is None:
        choice, floor_at, arg_at = _minpath_dp(
            cost, offs, adv, dlen, allowed, bend, n_ok, n_len,
            (b_r, b_c), seeds, pen_at)

    rise = np.diff(floor_at)
    early = max(2, int(0.3 * n_len))
    lo_n = early
    hi_n = n_len
    if len_prior:
        lo_n = max(lo_n, int(round(len_prior * (1.0 - len_slack))))
        hi_n = min(n_len, max(lo_n + 1,
                              int(round(len_prior * (1.0 + len_slack)))))
    typical = float(np.median(rise[early - 1:])) or 1.0
    over = np.nonzero(rise[lo_n - 1:hi_n] > gap_factor * max(typical, 1e-6))[0]
    n_end = int(lo_n + over[0] - 1) if len(over) else hi_n
    # The length prior may only hold the cut back where the filament fades
    # gradually -- a cilium vanishing into a crest.  Where the cheapest
    # continuation has to cross white space the filament has plainly ended,
    # and the prior must not force the path on: with a budget half a micron
    # over the drawn arc, every trace on an isolated cilium grew a spur
    # straight up into white, and one crossed a ten-pixel gap to run along
    # its neighbour's tip.  A rise of a full background step, well above
    # anything a fading ridge produces, ends the curve wherever it occurs.
    hard = np.nonzero(rise[early - 1:hi_n] > hard_gap_factor
                      * max(typical, 1e-6))[0]
    if len(hard):
        n_end = min(n_end, int(early + hard[0] - 1))
    n_end = int(np.clip(n_end, early, hi_n))

    k, r, c = np.unravel_index(int(arg_at[n_end]), (n_dir, h, w))
    score = floor_at[n_end] / max(n_end, 1)
    pts = [(c, r)]
    dirs = [int(k)]
    n = n_end
    while n > 0:
        k_prev = int(choice[n, k, r, c])
        d_r, d_c = int(offs[k][0]), int(offs[k][1])
        r -= d_r
        c -= d_c
        n -= int(adv[k])
        # A window narrower than the cilium is long lets the best curve reach
        # the border, where a step's predecessor can lie outside it; stop
        # rather than index off the array.
        if not (0 <= r < h and 0 <= c < w):
            break
        pts.append((c, r))
        dirs.append(k_prev)
        k = k_prev
    pts = np.array(pts[::-1], dtype=float)
    dirs = np.array(dirs[::-1], dtype=int)
    # Trimming the tail back to the last pixel with ridge support was how the
    # tip was found before the cost-rise cut above, and keeping both is
    # harmful: on microscope frames the support dips mid-cilium and the trim
    # threw away most of the curve, returning a ninth of a cilium.  Off by
    # default, and the cut decides the tip.
    if arc_cap:
        # Layers count rounded step lengths, so a curve made mostly of
        # 45-degree steps covers 1.414 pixels per layer and comes back up to
        # 40 per cent longer than the arc length asked for -- long enough
        # that the waveform fit refused it, since a shape cannot be
        # represented at a length shorter than its own trace.  Cut to the arc
        # length actually requested.
        run = np.concatenate([[0.0], np.cumsum(
            np.hypot(*np.diff(pts, axis=0).T))])
        if run[-1] > max_len:
            keep_n = int(np.searchsorted(run, max_len)) + 1
            pts = pts[:max(3, keep_n)]
    if tip_support > 0:
        sup = ridge[dirs, pts[:, 1].astype(int), pts[:, 0].astype(int)]
        keep_to = np.nonzero(sup >= tip_support)[0]
        if len(keep_to) > 2:
            pts = pts[:keep_to[-1] + 1]
    return pts, float(score)


# Above this fraction of pixels at the extremes a sequence is treated as a
# drawing (or a segmentation): black ink on a plain background.  A blurred
# rendering with a grey cell body under the carpet came out at 0.49, half
# its pixels plain background, and was handled as a microscope frame, which
# put its base in mid-air; 0.45 takes it in while the recordings here stay
# well clear at 0.09 to 0.25.
DRAWN_BIMODAL_FRACTION = 0.45


def frames_are_drawn(video) -> bool:
    """Is this a drawing or segmentation rather than a microscope recording?"""
    return bimodal_frames(video) > DRAWN_BIMODAL_FRACTION


def enhancement_draws(video) -> bool:
    """Does the enhancement turn the frames into drawings -- the line
    filter, the top-hat, or a network that draws the cilia?"""
    enh = getattr(video, "enhance", None) or {}
    return bool(float(enh.get("lines") or 0.0) > 0
                or float(enh.get("tophat") or 0.0) > 0
                or clean_model_path(enh.get("model")))


def frames_drawn_source(video) -> bool:
    """A drawing or segmentation *as recorded*: its wall is the bottom of
    the ink (wall_line).  A recording that the enhancement has drawn keeps
    the recording's tissue surface (detect_surface on the raw mean): the
    wall rule on the drawing of input_images1 put the base at the frame's
    bottom, 25 um from the cilium."""
    return frames_are_drawn(video) and not enhancement_draws(video)


def bimodal_frames(video, samples: int = 5, raw: bool = False) -> float:
    """What fraction of pixels sit at one extreme of the range or the other
    (``raw``: of the frames before enhancement).

    A drawn or segmented wave is nearly all background and ink; a microscope
    frame is not.  It decides which image map to trace on: subtracting the
    mean is right when there is static structure to remove, and wrong on a
    drawing, where the mean is itself dark wherever the cilia have swept and
    the difference goes to nothing in exactly the crowded region that matters.
    """
    frames = np.linspace(video.first, video.last, samples).astype(int)
    fracs = []
    for f in frames:
        img = np.asarray(video._raw(int(f)) if raw else video.get(int(f)),
                         dtype=np.float32)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)
        lo, hi = np.percentile(img, [1.0, 99.0])
        rng = max(float(hi - lo), 1e-6)
        z = (img - lo) / rng
        fracs.append(float(((z < 0.1) | (z > 0.9)).mean()))
    return float(np.median(fracs))


def wall_line(mean_img, thresh: float = 0.15, smooth: int = 61,
              valid=None):
    """The wall row of a drawn carpet, for every column.

    detect_surface looks for where structure begins from the top, which is
    the tip envelope when the cilia are drawn without a cell body under them.

    What marks a base is that it is inked in every frame: the body above it
    sweeps, so those pixels are grey in the mean, but the base pixel is dark
    in the mean as it is in each frame.  The wall is therefore the lowest
    row of the persistently dark pixels.  Earlier rules read the lowest ink
    per column and took a smoothed value or a percentile across columns;
    both put the wall on cilia bodies -- 22 pixels high when the columns
    between bases dominated, 20 pixels high when the margins did, 7 pixels
    high on a sparse carpet where base columns were 3 per cent of all
    columns.  A drawn carpet's wall is one horizontal row, and is returned
    for every column.
    """
    img = np.asarray(mean_img, dtype=np.float32)
    dark = 1.0 - img / max(float(img.max()), 1e-6)
    if valid is not None:
        dark = np.where(valid, dark, 0.0)
    h, w = dark.shape
    rows = np.arange(h)[:, None]
    persistent = dark > 0.8
    if persistent.sum() >= 4:
        low = np.where(persistent, rows, -1).max(axis=0)
        low = low[low >= 0]
        return np.full(w, float(np.percentile(low, 75)))
    # Faint or sub-pixel drawing: no pixel is dark in every frame.  Fall
    # back to the lowest ink at all, taken near its maximum.
    r = np.where(dark > thresh, rows, -1).max(axis=0).astype(float)
    good = r >= 0
    if not good.any():
        return np.full(w, float(h - 1))
    return np.full(w, float(np.percentile(r[good], 99)))


def base_runs(fmap, row: int, thresh: float = 0.4, widths: bool = False):
    """Column centres of the bright runs along one row -- the cilium bases.

    With ``widths`` the run lengths come back too, which is the drawn line
    width where the bases stand close to vertical.
    """
    row = int(np.clip(row, 0, fmap.shape[0] - 1))
    on = (fmap[row] > thresh).astype(np.int8)
    edge = np.nonzero(np.diff(np.concatenate([[0], on, [0]])) != 0)[0]
    centres = np.array([(edge[i] + edge[i + 1] - 1) / 2.0
                        for i in range(0, len(edge), 2)], dtype=float)
    if widths:
        return centres, np.array([edge[i + 1] - edge[i]
                                  for i in range(0, len(edge), 2)], dtype=float)
    return centres


def track_cilium(video, base_x: float, length_px: float, frames=None,
                 half_width: int | None = None, refine: bool = True,
                 curvature: float = 1.0, ridge_sigma: float = 1.4,
                 tube: float = 0.35, ppum: float | None = None,
                 bin_factor: int | None = None, target_ppum: float = 20.0,
                 upsample: int | None = None, blend: float = 1.0,
                 target_spacing: float = 7.0, pass1_slack: float = 0.12,
                 progress=None):
    """Follow the one cilium rooted at ``base_x`` through the frames.

    Two passes, not a chain.  Chaining each frame to the one before it
    propagates a mistake for as long as the tube holds -- measured on the
    drawn wave, chaining put 0.85 of the curve on the right filament against
    0.92 for frames traced independently.  So the first pass traces each
    frame on its own, and the second re-traces it with a tube around the mean
    of its two neighbours from the first pass, and the median length of all of
    them.  That keeps the independence and still halves the frame-to-frame
    jitter of the tip.

    Only the frames asked for are returned, but each of their immediate
    neighbours is traced too, since the second pass leans on them.

    Returns a dict with the per-frame curves in full-frame pixel coordinates.
    """
    frame_list = list(range(video.first, video.last + 1))
    total = max(1, len(frame_list))
    acc = None
    _prefetch(video)
    for i, f in enumerate(frame_list):
        if progress is not None and i % 8 == 0:
            progress(0.1 * i / total, "reading frames")
        img = np.asarray(video.get(f), dtype=np.float32)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)
        acc = img.copy() if acc is None else acc + img
    mean_full = acc / float(total)
    drawn = frames_are_drawn(video)
    drawn_src = drawn and not enhancement_draws(video)   # see frames_drawn_source
    peak = max(float(mean_full.max()), 1e-6)

    # Working resolution.  On a drawn carpet it is set by the spacing of the
    # bases: about seven pixels between neighbours after resampling, which
    # the ridge filter separates cleanly and which keeps the dynamic
    # programme -- cubic in the resolution -- affordable.  On a 600-pixel
    # rendering with 11-pixel spacing the earlier rule binned by two and
    # then enlarged by two again, four times the work for nothing, and one
    # trace took over four seconds.  Microscope frames keep the old
    # pixels-per-micron rule.
    if bin_factor is not None or upsample is not None:
        binf = max(1, int(bin_factor or 1))
        work_scale = float(max(1, int(upsample or 1))) / binf
    elif drawn_src:
        # Coarse enough that neighbours stand about target_spacing apart,
        # but never so coarse that the cilium itself is short of 120 pixels
        # or its line thinner than 1.5: on a sparse carpet with 66 pixels
        # between cilia the spacing alone took the frames to a quarter
        # size, a 6-micron cilium became 26 pixels with half-pixel lines,
        # the base detector lost the target's base and snapped to a
        # neighbour, and the fitted beat hopped between cilia.
        wall0 = wall_line(static_mean(video, mean_full))
        dark0 = 1.0 - mean_full / peak
        runs, run_w = base_runs(dark0, int(np.median(wall0)) - 2, thresh=0.3,
                                widths=True)
        spacing = float(np.median(np.diff(runs))) if len(runs) > 3 else 0.0
        width_px = float(np.median(run_w)) if len(run_w) else 2.0
        want = [1.0 if spacing <= 0 else target_spacing / spacing,
                120.0 / max(length_px, 1.0),
                1.5 / max(width_px, 1e-6)]
        work_scale = float(np.clip(max(want), 0.2, 1.0))
        binf = 1
    else:
        pp = ppum if ppum else video_ppum(video)
        binf = max(1, int(pp // max(target_ppum, 1.0))) if pp else 1
        work_scale = 1.0 / binf
    per_px = 1.0 / work_scale                     # source pixels per working pixel
    mean_img = resample_frame(mean_full, work_scale)
    h, w = mean_img.shape
    valid = resample_frame(video.valid.astype(np.float32), work_scale) > 0.5
    wall = (wall_line(static_mean(video, mean_img), valid=valid) if drawn_src
            else detect_surface(static_mean(video, mean_img), valid=valid))

    base_b = base_x / per_px
    len_b = length_px / per_px
    if half_width is None:
        # Wide enough for a cilium lying right over: half a length was not,
        # and on a small field the window came out narrower than the curve.
        half_width = int(max(30, min(0.45 * w, 1.05 * len_b)))
    c0 = int(np.clip(base_b - half_width, 0, w - 4))
    c1 = int(np.clip(base_b + half_width + 1, c0 + 8, w))
    wall_here = float(np.median(wall[c0:c1]))
    r0 = int(np.clip(wall_here - 1.1 * len_b, 0, h - 4))
    r1 = int(np.clip(wall_here + 4, r0 + 8, h))
    y_base = wall_here - r0 - 2.0
    row_limit = np.full(c1 - c0, wall_here - r0 + 1.0)

    def frame_map(f):
        img = np.asarray(video.get(f), dtype=np.float32)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)
        img = resample_frame(img, work_scale)
        m = (1.0 - img / peak) if drawn else np.abs(img - mean_img)
        m = np.clip(m[r0:r1, c0:c1], 0.0, None)
        scale = float(np.percentile(m[m > 0], 99)) if (m > 0).any() else 0.0
        return m / scale if scale > 0 else m

    want = [int(f) for f in (frame_list if frames is None else frames)
            if video.first <= int(f) <= video.last]
    if not want:
        raise ValueError("no frames to trace")
    # Neighbours for the second pass are the nearest frames *traced*, so a
    # strided set of frames costs what it says rather than three times it.
    need = sorted(set(want))
    start_angles = np.deg2rad(np.linspace(-120.0, -60.0, 7))
    kw = dict(curvature=curvature, max_turn_deg=30.0, row_limit=row_limit,
              ridge_sigma=ridge_sigma, cone_base_deg=25.0, floor=0.02,
              blend=blend)

    def base_at(fmap):
        runs = base_runs(fmap, int(round(y_base)))
        if not len(runs):
            return base_b - c0
        return float(runs[np.argmin(np.abs(runs - (base_b - c0)))])

    def step_report(lo, hi, label, count):
        if progress is None:
            return None
        return lambda frac: progress(lo + (hi - lo) * frac, "{} {} of {}".format(
            label, int(round(frac * count)), count))

    # Every frame of a pass is traced independently of the others, so the
    # frames of each pass are shared out over the cores (or run on the GPU);
    # see trace_many_minimal_paths.
    maps = {}
    for f in need:
        maps[f] = frame_map(f)
    # The model's length is a real prior, so the cut is allowed to shorten
    # the curve by a third at most.  Left free, on microscope frames --
    # where the ridge support is patchy rather than drawn -- it fired
    # almost at once and returned a tenth of a cilium.
    jobs = [((maps[f], (base_at(maps[f]), y_base), len_b, start_angles),
             dict(len_prior=len_b, len_slack=pass1_slack, **kw)) for f in need]
    first = dict(zip(need, trace_many_minimal_paths(
        jobs, progress=step_report(0.1, 0.55, "tracing", len(need)))))

    out = dict(first)
    if refine and len(need) > 2:
        med = float(np.median([
            np.hypot(*np.diff(first[f], axis=0).T).sum() for f in need]))
        order = sorted(first)
        jobs = []
        for f in want:
            k = order.index(f)
            near = [first[order[j]] for j in (k - 1, k + 1)
                    if 0 <= j < len(order)]
            # The length prior applies whether or not this frame has
            # neighbours in the set: asked for a handful of frames spread
            # across the record, the second pass had no tube and no prior and
            # cut one curve to a third of the others.
            extra = dict(len_prior=med, len_slack=0.12)
            if near:
                extra.update(prev_path=np.mean(
                    [_resample_curve(p, 80)[0] for p in near], axis=0),
                    tube=tube, tube_cap=6.0)
            extra.update(kw)
            jobs.append(((maps[f], (base_at(maps[f]), y_base), len_b,
                          start_angles), extra))
        out.update(zip(want, trace_many_minimal_paths(
            jobs, progress=step_report(0.55, 1.0, "refining", len(want)))))

    shift = np.array([c0, r0], dtype=float)
    curves = {f: (out[f] + shift) * per_px for f in want}
    return {"frames": want, "curves": curves, "wall": wall * per_px,
            "window": (c0 * per_px, r0 * per_px, c1 * per_px, r1 * per_px),
            "drawn": drawn, "bin_factor": binf, "scale": work_scale,
            "lengths": np.array([np.hypot(*np.diff(curves[f], axis=0).T).sum()
                                 for f in want])}


def save_trace_xyt(path: str, curves, times_ms, ppum: float,
                   base_xy=None, image_h: int | None = None) -> str:
    """Write per-frame centrelines in the four-column index, x, y, time form.

    Microns and milliseconds, y increasing away from the wall, the base of
    each curve at the origin -- the same layout load_trace_xyt reads and the
    reference fitting script expects.
    """
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("index,x_um,y_um,time_ms\n")
        for curve, t_ms in zip(curves, times_ms):
            pts = np.asarray(curve, dtype=float)
            ox, oy = (pts[0] if base_xy is None else np.asarray(base_xy, float))
            for i, (x, y) in enumerate(pts):
                fh.write("{:d},{:.5f},{:.5f},{:.4f}\n".format(
                    i + 1, (x - ox) / ppum, (oy - y) / ppum, t_ms))
    return path


def cycle_frames(video, model, n_out: int = 20, all_cycles: bool = True):
    """Pick frames to trace, spread over evenly spaced cycle phases.

    With ``all_cycles`` the samples are spread over the whole loaded record, so
    every beat in it contributes and per-frame tracing noise averages out; the
    phases still land evenly because the sampling interval and the period are
    unrelated.  Otherwise a single period from the main-cycle start is used.
    """
    lo, hi = int(video.first), int(video.last)
    period = model.period_frames
    start = model.macysta
    if all_cycles or not (period > 0) or start + period > hi + 1 or start < lo:
        n = int(min(n_out, hi - lo + 1))
        return [int(round(f)) for f in np.linspace(lo, hi, max(n, 2))]
    return [int(round(start + k * period / n_out)) for k in range(n_out)]


def auto_trace(video, model, frames=None, n_out: int = 20,
               step: float | None = None, look: float | None = None,
               look_samples: int = 8,
               bg_sigma: float = 6.0, smooth: float = 1.0,
               prior_weight: float = 0.25, turn_weight: float = 0.0,
               max_turn: float = 12.0, mask_pad: int = 3, ramp: float = 0.0,
               motion_weight: float = 0.0, start_fan: float = 45.0,
               method: str = "auto", max_traces: int = 120,
               info: dict | None = None, progress=None):
    """Trace the cilium at the specification's base in every frame of the record.

    Every frame is traced, in order, each one starting from the previous
    frame's path, because tracing frames independently lets the walk change
    cilium between samples.  ``frames``, or a sample of ``n_out`` of them, is
    then returned.

    Returns ``(steps, curves, times_ms)`` in the same form as the hand traces
    read from cilium_trace_px.csv, so the same fitting path serves both.

    ``method`` picks the tracer.  "minpath" solves for the whole curve at once
    (see track_cilium); "walk" is the greedy walk below, which decides each
    step from what it can see ahead and cannot undo a wrong turn.  "auto",
    the default, takes the minimal path on drawn or segmented frames and the
    walk on microscope frames, because measurement says each wins on one and
    loses on the other and by a wide margin.  On the drawn wave the walk
    follows the tip envelope across the carpet; on a 211x95 microscope
    recording the minimal path returned curves bending 13 degrees where the
    walk found 55, and the fit residual went from 0.13 to 0.43 microns,
    because a patchy ridge leaves the elastica term deciding the shape.
    """
    if method == "auto":
        method = "minpath" if frames_are_drawn(video) else "walk"
    if method == "minpath":
        x_base = model.pqd.getf("vid_cil_x0", video.shape[1] / 2.0)
        length_px = max(6.0, model.length * model.ppum)
        wanted = frames if frames is not None else cycle_frames(
            video, model, n_out)
        wanted = [int(f) for f in wanted
                  if video.first <= int(f) <= video.last]
        if not wanted:
            raise ValueError("no frames to trace")
        # A bounded number of traces, spread over the whole record.  Every
        # frame of a long record is more than the fold can use -- 24 phase
        # bins want a handful of traces each -- and on a 600-pixel rendering
        # tracing all 301 frames twice ran for most of an hour.  Striding
        # keeps the phase coverage, because successive beats fall at
        # different offsets within the stride.
        if len(wanted) > max_traces:
            stride = int(np.ceil(len(wanted) / float(max_traces)))
            wanted = wanted[::stride]
        run = track_cilium(video, x_base, length_px, frames=wanted,
                           ppum=model.ppum, progress=progress)
        if info is not None:
            info["surface"] = run["wall"]
            info["window"] = run["window"]
            info["drawn"] = run["drawn"]
            info["traced_px"] = run["lengths"]
            info["bin_factor"] = run["bin_factor"]
            info["method"] = "minpath"
            # There is no fading ridge to report here: the minimal path finds
            # the tip from where cheap extension stops, so the length it
            # actually measured is the honest equivalent.
            info["fade_um"] = float(np.median(run["lengths"])) / max(
                model.ppum, 1e-9)
        fps = model.fps or 1.0
        steps, curves, times = [], [], []
        for f in wanted:
            curve = run["curves"].get(f)
            if curve is None or len(curve) < 3:
                continue
            steps.append(len(steps))
            curves.append(np.asarray(curve, dtype=float))
            times.append(f / fps * 1000.0)
        if len(curves) < 4:
            raise ValueError("could not trace enough frames (got {})".format(
                len(curves)))
        return steps, curves, times

    # Streamed, one frame at a time: the background is the average of the
    # frames, which takes a single pass to accumulate, and each frame's map can
    # then be built and traced without ever holding the stack.  Holding it
    # forced a choice between memory and resolution -- 247 frames of 1587x685
    # is a gigabyte -- and reducing the resolution merged neighbouring cilia
    # that are 20 pixels apart into one another, which is exactly what tracing
    # must not do.
    binf = 1
    frame_list = list(range(video.first, video.last + 1))
    total = max(1, len(frame_list))
    acc = None
    _prefetch(video)
    for i, f in enumerate(frame_list):
        if progress is not None and i % 8 == 0:
            progress(0.35 * i / total, "reading frames")
        img = np.asarray(video.get(f), dtype=np.float32)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)
        acc = img.copy() if acc is None else acc + img
    mean_img = acc / float(total)
    h, w = mean_img.shape
    valid = video.valid
    surface = detect_surface(static_mean(video, mean_img), valid=valid)

    # Which side of the background a cilium falls on, from the skew of the
    # deviation inside the beating band (see structure_stack for why).
    rows_ix = np.arange(h)[:, None]
    band = (rows_ix < (surface[None, :] - mask_pad)) & valid
    sign = 0.0
    if band.any():
        probe_frames = frame_list[::max(1, total // 12)][:12]
        dev = []
        for f in probe_frames:
            img = np.asarray(video.get(f), dtype=np.float32)
            if img.ndim == 3:
                img = img[..., :3].mean(axis=2)
            dev.append((img - mean_img)[band])
        dev = np.concatenate(dev)
        sd = float(np.std(dev))
        if sd > 0:
            skew = float(np.mean((dev / sd) ** 3))
            sign = 0.0 if abs(skew) < 0.05 else np.sign(skew)
    above = band.astype(np.float32)

    def frame_map(f):
        img = np.asarray(video.get(f), dtype=np.float32)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)
        d = img - mean_img
        m = (np.abs(d) if sign == 0.0 else np.clip(sign * d, 0.0, None)) * above
        if smooth > 0:
            m = gaussian_blur(m, smooth)
        scale = float(np.percentile(m[m > 0], 90)) if (m > 0).any() else 0.0
        return m / scale if scale > 0 else m

    if info is not None:
        info["surface"] = surface
    # Traces may not cross into the cell.  A couple of pixels of slack absorbs
    # the surface detection's own error without letting a walk run downhill
    # into the tissue.
    row_limit = np.where(surface < h - 1, surface + 2.0, float(h))

    x0 = model.pqd.getf("vid_cil_x0", w * binf / 2.0) / binf
    y0 = (model.image_h - model.pqd.getf("vid_cil_y0_invy",
                                         model.image_h / 2.0)) / binf
    # The traces span exactly the model's length, so target and model cover the
    # same arc and can be compared fraction by fraction.  The walk goes further
    # than that only so the ridge can be watched fading past the tip, which is
    # reported as advice; it is not used to change the length.  Measuring the
    # length from the traces and then adopting it was tried and abandoned: on
    # the one recording with a trusted hand fit it scattered between 5.1 and
    # 10.3 microns around a true 6.0, depending only on where it started.
    length_px = max(6.0, model.length * model.ppum / binf)
    if step is None:
        step = max(1.0, length_px / 100.0)
    expect = length_px / step
    n_steps = int(round(1.4 * expect))
    min_steps = max(4, int(round(0.5 * expect)))
    fps = model.fps or 1.0

    # How far ahead each candidate direction is scored, in pixels.  Long enough
    # that a curved ridge departs from a straight one by more than the ridge is
    # wide -- for a bend of radius R and a ridge of half-width w that is about
    # sqrt(2*R*w) -- and no longer, because a probe that reaches past the
    # neighbouring cilia averages over them and the walk then follows the grain
    # of the bundle instead of one filament.  R comes from the tightest bend a
    # cilium is allowed (MAX_TRACE_TURN over its length), w from the image
    # scale.  On a densely packed drawn wave this gives 32 px where a fifth of
    # the cilium would have given 67, and 67 spanned three neighbours.
    if look is None:
        radius = length_px / max(np.deg2rad(MAX_TRACE_TURN), 1e-6)
        width = max(2.0, 0.05 * model.ppum)
        look = float(np.clip(np.sqrt(2.0 * radius * width), 4.0,
                             0.25 * length_px))

    # Start every trace along the model's base axis (the surface normal of
    # the beat's frame), give or take start_fan.  A cilium is anchored, so
    # its base direction is the same in every frame; a trace free to leave
    # the base in any direction produces targets the beat cannot represent
    # at all: matching a target that leans 40 degrees right at the base
    # would need unbounded curvature in the first fraction of a micron, and
    # the fit answers instead with a nearly straight cilium at those phases.
    # This is what made some frames refuse to bend.
    base_angle = base_tangent_angle(model)
    fan = np.deg2rad(start_fan)
    start_angles = base_angle + np.linspace(-fan, fan, 19)

    wanted = frames if frames is not None else cycle_frames(video, model, n_out)
    wanted = [int(f) for f in wanted
              if video.first <= int(f) <= video.last]
    if not wanted:
        raise ValueError("no frames to trace")

    all_frames = list(range(video.first, video.last + 1))

    def sweep(walk, floor):
        out, vals = {}, {}
        prior = None
        # The walk is a chain -- each frame starts from the path before -- so
        # it runs in order; the filtered frames it walks on do not depend on
        # one another and are prepared ahead on the other cores.
        maps = lookahead_map(frame_map, all_frames)
        for i, (f, img) in enumerate(zip(all_frames, maps)):
            if progress is not None and i % 4 == 0:
                progress(0.35 + 0.65 * i / len(all_frames),
                         "tracing {} of {}".format(i + 1, len(all_frames)))
            try:
                path = trace_cilium(img, (x0, y0), walk, step=step,
                                    max_turn=max_turn, prior=prior,
                                    prior_weight=prior_weight,
                                    turn_weight=turn_weight,
                                    start_angles=start_angles,
                                    min_steps=floor, row_limit=row_limit,
                                    look=look, look_samples=look_samples)
            except ValueError:
                continue
            out[f] = path
            vals[f] = _sample_bilinear(img, path[:, 0], path[:, 1])
            prior = path
        return out, vals

    traced, values = sweep(n_steps, min_steps)
    if traced:
        fade = tip_from_profiles(list(traced.values()), list(values.values()))
        span = int(round(expect))
        for f in list(traced):
            traced[f] = traced[f][:span + 1]
        if info is not None:
            info["fade_steps"] = fade
            info["fade_um"] = fade * step * binf / max(model.ppum, 1e-9)
            info["bin_factor"] = binf
            info["walked_steps"] = n_steps

    steps, curves, times = [], [], []
    for f in wanted:
        if f not in traced or len(traced[f]) < 3:
            continue
        steps.append(len(steps))
        curves.append(traced[f] * binf)
        times.append(f / fps * 1000.0)
    if len(curves) < 4:
        raise ValueError("could not trace enough frames (got {})".format(
            len(curves)))
    return steps, curves, times


def _resample_curve(points, n: int):
    """Resample an ordered polyline at n equal fractions of its arc length."""
    seg = np.hypot(*np.diff(points, axis=0).T)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    if s[-1] <= 0:
        return np.repeat(points[:1], n, axis=0), 0.0
    target = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(target, s, points[:, 0]),
                            np.interp(target, s, points[:, 1])]), float(s[-1])


def load_trace_xyt(path: str):
    """Read a trace in the four-column form: index, x, y, time.

    Real units, one row per digitised point, grouped by the time column; the
    first point of each time is the base.  This is the format the reference
    MATLAB fitting script expects.
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip().replace(",", " ")
            if not line or line[0] not in "-+.0123456789":
                continue
            parts = line.split()
            if len(parts) >= 4:
                rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not rows:
        raise ValueError("{}: no numeric rows".format(path))
    data = np.array(rows, dtype=float)
    times, inverse = np.unique(data[:, 2], return_inverse=True)
    curves = [data[inverse == i, :2] for i in range(len(times))]
    return times, curves


def align_beat_frame(curves):
    """Move every traced shape into the beat frame, the way the reference does.

    Each shape is shifted so its base is the origin and rotated so the mean
    base direction points along +y; the coordinate across the beat is then
    taken as the *distance* from the beat axis rather than a signed offset, so
    the cilium can never be placed below the surface it grows from.

    Returns the aligned curves and the beat direction that was used.
    """
    bases = np.array([np.asarray(c, dtype=float)[0] for c in curves])
    seconds = np.array([np.asarray(c, dtype=float)[min(1, len(c) - 1)]
                        for c in curves])
    d = seconds - bases
    base_theta = float(np.mean(np.arctan2(d[:, 1], d[:, 0]))) - np.pi / 2.0
    beat = np.array([np.cos(base_theta), np.sin(base_theta)])

    out = []
    for c, b in zip(curves, bases):
        rel = np.asarray(c, dtype=float) - b
        along = rel @ beat
        across = np.linalg.norm(rel - np.outer(along, beat), axis=1)
        out.append(np.column_stack([along, across]))
    return out, base_theta


def resample_unit_length(curves, num_locs: int = 20):
    """Sample each shape at equal arc-length fractions and scale it to length 1.

    Scaling every shape to unit length before fitting is what the reference
    method does, and it matters: the traced length wobbles from frame to frame,
    and without this that wobble enters the fit as if it were beat shape.
    """
    out, lengths = [], []
    for c in curves:
        c = np.asarray(c, dtype=float)
        pts, arc = _resample_curve(c, int(num_locs))
        if arc <= 0:
            continue
        out.append(pts / arc)
        lengths.append(arc)
    return np.array(out), np.array(lengths)


def fit_beat_reference(curves, n_modes: int | None = None,
                       num_locs: int = 20, n_space: int = 3,
                       period_samples: int | None = None,
                       damping: float | None = None):
    """Fit the beat the way the reference MATLAB script does.

    The order of operations is the point of it.  Each shape is aligned, sampled
    at equal fractions of its own arc length and scaled to unit length; then a
    Fourier series in time is fitted *independently at every node* by taking
    the discrete transform over exactly one period; and only then is a
    polynomial in s fitted to those Fourier coefficients, with no constant term
    so the base stays fixed.  Finally the whole thing is rescaled so the mean
    length over a period is one.

    Fitting positions globally, as a single least-squares problem in all the
    coefficients at once, is not the same thing: the time series at each node
    is then only fitted as well as the shared spatial polynomial allows, and
    the compromise flattens the strongly curved phases.  Doing time first, per
    node, lets each node follow its own trajectory exactly before any spatial
    smoothing is applied.

    ``damping`` shrinks the high time harmonics, mode k by 1/(1 + damping*k^2).
    The phase samples are equally spaced, so the harmonic basis is orthogonal
    and that shrinkage is exactly the ridge-regularised solution -- no matrix
    solve needed.  It is there because the traces are least reliable in the
    fastest part of the stroke: on one recording consecutive phase bins there
    differed by 2.4 to 3.4 microns while elsewhere they differed by 0.5, and a
    cilium cannot move three microns in one frame.  Retaining every harmonic
    reproduces those jumps faithfully; damping follows the sweep through them.
    Passing None applies no damping; isolated bad bins are repaired instead
    (see _repair_spikes), which is what the damping was for and does not
    flatten the stroke.

    ``curves`` are ordered base -> tip, one per time step, equally spaced in
    time and covering whole beats.
    """
    samples, lengths = resample_unit_length(curves, num_locs=num_locs)
    n_times = len(samples)
    if n_times < 4:
        raise ValueError("need at least 4 traced shapes")

    per_period = int(period_samples or n_times)
    per_period = max(4, min(per_period, n_times))
    max_modes = 1 + int(np.ceil(0.5 * (per_period - 1)))
    n_modes = int(n_modes or max_modes)
    n_modes = max(1, min(n_modes, max_modes))

    block = samples[:per_period]                      # (per_period, num_locs, 2)
    block, repaired = _repair_spikes(block)
    cos_c = np.zeros((n_modes, num_locs, 2))
    sin_c = np.zeros((n_modes, num_locs, 2))
    spectrum = np.fft.fft(block, axis=0) / per_period
    cos_c[0] = spectrum[0].real
    for k in range(1, n_modes):
        cos_c[k] = 2.0 * spectrum[k].real
        sin_c[k] = -2.0 * spectrum[k].imag
    # The top harmonic of an even-length record has no conjugate partner.
    if n_modes == max_modes and max_modes != 1 + (per_period - 1) / 2:
        cos_c[-1] *= 0.5

    if damping is None:
        damping = 0.0
    if damping > 0:
        cos_c *= _harmonic_shrink(n_modes, damping)[:, None, None]
        sin_c *= _harmonic_shrink(n_modes, damping)[:, None, None]

    out = _beat_from_harmonics(cos_c, sin_c, num_locs, n_space)
    sample_phases = np.arange(per_period) / float(per_period)
    err = _beat_residuals(out, block, sample_phases, num_locs)
    return {
        "coeffs": out,
        "n_modes": n_modes,
        "max_modes": max_modes,
        "per_period": per_period,
        "lengths": lengths,
        "mean_length": float(np.mean(lengths)) if len(lengths) else 0.0,
        "samples": samples,
        "repaired_bins": repaired,
        "resid": float(np.mean(err)),
        "resid_um": float(np.mean(err) * (np.mean(lengths) if len(lengths)
                                          else 1.0)),
    }


def _beat_from_harmonics(cos_c, sin_c, num_locs: int, n_space: int) -> dict:
    """Polynomial in s through per-node harmonics, then mean unit length.

    The second half of the reference method: with the time series at every
    node already reduced to cosine and sine coefficients, fit s^1..s^n_space
    to those coefficients (no constant term, so the base stays fixed) and
    rescale the whole table so the beat's mean length over a period is one.
    """
    n_modes = cos_c.shape[0]
    s = np.arange(1, num_locs) / float(num_locs - 1)
    S = np.stack([s ** m for m in range(1, int(n_space) + 1)], axis=1)
    out = {k: np.zeros((int(n_space), n_modes)) for k in FOURIER_KEYS}
    for k in range(n_modes):
        for comp, (a_key, b_key) in enumerate((("Ax", "Bx"), ("Az", "Bz"))):
            out[a_key][:, k] = np.linalg.lstsq(S, cos_c[k, 1:, comp],
                                               rcond=None)[0]
            out[b_key][:, k] = np.linalg.lstsq(S, sin_c[k, 1:, comp],
                                               rcond=None)[0]
    mean_len = _mean_fourier_length(out)
    if mean_len > 0:
        for key in FOURIER_KEYS:
            out[key] = out[key] / mean_len
    return out


def _beat_residuals(coeffs, samples, phases, num_locs: int):
    """RMS distance of each unit-length sample shape from the fitted beat."""
    wave = FourierWave(coeffs, length=1.0, cbf=1.0)
    s_eval = np.linspace(0.0, 1.0, num_locs)
    err = []
    for shape, ph in zip(samples, phases):
        x, z = wave.position(s_eval, float(ph))
        err.append(float(np.sqrt(np.mean((x - shape[:, 0]) ** 2
                                         + (z - shape[:, 1]) ** 2))))
    return np.array(err)


def curves_to_model_frame(curves, model):
    """Traced pixel curves expressed in the model's own microns, base at 0."""
    pqd = model.pqd
    ppum = model.ppum
    base = np.array([pqd.getf("vid_cil_x0"),
                     model.image_h - pqd.getf("vid_cil_y0_invy")])
    ori = np.deg2rad(pqd.getf("vid_cil_orientation", 0.0))
    cs, sn = np.cos(-ori), np.sin(-ori)
    out = []
    for c in curves:
        pts = _resample_curve(np.asarray(c, dtype=float), 40)[0]
        dx = (pts[:, 0] - base[0]) / ppum
        dz = (base[1] - pts[:, 1]) / ppum
        out.append(np.column_stack([dx * cs - dz * sn, dx * sn + dz * cs]))
    return out


def fit_beat_at_phases(curves, phases, n_modes: int = 8, num_locs: int = 20,
                       n_space: int = 3, robust: bool = True):
    """The reference fit, but with every trace at its own exact phase.

    The reference script transforms a record sampled uniformly over one
    period, so the fitter used to fold the traces into phase bins, take a
    median shape per bin and transform those.  That costs accuracy twice
    over.  The median of a bin stands for its centre while the transform
    reads it as the bin's start, so the whole beat lags by half a bin --
    1.7 frames of an 80-frame beat, measured as a 0.4 micron displacement
    during the fast stroke.  And a bin's median smears whatever moves within
    it, which is most of the effective stroke.

    Fitting the same harmonics to each node's coordinates by least squares
    over all traces, each at the phase of its own frame, has neither
    problem: the phases are exact and every frame counts as itself.  For a
    record sampled uniformly over one period it reduces to the transform.
    With ``robust`` the fit is reweighted twice (Huber), so a trace that
    followed the wrong cilium in a few frames is down-weighted rather than
    averaged in.  ``curves`` are ordered base -> tip in microns, base at the
    origin, y up; ``phases`` in [0, 1).
    """
    phases = np.asarray(phases, dtype=float) % 1.0
    samples, lengths = resample_unit_length(curves, num_locs=num_locs)
    n = len(samples)
    if n < 2 * n_modes + 2:
        raise ValueError("only {} traces for {} time modes; need at least {}"
                         .format(n, n_modes, 2 * n_modes + 2))
    n_modes = int(n_modes)
    k = np.arange(1, n_modes)
    X = np.column_stack([np.ones(n)]
                        + [np.cos(2 * np.pi * kk * phases) for kk in k]
                        + [np.sin(2 * np.pi * kk * phases) for kk in k])
    Y = samples.reshape(n, -1)                       # (n, num_locs*2)
    w = np.ones(n)
    for _ in range(3 if robust else 1):
        sw = np.sqrt(w)[:, None]
        coef, _, _, _ = np.linalg.lstsq(X * sw, Y * sw, rcond=None)
        resid = np.sqrt(np.mean((Y - X @ coef) ** 2, axis=1))
        scale = 1.5 * float(np.median(resid)) + 1e-9
        w = np.where(resid <= scale, 1.0, scale / resid)
    coef = coef.reshape(-1, num_locs, 2)             # (2*n_modes-1, locs, 2)
    cos_c = np.zeros((n_modes, num_locs, 2))
    sin_c = np.zeros((n_modes, num_locs, 2))
    cos_c[0] = coef[0]
    cos_c[1:] = coef[1:n_modes]
    sin_c[1:] = coef[n_modes:]
    out = _beat_from_harmonics(cos_c, sin_c, num_locs, n_space)
    err = _beat_residuals(out, samples, phases, num_locs)
    return {
        "coeffs": out,
        "n_modes": n_modes,
        "lengths": lengths,
        "mean_length": float(np.mean(lengths)) if len(lengths) else 0.0,
        "samples": samples,
        "weights": w,
        "n_downweighted": int((w < 1.0).sum()),
        "resid": float(np.mean(err)),
        "resid_um": float(np.mean(err) * (np.mean(lengths) if len(lengths)
                                          else 1.0)),
        "resid_per_trace": err,
    }


def usable_traces(curves, min_tip_rise: float = 0.0,
                  max_turn_deg: float = 180.0):
    """Which traces pass the same checks fold_curves applies."""
    keep = []
    for c in curves:
        c = np.asarray(c, dtype=float)
        if len(c) < 3:
            keep.append(False)
            continue
        ang, arc = curve_angles(c, 40)
        if arc <= 0 or abs(np.rad2deg(ang[-1] - ang[0])) > max_turn_deg:
            keep.append(False)
            continue
        keep.append((c[0, 1] - c[-1, 1]) >= min_tip_rise * arc)
    return np.array(keep, dtype=bool)


def choose_reference_modes(curves, num_locs: int = 20, n_space: int = 3,
                           period_samples: int | None = None):
    """How many time modes the traces support, by holding out every other one.

    Retaining every mode reproduces the input exactly and tells you nothing;
    the reference script asks the operator to choose by eye instead.  This
    makes the same choice by measuring how well each option predicts the
    shapes it was not fitted to.
    """
    curves = list(curves)
    n = len(curves)
    if n < 8:
        return None
    train = [curves[i] for i in range(n) if i % 2 == 0]
    test_idx = [i for i in range(n) if i % 2 == 1]
    samples, _ = resample_unit_length(curves, num_locs=num_locs)
    best = None
    max_modes = 1 + int(np.ceil(0.5 * (len(train) - 1)))
    for nm in range(2, max_modes + 1):
        try:
            fit = fit_beat_reference(train, n_modes=nm, num_locs=num_locs,
                                     n_space=n_space)
        except ValueError:
            continue
        wave = FourierWave(fit["coeffs"], length=1.0, cbf=1.0)
        s_eval = np.linspace(0.0, 1.0, num_locs)
        err = []
        for i in test_idx:
            x, z = wave.position(s_eval, i / float(n))
            err.append(np.sqrt(np.mean((x - samples[i][:, 0]) ** 2
                                       + (z - samples[i][:, 1]) ** 2)))
        score = float(np.mean(err))
        if best is None or score < best[0] - 1e-9:
            best = (score, nm)
    return None if best is None else best[1]


def _harmonic_shrink(n_modes: int, damping: float):
    """How much each time harmonic is shrunk by the damping.

    The mean and the fundamental are left alone.  The fundamental *is* the
    beat -- it carries the whole sweep from one end of the stroke to the other
    -- so damping it in with the noisy harmonics shrinks the pattern itself: on
    a drawn wave whose traces swept 7.2 microns, the fitted tips covered only
    2.6.  Only the harmonics above the fundamental are noise-prone, so the
    penalty starts there.
    """
    k = np.arange(int(n_modes))
    return 1.0 / (1.0 + damping * np.maximum(k - 1, 0) ** 2)


# =========================================================================
# Fitting the beat to the frames themselves
# =========================================================================
#
# Tracing one cilium frame by frame and fitting a beat through the traces
# works when the cilium can be told from its neighbours in every frame.  On
# a dense carpet it cannot: the walk follows whichever filament is nearest
# at each step, a third of the traces end up on a neighbour, and no fit
# recovers a beat from that -- on one recording 37 of 100 traces were
# outliers and the "beat" was a straight spoke swinging through a fan of
# scribbles.  The joint fit below never decides per frame.  It takes the
# whole beat -- all 96 coefficients of the Fourier series -- and moves it
# so that, in every frame at that frame's own phase, the model curve lies
# along a dark filament.  A frame in which a neighbour happens to lie
# closer pulls a little; the other hundred frames pull back.
#
# The data term is the orientation-selective ridge response of the frames
# (the second derivative across the curve of the Gaussian-smoothed image,
# less the derivative along it, so a curve crossing filaments at an angle
# earns nothing), on frame minus temporal mean for recordings and on the
# frame itself for drawings, where the mean is dark wherever cilia have
# swept.  Penalties keep the curvature below 1.5 per micron, the material
# speed below eight lengths per beat (a shape change over two per cent of
# the cycle is a glitch, not a stroke) and the curve above the surface;
# the length is held exactly at L_c as everywhere else.  The gradient is
# analytic, so a fit of 600 steps takes a few seconds, and that buys what
# matters most: many starting points.  The landscape has local optima --
# a beat locked onto a neighbour for a quarter of the cycle is one -- so
# the trace fit, a phase-averaged trace fit, the published template beat
# and its mirror image at several phase offsets all start, the coarse
# stage (sigma 3 px, two harmonics) runs for all of them, and only the best
# survive to the finer stages.  The energy also arbitrates between the
# trace fits and the joint fit at the end, so the joint fit cannot make a
# good trace fit worse by its own measure.

_FOURIER_BLOCKS = ("Ax", "Az", "Bx", "Bz")


def _fourier_vec(coeffs, n_modes: int = FOURIER_DEFAULT_MODES,
                 n_space: int = N_FOURIER_M):
    """Flatten a coefficient dict into one vector (Ax, Az, Bx, Bz blocks)."""
    v = []
    for k in _FOURIER_BLOCKS:
        a = np.zeros((n_space, n_modes))
        arr = np.asarray(coeffs[k], dtype=float)
        r, c = min(arr.shape[0], n_space), min(arr.shape[1], n_modes)
        a[:r, :c] = arr[:r, :c]
        v.append(a.ravel())
    return np.concatenate(v)


def _fourier_dict(vec, n_modes: int = FOURIER_DEFAULT_MODES,
                  n_space: int = N_FOURIER_M):
    out = {}
    k = n_space * n_modes
    for i, key in enumerate(_FOURIER_BLOCKS):
        out[key] = np.asarray(vec[i * k:(i + 1) * k], dtype=float).reshape(
            n_space, n_modes).tolist()
    return out


def spread_frames(first: int, last: int, n_max: int):
    """At most ``n_max`` frame numbers spread evenly over the record."""
    frames = list(range(int(first), int(last) + 1))
    if len(frames) <= n_max:
        return frames
    idx = np.unique(np.round(np.linspace(0, len(frames) - 1, n_max)).astype(int))
    return [frames[i] for i in idx]


def band_envelope(video, model, n_bins: int = 12, half_um: float = 0.6,
                  frac: float = 0.35, period=None, start=None):
    """How high the lawn's texture reaches at each phase of the beat.

    On a lawn of cilia beating in step, seen side-on, every cilium has the
    same shape at a given phase, so the top of the textured band is the
    height of the cilium's highest point at that phase -- measured here as
    the highest row, within ``half_um`` cilium lengths of the base, whose
    variation along the surface is more than ``frac`` of the way from the
    background's to the band's.  On a rendered lawn with known truth the
    error was +0.01 +- 0.38 um over the cycle.  Returns the heights per
    phase bin, in microns, and the background and band texture levels.
    """
    pqd = model.pqd
    bx = pqd.getf("vid_cil_x0")
    by = model.image_h - pqd.getf("vid_cil_y0_invy")
    L = float(model.length) * model.ppum
    half = int(round(half_um * L))
    x0, x1 = int(max(0, bx - half)), int(min(model.image_w, bx + half))
    top = int(by) + 1
    if top < 8 or x1 - x0 < 8:
        raise ValueError("the base is too close to the frame's edge")
    _prefetch(video)
    frames = list(range(video.first, video.last + 1))
    prof = []
    for f in frames:
        a = np.asarray(video.get(f), dtype=np.float32)
        if a.ndim == 3:
            a = a[..., :3].mean(axis=2)
        prof.append(a[:top, x0:x1].std(axis=1))
    prof = np.array(prof)                       # (frames, rows)
    height = (by - np.arange(top)) / model.ppum
    top_bg = max(1, int(by - 1.6 * L))
    bg = float(np.median(prof[:, :top_bg])) if top_bg > 2 else float(prof.min())
    period = float(period if period is not None else model.period_frames)
    start = float(start if start is not None else model.macysta)
    phases = ((np.array(frames, float) - start) / period) % 1.0
    bins = np.minimum((phases * n_bins).astype(int), n_bins - 1)
    out = np.zeros(n_bins)
    lo_band = max(0, int(by - 0.5 * L))
    levels = []
    for b in range(n_bins):
        sel = bins == b
        pr = prof[sel].mean(axis=0) if sel.any() else prof.mean(axis=0)
        band = float(np.percentile(pr[lo_band:int(by) + 1], 90))
        levels.append(band)
        # The top of the run of textured rows that starts at the base --
        # not the highest textured row anywhere: a particle drifting past
        # above the lawn (CO121) put the envelope at 9.9 um over a 5.5 um
        # cilium, and the fit, held to it, stood every shape upright.
        above = pr > bg + frac * (band - bg)
        r = int(by)
        while r > 0 and above[r - 1]:
            r -= 1
        out[b] = float(height[r]) if above[int(by)] or r < int(by) else 0.0
    return out, bg, float(np.median(levels))


# A recording's polarity is taken as settled when the skew of the frames'
# deviation from their mean, in the window around the base, has the same
# sign in every fifth of the record and its magnitude over the whole record
# is at least this.  Drawn carpets give -0.7 to -2.1; a rendered lawn with
# half its cilia bright and half dark, where fitting for dark cilia alone
# was the worst choice, -0.18 to -0.31; the bright-field recordings here
# 0.05 to 0.16 overall with fifths of either sign.
POLARITY_MIN_SKEW = 0.4

# Below this circular spread (in cycles) of the beat's phase across the
# window around the base, the lawn beats in step and the band's envelope
# is the height of every cilium.  Lawns beating in step measure 0.03 to
# 0.07, a metachronal wave of 8 um 0.24.
LAWN_SYNC_MAX_SPREAD = 0.12


def _base_window(video, model, half_um: float = 1.2, up_um: float = 1.2):
    """Frames of the record cut to a window around the base, greyscale:
    (stack, (r0, r1, c0, c1))."""
    pqd = model.pqd
    bx = pqd.getf("vid_cil_x0")
    by = model.image_h - pqd.getf("vid_cil_y0_invy")
    L = float(model.length) * model.ppum
    c0 = int(max(0, bx - half_um * L))
    c1 = int(min(model.image_w, bx + half_um * L))
    r0 = int(max(0, by - up_um * L))
    r1 = int(max(r0 + 2, min(model.image_h, by - 2)))
    _prefetch(video)
    out = []
    for f in range(video.first, video.last + 1):
        a = np.asarray(video.get(f), dtype=np.float32)
        if a.ndim == 3:
            a = a[..., :3].mean(axis=2)
        out.append(a[r0:r1, c0:c1])
    return np.array(out), (r0, r1, c0, c1)


def polarity_of(video, model, n_chunks: int = 5) -> dict:
    """Do the cilia show dark (-1), bright (+1), or as both (0)?

    Judged by the skew of frame - mean in the window around the base, as
    the tracer judges it -- but in each fifth of the record separately.
    Bright-field optics show a thin filament dark or bright depending on
    the focus, and a drift of focus turns one into the other: on one
    recording the fifths measured -0.31, 0.06, 0.01, 0.21 and 0.26.  A
    beat fitted for dark cilia there matched the first three fifths and
    not the last two.  Where the fifths disagree, or the skew is small, the
    polarity is "either" and the fit scores a line of either sign.
    """
    stack, _ = _base_window(video, model)
    dev = stack - stack.mean(axis=0)
    skews = []
    for chunk in np.array_split(np.arange(len(stack)), n_chunks):
        d = dev[chunk].ravel()
        sd = float(d.std())
        skews.append(float(np.mean((d / sd) ** 3)) if sd > 0 else 0.0)
    d = dev.ravel()
    sd = float(d.std())
    overall = float(np.mean((d / sd) ** 3)) if sd > 0 else 0.0
    agree = all(np.sign(k) == np.sign(overall) and abs(k) >= 0.5 * POLARITY_MIN_SKEW
                for k in skews)
    clear = agree and abs(overall) >= POLARITY_MIN_SKEW
    sign = (1.0 if overall > 0 else -1.0) if clear else 0.0
    return {"sign": sign, "clear": clear, "skew": overall, "skews": skews}


def lawn_synchrony(video, model, n_cols: int = 9) -> dict:
    """Does the lawn around the base beat in step?

    The height of the texture -- the top of the band -- is followed over
    time in each of ``n_cols`` strips across the window (the same measure
    as band_envelope, strip by strip), and the phase of each strip's
    oscillation at the beat frequency is taken.  The circular spread of
    those phases, in cycles, is small for a lawn beating in step and large
    under a metachronal wave; ``synchronous`` is the comparison with
    LAWN_SYNC_MAX_SPREAD.
    """
    stack, (r0, r1, c0, c1) = _base_window(video, model, half_um=0.6, up_um=1.1)
    n, hh, ww = stack.shape
    per = float(model.period_frames)
    if n < 2 * per or hh < 6 or ww < n_cols:
        return {"spread": float("nan"), "synchronous": False, "phases": []}
    dev = np.abs(stack - stack.mean(axis=0))
    bg = float(np.median(dev[:, :max(2, int(0.15 * hh))]))
    edges = np.linspace(0, ww, n_cols + 1).astype(int)
    t = np.arange(n)
    taper = np.hanning(n)
    phases, amps = [], []
    for i in range(n_cols):
        prof = dev[:, :, edges[i]:edges[i + 1]].mean(axis=2)      # (frames, rows)
        band = float(np.percentile(prof[:, -max(3, hh // 3):], 90))
        thr = bg + 0.35 * (band - bg)
        top = np.array([np.nonzero(pr > thr)[0].min() if (pr > thr).any()
                        else hh for pr in prof], dtype=float)
        sig = (hh - top - np.mean(hh - top)) * taper
        z = np.sum(sig * np.exp(-2j * np.pi * t / per))
        phases.append(float(np.angle(z) / (2 * np.pi)))
        amps.append(float(abs(z)))
    phases, amps = np.array(phases), np.array(amps)
    R = abs(np.sum(amps * np.exp(2j * np.pi * phases))) / max(amps.sum(), 1e-9)
    spread = float(np.sqrt(-2.0 * np.log(max(R, 1e-9))) / (2 * np.pi))
    return {"spread": spread, "synchronous": spread < LAWN_SYNC_MAX_SPREAD,
            "phases": phases.tolist()}


# The sweep penalty of the joint fit: a beat has one stroke each way --
# the recovery stroke sweeps one way at every moment, the effective stroke
# the other -- so the distal half's sideways position goes once down and
# once up per beat.  Its total variation over the beat is then exactly
# twice its range; what exceeds that is back-and-forth within a stroke and
# is penalised, as a fraction of the range.  On the HYDIN recording the
# unconstrained fit reversed direction nine times within the strokes
# (excess 84 %) and the exported shapes jittered.
MONO_SWEEP_LAMBDA = 0.5
MONO_SWEEP_EPS = 0.02        # um: smoothing of |step| for the gradient
MONO_SWEEP_PHASES = 48       # phases round the beat the sweep is judged at
MONO_SWEEP_TOL = 0.002       # excess below this is nothing (the smoothing's own)
# The penalty acts in the last stages of IMAGE_FIT_SCHEDULE only: on from
# the first, coarse stage it steered the beam into worse beats (rendered
# lawn 0.48 -> 0.62 um from the truth; the last two stages: 0.52, with the
# data term of the result higher than the unconstrained fit's).
MONO_SWEEP_LAST_STAGES = 2

# Which way the effective stroke sweeps, as a setting (specification key
# vid_effective_stroke): "right" -- x increasing, so the recovery stroke
# sweeps right to left and the effective stroke left to right; "left" --
# the other way; "auto" -- the faster of the two sweeps, judged after the
# fit (stroke_landmarks).  The setting places the main cycle's start (Find
# cycle, Auto fit) and, in the joint fit, holds the stroke sweeping that
# way to be the faster one, which on a dense recording the frames alone
# cannot always decide (HYDIN: beats of either timing within 0.03 in
# energy, the traced tip's sweeps within 10 % in speed).  The default is
# right: the user's recordings are mounted that way (HYDIN by the user's
# account, input_images by speed once the fit is held to one stroke each
# way), and so are the published template and every rendered truth.
EFFECTIVE_STROKE_KEY = "vid_effective_stroke"
EFFECTIVE_STROKE_CHOICES = ("right", "left", "auto")
EFFECTIVE_STROKE_DEFAULT = "right"

# The frame the pattern is drawn and exported in (specification key
# vid_pattern_frame): "upright" -- turned to the fitted fan's own axis, so
# the fan stands symmetric about the vertical (Auto fit ends by turning
# the frame; on CO121 the fan leans 24 deg right of the surface normal
# and the orientation goes from -6 to -30 deg); "surface" -- x along the
# cell surface, z its normal, as Find base sets it, the lean left in.
PATTERN_FRAME_KEY = "vid_pattern_frame"
PATTERN_FRAME_CHOICES = ("upright", "surface")
PATTERN_FRAME_DEFAULT = "upright"
# How far the frame has been turned from the surface's orientation, in
# degrees (0: the surface frame); the stroke landmarks are read in the
# surface frame whatever the pattern frame.
PATTERN_TURN_KEY = "vid_pattern_turn"


def pattern_frame_setting(pqd) -> str:
    try:
        v = str(pqd.get(PATTERN_FRAME_KEY, PATTERN_FRAME_DEFAULT) or "")
    except Exception:
        v = ""
    v = v.strip().lower()
    return v if v in PATTERN_FRAME_CHOICES else PATTERN_FRAME_DEFAULT


def effective_stroke_setting(pqd) -> str:
    """The effective-stroke setting of a pqd: right, left or auto."""
    try:
        v = str(pqd.get(EFFECTIVE_STROKE_KEY, EFFECTIVE_STROKE_DEFAULT) or "")
    except Exception:
        v = ""
    v = v.strip().lower()
    return v if v in EFFECTIVE_STROKE_CHOICES else EFFECTIVE_STROKE_DEFAULT


def effective_direction(pqd):
    """+1 (the effective stroke sweeps right), -1 (left) or None (auto)."""
    return {"right": 1, "left": -1}.get(effective_stroke_setting(pqd))


def sweep_report(wave, n_phases: int = 48) -> dict:
    """How monotone a beat's sweep is: the distal half's sideways position
    over the beat, its range, the excess of its total variation over twice
    the range (0 for one stroke each way), and the number of direction
    reversals (steps under 1 % of the range are not counted)."""
    pts = max(20, int(getattr(wave, "n_pts", 60) or 60))
    mid = np.array([np.mean(wave.shape_at(t, n_pts=pts)[0][pts // 2:])
                    for t in np.arange(n_phases) / float(n_phases)])
    d = np.roll(mid, -1) - mid
    rng = float(mid.max() - mid.min())
    tv = float(np.abs(d).sum())
    excess = max(tv / (2.0 * rng) - 1.0, 0.0) if rng > 1e-9 else 0.0
    big = d[np.abs(d) > 0.01 * rng] if rng > 1e-9 else d[:0]
    reversals = int(np.sum(np.sign(big) != np.sign(np.roll(big, 1)))) if len(big) else 0
    reversals = max(reversals - 2, 0)        # the two turns of a wiper are not reversals
    return {"range_um": rng, "excess": excess, "reversals": reversals}


def fan_lean(wave, n_phases: int = 48) -> dict:
    """Which way a beat's fan leans: the tip's direction from the base,
    in degrees from the model frame's normal (positive = towards +x, the
    right), at ``n_phases`` phases -- its extremes and their bisector,
    the fan's axis.  A fan standing upright has an axis near 0."""
    pts = max(20, int(getattr(wave, "n_pts", 60) or 60))
    ang = []
    for t in np.arange(n_phases) / float(n_phases):
        x, z = wave.shape_at(t, n_pts=pts)
        ang.append(np.degrees(np.arctan2(x[-1], z[-1])))
    ang = np.array(ang)
    lo, hi = float(ang.min()), float(ang.max())
    return {"axis_deg": 0.5 * (lo + hi), "lo_deg": lo, "hi_deg": hi,
            "half_width_deg": 0.5 * (hi - lo)}


class BeatImageFit:
    """Energy of a Fourier beat against the frames, with its gradient.

    The frames are reduced to a window around the base, binned to about
    ``target_ppum`` pixels per micron, and replaced by the three Gaussian
    Hessian maps of the cilium signal at scale ``sigma`` (``mode`` "dev":
    frame minus temporal mean, for recordings; "raw": the frame itself, for
    drawings).  ``beta`` adds a plain darkness term on top of the ridge
    response.  ``period`` and ``start`` default to the model's cycle.
    """

    def __init__(self, video, model, frames=None, n_frames: int = 96,
                 sigma: float = 3.0, n_s: int = 40, pad_um: float = 1.4,
                 n_modes: int = FOURIER_DEFAULT_MODES,
                 n_space: int = N_FOURIER_M, sign=None,
                 surf_pad_px: float = 0.0, period=None, start=None,
                 n_quad: int = 161, target_ppum: float = 14.0,
                 mode: str | None = None, beta: float | None = None,
                 envelope=None, lam_env: float = 4.0, lam_mono=None,
                 direction=None, progress=None):
        self.model = model
        self.video = video
        self.n_modes, self.n_space = int(n_modes), int(n_space)
        pqd = model.pqd
        full_ppum = model.ppum
        self.binf = max(1, int(round(full_ppum / target_ppum)))
        b = self.binf
        self.ppum = full_ppum / b
        self.L = float(model.length)
        self.x0 = pqd.getf("vid_cil_x0") / b
        self.y0 = (model.image_h - pqd.getf("vid_cil_y0_invy")) / b
        self.ori = np.deg2rad(pqd.getf("vid_cil_orientation", 0.0))
        self.frames = (list(frames) if frames is not None
                       else spread_frames(video.first, video.last, n_frames))
        self.period = float(period if period is not None else model.period_frames)
        self.start = float(start if start is not None else model.macysta)
        self.n_s = int(n_s)
        self.s_q = np.linspace(0.0, 1.0, int(n_quad))
        self.frac = np.linspace(0.0, 1.0, self.n_s)
        self.surf_pad = float(surf_pad_px)
        self.progress = progress
        # The lawn's envelope: the height the texture reaches at each phase
        # (band_envelope), which on a lawn beating in step is the height of
        # every cilium's highest point.  With it, the fit is held to that
        # height at every phase -- see evaluate.
        self.envelope = None if envelope is None else np.asarray(envelope, float)
        self.lam_env = float(lam_env)
        # The sweep: one stroke each way per beat, no back-and-forth (see
        # MONO_SWEEP_LAMBDA and evaluate).
        self.lam_mono = float(MONO_SWEEP_LAMBDA if lam_mono is None else lam_mono)
        # +1 / -1: the stroke sweeping that way is the faster one (the
        # effective-stroke setting); None: either
        self.direction = None if not direction else int(np.sign(direction))
        drawn = frames_are_drawn(video)
        self.mode = mode or ("raw" if drawn else "dev")
        self.beta = float(beta if beta is not None else (1.0 if drawn else 0.25))
        self.set_phases()
        acc = None
        total = video.last - video.first + 1
        _prefetch(video)
        for i, f in enumerate(range(video.first, video.last + 1)):
            if progress is not None and i % 10 == 0:
                progress(0.3 * i / total, "reading frames")
            im = self._frame(f)
            acc = im.copy() if acc is None else acc + im
        mean_full = acc / float(total)
        self.surface = detect_surface(static_mean(video, mean_full),
                                      valid=video.valid) / float(b)
        self.mean = bin_frame(mean_full, b)
        h, w = self.mean.shape
        Lpx = self.L * self.ppum
        r = pad_um * Lpx
        self.c0 = int(max(0, np.floor(self.x0 - r)))
        self.c1 = int(min(w, np.ceil(self.x0 + r)))
        self.r0 = int(max(0, np.floor(self.y0 - r)))
        self.r1 = int(min(h, np.ceil(self.y0 + 0.35 * Lpx)))
        if self.c1 - self.c0 < 8 or self.r1 - self.r0 < 8:
            raise ValueError("the base lies outside the frame")
        # +1: bright cilia, -1: dark, 0: either -- the ridge response then
        # takes the line's strength regardless of its polarity (see
        # evaluate), for bright-field recordings whose cilia show as both.
        self.sign = float(sign) if sign is not None else self._polarity(mean_full)
        if self.sign == 0.0:
            self.beta = 0.0            # darkness has no meaning either way
        self.set_sigma(sigma)

    def _frame(self, f):
        im = np.asarray(self.video.get(f), dtype=np.float32)
        if im.ndim == 3:
            im = im[..., :3].mean(axis=2)
        return im

    def __getstate__(self):
        # What a worker process needs to score and optimise beats: the maps
        # and the geometry.  The video, the model and a GUI progress callback
        # stay behind -- they are only used to build the maps.
        state = dict(self.__dict__)
        for key in ("video", "model", "progress"):
            state[key] = None
        return state

    def _polarity(self, mean_full):
        """-1 for dark cilia on a bright field, +1 for bright ones.

        Judged as auto_trace judges it, by the skew of the deviation from
        the temporal mean over the whole beating band.  A window around the
        base is not enough: in the dense middle of a carpet the bright
        halos of bright-field optics outweigh the dark filaments, the local
        skew comes out positive and every map is inverted.
        """
        surf = self.surface * self.binf
        h, w = mean_full.shape
        rows = np.arange(h)[:, None]
        band = rows < (surf[None, :] - 3)
        try:
            valid = np.asarray(self.video.valid, dtype=bool)
            if valid.shape == band.shape:
                band &= valid
        except Exception:
            pass
        if band.sum() < 16:
            return -1.0
        dev = []
        for f in self.frames[::max(1, len(self.frames) // 12)][:12]:
            dev.append((self._frame(f) - mean_full)[band])
        dev = np.concatenate(dev)
        sd = float(np.std(dev))
        if sd <= 0:
            return -1.0
        skew = float(np.mean((dev / sd) ** 3))
        return 1.0 if skew > 0.05 else -1.0

    def set_phases(self):
        self.phases = ((np.array(self.frames, float) - self.start)
                       / self.period) % 1.0
        n = np.arange(self.n_modes)
        ang = 2 * np.pi * n[None, :] * self.phases[:, None]
        self.C, self.S = np.cos(ang), np.sin(ang)
        self.W = 2 * np.pi * n[None, :]
        self.env_at = None
        if self.envelope is not None and len(self.envelope) >= 2:
            nb = len(self.envelope)
            centres = (np.arange(nb + 2) - 0.5) / nb         # bin centres, wrapped
            vals = np.concatenate([[self.envelope[-1]], self.envelope,
                                   [self.envelope[0]]])
            self.env_at = np.interp(self.phases, centres, vals)

    def set_sigma(self, sigma: float):
        """Build the ridge (and darkness) maps at scale ``sigma`` pixels."""
        self.sigma = float(sigma)
        F = len(self.frames)
        hh, ww = self.r1 - self.r0, self.c1 - self.c0
        H = np.zeros((3, F, hh, ww), np.float32)
        D = np.zeros((F, hh, ww), np.float32)
        m = max(8, int(round(4 * sigma)) + 2)
        h, w = self.mean.shape
        R0, R1 = max(0, self.r0 - m), min(h, self.r1 + m)
        C0, C1 = max(0, self.c0 - m), min(w, self.c1 + m)
        rows_w = np.arange(self.r0, self.r1)[:, None]
        cols_w = np.arange(self.c0, self.c1)[None, :]
        above_w = rows_w < (self.surface[np.clip(cols_w, 0, len(self.surface) - 1)] - 2)
        levels = None
        sl = (slice(self.r0 - R0, self.r0 - R0 + hh),
              slice(self.c0 - C0, self.c0 - C0 + ww))
        sign = self.sign if self.sign != 0.0 else 1.0
        for i, f in enumerate(self.frames):
            if self.progress is not None and i % 10 == 0:
                self.progress(0.3 + 0.2 * i / F,
                              "ridge maps at sigma {:g}".format(sigma))
            im = bin_frame(self._frame(f), self.binf)
            if self.mode == "dev":
                d = (im - self.mean)[R0:R1, C0:C1] * sign
            else:
                d = im[R0:R1, C0:C1] * sign
            hrr, hrc, hcc = gauss_hess(d, sigma)
            H[0, i] = -hrr[sl]
            H[1, i] = -hrc[sl]
            H[2, i] = -hcc[sl]
            if self.beta > 0:
                win = (im * sign)[self.r0:self.r1, self.c0:self.c1]
                if levels is None:
                    vals = win[above_w] if above_w.any() else win.ravel()
                    lo, hi = np.percentile(vals, [10, 95])
                    levels = (float(lo), float(hi))
                lo, hi = levels
                D[i] = np.clip((gaussian_blur(win, max(0.7, 0.5 * sigma)) - lo)
                               / max(hi - lo, 1e-6), 0.0, 1.0)
        lam = np.maximum(H[0], H[2])
        if self.sign == 0.0:
            lam = np.maximum(np.abs(H[0]), np.abs(H[2]))
        vals = lam[:, above_w] if above_w.any() else lam
        pos = vals[vals > 0]
        scale = float(np.percentile(pos, 95)) if pos.size else 1.0
        # Nothing below the surface counts.  The cell body is dark and its
        # edge is a strong horizontal ridge; left in, both drew the lying
        # shapes of the beat down into the tissue, where no cilium is.
        keep = (rows_w < (self.surface[np.clip(cols_w, 0, len(self.surface) - 1)] - 1))
        keep = keep.astype(np.float32)[None]
        self.H = H / (scale if scale > 0 else 1.0) * keep[None]
        self.D = D * keep

    # ------------------------------------------------------------- shapes

    def unpack(self, vec):
        nm, ns = self.n_modes, self.n_space
        k = nm * ns
        return (vec[0:k].reshape(ns, nm), vec[k:2 * k].reshape(ns, nm),
                vec[2 * k:3 * k].reshape(ns, nm), vec[3 * k:].reshape(ns, nm))

    def shape_coeffs(self, vec):
        """Per-frame s^m coefficients of x and z, and their phase rates."""
        Ax, Az, Bx, Bz = self.unpack(vec)
        cx = self.C @ Ax.T + self.S @ Bx.T
        cz = self.C @ Az.T + self.S @ Bz.T
        cxt = (-self.S * self.W) @ Ax.T + (self.C * self.W) @ Bx.T
        czt = (-self.S * self.W) @ Az.T + (self.C * self.W) @ Bz.T
        return cx, cz, cxt, czt

    def arc_table(self, cx, cz, grad: bool = False):
        """Raw arc length per frame and the equal-arc-length s samples."""
        m = np.arange(1, self.n_space + 1)
        sq = self.s_q
        dpow = (m[:, None] * sq[None, :] ** (m - 1)[:, None])
        dxq = cx @ dpow
        dzq = cz @ dpow
        speed = np.hypot(dxq, dzq)
        cum = np.concatenate(
            [np.zeros((len(cx), 1)),
             np.cumsum(0.5 * (speed[:, 1:] + speed[:, :-1]) * np.diff(sq)[None, :],
                       axis=1)], axis=1)
        raw = cum[:, -1]
        S_ = np.empty((len(cx), self.n_s))
        for i in range(len(cx)):
            S_[i] = np.interp(self.frac * raw[i], cum[i], sq)
        if not grad:
            return raw, S_
        wq = np.full(len(sq), sq[1] - sq[0])
        wq[0] *= 0.5
        wq[-1] *= 0.5
        spq = np.maximum(speed, 1e-9)
        draw_dcx = ((dxq / spq) * wq[None, :]) @ dpow.T
        draw_dcz = ((dzq / spq) * wq[None, :]) @ dpow.T
        return raw, S_, draw_dcx, draw_dcz

    def arc_slide(self, cx, cz, S_, w, draw_dcx, draw_dcz):
        """``sum_s w[f, s] * dS_[f, s] / dc`` for the x and z coefficients.

        The equal-arc-length samples of arc_table slide along the curve as
        the coefficients change; a term that depends on *which* points of
        the curve are sampled (the sweep penalty: the mean sideways
        position of the distal half) needs that part of its gradient.
        ``w`` is dE/dS_ per sample.  Returns (F, M) arrays to add to the
        coefficient gradients."""
        m = np.arange(1, self.n_space + 1)
        sq = self.s_q
        ds = sq[1] - sq[0]
        dpow = (m[:, None] * sq[None, :] ** (m - 1)[:, None])
        dxq = cx @ dpow
        dzq = cz @ dpow
        spq = np.maximum(np.hypot(dxq, dzq), 1e-9)
        out = []
        i = np.clip(np.floor(S_ / ds).astype(int), 0, len(sq) - 2)
        t = np.clip((S_ - sq[i]) / ds, 0.0, 1.0)
        rows = np.arange(len(cx))[:, None]
        slope = 0.5 * (spq[rows, i] + spq[rows, i + 1])          # d cum / dS
        for dq, draw in ((dxq, draw_dcx), (dzq, draw_dcz)):
            ig = (dq / spq)[:, None, :] * dpow[None, :, :]        # d speed / dc
            dcum = np.concatenate(
                [np.zeros(ig.shape[:2] + (1,)),
                 np.cumsum(0.5 * (ig[..., 1:] + ig[..., :-1]) * ds, axis=2)],
                axis=2)                                           # (F, M, Q)
            at = (np.take_along_axis(dcum, i[:, None, :], axis=2) * (1 - t)[:, None, :]
                  + np.take_along_axis(dcum, (i + 1)[:, None, :], axis=2) * t[:, None, :])
            dS = (self.frac[None, None, :] * draw[:, :, None] - at) / slope[:, None, :]
            out.append(np.einsum("fs,fms->fm", w, dS))
        return out[0], out[1]

    def evaluate(self, vec, grad: bool = False, lam_bend: float = 0.3,
                 kappa0: float = 1.5, lam_surf: float = 4.0,
                 lam_speed: float = 0.002, v0: float = 8.0, clip: float = 3.0):
        """Energy (higher is better), its parts, and optionally its gradient."""
        cx, cz, cxt, czt = self.shape_coeffs(vec)
        raw, S_, draw_dcx, draw_dcz = self.arc_table(cx, cz, grad=True)
        k = (self.L / np.maximum(raw, 1e-9))[:, None]
        m = np.arange(1, self.n_space + 1)
        P0 = np.stack([S_ ** mm for mm in m], axis=1)
        P1 = np.stack([mm * S_ ** (mm - 1) for mm in m], axis=1)
        P2 = np.stack([(mm * (mm - 1) * S_ ** (mm - 2)) if mm >= 2
                       else np.zeros_like(S_) for mm in m], axis=1)
        X = np.einsum("fm,fms->fs", cx, P0)
        Z = np.einsum("fm,fms->fs", cz, P0)
        X1 = np.einsum("fm,fms->fs", cx, P1)
        Z1 = np.einsum("fm,fms->fs", cz, P1)
        X2 = np.einsum("fm,fms->fs", cx, P2)
        Z2 = np.einsum("fm,fms->fs", cz, P2)
        Xt = np.einsum("fm,fms->fs", cxt, P0)
        Zt = np.einsum("fm,fms->fs", czt, P0)
        c, s_ = np.cos(self.ori), np.sin(self.ori)
        pp = self.ppum
        x = k * X
        z = k * Z
        px = self.x0 + (x * c - z * s_) * pp
        py = self.y0 - (x * s_ + z * c) * pp
        Tx = X1 * c - Z1 * s_
        Ty = -(X1 * s_ + Z1 * c)
        Tn = np.maximum(np.hypot(Tx, Ty), 1e-9)
        tx, ty = Tx / Tn, Ty / Tn
        # ---- data term
        u = px - self.c0
        v = py - self.r0
        hh, ww = self.H.shape[2:]
        inside = (u >= 0) & (u <= ww - 1.001) & (v >= 0) & (v <= hh - 1.001)
        uu = np.clip(u, 0, ww - 1.001)
        vv = np.clip(v, 0, hh - 1.001)
        i0 = np.floor(vv).astype(int)
        j0 = np.floor(uu).astype(int)
        fv = vv - i0
        fu = uu - j0
        F, Sn = px.shape
        fi = np.broadcast_to(np.arange(F)[:, None], (F, Sn))
        w00 = (1 - fv) * (1 - fu)
        w01 = (1 - fv) * fu
        w10 = fv * (1 - fu)
        w11 = fv * fu

        def samp(M):
            return (w00 * M[fi, i0, j0] + w01 * M[fi, i0, j0 + 1]
                    + w10 * M[fi, i0 + 1, j0] + w11 * M[fi, i0 + 1, j0 + 1])

        # Exact derivatives of the bilinear interpolant, so the gradient is
        # the gradient of the energy actually evaluated -- a central-
        # difference map sampled the same way is only a smoothed version,
        # and on a sharp drawing the two disagree.
        def d_col(M):
            return ((1 - fv) * (M[fi, i0, j0 + 1] - M[fi, i0, j0])
                    + fv * (M[fi, i0 + 1, j0 + 1] - M[fi, i0 + 1, j0]))

        def d_row(M):
            return ((1 - fu) * (M[fi, i0 + 1, j0] - M[fi, i0, j0])
                    + fu * (M[fi, i0 + 1, j0 + 1] - M[fi, i0, j0 + 1]))
        h = [samp(self.H[q]) for q in range(3)]
        nx, ny = -ty, tx
        across = nx * nx * h[2] + 2 * nx * ny * h[1] + ny * ny * h[0]
        along = tx * tx * h[2] + 2 * tx * ty * h[1] + ty * ty * h[0]
        # Either polarity: a line of either sign counts, so the response is
        # the magnitude across less the magnitude along.
        sgn_ac = np.sign(across) if self.sign == 0.0 else 1.0
        r_raw = sgn_ac * across - np.abs(along)
        if self.beta > 0:
            r_raw = r_raw + self.beta * samp(self.D)
        active = inside & (r_raw < clip)
        r = np.where(inside, np.minimum(r_raw, clip), -1.0)
        data = float(r.mean())
        # The support along the cilium, base to tip, averaged over the
        # frames: where it falls away is where the frames stop showing a
        # cilium (support_profile).
        self.last_profile = r.mean(axis=0)
        # ---- penalties
        cols = np.clip(np.round(px).astype(int), 0, len(self.surface) - 1)
        below = py - (self.surface[cols] - self.surf_pad)
        bpos = np.clip(below, 0, None)
        pen_surf = float(np.mean(bpos ** 2)) / pp ** 2
        wcur = X1 * Z2 - Z1 * X2
        sp = np.maximum(np.hypot(X1, Z1), 1e-9)
        kraw = np.abs(wcur) / sp ** 3
        kappa = kraw / k
        kex = np.clip(kappa - kappa0, 0, None)
        pen_bend = float(np.mean(kex ** 2))
        vn = np.maximum(np.hypot(Xt, Zt), 1e-9)
        vel = k * vn / self.L
        vex = np.clip(vel - v0, 0, None)
        pen_speed = float(np.mean(vex ** 2))
        # The envelope: the cilium's highest point at each phase, against
        # the height the lawn's texture reaches there.
        pen_env = 0.0
        if self.env_at is not None:
            hgt = x * s_ + z * c                       # height above the base, um
            arg_h = np.argmax(hgt, axis=1)
            hmax = hgt[np.arange(F), arg_h]
            dh = (hmax - self.env_at) / self.L
            pen_env = float(np.mean(dh ** 2))
        # The sweep: the distal half's sideways position over the beat,
        # frames in phase order.  A beat with one stroke each way has a
        # total variation of exactly twice its range; every reversal
        # within a stroke adds to it.  The excess, as a fraction of the
        # range, is penalised (MONO_SWEEP_LAMBDA).
        # The sweep: one stroke each way, the one the setting names the
        # faster (sweep_terms, on a uniform grid of phases -- the frames'
        # own phases leave gaps a wiggle can hide in).
        pen_mono = pen_dir = 0.0
        g_sweep = None
        if self.lam_mono > 0:
            if grad:
                pen_mono, pen_dir, g_sweep = self.sweep_terms(vec, grad=True)
            else:
                pen_mono, pen_dir = self.sweep_terms(vec)
        E = (data - lam_bend * pen_bend - lam_surf * pen_surf
             - lam_speed * pen_speed - self.lam_env * pen_env
             - self.lam_mono * (pen_mono + pen_dir))
        parts = dict(data=data, bend=pen_bend, surf=pen_surf, speed=pen_speed,
                     env=pen_env, mono=pen_mono, dir=pen_dir,
                     vmax=float(vel.max()), kmax=float(kappa.max()),
                     inside=float(inside.mean()))
        if not grad:
            return E, parts
        # ---- gradient: sensitivities to the sampled quantities first
        N = float(F * Sn)
        hc = [d_col(self.H[q]) for q in range(3)]
        hr = [d_row(self.H[q]) for q in range(3)]
        sgn_al = np.sign(along)
        dr_dh = [sgn_ac * ny * ny - sgn_al * ty * ty,
                 sgn_ac * 2 * nx * ny - sgn_al * 2 * tx * ty,
                 sgn_ac * nx * nx - sgn_al * tx * tx]
        dr_dpx = sum(dr_dh[q] * hc[q] for q in range(3)) * active / N
        dr_dpy = sum(dr_dh[q] * hr[q] for q in range(3)) * active / N
        if self.beta > 0:
            dr_dpx = dr_dpx + self.beta * d_col(self.D) * active / N
            dr_dpy = dr_dpy + self.beta * d_row(self.D) * active / N
        dA_dtx = (-2 * ty * h[1] + 2 * tx * h[0])
        dA_dty = (2 * ty * h[2] - 2 * tx * h[1])
        dAl_dtx = (2 * tx * h[2] + 2 * ty * h[1])
        dAl_dty = (2 * tx * h[1] + 2 * ty * h[0])
        dr_dtx = (sgn_ac * dA_dtx - sgn_al * dAl_dtx) * active / N
        dr_dty = (sgn_ac * dA_dty - sgn_al * dAl_dty) * active / N
        dr_dTx = (dr_dtx * (1 - tx * tx) - dr_dty * tx * ty) / Tn
        dr_dTy = (-dr_dtx * tx * ty + dr_dty * (1 - ty * ty)) / Tn
        g_X1 = dr_dTx * c - dr_dTy * s_
        g_Z1 = -dr_dTx * s_ - dr_dTy * c
        g_X = (dr_dpx * c - dr_dpy * s_) * k * pp
        g_Z = (-dr_dpx * s_ - dr_dpy * c) * k * pp
        dS_dpy = -lam_surf * 2 * bpos / (N * pp ** 2)
        g_X += (-dS_dpy * s_) * k * pp
        g_Z += (-dS_dpy * c) * k * pp
        dB = -lam_bend * 2 * kex / N / k
        sw = np.sign(wcur)
        g_X1 += dB * (sw * Z2 / sp ** 3 - 3 * np.abs(wcur) * X1 / sp ** 5)
        g_Z1 += dB * (-sw * X2 / sp ** 3 - 3 * np.abs(wcur) * Z1 / sp ** 5)
        g_X2 = dB * (-sw * Z1 / sp ** 3)
        g_Z2 = dB * (sw * X1 / sp ** 3)
        dV = -lam_speed * 2 * vex / N * k / self.L
        g_Xt = dV * Xt / vn
        g_Zt = dV * Zt / vn
        # the per-phase scale k_f = L / raw_f depends on the coefficients too
        gk = (np.sum(dr_dpx * (X * c - Z * s_) * pp
                     + (dr_dpy + dS_dpy) * (-(X * s_ + Z * c)) * pp, axis=1)
              + np.sum((-lam_bend * 2 * kex / N) * (-kraw / k ** 2), axis=1)
              + np.sum((-lam_speed * 2 * vex / N) * vn / self.L, axis=1))
        if self.env_at is not None:
            dE_dh = -self.lam_env * 2 * dh / (F * self.L)      # per frame
            rows_f = np.arange(F)
            g_X[rows_f, arg_h] += dE_dh * k[:, 0] * s_
            g_Z[rows_f, arg_h] += dE_dh * k[:, 0] * c
            gk += dE_dh * (X[rows_f, arg_h] * s_ + Z[rows_f, arg_h] * c)
        dk_draw = (-k[:, 0] / np.maximum(raw, 1e-9))
        gscale_x = (gk * dk_draw)[:, None] * draw_dcx
        gscale_z = (gk * dk_draw)[:, None] * draw_dcz
        gcx = (np.einsum("fs,fms->fm", g_X, P0) + np.einsum("fs,fms->fm", g_X1, P1)
               + np.einsum("fs,fms->fm", g_X2, P2) + gscale_x)
        gcz = (np.einsum("fs,fms->fm", g_Z, P0) + np.einsum("fs,fms->fm", g_Z1, P1)
               + np.einsum("fs,fms->fm", g_Z2, P2) + gscale_z)
        gcxt = np.einsum("fs,fms->fm", g_Xt, P0)
        gczt = np.einsum("fs,fms->fm", g_Zt, P0)
        gAx = gcx.T @ self.C + gcxt.T @ (-self.S * self.W)
        gBx = gcx.T @ self.S + gcxt.T @ (self.C * self.W)
        gAz = gcz.T @ self.C + gczt.T @ (-self.S * self.W)
        gBz = gcz.T @ self.S + gczt.T @ (self.C * self.W)
        g = np.concatenate([gAx.ravel(), gAz.ravel(), gBx.ravel(), gBz.ravel()])
        if g_sweep is not None:
            g = g - self.lam_mono * g_sweep
        return E, parts, g

    def sweep_terms(self, vec, grad: bool = False):
        """The sweep penalty of the beat ``vec``: (excess, shortfall) and,
        with ``grad``, the gradient of their sum against the coefficients.

        The distal half's mean sideways position is taken at
        MONO_SWEEP_PHASES phases evenly round the beat.  *Excess*: its
        total variation over twice its range, less one -- zero for one
        stroke each way, every reversal within a stroke adds to it.
        *Shortfall*, with a direction set: the squared steps that way
        must sum to at least the squared steps the other way (equal sums
        means equal times; the faster stroke has the larger), and what is
        missing counts, per step and per range squared.
        """
        G = int(MONO_SWEEP_PHASES)
        nm = self.n_modes
        basis = getattr(self, "_sweep_basis", None)
        if basis is None or basis[0].shape != (G, nm):
            ang = 2 * np.pi * np.arange(nm)[None, :] * (np.arange(G) / float(G))[:, None]
            basis = (np.cos(ang), np.sin(ang))
            self._sweep_basis = basis
        Cg, Sg = basis
        Ax, Az, Bx, Bz = self.unpack(vec)
        cx = Cg @ Ax.T + Sg @ Bx.T
        cz = Cg @ Az.T + Sg @ Bz.T
        raw, S_, draw_dcx, draw_dcz = self.arc_table(cx, cz, grad=True)
        k = (self.L / np.maximum(raw, 1e-9))[:, None]
        m = np.arange(1, self.n_space + 1)
        P0 = np.stack([S_ ** mm for mm in m], axis=1)
        X = np.einsum("fm,fms->fs", cx, P0)
        Sn = X.shape[1]
        half = Sn // 2
        mg = (k * X)[:, half:].mean(axis=1)                 # um, phase order
        d = np.roll(mg, -1) - mg                            # cyclic steps
        i_hi, i_lo = int(np.argmax(mg)), int(np.argmin(mg))
        rng = max(float(mg[i_hi] - mg[i_lo]), 0.1 * self.L)
        eps_m = MONO_SWEEP_EPS
        # |step| smoothed at zero without a bias at large steps:
        # s - eps^2 / 2s is |d| + O(eps^4 / d^3)
        s_m = np.sqrt(d * d + eps_m ** 2)
        tv = float(np.sum(s_m - eps_m ** 2 / (2.0 * s_m)))
        excess = max(tv / (2.0 * rng) - 1.0 - MONO_SWEEP_TOL, 0.0)
        short = 0.0
        if self.direction:
            short = max(-self.direction * float(np.sum(d * np.abs(d))), 0.0) * G / (rng * rng)
        if not grad:
            return excess, short
        # d/dm: the smoothed signs of the steps either side of each phase
        # (and the steps themselves for the shortfall), and the two extremes
        g_m = np.zeros(G)
        d_rng = 0.0
        if excess > 0:
            w = (d / s_m) * (1.0 + eps_m ** 2 / (2.0 * s_m * s_m))
            g_m += (np.roll(w, 1) - w) / (2.0 * rng)
            d_rng += -tv / (2.0 * rng * rng)
        if short > 0:
            a2 = -self.direction * 2.0 * np.abs(d) * G / (rng * rng)
            g_m += np.roll(a2, 1) - a2
            d_rng += -2.0 * short / rng
        if float(mg[i_hi] - mg[i_lo]) > 0.1 * self.L:
            g_m[i_hi] += d_rng
            g_m[i_lo] -= d_rng
        dP_dx = np.zeros_like(X)
        dP_dx[:, half:] = (g_m / float(Sn - half))[:, None]
        g_X = dP_dx * k
        gk = np.sum(dP_dx * X, axis=1)
        P1 = np.stack([mm * S_ ** (mm - 1) for mm in m], axis=1)
        X1 = np.einsum("fm,fms->fs", cx, P1)
        slide = self.arc_slide(cx, cz, S_, dP_dx * k * X1, draw_dcx, draw_dcz)
        dk_draw = (-k[:, 0] / np.maximum(raw, 1e-9))
        gcx = (np.einsum("fs,fms->fm", g_X, P0)
               + (gk * dk_draw)[:, None] * draw_dcx + slide[0])
        gcz = (gk * dk_draw)[:, None] * draw_dcz + slide[1]
        gAx, gBx = gcx.T @ Cg, gcx.T @ Sg
        gAz, gBz = gcz.T @ Cg, gcz.T @ Sg
        g = np.concatenate([gAx.ravel(), gAz.ravel(), gBx.ravel(), gBz.ravel()])
        return excess, short, g

    def energy(self, vec, **kw) -> float:
        return self.evaluate(vec, **kw)[0]

    def normalise(self, vec):
        """Fix the gauge: scale so the mean raw arc length over phases is one."""
        cx, cz, _, _ = self.shape_coeffs(vec)
        raw, _ = self.arc_table(cx, cz)
        m = float(np.mean(raw))
        return vec / m if m > 0 else vec

    def free_mask(self, modes=None):
        """Which coefficients move: never the sin terms of mode 0, and only
        the first ``modes`` harmonics when given."""
        nm, ns = self.n_modes, self.n_space
        k = nm * ns
        mask = np.ones(4 * k, bool)
        for blk in (2, 3):
            for mrow in range(ns):
                mask[blk * k + mrow * nm] = False
        if modes is not None:
            sel = np.zeros(4 * k, bool)
            for blk in range(4):
                for mrow in range(ns):
                    sel[blk * k + mrow * nm: blk * k + mrow * nm + int(modes)] = True
            mask &= sel
        return mask

    def optimise(self, vec, iters: int = 200, lr: float = 0.01, modes=None, **kw):
        """Adam ascent on the energy; returns (best energy, best vector)."""
        vec = self.normalise(np.array(vec, dtype=float))
        idx = np.where(self.free_mask(modes))[0]
        P = len(idx)
        mom = np.zeros(P)
        vel = np.zeros(P)
        b1, b2, eps = 0.9, 0.999, 1e-8
        best = (self.energy(vec, **kw), vec.copy())
        for it in range(1, int(iters) + 1):
            E, parts, g = self.evaluate(vec, grad=True, **kw)
            g = g[idx]
            mom = b1 * mom + (1 - b1) * g
            vel = b2 * vel + (1 - b2) * g * g
            vec[idx] += lr * (mom / (1 - b1 ** it)) / (np.sqrt(vel / (1 - b2 ** it)) + eps)
            vec = self.normalise(vec)
            if E > best[0]:
                best = (E, vec.copy())
        E = self.energy(vec, **kw)
        if E > best[0]:
            best = (E, vec.copy())
        return best

    def best_phase_shift(self, vec, n: int = 24):
        """The phase origin at which a given beat fits the frames best."""
        best = None
        for sh in np.linspace(0, 1, n, endpoint=False):
            vv = self.normalise(_fourier_vec(shift_fourier_phase(
                _fourier_dict(vec, self.n_modes, self.n_space), sh),
                self.n_modes, self.n_space))
            e = self.energy(vv)
            if best is None or e > best[0]:
                best = (e, vv, sh)
        return best


def phase_averaged_video(video, model, n_bins: int = 24, period=None,
                         start=None, progress=None):
    """A pseudo-video of ``n_bins`` frames: the mean frame of each phase bin.

    Everything periodic at the base adds coherently; whatever beats at
    another frequency, drifts or just passes through averages out.  Frame
    ``b`` of the result is phase ``b / n_bins``, and the companion model
    carries that period, so the tracer and the fitter use it unchanged.
    Returns (video, model, frames per bin).
    """
    period = float(period if period is not None else model.period_frames)
    start = float(start if start is not None else model.macysta)
    acc = [None] * n_bins
    cnt = np.zeros(n_bins, int)
    total = video.last - video.first + 1
    _prefetch(video)
    for i, f in enumerate(range(video.first, video.last + 1)):
        if progress is not None and i % 10 == 0:
            progress(i / float(total), "phase-averaging frames")
        ph = ((f - start) / period) % 1.0
        b = int(ph * n_bins) % n_bins
        im = np.asarray(video.get(f), dtype=np.float32)
        if im.ndim == 3:
            im = im[..., :3].mean(axis=2)
        acc[b] = im.copy() if acc[b] is None else acc[b] + im
        cnt[b] += 1
    frames = []
    for b in range(n_bins):
        if cnt[b] == 0:
            # an empty bin borrows the nearest filled one
            for d in range(1, n_bins):
                bb = next((q for q in ((b + d) % n_bins, (b - d) % n_bins)
                           if cnt[q] > 0), None)
                if bb is not None:
                    frames.append(acc[bb] / cnt[bb])
                    break
        else:
            frames.append(acc[b] / cnt[b])
    obj = VideoFrames.__new__(VideoFrames)
    obj.directory = getattr(video, "directory", "")
    obj.basename = "phase"
    obj.rotation = 0.0
    obj.crop = None
    obj.enhance = None
    obj._bg_mean = None
    obj._source_shape = frames[0].shape[:2]
    obj.all_paths = ["<phase bin {}>".format(b) for b in range(n_bins)]
    obj.first = 0
    obj.last = n_bins - 1
    obj._cache = {b: frames[b] for b in range(n_bins)}
    obj.is_placeholder = False
    obj.basename_stale = False
    pqd = copy.deepcopy(model.pqd)
    fps = model.fps or 1.0
    pqd.set("vid_first_frame_to_load", 0)
    pqd.set("vid_last_frame_to_load", n_bins - 1)
    pqd.set("vid_frame_main_cycle_start", 0)
    pqd.set("cbf", fps / float(n_bins))
    # The averaged frames are already rotated, cropped and enhanced, so the
    # model must describe them as they are: no rotation, no crop, no
    # enhancement, and the frame size they have.  The base and ppum (after
    # any upsampling) are unchanged.
    pqd.set("vid_image_rotation", 0.0)
    for key in CROP_KEYS:
        pqd.scalars.pop(key, None)
    for key, (pqd_key, default) in ENHANCE_KEYS.items():
        pqd.scalars.pop(pqd_key, None)
    pqd.set("vid_image_w", frames[0].shape[1])
    pqd.set("vid_image_h", frames[0].shape[0])
    pqd.set("vid_ppum", model.ppum)
    return obj, WaveModel(pqd), cnt


def phase_averaged_trace_fit(video, model, n_bins: int = 24, period=None,
                             start=None, progress=None):
    """Trace the cilium on the phase-averaged frames and fit a beat.

    Returns the fit_beat_at_phases report with ``coeffs``, plus the
    traces, their phases and the frames per bin.
    """
    pv, pm, cnt = phase_averaged_video(video, model, n_bins=n_bins,
                                       period=period, start=start,
                                       progress=progress)
    method = "minpath" if frames_are_drawn(video) else "walk"
    info = {}
    _, curves, times = auto_trace(pv, pm, frames=list(range(n_bins)),
                                  info=info, max_traces=n_bins, method=method,
                                  progress=None)
    phases = np.array([pm.phase_of_frame(t * pm.fps / 1000.0) for t in times])
    keep = usable_traces(curves)
    if keep.sum() < 2 * FOURIER_DEFAULT_MODES + 2:
        raise ValueError("only {} usable phase-averaged traces".format(
            int(keep.sum())))
    targets = curves_to_model_frame([c for c, k in zip(curves, keep) if k], pm)
    fit = fit_beat_at_phases(targets, phases[keep],
                             n_modes=FOURIER_DEFAULT_MODES, num_locs=20,
                             n_space=N_FOURIER_M)
    fit["curves"] = curves
    fit["phases"] = phases
    fit["keep"] = keep
    fit["counts"] = cnt
    fit["method"] = method
    return fit


def template_starts(fit, n_shifts: int = 6, extra=()):
    """Starting points for the joint fit: the template beat and its mirror
    image at several phase offsets, plus any supplied coefficient sets."""
    nm, ns = fit.n_modes, fit.n_space
    starts = []
    for name, coeffs in extra:
        if coeffs is not None:
            starts.append((name, fit.normalise(_fourier_vec(coeffs, nm, ns))))
    v0 = _fourier_vec(FOURIER_DEFAULT, nm, ns)
    mir = v0.copy()
    k = nm * ns
    mir[0:k] *= -1
    mir[2 * k:3 * k] *= -1
    for name, v in (("template", v0), ("mirrored template", mir)):
        for sh in np.linspace(0, 1, n_shifts, endpoint=False):
            vv = _fourier_vec(shift_fourier_phase(_fourier_dict(v, nm, ns), sh),
                              nm, ns)
            starts.append(("{} shift {:.2f}".format(name, sh), fit.normalise(vv)))
    return starts


# sigma (px), iterations, learning rate, harmonics released, survivors
IMAGE_FIT_SCHEDULE = (
    (3.0, 150, 0.03, 2, 8),
    (3.0, 150, 0.02, 4, 5),
    (2.0, 150, 0.01, None, 3),
    (1.5, 120, 0.005, None, 1),
)


# The fit a pool worker currently holds: a stage's maps are sent once per
# worker through a temporary file, not once per start.
_WORKER_FIT = {"token": None, "fit": None}


def _optimise_job(job):
    """One start of the beam, in a worker: (fit file, token, vec, kw)."""
    import pickle
    path, token, vec, kw = job
    if _WORKER_FIT["token"] != token:
        _WORKER_FIT["fit"] = None
        with open(path, "rb") as fh:
            _WORKER_FIT["fit"] = pickle.load(fh)
        _WORKER_FIT["token"] = token
    return _WORKER_FIT["fit"].optimise(vec, **kw)


def optimise_beam(fit, vecs, progress=None, **kw):
    """``[fit.optimise(v, **kw) for v in vecs]``, over the CPU cores."""
    vecs = list(vecs)
    n = min(compute_info()["workers"], len(vecs))
    if n > 1 and not _in_worker_process() and not _processes_allowed():
        return parallel_map(lambda v: fit.optimise(v, **kw), vecs,
                            progress=progress, threads=True, workers=n)
    if n <= 1 or _in_worker_process() or _POOLS["broken"]:
        out = []
        for i, v in enumerate(vecs):
            out.append(fit.optimise(v, **kw))
            if progress is not None:
                progress((i + 1) / float(len(vecs)))
        return out
    import pickle
    import tempfile
    import uuid
    fd, path = tempfile.mkstemp(prefix="ciliograph_fit_", suffix=".pkl")
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(fit, fh, protocol=pickle.HIGHEST_PROTOCOL)
        token = uuid.uuid4().hex
        return parallel_map(_optimise_job,
                            [(path, token, v, kw) for v in vecs],
                            progress=progress, workers=n)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def image_fit_beat(video, model, inits=None, n_template_shifts: int = 6,
                   schedule=IMAGE_FIT_SCHEDULE, frames=None, n_frames: int = 96,
                   period=None, start=None, progress=None, verbose=None,
                   sign=None, envelope=None, n_keep: int = 3, direction=None):
    """Fit the beat to the frames from many starts; keep the best energy.

    ``inits`` is a list of (name, coeffs) to start from as well -- the trace
    fits.  Every start runs the coarse stage; only the best survive to the
    finer ones.  ``sign`` and ``envelope`` are passed to BeatImageFit.  The
    last stage keeps ``n_keep`` starts rather than one, and ``spread_um``
    is how far the runners-up within 0.03 of the best energy lie from it:
    where several beats explain the frames equally well, that is how well
    the frames determine the beat.  Returns a dict with ``coeffs``,
    ``energy``, the ranked ``beam`` and the ``fit`` object, which scores
    any beat with ``fit.energy(_fourier_vec(coeffs))``.
    """
    t0 = time.time()
    fit = BeatImageFit(video, model, frames=frames, n_frames=n_frames,
                       sigma=schedule[0][0], period=period, start=start,
                       progress=progress, sign=sign, envelope=envelope,
                       direction=direction)
    nm, ns = fit.n_modes, fit.n_space
    lam_mono_full = fit.lam_mono
    first_mono = max(len(schedule) - int(MONO_SWEEP_LAST_STAGES), 0)
    beam = [dict(name=n, vec=v, energy_init=fit.energy(v))
            for n, v in template_starts(fit, n_template_shifts, inits or [])]
    n_stage = len(schedule)
    for si, (sigma, iters, lr, modes, keep) in enumerate(schedule):
        if si == n_stage - 1:
            keep = max(int(keep), int(n_keep))
        if fit.sigma != sigma:
            fit.set_sigma(sigma)
        fit.lam_mono = lam_mono_full if si >= first_mono else 0.0
        report = None
        if progress is not None:
            def report(frac, si=si, n_beam=len(beam)):
                progress(0.5 + 0.5 * (si + frac) / n_stage,
                         "fitting the beat to the frames: stage {} of {}, "
                         "start {} of {}".format(si + 1, n_stage,
                                                 max(1, int(round(frac * n_beam))),
                                                 n_beam))
            report(0.0)
        # Every start climbs on its own, so the starts of a stage are shared
        # out over the CPU cores; each gets exactly the answer it would alone.
        results = optimise_beam(fit, [b["vec"] for b in beam], iters=iters,
                                lr=lr, modes=modes, progress=report)
        for b, (e, v) in zip(beam, results):
            b["vec"] = v
            b["energy"] = e
        beam.sort(key=lambda b: -b["energy"])
        if verbose:
            verbose("  stage {} (sigma {}, modes {}): ".format(
                si + 1, sigma, modes or "all") + ", ".join(
                "{} {:+.3f}".format(b["name"][:14], b["energy"])
                for b in beam[:keep + 2]))
        beam = beam[:keep]
    if fit.sigma != schedule[-1][0]:
        fit.set_sigma(schedule[-1][0])
    fit.lam_mono = lam_mono_full
    best = beam[0]
    E, parts = fit.evaluate(best["vec"])
    spread = 0.0
    wave_best = FourierWave(_fourier_dict(best["vec"], nm, ns),
                            length=model.length, cbf=model.cbf)
    for b in beam[1:]:
        if best["energy"] - b["energy"] > 0.03:
            continue
        other = FourierWave(_fourier_dict(b["vec"], nm, ns),
                            length=model.length, cbf=model.cbf)
        spread = max(spread, compare_beats(wave_best, other)[0])
    return dict(coeffs=_fourier_dict(best["vec"], nm, ns), energy=E, parts=parts,
                best_start=best["name"], beam=beam, fit=fit,
                n_frames=len(fit.frames), binf=fit.binf, sign=fit.sign,
                mode=fit.mode, spread_um=float(spread),
                seconds=time.time() - t0)


# A beat already fitted is kept by the next Auto fit unless a new start
# beats it by more than this much energy -- the same margin within which
# the readout calls two beats equally supported.  Without it every run
# started afresh from the traces and the template, landed on another of
# the equally supported beats of a dense bundle, and moved the cycle start
# with it: on input_images three runs in a row gave energies +0.235,
# +0.246, +0.237 with beats 0.5 and 1.0 um apart, the last one sweeping
# below the base.
AUTOFIT_KEEP_MARGIN = 0.03


def fourier_autofit(video, model, max_traces: int = 120, progress=None,
                    current=None) -> dict:
    """Auto fit for the Fourier waveform, without the window.

    ``current``: the beat already fitted, if any.  It is a start for the
    joint fit like the others and is scored like them, and it stays the
    answer unless another start beats it by AUTOFIT_KEEP_MARGIN: a second
    Auto fit polishes the first rather than replacing it.

    Three candidates, one judge.  The cilium is traced frame by frame and a
    beat fitted through the traces (the reference method); it is traced
    again on the phase-averaged frames, where everything that does not
    repeat with the beat has blurred away; and the beat is fitted to the
    frames themselves, jointly over the whole record, from those two fits
    and from the template beat at several phase offsets (image_fit_beat).
    The image energy -- how well each beat lies along a dark filament in
    every frame at its own phase -- picks the winner.  On a sparse
    recording the trace fit usually wins or the joint fit polishes it; on
    a dense carpet, where the traces hop between neighbours and the trace
    fit is a fan of scribbles, the joint fit is the only one of the three
    that stays on one beat.

    ``progress(fraction, note)`` reports.  Raises ValueError when no beat
    fits.  Returns the chosen coefficients with everything the fitter's
    readout says about them (the GUI formats it; Auto enhance's pattern
    round compares them across settings).
    """
    def report(frac, note=None):
        if progress is not None:
            progress(frac, note)

    beat = beat_strength(video, progress=report)
    old = model.period_frames
    # The period found on the cycle tab stands.  This used to replace it
    # whenever beat_strength disagreed by a tenth, and on a window of one
    # beat beat_strength answered 4 frames for an 82-frame beat: every
    # phase was then wrong, the fold averaged unrelated shapes into one
    # straight line, and the fan collapsed to a sweep of two microns --
    # with a small residual, since the fit matched the averaged nonsense
    # well.  A disagreement is worth reporting, not acting on.
    period_note = ""
    if beat["period"] > 0 and abs(beat["period"] - old) > 0.1 * old:
        period_note = (
            "NOTE the field repeats every {:.1f} frames by "
            "self-similarity, not {:.1f}; check Find cycle"
            .format(beat["period"], old))

    # 1. the reference method: trace every frame, fit through the traces
    #    at their exact phases
    def sub_report(lo, hi):
        return lambda frac, note=None: report(
            lo + (hi - lo) * float(np.clip(frac, 0.0, 1.0)), note)

    info: dict = {}
    frames = list(range(video.first, video.last + 1))
    trace_fit = None
    trace_err = ""
    n_used = n_down = n_traces = 0
    resid_trace = float("nan")
    try:
        _, curves, times = auto_trace(
            video, model, frames=frames, info=info,
            max_traces=int(max_traces), progress=sub_report(0.0, 0.45))
        phases = np.array([model.phase_of_frame(t * model.fps / 1000.0)
                           for t in times])
        keep = usable_traces(curves)
        n_traces = len(curves)
        if keep.sum() < 2 * FOURIER_DEFAULT_MODES + 2:
            raise ValueError("only {} usable traces".format(int(keep.sum())))
        targets = curves_to_model_frame(
            [c for c, k in zip(curves, keep) if k], model)
        fit = fit_beat_at_phases(targets, phases[keep],
                                 n_modes=FOURIER_DEFAULT_MODES,
                                 num_locs=20, n_space=N_FOURIER_M)
        trace_fit = fit["coeffs"]
        resid_trace = float(fit["resid_um"])
        n_used, n_down = int(keep.sum()), int(fit["n_downweighted"])
    except Exception as exc:
        trace_err = str(exc)

    # 2. the same on the phase-averaged frames
    report(0.47, "tracing the phase-averaged frames")
    avg_fit = None
    avg_err = ""
    resid_avg = float("nan")
    try:
        pa = phase_averaged_trace_fit(video, model)
        avg_fit = pa["coeffs"]
        resid_avg = float(pa["resid_um"])
    except Exception as exc:
        avg_err = str(exc)

    # 3. the joint fit to the frames, from every start.  The polarity is
    #    measured rather than assumed -- bright-field cilia show dark or
    #    bright with the focus, and on these recordings the focus drifts
    #    within a record -- and on a lawn that beats in step, the height
    #    its texture reaches at each phase holds the fit to the height of
    #    the cilium (see band_envelope).
    inits = [(n, c) for n, c in (("trace fit", trace_fit),
                                 ("phase-averaged fit", avg_fit))
             if c is not None]
    if current is not None and np.all(np.isfinite(_fourier_vec(current))):
        inits.append(("previous fit", current))
    report(0.48, "judging the polarity and the lawn")
    try:
        pol = polarity_of(video, model)
    except Exception as exc:
        pol = {"sign": None, "clear": False, "skew": float("nan"),
               "skews": [], "error": str(exc)}
    try:
        sync = lawn_synchrony(video, model)
    except Exception:
        sync = {"spread": float("nan"), "synchronous": False}
    envelope = None
    if sync["synchronous"]:
        try:
            envelope = band_envelope(video, model)[0]
        except (ValueError, IndexError):
            envelope = None
    joint = image_fit_beat(video, model, inits=inits, sign=pol["sign"],
                           envelope=envelope, progress=sub_report(0.5, 1.0),
                           direction=effective_direction(model.pqd))
    scorer = joint["fit"]
    scores = {"joint fit ({})".format(joint["best_start"]):
              (joint["energy"], joint["coeffs"])}
    # The previous beat as the joint fit polished it, when its start
    # survived to the last stage and another start won: with the sweep
    # penalty a beat that had back-and-forth in it scores low as it is,
    # and the polished one is the same beat without the back-and-forth.
    polished = next((b for b in joint["beam"] if b["name"] == "previous fit"),
                    None)
    if polished is not None and joint["best_start"] != "previous fit":
        scores["previous fit polished"] = (
            float(polished["energy"]),
            _fourier_dict(polished["vec"], scorer.n_modes, scorer.n_space))
    for name, c in inits:
        scores[name] = (float(scorer.energy(scorer.normalise(
            _fourier_vec(c)))), c)
    chosen = max(scores, key=lambda k: scores[k][0])
    kept, best_new = False, None
    if "previous fit" in scores:
        others = [k for k in scores
                  if k not in ("previous fit", "previous fit polished")]
        best_new = max(others, key=lambda k: scores[k][0]) if others else None
        if (best_new is None or scores[best_new][0] - scores["previous fit"][0]
                < AUTOFIT_KEEP_MARGIN):
            chosen, kept = "previous fit", True
        elif ("previous fit polished" in scores and scores[best_new][0]
              - scores["previous fit polished"][0] < AUTOFIT_KEEP_MARGIN):
            chosen, kept = "previous fit polished", True
    energy, coeffs = scores[chosen]
    if not np.all(np.isfinite(_fourier_vec(coeffs))):
        raise ValueError("the fit produced non-finite coefficients")
    # The data term lies between -1 (nothing under the curve anywhere) and
    # 3; a strongly negative total means the penalties dominate -- a period
    # of a few frames making every shape change a lurch, or a base inside
    # the tissue -- and no beat in the frames was found.  Better to say so
    # than to draw it.
    if energy < -0.5:
        raise ValueError(
            "no beat fits these frames (energy {:+.2f}): check the period "
            "on the cycle tab ({:.1f} frames) and the base position"
            .format(energy, model.period_frames))
    return dict(coeffs=coeffs, energy=energy, chosen=chosen, scores=scores,
                joint=joint, scorer=scorer, beat=beat, period_note=period_note,
                info=info, trace_fit=trace_fit, trace_err=trace_err,
                n_used=n_used, n_down=n_down, n_traces=n_traces,
                resid_trace=resid_trace, avg_fit=avg_fit, avg_err=avg_err,
                resid_avg=resid_avg, pol=pol, sync=sync, envelope=envelope,
                kept=kept, best_new=best_new)


def _harmonic_series(cos_c, sin_c, per_period: int):
    """Rebuild the node trajectories from harmonic coefficients."""
    k = np.arange(cos_c.shape[0])
    phi = 2.0 * np.pi * np.outer(np.arange(per_period), k) / per_period
    return (np.cos(phi) @ cos_c.reshape(len(k), -1)
            + np.sin(phi) @ sin_c.reshape(len(k), -1)).reshape(
                per_period, *cos_c.shape[1:])


def _repair_spikes(block, max_lurch: float = 2.5):
    """Replace isolated bad phase bins by the mean of their neighbours.

    A lurch -- one bin where the traced shape jumps away and the next comes
    back -- is a tracing failure, and the cure for it used to be damping the
    high time harmonics of every node until no step exceeded twice the
    typical one.  That criterion cannot tell a lurch from the beat itself: a
    real effective stroke moves the tip five times faster than the median
    step (measured on the fitted beats of two recordings), so it was always
    tripped, the damping chosen was the strongest on offer, and modes 2 and
    3 -- the ones carrying the recovery-stroke bend -- were shrunk to a third
    and a fifth.  On a drawn wave whose folded traces swept the tip from -4
    to +2.6 microns the fitted fan reached from -2.7 to +0.3.

    A spike is different from a stroke in shape, not size: both steps around
    it are large and the shape comes back to nearly where it left, whereas a
    stroke's consecutive steps carry on in the same direction.  Only such
    bins are touched, and they are interpolated rather than the whole beat
    being smoothed.  Returns the repaired block and the bins repaired.
    """
    b = np.array(block, dtype=float)
    n = len(b)
    if n < 6:
        return b, []
    def dist(a, c):
        return float(np.sqrt(np.mean(np.sum((a - c) ** 2, axis=-1))))
    steps = np.array([dist(b[i], b[(i + 1) % n]) for i in range(n)])
    typical = float(np.median(steps))
    if typical <= 0:
        return b, []
    fixed = []
    for i in range(n):
        before, after = b[(i - 1) % n], b[(i + 1) % n]
        s_in, s_out = steps[(i - 1) % n], steps[i]
        if s_in > max_lurch * typical and s_out > max_lurch * typical \
                and dist(before, after) < 0.5 * (s_in + s_out):
            fixed.append(i)
    for i in fixed:
        b[i] = 0.5 * (block[(i - 1) % n] + block[(i + 1) % n])
    return b, fixed


def _choose_damping(block, cos_c, sin_c, per_period: int, max_lurch: float = 2.0,
                    grid=(0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)):
    """Pick the harmonic damping: the least that stops the beat lurching.

    A beating cilium moves smoothly, so from one phase sample to the next the
    shape should move by about the same amount throughout the cycle.  Where the
    traces are unreliable -- the fastest part of the stroke, where the blur is
    worst -- consecutive samples disagree, and an undamped fit reproduces that
    disagreement as a lurch: one recording moved 2.1 microns between two frames
    against a typical 0.7, which is not something a cilium does.  So the
    smallest damping that keeps the largest step within ``max_lurch`` times the
    typical one is chosen.

    Held-out error was tried first as the criterion and does not work here: the
    bins held out are noisy in the same way as the bins fitted, so reproducing
    their noise scores well and nothing is ever damped.  Keeping fewer modes
    outright is worse still -- a hard cut removes the harmonics carrying the
    sharply curved part of the stroke along with the noise.
    """
    n = per_period
    if n < 8:
        return 0.0
    chosen = grid[-1]
    for lam in grid:
        shrink = _harmonic_shrink(cos_c.shape[0], lam)
        series = _harmonic_series(cos_c * shrink[:, None, None],
                                  sin_c * shrink[:, None, None], n)
        step = np.sqrt(np.mean(np.sum(np.diff(series, axis=0,
                                              append=series[:1]) ** 2, axis=-1),
                               axis=-1))
        typical = float(np.median(step))
        if typical <= 0:
            return lam
        if float(step.max()) <= max_lurch * typical:
            chosen = lam
            break
    return chosen


def _mean_fourier_length(coeffs, n_time: int = 300, n_space_pts: int = 200):
    """Mean arc length of a unit-scaled Fourier beat over one period.

    Raw polynomial, deliberately: with the length correction on, every phase
    would report exactly one and the renormalisation this feeds would be a
    no-op.
    """
    wave = FourierWave(coeffs, length=1.0, cbf=1.0, fixed_length=False)
    s = np.linspace(0.0, 1.0, n_space_pts)
    total = 0.0
    for i in range(n_time):
        x, z = wave.position(s, i / float(n_time))
        total += float(np.hypot(*np.diff(np.column_stack([x, z]), axis=0).T).sum())
    return total / n_time


def curve_angles(points, n: int):
    """Tangent angle at n-1 equal steps along a polyline, and its arc length."""
    pts, _ = _resample_curve(np.asarray(points, dtype=float), n)
    d = np.diff(pts, axis=0)
    ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return ang, float(np.hypot(d[:, 0], d[:, 1]).sum())


def curve_turning(points, n: int = 40):
    """Total turning along a polyline, in degrees."""
    ang, _ = curve_angles(points, n)
    return float(np.rad2deg(ang[-1] - ang[0]))


def fold_curves(curves, phases, n_bins: int = 12, n_sample: int = 40,
                min_tip_rise: float = 0.0, fill: bool = True,
                max_turn_deg: float = 180.0):
    """Average traced centrelines by cycle phase, robustly.

    In a dense carpet a single frame's trace is unreliable -- the walk can
    follow a neighbour -- but the cilium at the base is the only thing in the
    picture that repeats with the beat period, so binning the traces by phase
    and averaging keeps it and discards the excursions.  The median, not the
    mean, because a trace that jumped to a neighbour is an outlier of exactly
    the size that would drag a mean off.

    The averaging is done on the tangent angle along the arc, not on x and y.
    Averaging positions pulls a bundle of differently-bent curves towards their
    common middle, which shortens them and can produce a result that loops --
    folded medians turning through 376 degrees, which is not a cilium and which
    a fit can only answer with a nearly straight model.  Averaging the angle
    profile and integrating it back keeps every result a smooth curve of the
    right length, and it is the same description the waveform model uses.

    Traces that turn more than ``max_turn_deg`` in total are dropped first: a
    cilium of a few microns does not curl through half a circle, so those are
    tracing failures rather than data.  So are traces whose tip has come
    back to within ``min_tip_rise`` of a length of the base.

    Returns ``(bin_phases, curves, counts)`` for every bin when ``fill`` is
    on; a count of zero marks a bin interpolated from its neighbours because
    it kept no usable trace.
    """
    phases = np.asarray(phases, dtype=float)
    keep, angles, lengths, bases = [], [], [], []
    for i, c in enumerate(curves):
        c = np.asarray(c, dtype=float)
        if len(c) < 3:
            continue
        ang, arc = curve_angles(c, n_sample)
        if arc <= 0 or abs(np.rad2deg(ang[-1] - ang[0])) > max_turn_deg:
            continue
        # A cilium is anchored at the wall, so its tip stands clear of the
        # base at every phase; a trace whose tip has come back down to the
        # base -- or below it, which the walk managed in a third of the
        # frames of one recording -- has followed something else.  Folding
        # those in gave a fan whose tip touched the wall.  Image rows grow
        # downwards, so "above" is a smaller row.
        if (c[0, 1] - c[-1, 1]) < min_tip_rise * arc:
            continue
        keep.append(i)
        angles.append(ang)
        lengths.append(arc)
        bases.append(c[0])
    if not keep:
        raise ValueError("every trace failed the turning check")

    keep = np.array(keep)
    angles = np.array(angles)
    lengths = np.array(lengths)
    bases = np.array(bases)

    out_phase, out_curve, counts = [], [], []
    for b in range(n_bins):
        sel = (phases[keep] >= b / n_bins) & (phases[keep] < (b + 1) / n_bins)
        if not sel.any():
            continue
        ang = np.median(angles[sel], axis=0)
        arc = float(np.median(lengths[sel]))
        step = arc / len(ang)
        base = np.median(bases[sel], axis=0)
        pts = np.empty((len(ang) + 1, 2))
        pts[0] = base
        pts[1:, 0] = base[0] + np.cumsum(np.cos(ang) * step)
        pts[1:, 1] = base[1] + np.cumsum(np.sin(ang) * step)
        out_phase.append((b + 0.5) / n_bins)
        out_curve.append(pts)
        counts.append(int(sel.sum()))
    if not out_curve:
        raise ValueError("no phase bin kept any trace")
    if fill and len(out_curve) < n_bins and len(out_curve) >= 3:
        # The transform in time downstream needs every bin, equally spaced.
        # A bin that kept no usable trace is interpolated between its
        # nearest kept neighbours on either side, going round the cycle, so
        # the phases stay uniform and the gap is bridged smoothly rather
        # than closed up -- closing it up would shift every later phase.
        kept = {int(round(ph * n_bins - 0.5)): (ph, c, n)
                for ph, c, n in zip(out_phase, out_curve, counts)}
        full_ph, full_c, full_n = [], [], []
        for b in range(n_bins):
            if b in kept:
                ph, c, n = kept[b]
                full_ph.append(ph); full_c.append(c); full_n.append(n)
                continue
            lo = next(k for k in range(1, n_bins) if (b - k) % n_bins in kept)
            hi = next(k for k in range(1, n_bins) if (b + k) % n_bins in kept)
            c_lo = kept[(b - lo) % n_bins][1]
            c_hi = kept[(b + hi) % n_bins][1]
            t = lo / float(lo + hi)
            full_ph.append((b + 0.5) / n_bins)
            full_c.append((1.0 - t) * c_lo + t * c_hi)
            full_n.append(0)
        out_phase, out_curve, counts = full_ph, full_c, full_n
    return np.array(out_phase), out_curve, np.array(counts)


def unit_shapes(wave, n_cs: int, n_pts: int = 201):
    """The beat's shapes at unit length, at n_cs phases: (x, z) arrays that
    scale to any cilium length (the series is proportional to L)."""
    keep = wave.length
    wave.length = 1.0
    try:
        return [wave.shape_at(i / n_cs, n_pts=n_pts) for i in range(n_cs)]
    finally:
        wave.length = keep


def estimate_length(video, model, lo: float = 2.0, hi: float = 14.0,
                    step: float = 0.1, n_cs: int = 20, mask_pad: int = 8,
                    progress=None):
    """Estimate the cilium length from how far the beating reaches.

    The model's swept region is grown and shrunk along its own curvature
    profile and matched against the motion map; the best length is where the
    swept region covers the moving area without spilling into still fluid.

    This is deliberately reported with a range.  Segmenting a single cilium
    from these images is not reliable -- neighbouring cilia touch, contrast is
    low, and a region grown from the base leaks into its neighbours -- so treat
    the number as a starting point and confirm it against the image.  Clicking
    the tip in the GUI is considerably more accurate.
    """
    motion, mean_img = motion_and_mean(video, progress=progress)
    h, w = motion.shape
    surface = detect_surface(static_mean(video, mean_img), valid=video.valid)
    rows = np.arange(h)[:, None]
    M = motion.copy()
    M[rows > (surface[None, :] - mask_pad)] = 0.0
    Mc = M - M.mean()
    norm = np.linalg.norm(Mc)
    if norm <= 0:
        raise ValueError("no motion detected in the frames")
    Mn = Mc / norm

    ppum = model.ppum
    x0 = model.pqd.getf("vid_cil_x0")
    y0 = h - model.pqd.getf("vid_cil_y0_invy")

    # The beat's shapes scale with the length (R = L * sum ...), so the
    # unit-length shapes are computed once, with nodes dense enough for the
    # longest candidate, and scaled to each length tried.
    keep = model.length
    n_pts = max(int(hi / model.dl) + 1, 2)
    unit = unit_shapes(model.wave, n_cs, n_pts)
    lengths = np.arange(lo, hi + 1e-9, step)
    scores = []
    for li, L in enumerate(lengths):
        if progress is not None:
            progress(li / len(lengths), "trying lengths")
        mask = np.zeros((h, w))
        for x, y in unit:
            px = (x0 + L * x * ppum).astype(int)
            py = (y0 - L * y * ppum).astype(int)
            ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            mask[py[ok], px[ok]] = 1.0
        mc = mask - mask.mean()
        nn = np.linalg.norm(mc)
        scores.append(float((mc * Mn).sum() / nn) if nn > 0 else -np.inf)

    scores = np.array(scores)
    best = float(lengths[int(np.argmax(scores))])
    span = scores.max() - scores.min()
    near = scores >= scores.max() - 0.02 * span
    # On a drawn carpet the cilia are individually visible, and the highest
    # ink above the wall over the record is the tip of the most upright
    # shape -- a cilium is never shorter than its own reach.  The motion
    # match above gave 5.5 microns for a 6-micron cilium; the reach was
    # 5.86.  It is a floor, not the answer, because the most upright shape
    # still bends a little; two per cent is what the published pattern
    # bends by at its straightest.
    reach_um = None
    if frames_drawn_source(video):
        wall = wall_line(static_mean(video, mean_img))
        highest = None
        for f in range(video.first, video.last + 1,
                       max(1, (video.last - video.first + 1) // 40)):
            img = np.asarray(video.get(f), dtype=np.float32)
            if img.ndim == 3:
                img = img[..., :3].mean(axis=2)
            ink = (1.0 - img / max(float(img.max()), 1e-6)) > 0.5
            rows_ink = np.nonzero(ink.any(axis=1))[0]
            if len(rows_ink):
                top = int(rows_ink.min())
                highest = top if highest is None else min(highest, top)
        if highest is not None and ppum:
            reach_um = float((np.median(wall) - highest) / ppum)
            floor_um = reach_um / 0.98
            if floor_um > best:
                best = float(np.clip(floor_um, lo, hi))
                near = near | (np.abs(lengths - best) < step)

    # The estimate is measured outwards from the base, so a base in the wrong
    # place shifts it directly.  Report how far the base sits from the surface
    # the cilia grow out of, since that is the usual reason for a poor answer.
    x0i = int(np.clip(x0, 0, w - 1))
    surface_row = float(surface[x0i])
    above_surface_px = surface_row - y0
    return {
        "length": best,
        "reach_um": reach_um,
        "low": float(lengths[near].min()),
        "high": float(lengths[near].max()),
        "lengths": lengths,
        "scores": scores,
        "current": keep,
        "base_x0": float(x0),
        "base_row": float(y0),
        "surface_row": surface_row,
        "base_above_surface_px": float(above_surface_px),
        "base_above_surface_um": float(above_surface_px / ppum) if ppum else 0.0,
    }


def crop_tuple(crop):
    """Normalise a crop to four ints (x0, y0, x1, y1) or None."""
    if crop is None:
        return None
    x0, y0, x1, y1 = [int(round(float(v))) for v in crop]
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return (x0, y0, x1, y1)


def clip_crop(crop, shape):
    """The crop held inside an image of ``shape``."""
    h, w = shape[0], shape[1]
    x0, y0, x1, y1 = crop
    x0, x1 = max(0, min(x0, w - 1)), max(1, min(x1, w))
    y0, y1 = max(0, min(y0, h - 1)), max(1, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        return (0, 0, w, h)
    return (x0, y0, x1, y1)


CROP_KEYS = ("vid_crop_x0", "vid_crop_y0", "vid_crop_x1", "vid_crop_y1")

# Frame enhancement that feeds the measurements.  Every step downstream --
# the surface, the base, the period, the traces, the fit -- reads frames
# through VideoFrames.get, so a filter applied there is applied to all of
# them, and the overlay is drawn on what was measured.  The display-only
# contrast controls are deliberately separate.
ENHANCE_KEYS = {
    "background": ("vid_enh_background", 0),   # 1: subtract the temporal mean
    "flatten": ("vid_enh_flatten", 0),         # window px: remove illumination
    "denoise": ("vid_enh_denoise", 0.0),       # Gaussian sigma px
    "sharpen": ("vid_enh_sharpen", 0.0),       # unsharp-mask amount
    "invert": ("vid_enh_invert", 0),           # 1: bright cilia -> dark
    "upsample": ("vid_enh_upsample", 1),       # x1, x2, x3, x4
    # Classical counterparts of the photo-enhancer tools (HitPaw and the
    # like), chosen so that none of them can draw a cilium that is not in
    # the frames: each is a fixed filter whose effect is the same wherever
    # it is applied.
    "deflicker": ("vid_enh_deflicker", 0),     # 1: even out frame brightness
    "deblock": ("vid_enh_deblock", 0),         # 1: remove JPEG 8x8 blocks
    "linesmooth": ("vid_enh_linesmooth", 0.0), # px: denoise along the cilia
    "deblur": ("vid_enh_deblur", 0.0),         # px: blur sigma to undo
    "contrast": ("vid_enh_contrast", 0.0),     # CLAHE clip limit (0: off)
    "upmethod": ("vid_enh_upmethod", 0),       # upsampling: 0 bilinear, 1 Lanczos
    # A neural network from a file, run through onnxruntime (optional):
    # any image-to-image ONNX model -- a denoiser, a deblurrer, a
    # super-resolver.  See OnnxEnhancer.
    "model": ("vid_enh_model", ""),            # path of an .onnx file, or none
    "modelrange": ("vid_enh_model_range", 0),  # pixel range the model expects:
                                               # 0: 0..1, 1: 0..255, 2: -1..1
    # Fiji's tools for thin structures (Oct 2026): what a Fiji user reaches
    # for to make one cilium stand out -- see the functions named.
    "bgmedian": ("vid_enh_bg_median", 0),      # 1: background = temporal median
                                               # (Z Project median), not the mean
    "bandpass": ("vid_enh_bandpass", 0.0),     # px: FFT bandpass, large cutoff (0 off)
    "bpsmall": ("vid_enh_bp_small", 2.0),      # px: its small cutoff
    "tophat": ("vid_enh_tophat", 0.0),         # px: black top-hat disc radius (0 off)
    "lines": ("vid_enh_lines", 0.0),           # px: line filter, line width (0 off)
}
MODEL_RANGES = ("0..1", "0..255", "-1..1")
LINES_SOFTEN = 0.8        # px: the line filter's drawing is softened by this
LINES_SNR = 6.0           # a line this many noise levels strong is fully dark
LINES_FLOOR = 2.0         # ... and below this many it is not drawn at all
LINES_ALONG = 2.0         # px: the response is averaged this far along lines
LINES_SURFACE_MARGIN = 4  # px above the tissue surface left blank (its edge
                          # is a strong line too)

# Wiener regularisation for deblur: caps the gain at (1 + r) / (2 sqrt r),
# about 5x, wherever the blur had taken the signal below the noise.
DEBLUR_REG = 0.01


def clean_model_path(value) -> str:
    """The model setting as a path, or "" for none."""
    text = str(value if value is not None else "").strip()
    return "" if text.lower() in ("", "none", "0", "0.0", "-") else text


def enhance_from_pqd(pqd) -> dict:
    """The enhancement settings recorded in a specification (defaults when absent)."""
    out = {}
    for name, (key, default) in ENHANCE_KEYS.items():
        if isinstance(default, str):
            out[name] = clean_model_path(pqd.get(key, default))
            continue
        v = pqd.getf(key, float(default))
        out[name] = int(round(v)) if isinstance(default, int) else float(v)
    out["upsample"] = int(np.clip(out["upsample"], 1, 4))
    out["upmethod"] = int(np.clip(out["upmethod"], 0, 1))
    out["modelrange"] = int(np.clip(out["modelrange"], 0, len(MODEL_RANGES) - 1))
    return out


def enhance_defaults() -> dict:
    return {n: d for n, (k, d) in ENHANCE_KEYS.items()}


def enhance_is_default(settings: dict) -> bool:
    for n, (k, d) in ENHANCE_KEYS.items():
        v = settings.get(n, d)
        if isinstance(d, str):
            if clean_model_path(v):
                return False
        elif n == "bpsmall" or n == "bgmedian":
            continue            # mean nothing on their own (see below)
        elif v != d:
            return False
    return True


def enhance_to_pqd(pqd, settings: dict) -> None:
    """Write the settings into a specification.  The model path is written as
    "none" when there is none: an empty value would not survive a reload
    (the file's parser strips the line), and the key would be appended again
    on every save."""
    for name, (key, default) in ENHANCE_KEYS.items():
        v = settings.get(name, default)
        if isinstance(default, str):
            v = clean_model_path(v) or "none"
        pqd.set(key, v)


def enhance_frame(img, settings: dict, background=None, refs=None,
                  frame=None):
    """Apply the enhancement settings to one frame (0..255 scale).

    Order: deflicker, invert, the network, background removal, bandpass,
    illumination flattening, top-hat, denoise, line-preserving denoise,
    deblur, sharpen, the line filter, local contrast, upsample.  (JPEG
    deblocking comes earlier still, on the decoded file before rotation --
    see VideoFrames._full -- because the block grid is the file's.)
    Background removal subtracts the temporal mean (or median) of the
    frames and puts the result back around mid-grey, which takes out
    everything that does not move -- the tissue edge, debris, uneven
    lighting.  Flattening subtracts a wide local average, so a brightness
    gradient across the field goes while lines stay.  Denoising is a small
    Gaussian blur; sharpening an unsharp mask.  Upsampling is bilinear or
    Lanczos and only helps a tracer whose filters are sized in pixels see a
    small recording at a workable scale -- ppum and the base are scaled with
    it.  The other steps are described at their own functions.

    ``refs`` holds what some steps need from the whole record, computed
    once by VideoFrames: the deflicker parameters (which need the frame's
    number, ``frame``), the contrast map, and the neural network ("ai",
    an OnnxEnhancer) when the model setting names one -- it runs right
    after deflicker and invert, on the frame as recorded, which is what
    such networks are trained on.  A model that enlarges the frame keeps
    its size only when the upsample setting asks for that same factor;
    otherwise its output is brought back to the frame's own size.
    """
    a = np.asarray(img)
    native = a.shape[:2]
    return _enhance_post(_enhance_pre(img, settings, background, refs, frame),
                         settings, refs, native)


def _enhance_pre(img, settings: dict, background=None, refs=None,
                 frame=None):
    """enhance_frame up to, not including, the local contrast step."""
    refs = refs or {}
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    native = a.shape
    fl = refs.get("flicker")
    if settings.get("deflicker") and fl is not None and frame is not None:
        i = int(frame) - int(fl["first"])
        if 0 <= i < len(fl["gain"]):
            a = deflicker_frame(a, fl, i)
    if settings.get("invert"):
        a = 255.0 - a
    ai = refs.get("ai")
    if clean_model_path(settings.get("model")) and ai is not None:
        cache = refs.get("ai_cache")
        key = None if (cache is None or frame is None) else cache(frame)
        out = model_output_cached(key) if key else None
        if out is None:
            out = ai.run(a)
            if key:
                model_output_store(key, out)
        a = out
        up = int(settings.get("upsample") or 1)
        if a.shape != native and a.shape != (native[0] * up, native[1] * up):
            a = resample_lanczos(a, 1.0, shape=native)
    if settings.get("background") and background is not None:
        # The background is the mean of the frames as they were read; once
        # the frame is inverted it must be inverted too.  Subtracting the
        # plain mean from an inverted frame doubled every static feature
        # instead of removing it.
        bg = (255.0 - background) if settings.get("invert") else background
        if bg.shape != a.shape:
            bg = resample_lanczos(bg, 1.0, shape=a.shape)
        a = a - bg + float(np.mean(bg))
    bp = float(settings.get("bandpass") or 0.0)
    if bp > 0:
        a = bandpass_filter(a, float(settings.get("bpsmall") or 0.0), bp)
    win = int(settings.get("flatten") or 0)
    if win >= 3:
        local = gaussian_blur(a, win / 2.0)
        a = a - local + float(a.mean())
    th = float(settings.get("tophat") or 0.0)
    if th > 0:
        a = _keep_drawing(tophat_drawing(a, th), refs)
    sig = float(settings.get("denoise") or 0.0)
    if sig > 0:
        a = gaussian_blur(a, sig)
    ls = float(settings.get("linesmooth") or 0.0)
    if ls > 0:
        a = line_smooth(a, ls)
    db = float(settings.get("deblur") or 0.0)
    if db > 0:
        a = deblur_gaussian(a, db)
    amt = float(settings.get("sharpen") or 0.0)
    if amt > 0:
        a = a + amt * (a - gaussian_blur(a, 1.0))
    lw = float(settings.get("lines") or 0.0)
    if lw > 0:
        a = _keep_drawing(line_drawing(a, lw), refs)
    return a


def _keep_drawing(a, refs):
    """A drawing (top-hat, line filter) blanked where it must not draw:
    outside the real image and from the tissue surface down (see
    drawing_keep_mask).  Without this the tracer, starting at the base,
    ran along the tissue's edge, and on a levelled recording to the edge
    of the rotation padding, 25 um from the cilium."""
    keep = (refs or {}).get("draw_keep")
    if keep is None or keep.shape != a.shape:
        return a
    return np.where(keep, a, np.float32(255.0)).astype(np.float32)


def drawing_keep_mask(valid, surface, margin: int):
    """Pixels a drawing may hold: ``margin`` px inside the real image
    (``valid``) and more than ``margin`` rows above ``surface``."""
    valid = np.asarray(valid, dtype=bool)
    h, w = valid.shape
    r = int(margin)
    b = np.pad(valid, r, mode="constant", constant_values=False)
    keep = np.ones((h, w), dtype=bool)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            keep &= b[r + dy:r + dy + h, r + dx:r + dx + w]
    rows = np.arange(h)[:, None]
    surf = np.asarray(surface, dtype=float)
    if surf.shape == (w,):
        keep &= rows < (surf[None, :] - r)
    return keep


def _enhance_post(a, settings: dict, refs=None, native=None):
    """The rest of enhance_frame: local contrast, upsampling, the clip.
    ``native`` is the frame's size before any step; the result is always
    that size times the upsample setting."""
    refs = refs or {}
    if float(settings.get("contrast") or 0.0) > 0 and refs.get("clahe") is not None:
        a = apply_clahe(a, refs["clahe"])
    up = int(settings.get("upsample") or 1)
    if native is None:
        native = a.shape
    target = (int(native[0]) * up, int(native[1]) * up)
    if tuple(a.shape) != target:
        if a.shape == tuple(native) and int(settings.get("upmethod") or 0) == 0:
            a = resample_frame(a, float(up))
        else:
            a = resample_lanczos(a, float(up), shape=target)
    return np.clip(a, 0.0, 255.0)


# ---- the classical enhancement tools --------------------------------------
#
# Each one is the textbook, non-learned counterpart of a tool in the AI photo
# enhancers: "remove JPEG artifacts" (re-application of JPEG), "low light"
# (contrast-limited adaptive histogram equalisation), "unblur" (Wiener
# deconvolution), "denoise" (smoothing along the local line direction), and
# "upscale" (Lanczos interpolation).  The AI versions are trained on everyday
# photographs and fill in what they expect to see; these only rearrange what
# is in the frame, so a filament they show is a filament the camera saw.

# Lamp flicker and exposure jitter change from one frame to the next; the
# beat changes the brightness of a small field too -- on input_images4 the
# mean of its stillest pixels swings by 4.7 grey levels at exactly the beat
# period, because there almost everything in view is cilia.  So only the
# part of the brightness that changes faster than this many frames is taken
# to be flicker, and the beat's own slower swing is left alone.  It assumes a
# beat longer than about ten frames (any CBF below 30 Hz at 300 fps).
FLICKER_WINDOW = 5


def _running_mean(x, window: int):
    x = np.asarray(x, dtype=np.float64)
    k = int(max(1, window))
    pad = np.pad(x, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def _still_pixels(stack, valid=None):
    st = np.asarray(stack, dtype=np.float32)
    mean = st.mean(axis=0)
    sd = st.std(axis=0)
    ok = np.ones(mean.shape, bool) if valid is None else np.asarray(valid, bool)
    ok = ok & (mean > 2.0) & (mean < 253.0)
    if ok.sum() < 16:
        ok = np.ones(mean.shape, bool)
    return mean, ok & (sd <= np.percentile(sd[ok], 40))


def frame_flicker(stack, valid=None) -> float:
    """Frame-to-frame brightness jitter, in grey levels: the part of the
    mean of the stillest pixels that changes faster than FLICKER_WINDOW
    frames.  Pass consecutive frames."""
    st = np.asarray(stack, dtype=np.float32)
    if len(st) < 2 * FLICKER_WINDOW:
        return 0.0
    _mean, still = _still_pixels(st, valid)
    series = st[:, still].mean(axis=1).astype(np.float64)
    return float(np.std(series - _running_mean(series, FLICKER_WINDOW)))


def flicker_reference(frames, valid=None):
    """(temporal mean, mask of still pixels) for flicker_params: the still
    pixels are the 40 per cent that vary least over the record and are
    neither padding nor saturated."""
    return _still_pixels(frames, valid)


def flicker_params(frames, mean, still):
    """Per-frame gain and offset that bring each frame's still pixels onto
    the record's mean (least squares), split into their slow part -- the
    beat, kept -- and their fast part -- flicker, removed.  ``frames`` are
    every frame of the record in order (an iterable)."""
    y = mean[still].astype(np.float64)
    gains, offsets = [], []
    for a in frames:
        x = np.asarray(a, dtype=np.float32)[still].astype(np.float64)
        vx = x.var()
        g = 1.0
        if vx > 1e-6 and x.size >= 16:
            g = float(np.clip(((x - x.mean()) * (y - y.mean())).mean() / vx,
                              0.5, 2.0))
        gains.append(g)
        offsets.append(float(y.mean() - g * x.mean()))
    g = np.array(gains)
    b = np.array(offsets)
    return {"gain": g, "offset": b,
            "gain_slow": _running_mean(g, FLICKER_WINDOW),
            "offset_slow": _running_mean(b, FLICKER_WINDOW)}


def deflicker_frame(a, params: dict, i: int):
    """Take the flicker out of frame ``i`` (counted from the first frame).

    A microscope lamp that pulses with the mains, or a camera's exposure
    control, moves every pixel of a frame together; left in, it lands in
    frame - mean as a film over the whole field that the tracer reads as
    motion.  The frame is mapped by its own gain and offset onto the
    record's mean and back by the slow (beat) part of them, so what changes
    is only the frame-to-frame jitter."""
    g, b = params["gain"][i], params["offset"][i]
    gs, bs = params["gain_slow"][i], params["offset_slow"][i]
    return ((a * np.float32(g) + np.float32(b - bs)) / np.float32(gs)).astype(
        np.float32)


def jpeg_quant_table(path):
    """The luminance quantisation table of a JPEG file, or None."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            q = getattr(im, "quantization", None)
            if q:
                return [int(v) for v in q[0]]
    except Exception:
        pass
    return None


def jpeg_deblock(grey, qtable, step: int = 1):
    """Remove JPEG block edges by re-application of JPEG (Nosratinia 2001).

    The frame is compressed again with its own quantisation table on every
    shift of the 8x8 block grid and the shifted results are averaged.  Each
    recompression keeps what the file could represent and discards the
    rest; only the original grid lines up with the file's own blocks, so the
    average keeps the picture and spreads the block edges into nothing.  On
    input_images2 (95 % quality) the edge excess at block borders fell from
    26 % to 8 % across and 83 % to 21 % down, changing the frame by 0.6 grey
    levels on average.  ``step`` 1 uses all 64 shifts, 2 uses 16.
    """
    import io
    from PIL import Image
    a = np.asarray(grey, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    h, w = a.shape
    pad = 8
    big = np.pad(a, pad, mode="reflect")
    acc = np.zeros((h, w), dtype=np.float64)
    n = 0
    q = [int(v) for v in qtable]
    for dy in range(0, 8, int(step)):
        for dx in range(0, 8, int(step)):
            sub = big[pad - dy:pad - dy + h + 8, pad - dx:pad - dx + w + 8]
            buf = io.BytesIO()
            Image.fromarray(np.clip(np.rint(sub), 0, 255).astype(np.uint8),
                            "L").save(buf, "JPEG", qtables=[q])
            buf.seek(0)
            with Image.open(buf) as im:
                dec = np.asarray(im.convert("L"), dtype=np.float64)
            acc += dec[dy:dy + h, dx:dx + w]
            n += 1
    return (acc / n).astype(np.float32)


def jpeg_blockiness(stack) -> float:
    """How much stronger edges are at 8-px block borders than elsewhere, the
    worse of across and down: 1.0 for a frame with no blocking, 1.83 for
    input_images2."""
    st = np.asarray(stack, dtype=np.float64)
    out = []
    for axis in (-1, -2):
        g = np.abs(np.diff(st, axis=axis))
        g = np.moveaxis(g, axis, -1).reshape(-1, g.shape[axis]).mean(axis=0)
        if len(g) < 16:
            continue
        at = g[7::8].mean()
        other = np.delete(g, np.arange(7, len(g), 8)).mean()
        out.append(at / max(other, 1e-9))
    return float(max(out)) if out else 1.0


def _gauss32(a, sigma: float):
    """A true Gaussian blur in float32, by shifted sums (edges held).  For
    the structure tensor in line_smooth, where gaussian_blur's float64 box
    passes were half the cost."""
    a = np.asarray(a, dtype=np.float32)
    r = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k = (k / k.sum()).astype(np.float32)
    h, w = a.shape
    b = np.pad(a, ((r, r), (0, 0)), mode="edge")
    tmp = np.zeros((h, w), dtype=np.float32)
    for i, kk in enumerate(k):
        tmp += kk * b[i:i + h]
    b = np.pad(tmp, ((0, 0), (r, r)), mode="edge")
    out = np.zeros((h, w), dtype=np.float32)
    for i, kk in enumerate(k):
        out += kk * b[:, i:i + w]
    return out


def line_smooth(a, sigma: float, n_orient: int = 8):
    """Denoise along the cilia, not across them.

    The local direction of every line comes from the structure tensor (the
    smoothed outer product of the gradient), and each pixel is averaged with
    its neighbours *along* that direction only, over a Gaussian of
    ``sigma`` pixels.  Where nothing is oriented -- open fluid, noise -- the
    tensor's coherence is low and a small isotropic blur is used instead.  A
    plain Gaussian blur of the same noise reduction spreads a 2-pixel cilium
    across its neighbours: on a test pattern of blurred curved lines with
    noise it cost 17 % of their contrast, and this, at ``sigma`` 2, 4 %.

    Eight directions are filtered and each pixel blends the two nearest to
    its own (more made no measurable difference).  Each directional filter
    steps one whole pixel at a time along the axis the line runs closer to
    and interpolates only across it, which halves the work of sampling the
    line bilinearly.
    """
    a = np.asarray(a, dtype=np.float32)
    h, w = a.shape
    sigma = float(sigma)
    s = _gauss32(a, 1.0)
    gy, gx = np.gradient(s)
    rho = max(1.5, sigma)
    jxx = _gauss32(gx * gx, rho)
    jxy = _gauss32(gx * gy, rho)
    jyy = _gauss32(gy * gy, rho)
    line_dir = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.float32(np.pi / 2)
    coh = (np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
           / np.maximum(jxx + jyy, np.float32(1e-12))) ** 2
    reach = 2.5 * sigma
    P = int(np.ceil(reach)) + 2
    b = np.pad(a, P, mode="edge")
    out = np.zeros_like(a)
    tmp = np.empty_like(a)
    step = np.pi / n_orient
    for k in range(n_orient):
        phi = k * step
        c, sn = np.cos(phi), np.sin(phi)         # direction (x, y)
        along_x = abs(c) >= abs(sn)
        unit = abs(c) if along_x else abs(sn)    # line length per pixel step
        J = int(np.floor(reach * unit))
        js = np.arange(-J, J + 1)
        g = np.exp(-0.5 * (js / unit / sigma) ** 2)
        g /= g.sum()
        acc = np.zeros_like(a)
        for j, gg in zip(js, g):
            if along_x:
                dx, dy = int(j), j * sn / c
                iy = int(np.floor(dy))
                f = np.float32(dy - iy)
                t0 = b[P + iy:P + iy + h, P + dx:P + dx + w]
                t1 = b[P + iy + 1:P + iy + 1 + h, P + dx:P + dx + w]
            else:
                dy, dx = int(j), j * c / sn
                ix = int(np.floor(dx))
                f = np.float32(dx - ix)
                t0 = b[P + dy:P + dy + h, P + ix:P + ix + w]
                t1 = b[P + dy:P + dy + h, P + ix + 1:P + ix + 1 + w]
            np.multiply(t0, np.float32(gg) * (1 - f), out=tmp)
            acc += tmp
            if f:
                np.multiply(t1, np.float32(gg) * f, out=tmp)
                acc += tmp
        d = np.abs(((line_dir - np.float32(phi)) + np.float32(np.pi / 2))
                   % np.float32(np.pi) - np.float32(np.pi / 2))
        out += np.clip(1.0 - d / np.float32(step), 0.0, 1.0) * acc
    iso = _gauss32(a, 0.75 * sigma)
    return coh * out + (1 - coh) * iso


def deblur_gaussian(a, sigma: float, reg: float = DEBLUR_REG):
    """Undo a Gaussian blur of ``sigma`` pixels by Wiener deconvolution.

    The optics spread every point over a small disc -- at 40x and NA 0.65
    the diffraction limit alone is a Gaussian of about 0.2 um, two pixels at
    12 px/um, and defocus adds to it -- and that is what makes a cilium a
    pale smear rather than a dark line.  Dividing the blur back out in the
    Fourier domain restores the line's contrast; the regularisation stops
    the division where the blur left only noise.  On a test pattern blurred
    by 2 px the line contrast rose by half.  The PSF should match the real
    blur: too small a sigma amplifies noise without sharpening anything.
    The frame is mirrored at its edges first, so the edges do not ring.
    """
    a = np.asarray(a, dtype=np.float64)
    sigma = float(sigma)
    pad = int(np.ceil(4 * sigma)) + 8
    b = np.pad(a, pad, mode="reflect")
    fy = np.fft.fftfreq(b.shape[0])[:, None]
    fx = np.fft.rfftfreq(b.shape[1])[None, :]
    otf = np.exp(-2.0 * np.pi ** 2 * sigma ** 2 * (fx * fx + fy * fy))
    filt = otf * (1.0 + reg) / (otf * otf + reg)      # unit gain at zero
    x = np.fft.irfft2(np.fft.rfft2(b) * filt, s=b.shape)
    return x[pad:pad + a.shape[0], pad:pad + a.shape[1]].astype(np.float32)


# ---- Fiji's tools for thin structures ---------------------------------------
#
# What a Fiji user does to make a filament stand out before measuring it:
# Process > FFT > Bandpass Filter, Process > Filters > Top Hat, Plugins >
# Analyze > Tubeness (or Frangi Vesselness), and Image > Stacks > Z Project
# with "median" for a background to subtract.  The same four, in numpy.
# All of them take dark cilia on a lighter field, as bright-field shows
# them; for bright cilia, Invert comes first.

def bandpass_filter(a, small: float, large: float):
    """Fiji's FFT bandpass: keep structures between ``small`` and ``large``
    pixels across.  A difference of Gaussians does the same: the frame
    smoothed to the small size, less the frame smoothed to the large size,
    put back around its mean.  Pixel noise and slow illumination go in one
    step; a 2-pixel cilium between 2 and 40 stays."""
    a = np.asarray(a, dtype=np.float32)
    small, large = float(small), float(large)
    if large <= 0:
        return a
    lo = _gauss32(a, max(0.3, small / 2.0)) if small > 0 else a
    hi = _gauss32(a, max(0.5, large / 2.0))
    return (lo - hi + float(a.mean())).astype(np.float32)


def _disc_offsets(radius: float):
    r = int(np.ceil(radius))
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
    keep = (ox * ox + oy * oy) <= radius * radius + 1e-9
    return r, list(zip(oy[keep].tolist(), ox[keep].tolist()))


def grey_closing(a, radius: float):
    """Morphological closing with a disc: dilation (max) then erosion (min).
    Fills every dark feature narrower than the disc."""
    a = np.asarray(a, dtype=np.float32)
    r, offs = _disc_offsets(radius)
    h, w = a.shape
    b = np.pad(a, r, mode="edge")
    dil = np.full_like(a, -np.inf)
    for dy, dx in offs:
        np.maximum(dil, b[r + dy:r + dy + h, r + dx:r + dx + w], out=dil)
    b = np.pad(dil, r, mode="edge")
    out = np.full_like(a, np.inf)
    for dy, dx in offs:
        np.minimum(out, b[r + dy:r + dy + h, r + dx:r + dx + w], out=out)
    return out


def tophat_drawing(a, radius: float):
    """Fiji's Top Hat with a light background (the black top-hat): what the
    closing fills in is what is thinner than the disc -- the cilia -- and
    that is kept, with its own contrast, on a white field; everything wider
    (cell texture, halos, uneven light) is gone.  The radius should be a
    little more than half the cilium's width: 3 px for a 2-4 px cilium."""
    a = np.asarray(a, dtype=np.float32)
    depth = grey_closing(a, radius) - a
    return (255.0 - np.clip(depth, 0.0, 255.0)).astype(np.float32)


def ridge_response_at(a, sigma: float):
    """Frangi's dark-ridge response at one scale: the Hessian's across
    eigenvalue (scaled by sigma^2), damped where the along eigenvalue is as
    large (a blob).  Signed: negative on a bright line."""
    hrr, hrc, hcc = gauss_hess(np.asarray(a, dtype=np.float64), sigma)
    mean = 0.5 * (hrr + hcc)
    half = np.sqrt((0.5 * (hrr - hcc)) ** 2 + hrc ** 2)
    lam_hi, lam_lo = mean + half, mean - half      # across, along
    ratio = lam_lo / np.where(np.abs(lam_hi) > 1e-12, lam_hi, 1e-12)
    blob = np.exp(-(ratio ** 2) / (2 * 0.5 ** 2))
    return lam_hi * np.where(lam_hi > 0, blob, 1.0)


def line_response(a, width: float):
    """How much like a dark line of about ``width`` pixels each pixel is:
    Frangi's vesselness, in two dimensions, for dark ridges.  The Hessian
    at two scales fitted to the width gives its two eigenvalues; a line has
    one large positive (across) and one near zero (along), a blob two
    alike, a flat patch two near zero.  The response is the across
    eigenvalue, damped where the along one is as large (a blob), and the
    larger of the two scales' answers.  Signed: negative on a bright line."""
    best = None
    for sigma in (max(0.6, 0.5 * width), max(0.9, 0.8 * width)):
        resp = ridge_response_at(a, sigma)
        best = resp if best is None else np.where(resp > best, resp, best)
    return best


def line_drawing(a, width: float, snr: float = LINES_SNR,
                 floor: float = LINES_FLOOR, soften: float = LINES_SOFTEN):
    """Fiji's Tubeness / Frangi Vesselness, drawn: the line response
    (line_response) against the frame's own noise -- the robust spread of
    the response over the frame, which is mostly background -- so that a
    line ``snr`` noise levels strong is fully dark, one under ``floor``
    levels is left out, and the rest shades between; softened by
    ``soften`` pixels, on a white field.  A drawing like the ones
    Ciliograph's validation scores best on, and like the trained network's
    (train_cilia_net.py), from a fixed filter that can
    draw nothing the frame does not contain."""
    resp = line_response(a, width)
    if LINES_ALONG > 0:
        # Averaged along the line direction (line_smooth): a cilium is
        # long and keeps its response, a stroke of noise is short and
        # loses most of its own.
        resp = line_smooth(resp.astype(np.float32), LINES_ALONG)
    med = float(np.median(resp))
    noise = max(1.4826 * float(np.median(np.abs(resp - med))), 1e-9)
    z = (resp - med) / noise
    dark = np.clip((z - floor) / max(snr - floor, 1e-6), 0.0, 1.0).astype(np.float32)
    if soften > 0:
        dark = _gauss32(dark, soften)
    return (255.0 * (1.0 - np.clip(dark, 0.0, 1.0))).astype(np.float32)


def clahe_reference(frames, clip: float, tiles=(8, 8), nbins: int = 256):
    """Contrast-limited adaptive histogram equalisation, fitted once.

    The frame is cut into tiles (8 x 8, or fewer on a small frame), each
    tile's grey levels are spread to fill the range -- with the histogram
    clipped at ``clip`` times its mean height, so flat areas are not blown
    up into noise -- and the mapping is interpolated between tile centres.
    Unlike the usual per-image CLAHE the mapping is fitted to frames pooled
    over the whole record and then applied to every frame unchanged: a
    mapping refitted on each frame would rise and fall as the cilia moved
    through a tile, and frame - mean would fill with that flicker.
    """
    f0 = np.asarray(frames[0])
    h, w = f0.shape
    ty = int(min(tiles[0], max(1, h // 16)))
    tx = int(min(tiles[1], max(1, w // 16)))
    ys = np.linspace(0, h, ty + 1).round().astype(int)
    xs = np.linspace(0, w, tx + 1).round().astype(int)
    rt = np.searchsorted(ys, np.arange(h), side="right") - 1
    ct = np.searchsorted(xs, np.arange(w), side="right") - 1
    tile = (np.clip(rt, 0, ty - 1)[:, None] * tx
            + np.clip(ct, 0, tx - 1)[None, :]).ravel()
    hist = np.zeros(ty * tx * nbins, dtype=np.float64)
    for f in frames:
        v = np.clip(np.asarray(f, dtype=np.float64), 0, 255).ravel()
        bi = np.minimum((v * nbins / 256.0).astype(int), nbins - 1)
        hist += np.bincount(tile * nbins + bi, minlength=ty * tx * nbins)
    hist = hist.reshape(ty, tx, nbins)
    limit = float(clip) * hist.sum(axis=2, keepdims=True) / nbins
    excess = np.clip(hist - limit, 0, None).sum(axis=2, keepdims=True)
    hist = np.minimum(hist, limit) + excess / nbins
    cdf = np.cumsum(hist, axis=2)
    cdf = (cdf - cdf[..., :1]) / np.maximum(cdf[..., -1:] - cdf[..., :1], 1e-12)
    return {"luts": (cdf * 255.0).astype(np.float32),
            "cy": 0.5 * (ys[:-1] + ys[1:]) - 0.5,
            "cx": 0.5 * (xs[:-1] + xs[1:]) - 0.5,
            "nbins": nbins, "shape": (h, w)}


def _tile_weights(n: int, centres):
    p = np.arange(n, dtype=np.float64)
    i0 = np.clip(np.searchsorted(centres, p, side="right") - 1, 0,
                 len(centres) - 1)
    i1 = np.minimum(i0 + 1, len(centres) - 1)
    span = np.where(i1 > i0, centres[i1] - centres[i0], 1.0)
    wgt = np.where(i1 > i0, np.clip((p - centres[i0]) / span, 0.0, 1.0), 0.0)
    return i0, i1, wgt


def apply_clahe(a, ref):
    """Map a frame through a clahe_reference, bilinearly between tiles."""
    a = np.clip(np.asarray(a, dtype=np.float64), 0, 255)
    h, w = a.shape
    if (h, w) != tuple(ref["shape"]):
        return a.astype(np.float32)
    luts, nb = ref["luts"], ref["nbins"]
    r0, r1, wr = _tile_weights(h, ref["cy"])
    c0, c1, wc = _tile_weights(w, ref["cx"])
    v = a * (nb - 1) / 255.0
    b0 = np.clip(np.floor(v).astype(int), 0, nb - 1)
    b1 = np.minimum(b0 + 1, nb - 1)
    fb = v - b0
    out = np.zeros((h, w), dtype=np.float64)
    for ri, wy in ((r0, 1 - wr), (r1, wr)):
        for ci, wx in ((c0, 1 - wc), (c1, wc)):
            rr, cc = ri[:, None], ci[None, :]
            val = luts[rr, cc, b0] * (1 - fb) + luts[rr, cc, b1] * fb
            out += (wy[:, None] * wx[None, :]) * val
    return out.astype(np.float32)


def _lanczos_taps(n_in: int, scale: float, lobes: int = 3):
    n_out = max(2, int(round(n_in * scale)))
    src = (np.arange(n_out) + 0.5) / scale - 0.5      # as resample_frame
    base = np.floor(src).astype(int)
    idx, wts = [], []
    for k in range(-lobes + 1, lobes + 1):
        i = base + k
        x = src - i
        wgt = np.sinc(x) * np.sinc(x / lobes)
        wgt[np.abs(x) >= lobes] = 0.0
        idx.append(np.clip(i, 0, n_in - 1))
        wts.append(wgt)
    wts = np.array(wts)
    return np.array(idx), wts / wts.sum(axis=0, keepdims=True)


def resample_lanczos(img, scale: float, shape=None):
    """Resize by ``scale`` with a Lanczos-3 kernel, on the same pixel grid as
    resample_frame.  Sharper than bilinear for the same samples -- but like
    any interpolation it adds no detail the frame did not have.  ``shape``
    (height, width) asks for that exact size instead."""
    a = np.asarray(img, dtype=np.float64)
    h, w = a.shape
    if shape is not None:
        sy, sx = float(shape[0]) / h, float(shape[1]) / w
    else:
        sy = sx = float(scale)
    iy, wy = _lanczos_taps(h, sy)
    ix, wx = _lanczos_taps(w, sx)
    rows = sum(wy[k][:, None] * a[iy[k], :] for k in range(len(iy)))
    out = sum(wx[k][None, :] * rows[:, ix[k]] for k in range(len(ix)))
    if shape is not None and out.shape != tuple(shape):
        out = out[:shape[0], :shape[1]]
        if out.shape != tuple(shape):
            out = np.pad(out, ((0, shape[0] - out.shape[0]),
                               (0, shape[1] - out.shape[1])), mode="edge")
    return out.astype(np.float32)


# ---- a neural network from a file -----------------------------------------
#
# The AI photo enhancers are neural networks; this runs one of your choice.
# Any image-to-image network saved as an .onnx file goes: a denoiser
# (NAFNet, Restormer, SCUNet, DnCNN ...), a deblurrer, a super-resolver
# (Real-ESRGAN, SwinIR ...), or a network trained on your own recordings
# (Noise2Void, CARE from Fiji or Python, exported to ONNX).  The frames are
# fed through it tile by tile, in the layout, channel count and pixel range
# it expects, and the result takes the frame's place for every measurement.
# It needs the onnxruntime package (pip install onnxruntime, or
# onnxruntime-gpu for a CUDA card), which is otherwise never required.
#
# Two cautions.  A network trained on photographs fills in what it expects
# to see -- a super-resolver draws edges that were never recorded -- so
# before trusting one, score it on a rendered carpet with known truth
# (ciliograph validate --model ...), as README.md describes.  And none of
# this separates cilia that overlap: on a dense lawn the limit is the
# bundle, not the pixels.

MODEL_TILE = 256            # tile side for models that take any size
MODEL_OVERLAP = 48          # overlap between tiles, px
MODEL_MARGIN = 8            # px of every tile's edge that are not used: a
                            # network sees nothing beyond its input and its
                            # output is wrong there
MODEL_MULTIPLE = 16         # tiles are padded to a multiple of this
MODEL_PROBE_SIDES = (64, 128, 256, 384, 512, 768, 1024)
                            # piece sizes tried, smallest first, until the
                            # model accepts one: some refuse small pieces (a
                            # NAFNet exported with its local pooling, "TLC",
                            # has windows fixed at export -- 384 px for one
                            # trained on 256 px crops -- and fails below)
MODEL_CACHE_MAX_BYTES = 8_000_000   # largest output cached on disk, per frame


def resolve_model_path(path, directory=None) -> str:
    """The model file: as given, beside the images, beside ciliograph.py, or
    in its models/ folder.  A path that is nowhere is returned as given,
    for the loader to report."""
    path = clean_model_path(path)
    if not path:
        return ""
    candidates = [path]
    if not os.path.isabs(path):
        if directory:
            candidates.append(os.path.join(directory, path))
        root = _app_dir()
        candidates += [os.path.join(root, path), os.path.join(root, "models", path)]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return path


def model_identity(path: str) -> str:
    """A short hash of the model file's path, size and date, for the cache."""
    import hashlib
    try:
        st = os.stat(path)
        tag = "{}|{}|{}".format(os.path.abspath(path), st.st_size, int(st.st_mtime))
    except OSError:
        tag = os.path.abspath(path)
    return hashlib.sha1(tag.encode("utf-8")).hexdigest()[:16]


def model_cache_dir() -> str:
    """Where model outputs are kept between runs: a folder in the system's
    temporary directory.  Safe to delete at any time; a run that finds a
    frame there is spared the network, and gives exactly the same frame."""
    import tempfile
    return os.path.join(tempfile.gettempdir(), "ciliograph_models")


# ---- is onnxruntime safe to load here? ------------------------------------
#
# A broken onnxruntime does not always raise an error that Python can catch:
# on Windows, when an older Microsoft C++ runtime (msvcp140.dll) is found
# before the one onnxruntime was built against -- typically the copy inside a
# conda environment -- importing it ends the whole process at once, and the
# fitter window simply closes.  So onnxruntime is first imported, and the
# model opened, in a separate Python process; only if that process survives
# is it loaded here.  The verdict is kept for the session.  It is not kept
# on disk: installing another package can break a working onnxruntime (one
# that brings its own old msvcp140.dll), and a stale "it worked" would bring
# the crash back.

_ORT_VERDICTS = {}

_WINDOWS_CRASHES = {
    0xC0000005: "access violation",
    0xC0000409: "fail-fast / stack buffer overrun",
    0xC000001D: "illegal instruction (the processor lacks an instruction "
                "set onnxruntime was built for, such as AVX)",
    0xC0000135: "a DLL it needs was not found",
    0xC0000142: "a DLL failed to initialise",
    0xC0000139: "a DLL entry point was not found (mismatched DLL versions)",
}


def _windows_file_version(path):
    """(major, minor, build, revision) of a Windows DLL, or None."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        ver = ctypes.WinDLL("version")
        ver.GetFileVersionInfoSizeW.argtypes = [wintypes.LPCWSTR,
                                                ctypes.POINTER(wintypes.DWORD)]
        ver.GetFileVersionInfoSizeW.restype = wintypes.DWORD
        ver.GetFileVersionInfoW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
                                            wintypes.DWORD, ctypes.c_void_p]
        ver.GetFileVersionInfoW.restype = wintypes.BOOL
        ver.VerQueryValueW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR,
                                       ctypes.POINTER(ctypes.c_void_p),
                                       ctypes.POINTER(wintypes.UINT)]
        ver.VerQueryValueW.restype = wintypes.BOOL
        dummy = wintypes.DWORD(0)
        size = ver.GetFileVersionInfoSizeW(path, ctypes.byref(dummy))
        if not size:
            return None
        buf = ctypes.create_string_buffer(size)
        if not ver.GetFileVersionInfoW(path, 0, size, buf):
            return None
        ptr = ctypes.c_void_p()
        length = wintypes.UINT()
        if not ver.VerQueryValueW(buf, "\\", ctypes.byref(ptr), ctypes.byref(length)):
            return None

        class _FixedInfo(ctypes.Structure):
            _fields_ = [("signature", wintypes.DWORD),
                        ("struc_version", wintypes.DWORD),
                        ("ms", wintypes.DWORD), ("ls", wintypes.DWORD)]

        info = ctypes.cast(ptr, ctypes.POINTER(_FixedInfo)).contents
        return (info.ms >> 16, info.ms & 0xFFFF, info.ls >> 16, info.ls & 0xFFFF)
    except Exception:
        return None


def _cpp_runtime_copies() -> list:
    """The copies of msvcp140.dll a Python extension may find, in the order
    Windows looks: next to python.exe and in the environment (kind "env"),
    then Windows' own (kind "system").  [(path, version or None, kind)];
    empty off Windows."""
    if sys.platform != "win32":
        return []
    system = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    seen, out = set(), []
    for d, kind in ((os.path.dirname(sys.executable), "env"), (sys.prefix, "env"),
                    (os.path.join(sys.prefix, "Library", "bin"), "env"),
                    (system, "system")):
        d = os.path.abspath(d)
        if os.path.normcase(d) in seen:
            continue
        seen.add(os.path.normcase(d))
        f = os.path.join(d, "msvcp140.dll")
        if os.path.isfile(f):
            out.append((f, _windows_file_version(f), kind))
    return out


def _runtime_too_old(version) -> bool:
    """A msvcp140.dll older than 14.40 crashes programs built with newer
    compilers (the std::mutex change of Visual Studio 2022 17.10)."""
    return version is not None and (version[0], version[1]) < (14, 40)


def cpp_runtime_report(copies=None) -> list:
    """The msvcp140.dll copies (_cpp_runtime_copies) with their versions,
    as lines of text; empty off Windows."""
    if copies is None:
        copies = _cpp_runtime_copies()
    return ["{} version {}{}".format(
        f, ".".join(str(x) for x in v) if v else "?",
        "  <- older than 14.40" if _runtime_too_old(v) else "")
        for f, v, kind in copies]


def _admin_only(path) -> bool:
    """True for a folder under Program Files or ProgramData: software
    installed for all users, which only an administrator can change."""
    p = os.path.normcase(os.path.abspath(path))
    for var in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432", "ProgramData"):
        root = os.environ.get(var)
        if root:
            r = os.path.normcase(os.path.abspath(root)).rstrip("\\/")
            if p == r or p.startswith(r + os.sep):
                return True
    return False


# The environment of one's own that README recommends: everything from
# conda-forge (the C++ runtime and an onnxruntime built against it), and
# packages pip put in the user folder (for another Python) kept out.
OWN_ENV_COMMANDS = (
    "conda create -n cilio --override-channels -c conda-forge python=3.12 "
    "numpy scipy matplotlib-base pillow onnxruntime",
    "conda env config vars set -n cilio PYTHONNOUSERSITE=1",
    "conda activate cilio")


def _cpp_runtime_advice(copies, prefix, is_conda, admin_only) -> str:
    """What to do when importing onnxruntime ends the process on Windows,
    from the msvcp140.dll copies found (see _cpp_runtime_copies)."""
    def ver(v):
        return "{}.{}".format(v[0], v[1]) if v else "?"
    sys_v = next((v for f, v, kind in copies if kind == "system"), None)
    # The first copy in the environment is the one loaded; Windows' own
    # matters only when the environment has none.
    first = next(((f, v) for f, v, kind in copies if kind == "env"), None)
    own_env = ("make an environment of your own, which needs no administrator "
               "rights:  " + "  then  ".join(OWN_ENV_COMMANDS) +
               "  and start Ciliograph there"
               if is_conda else "install onnxruntime in a fresh Python environment")
    if first is not None and (first[1] is None or _runtime_too_old(first[1])):
        f, v = first
        text = ("Python loads {} ({}) before Windows' own copy, and onnxruntime "
                "needs 14.40 or later".format(
                    f, "version {}".format(ver(v)) if v else
                    "its version could not be read"))
        if sys_v is not None and not _runtime_too_old(sys_v):
            text += ("; Windows already has {}, so the Visual C++ "
                     "Redistributable will not help".format(ver(sys_v)))
        if not is_conda:
            return text + (".  Fix: replace that file with version 14.40 or "
                           "later, or " + own_env)
        update = 'conda install "vc14_runtime>=14.40"'
        if admin_only:
            return text + (".  {} is installed for all users, so pip and conda "
                           "cannot change it without administrator rights.  "
                           "Fix: {}; or, as administrator (Anaconda Prompt, Run "
                           "as administrator):  {}".format(prefix, own_env, update))
        return text + ".  Fix:  {}  in this environment, or {}".format(update, own_env)
    if first is None and (sys_v is None or _runtime_too_old(sys_v)):
        return ("Fix: install the latest Microsoft Visual C++ Redistributable "
                "from https://aka.ms/vs/17/release/vc_redist.x64.exe")
    return ("the C++ runtime it finds is recent enough, so the cause is "
            "something else; try the previous onnxruntime release, or " + own_env)


def onnxruntime_probe(model_path: str | None = None, timeout: float = 300.0) -> dict:
    """Import onnxruntime -- and open ``model_path`` on the CPU, when given --
    in a separate Python process.  Returns {"ok": True/False/None, "version",
    "error"}: None when the check itself could not run (then nothing is
    known, and loading goes ahead as before)."""
    import subprocess
    try:
        from importlib import metadata as _md
        dist = _md.version("onnxruntime")
    except Exception:
        dist = "?"
    ident = model_identity(model_path) if model_path else ""
    key = "|".join([sys.executable, dist, ident])
    if key in _ORT_VERDICTS:
        return _ORT_VERDICTS[key]
    code = (
        "import sys\n"
        "import onnxruntime as ort\n"
        "print('ORT-VERSION', ort.__version__, flush=True)\n"
        "if len(sys.argv) > 1:\n"
        "    o = ort.SessionOptions(); o.log_severity_level = 3\n"
        "    ort.InferenceSession(sys.argv[1], o, providers=['CPUExecutionProvider'])\n"
        "    print('ORT-MODEL-OK', flush=True)\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    if getattr(sys, "frozen", False):
        # the standalone .exe has no Python to run code with: it checks
        # itself, through a hidden command (see main)
        env.pop("PYTHONPATH", None)
        args = [sys.executable, "--ort-probe"]
    else:
        args = [sys.executable, "-X", "faulthandler", "-c", code]
    if model_path:
        args.append(model_path)
    flags = 0x08000000 if sys.platform == "win32" else 0      # no console window
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                           env=env, creationflags=flags)
    except subprocess.TimeoutExpired:
        verdict = {"ok": False, "version": dist, "error":
                   "loading the model took longer than {:.0f} s".format(timeout)}
        _ORT_VERDICTS[key] = verdict
        return verdict
    except Exception as exc:
        return {"ok": None, "version": dist,
                "error": "could not run the check: {}".format(exc)}
    version = dist
    for line in (r.stdout or "").splitlines():
        if line.startswith("ORT-VERSION"):
            version = line.split(None, 1)[1].strip()
    if r.returncode == 0:
        verdict = {"ok": True, "version": version, "error": ""}
    else:
        verdict = {"ok": False, "version": version,
                   "error": _describe_ort_failure(r, model_path)}
    _ORT_VERDICTS[key] = verdict
    return verdict


def _describe_ort_failure(r, model_path) -> str:
    """What a failed onnxruntime check means, and what to do about it."""
    rc = r.returncode
    err = (r.stderr or "").strip().splitlines()
    imported = "ORT-VERSION" in (r.stdout or "")
    stage = ("opening the model" if (imported and model_path)
             else "importing onnxruntime")
    last = next((l for l in reversed(err) if l.strip()), "")
    python_error = rc == 1 and ("Error" in last or "error" in last)
    if python_error:
        text = "{} failed: {}".format(stage, last.strip())
    else:
        u = rc & 0xFFFFFFFF
        if rc < 0 and sys.platform != "win32":
            what = "signal {}".format(-rc)
        else:
            what = _WINDOWS_CRASHES.get(u, "exit code {} (0x{:08X})".format(rc, u))
        text = ("{} ended the Python process ({}) -- onnxruntime itself "
                "crashes on this computer".format(stage, what))
        fatal = [l.strip() for l in err if "fatal" in l.lower()]
        if fatal:
            text += "; " + fatal[0]
    if not imported and sys.platform == "win32":
        copies = _cpp_runtime_copies()
        text += ".  " + _cpp_runtime_advice(
            copies, sys.prefix, os.path.isdir(os.path.join(sys.prefix, "conda-meta")),
            _admin_only(sys.prefix))
        report = cpp_runtime_report(copies)
        if report:
            text += ".  msvcp140.dll here: " + "; ".join(report)
    return text


class OnnxEnhancer:
    """Run an image-to-image ONNX model on grey frames.

    The model's input is inspected: a batch of images in NCHW or NHWC
    layout, 1 or 3 channels, float32 or float16, with a free or a fixed
    size.  The frame is mirrored outward by MODEL_MARGIN pixels and given
    to the model in pieces -- ``tile`` pixels square with ``overlap``
    pixels shared between neighbours, or the model's fixed size -- and the
    pieces are blended with linear ramps that leave out MODEL_MARGIN pixels
    at every edge, where a network's output is wrong because it saw nothing
    beyond.  A grey frame fed to a 3-channel model is repeated into its
    channels and the channels of the result averaged.  ``value_range`` says
    what pixel values the model expects (MODEL_RANGES).  The model's scale
    -- 1 for a denoiser, 2 or 4 for a super-resolver -- is found by running
    it once on a small frame, and ``run`` returns the frame scaled by it.
    A model of free size that refuses small frames is tried on larger ones
    (MODEL_PROBE_SIDES); the smallest it accepts, ``min_side``, is then the
    least any piece is given, a small frame being mirrored out to it.
    """

    def __init__(self, path: str, value_range: int = 0, threads=None):
        if not os.path.isfile(path):
            raise FileNotFoundError("no model file at {}".format(path))
        try:
            from importlib import util as _iu
            installed = _iu.find_spec("onnxruntime") is not None
        except Exception:
            installed = True
        if not installed:
            raise RuntimeError(
                "onnxruntime is not installed: pip install onnxruntime "
                "(or onnxruntime-gpu for a CUDA card)")
        if "onnxruntime" not in sys.modules:
            # First time in this process: make sure it cannot take the
            # program down (see onnxruntime_probe).
            verdict = onnxruntime_probe(path)
            if verdict["ok"] is False:
                raise RuntimeError(verdict["error"])
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime could not be imported: {}".format(
                str(exc).splitlines()[0] if str(exc) else exc)) from exc
        self.path = path
        self.name = os.path.basename(path)
        self.value_range = int(value_range)
        import threading
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        if threads:
            opts.intra_op_num_threads = int(threads)
        # The CPU unless the GPU was asked for (--device gpu or
        # CILIOGRAPH_DEVICE=gpu): a GPU provider whose driver or DLLs are
        # missing can take the whole program down without an exception.
        available = list(ort.get_available_providers())
        providers = ["CPUExecutionProvider"]
        if compute_info().get("requested") == "gpu":
            providers = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider",
                                     "CoreMLExecutionProvider") if p in available] \
                + providers
        self.session = ort.InferenceSession(path, opts, providers=providers)
        # The size search below expects refusals; they are not printed.
        self._quiet = ort.RunOptions()
        self._quiet.log_severity_level = 4
        # One frame through the network at a time, whatever threads read the
        # frames: a restoration network needs a gigabyte or more per run,
        # and several at once exhausted the memory and ended the program.
        self.lock = threading.Lock()
        self.provider = self.session.get_providers()[0]
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or not outputs:
            raise ValueError("the model must take one image and give one back "
                             "({} inputs, {} outputs)".format(len(inputs),
                                                             len(outputs)))
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        shape = list(inputs[0].shape)
        if len(shape) != 4:
            raise ValueError("the model's input is not a batch of images "
                             "(shape {})".format(shape))
        dims = [d if isinstance(d, int) and d > 0 else None for d in shape]
        if dims[3] in (1, 3) and dims[1] not in (1, 3):
            self.layout = "NHWC"
            self.channels = dims[3]
            fixed = (dims[1], dims[2])
        else:
            self.layout = "NCHW"
            self.channels = dims[1]          # None: found by the probe
            fixed = (dims[2], dims[3])
        self.fixed = fixed if (fixed[0] and fixed[1]) else None
        self.dtype = np.float16 if "float16" in inputs[0].type else np.float32
        self.scale = 1
        self.tile = MODEL_TILE
        self.overlap = MODEL_OVERLAP
        self.min_side = 0               # smallest piece the model accepts, if
                                        # it refuses MODEL_PROBE_SIDES[0]
        self.refused_side = 0           # the size tried before that
        self._probe()

    # ------------------------------------------------------------ plumbing

    def _probe(self):
        """Run once on a small flat frame: learns the channel count when the
        model leaves it free, the model's scale, and, for a model of free
        size that refuses small frames, the smallest it accepts."""
        sizes = [self.fixed] if self.fixed else [(s, s) for s in MODEL_PROBE_SIDES]
        tried = [self.channels] if self.channels else [1, 3]
        last, out = None, None
        for size in sizes:
            for ch in tried:
                try:
                    self.channels = ch
                    out = self._run_tile(np.full(size, 0.5, dtype=np.float32),
                                         quiet=True)
                    break
                except Exception as exc:
                    last = exc
            if out is not None:
                break
        if out is None:
            why = str(last).strip().splitlines()[0] if str(last).strip() else repr(last)
            raise ValueError("the model refused {}: {}".format(
                "a {}x{} frame".format(*sizes[0]) if len(sizes) == 1 else
                "frames of {0}x{0} to {1}x{1}".format(sizes[0][0], sizes[-1][0]),
                why))
        if not self.fixed and size != sizes[0]:
            self.min_side = int(size[0])
            self.refused_side = int(sizes[sizes.index(size) - 1][0])
            self.tile = max(self.tile, self.min_side)
        sy = out.shape[0] / float(size[0])
        sx = out.shape[1] / float(size[1])
        if abs(sy - sx) > 1e-6 or abs(sy - round(sy)) > 1e-6 or sy < 1:
            raise ValueError("the model changes a {}x{} frame to {}x{}, which "
                             "is not a whole-number scale".format(
                                 size[0], size[1], out.shape[0], out.shape[1]))
        self.scale = int(round(sy))

    def _run_tile(self, tile, quiet=False):
        """One piece, values already in the model's range: (h, w) in, scaled
        (h', w') out, one channel.  ``quiet``: a refusal is expected and
        onnxruntime does not print it."""
        x = np.asarray(tile, dtype=np.float32)[None, None]      # N C H W
        if self.channels == 3:
            x = np.repeat(x, 3, axis=1)
        if self.layout == "NHWC":
            x = np.transpose(x, (0, 2, 3, 1))
        with self.lock:
            y = self.session.run([self.output_name],
                                 {self.input_name: x.astype(self.dtype)},
                                 self._quiet if quiet else None)[0]
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 3:
            y = y[None]
        if self.layout == "NHWC":
            y = np.transpose(y, (0, 3, 1, 2))
        if y.ndim != 4:
            raise ValueError("the model returned shape {}".format(y.shape))
        return y[0].mean(axis=0)

    def _to_model(self, a):
        a = np.asarray(a, dtype=np.float32)
        if self.value_range == 1:
            return a
        if self.value_range == 2:
            return a / 127.5 - 1.0
        return a / 255.0

    def _from_model(self, y):
        if self.value_range == 1:
            return y
        if self.value_range == 2:
            return (y + 1.0) * 127.5
        return y * 255.0

    @staticmethod
    def _pad_to(a, h, w):
        """Pad by reflection to at least h x w -- repeated, when the frame is
        smaller than the padding (a model that needs large pieces), and by
        the edge values where numpy cannot reflect."""
        ph, pw = max(0, h - a.shape[0]), max(0, w - a.shape[1])
        if ph == 0 and pw == 0:
            return a
        try:
            return np.pad(a, ((0, ph), (0, pw)), mode="reflect")
        except ValueError:
            return np.pad(a, ((0, ph), (0, pw)), mode="edge")

    def run(self, img):
        """The model's view of a frame (values 0..255 in, 0..255 out), at
        the model's scale."""
        a = self._to_model(img)
        h, w = a.shape
        s = self.scale
        mg = MODEL_MARGIN
        mode = "reflect" if (mg < h and mg < w) else "edge"
        a = np.pad(a, mg, mode=mode)
        if self.fixed:
            th, tw = self.fixed
        else:
            m = MODEL_MULTIPLE
            tile = max(self.tile, self.min_side)
            if a.shape[0] <= tile and a.shape[1] <= tile:
                th = max(int(np.ceil(a.shape[0] / m) * m), self.min_side)
                tw = max(int(np.ceil(a.shape[1] / m) * m), self.min_side)
            else:
                th = tw = tile
        ov = min(self.overlap, th // 3, tw // 3)
        mg_t = min(mg, max(0, (ov - 4) // 2))          # margin left out of each tile
        step_h, step_w = max(1, th - ov), max(1, tw - ov)
        a = self._pad_to(a, th, tw)
        H, W = a.shape
        ys = list(range(0, max(1, H - th + 1), step_h))
        xs = list(range(0, max(1, W - tw + 1), step_w))
        if ys[-1] + th < H:
            ys.append(H - th)
        if xs[-1] + tw < W:
            xs.append(W - tw)
        acc = np.zeros((H * s, W * s), dtype=np.float64)
        wsum = np.zeros((H * s, W * s), dtype=np.float64)
        ramp_h = self._ramp(th * s, ov * s, mg_t * s)
        ramp_w = self._ramp(tw * s, ov * s, mg_t * s)
        weight = np.minimum(ramp_h[:, None], ramp_w[None, :])
        for y0 in ys:
            for x0 in xs:
                piece = a[y0:y0 + th, x0:x0 + tw]
                out = self._run_tile(piece)
                if out.shape != (th * s, tw * s):
                    raise ValueError("the model returned {} for a {}x{} piece"
                                     .format(out.shape, th, tw))
                sl = (slice(y0 * s, (y0 + th) * s), slice(x0 * s, (x0 + tw) * s))
                acc[sl] += out * weight
                wsum[sl] += weight
        out = acc / np.maximum(wsum, 1e-9)
        out = out[mg * s:(mg + h) * s, mg * s:(mg + w) * s]
        return np.clip(self._from_model(out), 0.0, 255.0).astype(np.float32)

    @staticmethod
    def _ramp(n: int, ov: int, margin: int = 0):
        """Weights along one axis of a tile: zero over ``margin`` at both
        ends, rising linearly over the rest of the overlap, one between."""
        w = np.ones(n, dtype=np.float64)
        ov = int(min(ov, n // 2))
        margin = int(min(margin, ov))
        if ov > 0:
            rise = ov - margin
            r = np.concatenate([np.zeros(margin),
                                (np.arange(rise) + 1.0) / (rise + 1.0)])
            w[:ov] = np.minimum(w[:ov], r)
            w[n - ov:] = np.minimum(w[n - ov:], r[::-1])
        return w

    def describe(self) -> str:
        return "{} ({}; {} {}-channel {}, {}; x{}{})".format(
            self.name, self.provider.replace("ExecutionProvider", ""),
            self.layout, self.channels,
            "float16" if self.dtype == np.float16 else "float32",
            MODEL_RANGES[self.value_range], self.scale,
            "; fixed {}x{}".format(*self.fixed) if self.fixed
            else "; tiles of {}{}".format(
                max(self.tile, self.min_side),
                " (it refused {} px pieces)".format(self.refused_side)
                if self.min_side else ""))


def model_output_cached(cache_key: str):
    """The cached output of a model for one frame, or None."""
    path = os.path.join(model_cache_dir(), cache_key + ".npy")
    try:
        if os.path.exists(path):
            return np.load(path).astype(np.float32)
    except Exception:
        pass
    return None


def model_output_store(cache_key: str, out) -> None:
    """Keep a model's output for one frame, exactly as computed, when it is
    not too large; written to a temporary name first so a reader never
    sees half a file."""
    out = np.asarray(out, dtype=np.float32)
    if out.nbytes > MODEL_CACHE_MAX_BYTES:
        return
    try:
        folder = model_cache_dir()
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, cache_key + ".npy")
        tmp = os.path.join(folder, "{}.{}.tmp.npy".format(cache_key, os.getpid()))
        np.save(tmp, out)
        os.replace(tmp, path)
    except Exception:
        pass


def write_blur_onnx(path: str, sigma: float = 1.0, min_side: int = 0) -> str:
    """Write a small ONNX model -- a Gaussian blur, one 5x5 convolution on a
    1-channel NCHW image of any size -- for checking the model plumbing
    without downloading anything.  Written byte by byte in the protobuf
    wire format, so no onnx package is needed.  ``min_side`` > 0 adds what
    makes an exported NAFNet refuse small images: a slice that starts
    ``min_side - 1`` pixels in, padded back with "edge" -- impossible when
    nothing is left -- and added to the blur times zero, so the output is
    still the blur wherever the model runs."""
    import struct

    def varint(n):
        out = b""
        n = int(n)
        while True:
            b = n & 0x7F
            n >>= 7
            if n:
                out += bytes([b | 0x80])
            else:
                return out + bytes([b])

    def field(num, wire, payload):
        return varint((num << 3) | wire) + payload

    def msg(num, payload):
        return field(num, 2, varint(len(payload)) + payload)

    def text(num, s):
        return msg(num, s.encode("utf-8"))

    def int_field(num, v):
        return field(num, 0, varint(v))

    def tensor_float(name, dims, values):
        raw = struct.pack("<{}f".format(len(values)), *values)
        body = b"".join(int_field(1, d) for d in dims)       # dims
        body += int_field(2, 1)                                # FLOAT
        body += text(8, name) + msg(9, raw)                    # name, raw_data
        return body

    def tensor_int64(name, values):
        raw = struct.pack("<{}q".format(len(values)), *values)
        return (int_field(1, len(values)) + int_field(2, 7)    # dims, INT64
                + text(8, name) + msg(9, raw))

    def dim(value=None, param=None):
        return msg(1, int_field(1, value) if value is not None else text(2, param))

    def value_info(name, dims):
        shape = msg(2, b"".join(dim(**d) for d in dims))
        tensor_type = msg(1, int_field(1, 1) + shape)           # elem_type FLOAT
        return text(1, name) + msg(2, tensor_type)

    r = 2
    x = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-x * x / (2.0 * sigma * sigma))
    k2 = np.outer(g, g)
    k2 /= k2.sum()
    weight = tensor_float("W", [1, 1, 5, 5], k2.ravel().tolist())
    attr_pads = msg(5, text(1, "pads") + b"".join(int_field(8, v) for v in (2, 2, 2, 2))
                    + int_field(20, 7))                        # INTS
    attr_ks = msg(5, text(1, "kernel_shape") + b"".join(int_field(8, v) for v in (5, 5))
                  + int_field(20, 7))
    nodes = msg(1, text(1, "input") + text(1, "W")
                + text(2, "blurred" if min_side > 0 else "output")
                + text(3, "blur") + text(4, "Conv") + attr_pads + attr_ks)
    inits = msg(5, weight)
    if min_side > 0:
        k = int(min_side) - 1
        inits += (msg(5, tensor_int64("starts", [k, k]))
                  + msg(5, tensor_int64("ends", [2 ** 62, 2 ** 62]))
                  + msg(5, tensor_int64("axes", [2, 3]))
                  + msg(5, tensor_int64("pads", [0, 0, 0, 0, 0, 0, k, k]))
                  + msg(5, tensor_float("zero", [1], [0.0])))
        attr_mode = msg(5, text(1, "mode") + msg(4, b"edge") + int_field(20, 3))
        nodes += (msg(1, text(1, "input") + text(1, "starts") + text(1, "ends")
                      + text(1, "axes") + text(2, "sliced") + text(3, "slice")
                      + text(4, "Slice"))
                  + msg(1, text(1, "sliced") + text(1, "pads") + text(2, "padded")
                        + text(3, "pad") + text(4, "Pad") + attr_mode)
                  + msg(1, text(1, "padded") + text(1, "zero") + text(2, "zeroed")
                        + text(3, "times0") + text(4, "Mul"))
                  + msg(1, text(1, "blurred") + text(1, "zeroed") + text(2, "output")
                        + text(3, "sum") + text(4, "Add")))
    inp = msg(11, value_info("input", [dict(param="N"), dict(value=1),
                                        dict(param="H"), dict(param="W")]))
    out = msg(12, value_info("output", [dict(param="N"), dict(value=1),
                                         dict(param="H"), dict(param="W")]))
    graph = msg(7, nodes + text(2, "blur5") + inits + inp + out)
    opset = msg(8, text(1, "") + int_field(2, 13))
    model = int_field(1, 7) + text(2, "ciliograph") + graph + opset
    with open(path, "wb") as fh:
        fh.write(model)
    return path


def crop_from_pqd(pqd):
    """The crop recorded in a specification, or None.  Pixels of the levelled frame."""
    vals = [pqd.getf(k, -1.0) for k in CROP_KEYS]
    if any(v < 0 for v in vals):
        return None
    return crop_tuple(vals)


class VideoFrames:
    """A lazily-loaded, cached sequence of video frames."""

    def __init__(self, directory: str, basename: str = "im",
                 first: int | None = None, last: int | None = None,
                 rotation: float = 0.0, crop=None, enhance=None):
        self.directory = directory
        self.basename = basename
        self.rotation = float(rotation)
        self.enhance = dict(enhance) if enhance else None
        self._bg_mean = None
        # (x0, y0, x1, y1) in the rotated frame, or None: the region every
        # step sees.  Cropping here, inside the frame source, means the base,
        # the traces, the fit and the overlay all live in the same reduced
        # frame without any of them knowing.
        self.crop = crop_tuple(crop)

        # An explicit listing, not glob: glob is case-insensitive on Windows
        # and case-sensitive on Linux, so a specification recording the prefix as
        # "Im" beside files named "im0000.jpg" -- or .JPG frames -- found
        # every frame on one and none on the other, and the same folder that
        # loaded on Windows gave "no frames matching" on Linux.  glob also
        # read [ ] ? in a folder name as wildcards.  The prefix is matched
        # ignoring case everywhere, and if it matches nothing at all while
        # the folder does hold frames, the prefix is treated as stale and the
        # folder's frames are taken: the images in front of you outrank a
        # name copied from another machine.
        if not os.path.isdir(directory):
            raise FileNotFoundError("no such folder: {}".format(directory))
        names = [n for n in os.listdir(directory)
                 if _is_image(os.path.join(directory, n))]
        prefix = str(basename or "").lower()
        matching = [n for n in names if n.lower().startswith(prefix)]
        self.basename_stale = bool(names) and not matching
        if self.basename_stale:
            matching = names
            self.basename = detect_basename(names)
        paths = sorted((os.path.join(directory, n) for n in matching),
                       key=natural_key)
        if not paths:
            raise FileNotFoundError(
                "no image frames in {} (looked for files starting with "
                "{!r})".format(directory, basename))

        # A multi-page TIFF holds the whole sequence in one file, so expand
        # each file into (path, page) entries and treat pages as frames.
        self.index = []
        for path in paths:
            pages = image_pages(path) if _is_tiff(path) else 1
            self.index.extend((path, page) for page in range(pages))

        self.all_paths = [entry[0] for entry in self.index]
        self.first = 0 if first is None else int(first)
        self.last = len(self.index) - 1 if last is None else int(last)
        self.last = min(self.last, len(self.index) - 1)
        self.first = max(0, min(self.first, self.last))

        self._cache: dict = {}

    @classmethod
    def from_pqd(cls, pqd: Pqd, directory: str) -> "VideoFrames":
        basename = pqd.get("vid_image_filename", "im")
        if not isinstance(basename, str):
            basename = "im"
        return cls(directory, basename=basename,
                   first=int(pqd.getf("vid_first_frame_to_load", 0)),
                   last=int(pqd.getf("vid_last_frame_to_load", 10 ** 9)),
                   rotation=pqd.getf("vid_image_rotation", 0.0),
                   crop=crop_from_pqd(pqd), enhance=enhance_from_pqd(pqd))

    def __len__(self) -> int:
        return self.last - self.first + 1

    @property
    def frame_numbers(self) -> range:
        return range(self.first, self.last + 1)

    def _drop_derived(self) -> None:
        """Forget everything computed from the frames as a whole."""
        self._bg_mean = None
        self._enh_refs = None
        self._pre_cache = {}
        self._markers = None

    def markers(self):
        """Burnt-in graphics in the frames, as a mask at the raw frames'
        size, or None: pixels at the black or white end of the record's
        scale that are the same in every sampled frame, in blobs under
        MARKER_MAX_FRAC of the frame.  They are not image data and are
        kept out of ``valid``.  On CO121 a black triangle in one corner
        set the dark threshold of the surface detection: no tissue was
        dark enough, the surface was put at the frame's bottom, Find base
        went there, and Auto fit found "no periodic motion"."""
        mk = getattr(self, "_markers", None)
        if mk is None:
            mk = False
            try:
                # A drawing has no camera graphics, and its wall -- a
                # constant black line -- would be taken for one.
                if bimodal_frames(self, raw=True) > DRAWN_BIMODAL_FRACTION:
                    raise ValueError("drawn frames")
                mn = mx = None
                for f in self._sample_frames(40):
                    a = np.asarray(self._raw(f), dtype=np.float32)
                    if a.ndim == 3:
                        a = a[..., :3].mean(axis=2)
                    mn = a.copy() if mn is None else np.minimum(mn, a)
                    mx = a.copy() if mx is None else np.maximum(mx, a)
                if mn is not None:
                    lo, hi = float(mn.min()), float(mx.max())
                    if hi - lo > 1e-6:
                        cand = ((mx == mn) & ((mn <= lo + 1e-6)
                                              | (mn >= hi - 1e-6)))
                        if cand.any():
                            found = small_blobs(cand, MARKER_MAX_FRAC)
                            if found.any():
                                mk = found
            except Exception:
                mk = False
            self._markers = mk
        return None if mk is False else mk

    def set_rotation(self, degrees: float) -> None:
        """Set the rotation applied to every frame, in degrees."""
        degrees = float(degrees)
        if degrees != self.rotation:
            self.rotation = degrees
            self._cache.clear()
            self._source_shape = None
            self._drop_derived()

    def set_crop(self, crop) -> None:
        """Restrict every frame to (x0, y0, x1, y1) of the rotated frame."""
        crop = crop_tuple(crop)
        if crop != getattr(self, "crop", None):
            self.crop = crop
            self._cache.clear()
            self._drop_derived()

    def set_enhance(self, settings) -> None:
        """Change the enhancement applied to every frame."""
        settings = dict(settings) if settings else None
        if settings != getattr(self, "enhance", None):
            old = getattr(self, "enhance", None) or {}
            self.enhance = settings
            self._cache.clear()
            self._enh_refs = None
            self._pre_cache = {}
            # Deblocking changes the frames as read, so the mean of them too.
            if bool(old.get("deblock")) != bool((settings or {}).get("deblock")):
                self._bg_mean = None

    def _raw(self, frame: int) -> np.ndarray:
        """The rotated, cropped frame before any enhancement."""
        img = self._full(frame)
        self._source_shape = (img.shape[0], img.shape[1])
        if self.crop is not None:
            x0, y0, x1, y1 = clip_crop(self.crop, img.shape)
            img = img[y0:y1, x0:x1]
        return img

    def _sample_frames(self, n_max: int = 40):
        n = self.last - self.first + 1
        return list(range(self.first, self.last + 1, max(1, n // n_max)))

    def background(self, median: bool = False) -> np.ndarray:
        """Temporal mean of the raw frames, for background removal -- or
        their temporal median (Fiji's Z Project > median): a pixel a cilium
        sweeps through less than half the time then shows the field behind
        it, not a smear of the cilium, so the background taken out of each
        frame holds nothing of the beat."""
        if median:
            if getattr(self, "_bg_median", None) is None:
                stack = []
                for f in self._sample_frames(40):
                    a = np.asarray(self._raw(f), dtype=np.float32)
                    if a.ndim == 3:
                        a = a[..., :3].mean(axis=2)
                    stack.append(a)
                self._bg_median = np.median(np.stack(stack), axis=0).astype(np.float32)
            return self._bg_median
        if getattr(self, "_bg_mean", None) is None:
            acc = None
            count = 0
            for f in self._sample_frames(40):
                a = np.asarray(self._raw(f), dtype=np.float32)
                if a.ndim == 3:
                    a = a[..., :3].mean(axis=2)
                acc = a.copy() if acc is None else acc + a
                count += 1
            self._bg_mean = acc / float(max(count, 1))
        return self._bg_mean

    def _raw_valid(self):
        """valid, at the resolution of the raw frames (before upsampling)."""
        mask = rotation_valid(self.source_shape, self.rotation)
        if self.crop is not None:
            x0, y0, x1, y1 = clip_crop(self.crop, mask.shape)
            mask = mask[y0:y1, x0:x1]
        mk = self.markers()
        if mk is not None and mk.shape == mask.shape:
            mask = mask & ~mk
        return mask

    def enhance_refs(self) -> dict:
        """What the enhancement needs from the whole record, computed once:
        the deflicker reference and the contrast (CLAHE) mapping."""
        enh = getattr(self, "enhance", None) or {}
        refs = getattr(self, "_enh_refs", None)
        if refs is not None:
            return refs
        refs = {}
        if enh.get("deflicker"):
            def grey_raw(f):
                a = np.asarray(self._raw(f), dtype=np.float32)
                return a[..., :3].mean(axis=2) if a.ndim == 3 else a
            mean, still = None, None
            sample = [grey_raw(f) for f in self._sample_frames(40)]
            valid = self._raw_valid()
            mean, still = flicker_reference(
                sample, valid if valid.shape == sample[0].shape else None)
            # Every frame's gain is needed, in order; reading them is the
            # costly part, so it is done on threads.
            frames = list(range(self.first, self.last + 1))
            n = frame_threads()
            if n > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=n) as ex:
                    raws = list(ex.map(grey_raw, frames))
            else:
                raws = [grey_raw(f) for f in frames]
            params = flicker_params(raws, mean, still)
            params["first"] = self.first
            refs["flicker"] = params
        model = clean_model_path(enh.get("model"))
        if model:
            # The network, and the key under which its output for a frame
            # is kept on disk: the model file, its pixel range, the frame
            # file, and every step the frame went through before it.
            path = resolve_model_path(model, self.directory)
            try:
                refs["ai"] = OnnxEnhancer(path, value_range=int(enh.get("modelrange") or 0),
                                          threads=_available_cpus())
            except Exception as exc:
                refs["ai"] = None
                refs["ai_error"] = str(exc).splitlines()[0]
                _compute_warn("model:" + path, "the model {} is not used: {}"
                              .format(os.path.basename(path), refs["ai_error"]))
            if refs.get("ai") is not None:
                import hashlib
                net = refs["ai"]
                stem = "|".join([model_identity(path), str(int(enh.get("modelrange") or 0)),
                                 # how the frame is cut into pieces (v2:
                                 # small frames mirrored out to min_side)
                                 "v2 {} {} {}".format(net.tile, net.min_side, net.overlap),
                                 str(self.rotation), str(self.crop),
                                 str(int(bool(enh.get("deblock")))),
                                 str(int(bool(enh.get("invert")))),
                                 str(int(bool(enh.get("deflicker")))),
                                 "{}-{}".format(self.first, self.last)
                                 if enh.get("deflicker") else ""])
                index = getattr(self, "index", None)

                def cache_key(frame, stem=stem, index=index):
                    if index is None:
                        return None
                    try:
                        path_f, page = index[int(frame)]
                        st = os.stat(path_f)
                        tag = "{}|{}|{}|{}|{}".format(stem, os.path.abspath(path_f),
                                                      page, st.st_size, int(st.st_mtime))
                    except (IndexError, OSError, TypeError):
                        return None
                    return hashlib.sha1(tag.encode("utf-8")).hexdigest()
                refs["ai_cache"] = cache_key
        if float(enh.get("lines") or 0.0) > 0 or float(enh.get("tophat") or 0.0) > 0:
            # Where the line filter and the top-hat may draw: inside the
            # real image (the edge of the rotation padding is a strong
            # line) and above the tissue surface, found in the raw frames
            # (its edge is the strongest line of all, and the tissue's
            # texture is thin structure too).
            try:
                bg = self.background()
                valid = self._raw_valid()
                if valid.shape != bg.shape:
                    valid = np.ones(bg.shape, dtype=bool)
                surface = detect_surface(bg, valid=valid)
                refs["draw_keep"] = drawing_keep_mask(valid, surface,
                                                      LINES_SURFACE_MARGIN)
            except Exception:
                refs["draw_keep"] = None
        if float(enh.get("contrast") or 0.0) > 0:
            # Fitted on frames that have been through every earlier step;
            # those are kept, so get() does not compute them twice.
            bg = self._background_for(enh, refs)
            pre = {f: _enhance_pre(self._raw(f), enh, bg, refs, f)
                   for f in self._sample_frames(24)}
            refs["clahe"] = clahe_reference(
                [np.clip(a, 0.0, 255.0) for a in pre.values()],
                float(enh["contrast"]))
            self._pre_cache = pre
        self._enh_refs = refs
        return refs

    def model_note(self) -> str:
        """What the network setting amounts to: its description, the error
        that stopped it, or ""."""
        enh = getattr(self, "enhance", None) or {}
        if not clean_model_path(enh.get("model")):
            return ""
        refs = self.enhance_refs()
        if refs.get("ai") is not None:
            return refs["ai"].describe()
        return "NOT USED: {}".format(refs.get("ai_error", "?"))

    def _background_for(self, enh: dict, refs: dict):
        if not enh.get("background"):
            return None
        return self.background(median=bool(enh.get("bgmedian")))

    def _full(self, frame: int) -> np.ndarray:
        path, page = self.index[frame]
        img = _load_image(path, page)
        enh = getattr(self, "enhance", None) or {}
        if enh.get("deblock"):
            # The 8x8 grid is the file's, so this comes before rotation.
            q = jpeg_quant_table(path) if not page else None
            if q is not None:
                img = jpeg_deblock(img, q)
        if self.rotation:
            img = rotate_image(img, self.rotation)
        return img

    def get(self, frame: int) -> np.ndarray:
        frame = int(np.clip(frame, self.first, self.last))
        if frame not in self._cache:
            img = self._raw(frame)
            enh = getattr(self, "enhance", None)
            if enh and not enhance_is_default(enh):
                refs = self.enhance_refs()
                native = img.shape[:2]
                pre = (getattr(self, "_pre_cache", None) or {}).pop(frame, None)
                if pre is None:
                    pre = _enhance_pre(img, enh, self._background_for(enh, refs),
                                       refs, frame)
                img = _enhance_post(pre, enh, refs, native)
            self._cache[frame] = img
        return self._cache[frame]

    def prefetch(self, frames=None, threads: int | None = None) -> None:
        """Read (and enhance) many frames at once, on several threads.

        Every step that calls this reads all of these frames anyway, one
        after another; decoding, rotating and enhancing them is independent
        per frame and mostly runs outside Python's interpreter lock, so it
        is done side by side first.  The cache ends up exactly as the plain
        loop would have left it, frame for frame.
        """
        try:
            want = (range(self.first, self.last + 1) if frames is None
                    else frames)
            todo = [f for f in dict.fromkeys(
                int(np.clip(int(f), self.first, self.last)) for f in want)
                if f not in self._cache]
            n = int(threads or frame_threads())
            if n <= 1 or len(todo) < 2 or not hasattr(self, "index"):
                return
            enh = getattr(self, "enhance", None)
            if enh and not enhance_is_default(enh):
                # once, before the threads need them
                self._background_for(enh, self.enhance_refs())
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(n, len(todo))) as ex:
                list(ex.map(self.get, todo))
        except Exception:
            # Nothing is lost: whatever failed is read again, one frame at a
            # time, by the caller -- which then reports the error itself.
            pass

    @property
    def upsample(self) -> int:
        enh = getattr(self, "enhance", None)
        return int((enh or {}).get("upsample") or 1)

    @property
    def source_shape(self):
        """(height, width) of the rotated frames before any crop."""
        if getattr(self, "_source_shape", None) is None:
            self.get(self.first)
        return self._source_shape

    @property
    def shape(self):
        """(height, width) of the frames as delivered -- after the crop."""
        img = self.get(self.first)
        return img.shape[0], img.shape[1]

    @property
    def valid(self):
        """Mask of pixels that are real image data: not rotation padding,
        not burnt-in graphics (markers)."""
        mask = self._raw_valid()
        if self.upsample > 1:
            mask = resample_frame(mask.astype(np.float32),
                                  float(self.upsample)) > 0.5
        return mask

    def path_of(self, frame: int) -> str:
        return self.all_paths[int(np.clip(frame, self.first, self.last))]

    @classmethod
    def placeholder(cls, w: int = 320, h: int = 240) -> "VideoFrames":
        """A single blank frame, so the window can open before any images.

        Everything downstream expects a sequence to exist; this keeps that
        true while the user has not imported anything yet.
        """
        obj = cls.__new__(cls)
        obj.directory = ""
        obj.basename = ""
        obj.rotation = 0.0
        obj.crop = None
        obj.enhance = None
        obj._bg_mean = None
        obj._source_shape = (h, w)
        obj.all_paths = ["<no images>"]
        obj.first = 0
        obj.last = 0
        obj._cache = {0: np.full((h, w), 0.45, dtype=float)}
        obj.is_placeholder = True
        return obj


# =========================================================================
# Shape generation over the beat cycle
# =========================================================================

def split_index(pqd, n_cs: int, override: int | None = None) -> int:
    """Index separating the recovery stroke from the effective stroke.

    Derived from !phasef_ert when present; for the reference dataset this
    reproduces the value 7 that the original aa10.py had hard-coded.  The
    Fourier model carries no such marker, so it falls back to half a cycle.
    """
    if override is not None:
        return int(override)
    ert = pqd.getf("phasef_ert", 0.0) if pqd is not None else 0.0
    return int(round(ert * n_cs)) if ert > 0 else n_cs // 2


class ShapeSeries:
    """Cilium centreline shapes of a :class:`FourierWave` sampled over one
    beat cycle: ``n_cs`` time steps from phase 0, ``n_pts`` nodes each."""

    def __init__(self, model, n_cs: int = 14, n_pts: int | None = None,
                 equal_arclength: bool = True):
        self.model = model
        self.n_cs = n_cs
        self.pqd = getattr(model, "pqd", None)
        self.is_fourier = True
        self.n_xy = int(n_pts or 201)
        self.phases, self.xs, self.ys = model.shapes(
            n_cs, n_pts=self.n_xy, equal_arclength=equal_arclength)

        # Arc length measured from the actual geometry.
        self.arclength = np.zeros((n_cs, self.n_xy))
        for ti in range(n_cs):
            seg = np.hypot(np.diff(self.xs[ti]), np.diff(self.ys[ti]))
            self.arclength[ti, 1:] = np.cumsum(seg)

    # ------------------------------------------------------------- exports

    def save_data(self, path: str) -> str:
        """Write the z01a.data column format used by the original tool."""
        parts = ["#\n", "# i "]
        for ti in range(self.n_cs):
            parts.append(" ; x{:02d}  ".format(ti))
            parts.append(" ; y{:02d}  ".format(ti))
        parts.append("\n")

        for li in range(self.n_xy):
            parts.append("{:4}".format(li))
            for ti in range(self.n_cs):
                parts.append(" ; {:5.2f}".format(self.xs[ti][li]))
                parts.append(" ; {:5.2f}".format(self.ys[ti][li]))
            parts.append("\n")

        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("".join(parts))
        return path

    def save_csv(self, path: str) -> str:
        """Tidy CSV: step, phase, time_s, s_um, x_um, y_um."""
        cbf = self.model.cbf
        period_s = (1.0 / cbf) if cbf > 0 else float("nan")

        lines = ["step,phase,time_s,s_um,x_um,y_um\n"]
        for ti in range(self.n_cs):
            phase = self.phases[ti]
            t_s = phase * period_s
            x, y = self.xs[ti], self.ys[ti]
            for li in range(self.n_xy):
                lines.append("{},{:.6f},{:.6f},{:.4f},{:.6f},{:.6f}\n".format(
                    ti, phase, t_s, self.arclength[ti, li], x[li], y[li]))

        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(lines)
        return path

    def save_npz(self, path: str) -> str:
        """Numpy archive with x_um[step, s] and y_um[step, s]."""
        np.savez_compressed(
            path,
            phase=np.array(self.phases),
            s_um=self.arclength,
            x_um=np.array(self.xs),
            y_um=np.array(self.ys),
            cbf_hz=self.model.cbf,
            length_um=self.model.length,
        )
        return path

    def tip_track(self):
        """(phase, x, y) of the cilium tip at each time step."""
        ph = np.array(self.phases)
        tx = np.array([x[-1] for x in self.xs])
        ty = np.array([y[-1] for y in self.ys])
        return ph, tx, ty

    # --------------------------------------------------------------- plots

    def _color(self, i: int, split: int) -> str:
        if i == 0:
            return COLOR_FIRST
        if i < split:
            return COLOR_RECOVERY
        if i == split:
            return COLOR_SPLIT
        return COLOR_EFFECTIVE

    def plot(self, path: str, which: str = "all", split: int | None = None,
             title: str | None = None, figtext: str | None = None) -> str:
        """Save a figure.  `which` is "all", "recovery" or "effective"."""
        _load_pyplot(interactive=False)

        sp = split_index(self.pqd, self.n_cs, split)
        lim = max(6.1, self.model.length * 1.05)
        top = self.model.length * 1.25

        fig, ax = plt.subplots(figsize=(13.54, 10.42), dpi=96)
        ax.plot([-lim, lim, None, 0, 0], [0, 0, None, -0.3, top], color="#999999")
        ax.set_xlabel("x / um")
        ax.set_ylabel("z / um")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.3, top)
        ax.set_aspect("equal", adjustable="box")
        if title:
            ax.set_title(title)
        if figtext:
            fig.text(0.92, 0.19, figtext,
                     fontdict={"family": "serif", "weight": "bold", "size": 12})

        if which == "recovery":
            indices = [i for i in range(self.n_cs) if i <= sp]
        elif which == "effective":
            indices = [i for i in range(self.n_cs) if i >= sp] + [0]
        else:
            indices = list(range(self.n_cs))

        for i in indices:
            ax.plot(self.xs[i], self.ys[i], color=self._color(i, sp),
                    linestyle="-", linewidth=2.3, label="cs{}".format(i))

        if which == "recovery":
            ax.arrow(2, self.model.length * 1.08, -4, 0, color="#000",
                     width=0.02, head_width=0.2)
        elif which == "effective":
            ax.arrow(-2, self.model.length * 1.08, 4, 0, color="#000",
                     width=0.02, head_width=0.2)

        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path


def run_shapes(pqd_path: str, outdir: str = ".", n_cs: int = 14,
               split: int | None = None, figtext: str | None = None,
               make_plots: bool = True, verbose: bool = True,
               coeffs_path: str | None = None,
               n_pts: int | None = None, equal_arclength: bool = True,
               n_modes: int | None = FOURIER_DEFAULT_MODES,
               n_space: int | None = None) -> ShapeSeries:
    """Full pipeline: specification file -> shapes -> data files and figures.

    The beat is the specification's (or ``coeffs_path``, a coefficient text
    file, or the published template); L_c and the beat frequency come from
    the specification.
    """
    os.makedirs(outdir, exist_ok=True)

    pqd = Spec(pqd_path)
    coeffs = load_fourier_coeffs(coeffs_path) if coeffs_path else None
    model = FourierWave.from_spec(pqd, coeffs=coeffs, n_modes=n_modes,
                                  n_space=n_space)
    series = ShapeSeries(model, n_cs=n_cs, n_pts=n_pts,
                         equal_arclength=equal_arclength)

    def out(name):
        return os.path.join(outdir, name)

    written = [
        series.save_data(out("z01a.data")),
        series.save_csv(out("cilium_shapes.csv")),
        series.save_npz(out("cilium_shapes.npz")),
    ]

    if make_plots:
        written += [
            series.plot(out("z80a.png"), "all", split, figtext=figtext),
            series.plot(out("z80b_recovery.png"), "recovery", split, figtext=figtext),
            series.plot(out("z80c_effective.png"), "effective", split, figtext=figtext),
        ]

    if verbose:
        print("source      : {}".format(pqd_path))
        print("time steps  : {}".format(n_cs))
        print("points/shape: {}  ({} spacing)".format(
            series.n_xy,
            "equal arc length" if equal_arclength else "equal s"))
        print("coefficients: {}  ({} Fourier modes, n = 0..{})".format(
            coeffs_path or ("the specification's beat" if pqd.coeffs is not None
                            else "the published template"),
            model.n_modes, model.n_modes - 1))
        _, stretch = model.stretch_over_cycle(120)
        print("length      : {} um at every phase (raw fit rescaled by "
              "{:.3f}..{:.3f} over the cycle)".format(
                  model.length, 1.0 / stretch.max(), 1.0 / stretch.min()))
        print("beat freq   : {} Hz  -> period {:.4f} s".format(
            model.cbf, 1.0 / model.cbf if model.cbf else float("nan")))
        print("stroke split: index {}".format(split_index(pqd, n_cs, split)))
        print("written:")
        for w in written:
            print("  {}".format(w))

    return series


# =========================================================================
# Interactive fitter  (the replacement for axwa-bin64)
# =========================================================================

class _Readout:
    """A text that is printed rather than drawn: the fit's readout.  Keeps
    set_text / get_text so that scripts reading the readout still work."""

    def __init__(self, title: str):
        self.title = title
        self._text = ""

    def set_text(self, text) -> None:
        self._text = str(text)
        if self._text and not self._text.endswith("..."):
            print("--- {} ---".format(self.title))
            print(self._text)
            sys.stdout.flush()

    def get_text(self) -> str:
        return self._text


class CiliographGui:
    """The interactive fitting window."""

    # One tab per mode.  Each tab shows only the controls that mode changes.
    MODES = [
        "video", "enhance", "position", "cycle", "length", "params", "MCC",
    ]

    # The original tool's separate "cycle param" and "change param" modes are
    # one tab here: with all the sliders visible you select and change in the
    # same place.  ppum has no tab either -- it is a box on the video tab.
    # What each tab changes, shown in the tab panel header.
    MODE_BLURB = {
        "video":   "step through the video frames",
        "enhance": "clean the frames up for the measurements (not just the view)",
        "position": "where the cilium sits: base x0/y0 and its rotation",
        "cycle":   "main beat cycle: first and last frame (sets cbf)",
        "length":  "cilium length and how many nodes along it",
        "params":  "Fourier coefficients of mode n  (x components left, z right)",
        "MCC":     "track a particle and measure its transport velocity",
    }

    # Geometry of the frame timeline, the tab strip and the panel beneath it.
    TIME_BOX = [0.045, 0.345, 0.515, 0.055]
    TAB_Y, TAB_H = 0.288, 0.028
    TAB_X0, TAB_X1 = 0.035, 0.560
    PANEL_X, PANEL_W = 0.105, 0.395
    PANEL_TOP, PANEL_DY = 0.238, 0.026

    def __init__(self, pqd_path: str, images_dir: str, fps_play: float = 12.0,
                 fourier_coeffs: dict | None = None,
                 n_modes: int | None = FOURIER_DEFAULT_MODES,
                 n_space: int | None = None):
        _load_pyplot(interactive=True)

        self._fourier_coeffs = fourier_coeffs
        self._n_modes = n_modes
        self._n_space = n_space
        self.sel_n = 0        # which Fourier mode is on the sliders
        self.sel_slot = 0     # which of that mode's 12 coefficients is selected
        self.n_cs = 14        # time steps written by "Export nodes"
        self.pos_target = "height"   # what i/k drives on the position tab
        # Display-only image adjustments; measurements always use raw frames.
        self.contrast = {"bytescale": False, "gamma": 1.0,
                         "sharpness": 1.0, "brightness": 1.0}
        # With no dataset yet the window still opens: a blank frame and a
        # template parameter set stand in until "Import images..." is pressed.
        self.no_images = not (images_dir and os.path.isdir(images_dir))
        self.pqd_path = pqd_path
        info = None
        if not self.no_images:
            try:
                info = scan_image_folder(images_dir)
            except Exception:
                info = None
        opened = None                 # what the parameters were read from
        if pqd_path and os.path.exists(pqd_path):
            self.pqd = Spec(pqd_path)
            opened = "loaded {}".format(os.path.basename(pqd_path))
            if pqd_path.lower().endswith(".pqd"):
                # a wapa.pqd of the earlier versions: read, and saved as
                # ciliograph.json beside it from now on
                self.pqd_path = os.path.join(os.path.dirname(os.path.abspath(pqd_path)),
                                             SPEC_NAME)
                opened += " (Save writes {} beside it)".format(SPEC_NAME)
            elif self.pqd.coeffs is None:
                opened += " (no beat fitted yet: the template stands in)"
        elif info is not None:
            # Images but no specification file: the defaults for these
            # images, in memory.  pqd_path stays as where Save will write.
            self.pqd = default_pqd_for_images(info)
            self.pqd_path = pqd_path or os.path.join(
                os.path.abspath(images_dir), SPEC_NAME)
            opened = ("{} frames, parameters at their defaults (no "
                      "specification here yet; Save writes {})".format(
                          info.get("count", "?"), SPEC_NAME))
        else:
            self.pqd = Spec.defaults()
            self.pqd_path = None
        self._new_model()
        self.images_dir = images_dir or ""
        if self.no_images:
            self.video = VideoFrames.placeholder()
        else:
            try:
                self.video = VideoFrames.from_pqd(self.pqd, images_dir)
            except Exception:
                self.video = VideoFrames.placeholder()
                self.no_images = True

        self.img_h, self.img_w = self.video.shape
        self._check_image_size()

        self.frame = self.video.first
        self.playing = False
        self.fps_play = fps_play

        self.mode = 0
        self.show_model = True
        self.span = "all"
        self.status = ("no images -- press 'Import images...' on the video tab"
                       if self.no_images or not opened else opened)

        self._updating = False
        self._build()
        self.refresh()

    def _new_model(self) -> None:
        """The geometry and the beat of the current specification: the
        fitted coefficients, or ones given on the command line, or the
        template.  ``self.fourier`` is the beat the params tab edits."""
        wave = FourierWave.from_spec(self.pqd, coeffs=self._fourier_coeffs,
                                     n_modes=self._n_modes, n_space=self._n_space)
        self.model = WaveModel(self.pqd, wave=wave)
        self.fourier = self.model.wave

    def _check_image_size(self) -> None:
        want_w = int(self.pqd.getf("vid_image_w", self.img_w))
        want_h = int(self.pqd.getf("vid_image_h", self.img_h))
        try:
            have_h, have_w = self.video.source_shape
        except Exception:
            have_h, have_w = self.img_h, self.img_w
        if (want_w, want_h) != (have_w, have_h):
            print("Note: the specification says the images are {}x{} but the "
                  "files on disk are {}x{}.  Using the real size for the "
                  "overlay; consider updating vid_image_w / vid_image_h.".format(
                      want_w, want_h, have_w, have_h))

    # -------------------------------------------------------------- layout

    def _build(self) -> None:
        self.fig = plt.figure(figsize=(15.5, 9.0))
        try:
            # just the name: the file and folder are in the readout
            self.fig.canvas.manager.set_window_title("Ciliograph")
        except Exception:
            pass

        # Video + overlay.
        self.ax_img = self.fig.add_axes([0.04, 0.42, 0.52, 0.54])
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])
        self.im_artist = self.ax_img.imshow(
            apply_contrast(self.video.get(self.frame), **self.contrast),
            cmap="gray", vmin=0.0, vmax=1.0,
            extent=[0, self.img_w, self.img_h, 0])
        (self.line_model,) = self.ax_img.plot(
            [], [], color="#00ff88", linewidth=2.2, zorder=5)
        (self.line_base,) = self.ax_img.plot(
            [], [], marker="o", color="#ff3355", markersize=7,
            linestyle="none", zorder=6)
        self.ax_img.set_xlim(0, self.img_w)
        self.ax_img.set_ylim(self.img_h, 0)
        self._img_extent = (self.img_w, self.img_h)

        # Model shapes over the cycle.
        self.ax_shape = self.fig.add_axes([0.62, 0.635, 0.35, 0.325])
        self.ax_shape.set_xlabel("x / um")
        self.ax_shape.set_ylabel("z / um")
        self.ax_shape.set_aspect("equal", adjustable="box")
        self.cycle_lines = [
            self.ax_shape.plot([], [], color="#bbbbbb", linewidth=1.0)[0]
            for _ in range(16)
        ]
        (self.line_now,) = self.ax_shape.plot(
            [], [], color="#0066cc", linewidth=2.4, zorder=5)

        # Status readout.
        self.ax_hud = self.fig.add_axes([0.62, 0.135, 0.35, 0.45])
        self.ax_hud.axis("off")
        self.hud = self.ax_hud.text(
            0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=9.0)
        self.fig.canvas.mpl_connect("resize_event", lambda e: self._fit_hud_font())

        self._build_progress()

        # Frame timeline, then the tab strip and per-mode control panels.
        self._build_timeline()
        self._build_tabs()

        # Buttons.  Reload is the r key; the tabs are the strip and the
        # a/s keys.  The rows sit below the readout's lowest line.
        self.buttons = {}
        specs = [
            ("play", "Play/Pause", 0.62, self._toggle_play),
            ("prev", "< Frame", 0.70, lambda e: self.step_frame(-1)),
            ("next", "Frame >", 0.775, lambda e: self.step_frame(1)),
            ("save", "Save", 0.85, lambda e: self.save()),
            ("help", "Help", 0.925, self._on_help),
        ]
        for key, label, x, cb in specs:
            ax = self.fig.add_axes([x, 0.078, 0.070 if key != "help" else 0.055,
                                    0.042])
            b = Button(ax, label)
            b.on_clicked(cb)
            self.buttons[key] = b

        ax = self.fig.add_axes([0.62, 0.022, 0.108, 0.042])
        self.btn_export = Button(ax, "Export nodes", color="#d8f0d8",
                                 hovercolor="#b8e4b8")
        self.btn_export.label.set_fontsize(9)
        self.btn_export.on_clicked(self._on_export)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.timer = self.fig.canvas.new_timer(interval=int(1000 / self.fps_play))
        self.timer.add_callback(self._on_tick)
        self.timer.start()

    # ------------------------------------------------------ frame timeline

    def _build_timeline(self) -> None:
        """The frame strip: every frame, with the main beat cycle in purple.

        Mirrors the strip in the original axwa window -- a green bar spanning
        the whole video with one tick per frame, the main cycle drawn over it
        in purple, and a yellow marker at the current frame.  Click anywhere on
        it to jump to that frame.
        """
        # Rebuildable: importing a new sequence changes the frame range, so
        # this redraws in place rather than stacking up extra axes.
        if getattr(self, "ax_time", None) is None:
            self.ax_time = self.fig.add_axes(self.TIME_BOX)
            self.fig.canvas.mpl_connect("button_press_event",
                                        self._on_timeline_click)
        ax = self.ax_time
        ax.clear()
        ax.set_facecolor("#2b2b2b")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        first, last = self.video.first, self.video.last
        ax.set_xlim(first - 0.5, last + 0.5)
        ax.set_ylim(0.0, 1.0)

        # Full span of the video, with end caps.
        ax.plot([first, last], [0.70, 0.70], color="#66ff66", lw=1.4, zorder=2)
        for f in (first, last):
            ax.plot([f, f], [0.54, 0.88], color="#dddddd", lw=1.2, zorder=2)

        # One tick per frame; thinned out if the video is very long.
        n = last - first + 1
        step = 1 if n <= 500 else int(np.ceil(n / 500.0))
        for f in range(first, last + 1, step):
            ax.plot([f, f], [0.06, 0.40], color="#55dd55", lw=0.8, zorder=1)

        # Main cycle (purple) and the current frame (yellow); set in refresh.
        (self.tl_cycle,) = ax.plot([], [], color="#b44ae0", lw=6.0,
                                   solid_capstyle="butt", zorder=3)
        (self.tl_edge0,) = ax.plot([], [], color="#ff2fff", lw=2.0, zorder=4)
        (self.tl_edge1,) = ax.plot([], [], color="#ff2fff", lw=2.0, zorder=4)
        (self.tl_now,) = ax.plot([], [], color="#ffee00", lw=2.6, zorder=5)

        # Caption sits just above the strip so the yellow marker cannot cross it.
        self.tl_label = ax.text(first - 0.5, 1.12, "", va="bottom", ha="left",
                                fontsize=7.5, color="#555555", zorder=6,
                                clip_on=False)

    def _refresh_timeline(self) -> None:
        start = self.model.macysta
        end = start + self.model.period_frames

        self.tl_cycle.set_data([start, end], [0.70, 0.70])
        self.tl_edge0.set_data([start, start], [0.03, 0.92])
        self.tl_edge1.set_data([end, end], [0.03, 0.92])
        self.tl_now.set_data([self.frame, self.frame], [0.0, 1.0])
        self.tl_label.set_text(
            "all frames {}..{}    main cycle {:.1f}..{:.1f} "
            "({:.2f} frames)    now {}  [{}]".format(
                self.video.first, self.video.last, start, end,
                self.model.period_frames, self.frame, self.span))

    def _on_timeline_click(self, event) -> None:
        if self._crop_press(event):
            return
        if self._mcc_click(event):
            return
        if self._tip_click(event):
            return
        if event.inaxes is not self.ax_time or event.xdata is None:
            return
        self.goto(int(round(event.xdata)))

    # ----------------------------------------------------------- tab panel

    @staticmethod
    def _set(slider, value) -> None:
        """Set a slider without letting it run outside its own range.

        ``drawon`` is cleared on every slider at build time: matplotlib's
        ``set_val`` issues a full ``draw_idle`` per call, so syncing ~30 sliders
        would redraw the whole figure ~30 times per refresh.  ``refresh()``
        draws once at the end instead.
        """
        value = float(np.clip(value, slider.valmin, slider.valmax))
        if slider.val != value:
            slider.set_val(value)

    def _panel_axes(self, tab: str, row: int, height: float = 0.020):
        ax = self.fig.add_axes([self.PANEL_X,
                                self.PANEL_TOP - row * self.PANEL_DY,
                                self.PANEL_W, height])
        self._tab_axes.setdefault(tab, []).append(ax)
        return ax

    def _panel_box(self, tab, x, y, w, h):
        """A free-positioned axes belonging to one tab."""
        ax = self.fig.add_axes([x, y, w, h])
        self._tab_axes.setdefault(tab, []).append(ax)
        return ax

    def _arrow_pair(self, tab, x, y, w, h, on_step, step):
        """Small up/down buttons -- matplotlib has no spin control."""
        aw, ax_x = 0.013, x + w + 0.002
        for glyph, delta, yy in (("▲", step, y + h / 2.0), ("▼", -step, y)):
            ax = self._panel_box(tab, ax_x, yy, aw, h / 2.0)
            btn = Button(ax, glyph, color="#ededed", hovercolor="#d6d6d6")
            btn.label.set_fontsize(5.0)
            btn.on_clicked(lambda _e, d=delta: on_step(d))
            self._spin_buttons.append(btn)

    def _step_box(self, tb, delta: float, fmt: str = "{:g}") -> None:
        """Nudge a text box's value, then let its own handler validate it."""
        try:
            current = float(str(tb.text).strip())
        except (TypeError, ValueError):
            self.status = "cannot step a non-numeric value"
            self.refresh()
            return
        tb.set_val(fmt.format(round(current + delta, 6)))

    def _spin_textbox(self, tab, x, y, w, label, initial, key_step,
                      fmt="{:g}", h=0.026):
        """A text box with arrows, validated by whatever handler it is given."""
        tb = self._panel_textbox(tab, x, y, w, label, initial, h=h)
        self._arrow_pair(tab, x, y, w, h,
                         lambda d, t=tb, f=fmt: self._step_box(t, d, f),
                         key_step)
        return tb

    def _readonly_box(self, tab, x, y, w, label, h=0.024):
        """A greyed, non-editable field for a derived value."""
        ax = self._panel_box(tab, x, y, w, h)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#ececec")
        for sp in ax.spines.values():
            sp.set_color("#b4b4b4")
        ax.text(-0.05, 0.5, label, transform=ax.transAxes,
                ha="right", va="center", fontsize=8.5)
        return ax.text(0.06, 0.5, "", transform=ax.transAxes,
                       ha="left", va="center", fontsize=8.5, color="#333333")

    def _panel_textbox(self, tab, x, y, w, label, initial, h=0.026):
        ax = self._panel_box(tab, x, y, w, h)
        tb = TextBox(ax, label, initial=initial)
        tb.label.set_fontsize(8.5)
        try:
            tb.text_disp.set_fontsize(8.5)
        except Exception:
            pass
        self._textboxes = getattr(self, "_textboxes", []) + [tb]
        return tb

    def _typing(self) -> bool:
        """Is a text box being edited?  Then the keys are its, not the
        shortcuts': typing "12.5" into ppum used to step the frame at the
        "." (and the redraw put the old value back into the box)."""
        return any(getattr(tb, "capturekeystrokes", False)
                   for tb in getattr(self, "_textboxes", []))

    def _set_box(self, tb, text) -> None:
        """Show a value in a text box -- unless it is being typed into."""
        if getattr(tb, "capturekeystrokes", False):
            return
        if tb.text != text:
            tb.set_val(text)

    def _tb_number_cb(self, key, cast=float, unit="", after=None, rebuild=False):
        """Text box handler writing one numeric parameter into the specification."""
        def cb(text):
            # set_val() re-fires submit, so ignore echoes from _sync_sliders.
            if self._updating:
                return
            try:
                value = cast(float(str(text).strip()))
            except (TypeError, ValueError):
                self.status = "not a number: {!r}".format(text)
                self.refresh()
                return
            self.pqd.set(key, value)
            if rebuild:
                self.model.rebuild()
            if after is not None:
                after()
            self.status = "{} = {}{}".format(key, value, unit)
            self.refresh()
        return cb

    def _panel_slider(self, tab, row, label, lo, hi, value,
                      valfmt="%.4f", valstep=None):
        ax = self._panel_axes(tab, row)
        s = Slider(ax, label, lo, hi,
                   valinit=float(np.clip(value, lo, hi)),
                   valfmt=valfmt, valstep=valstep)
        s.label.set_fontsize(9)
        s.drawon = False          # refresh() issues a single draw instead
        return s

    def _scalar_cb(self, key, rebuild=False, unit="", after=None):
        """Callback writing one parameter value into the specification."""
        def cb(val):
            if self._updating:
                return
            self.pqd.set(key, float(val))
            if rebuild:
                self.model.rebuild()
            if after is not None:
                after()
            self.status = "{} = {:g}{}".format(key, float(val), unit)
            self.refresh()
        return cb

    # ------------------------------------------------------- main cycle

    def _sync_cycle_span(self) -> None:
        """Keep !vid_suimvi_first / _last in step with the main cycle.

        Those two entries record the same span the purple bar shows, so they
        are kept consistent whenever the cycle start or the beat frequency
        changes.
        """
        first = self.model.macysta
        last = first + self.model.period_frames
        self.pqd.set("vid_suimvi_first", int(round(first)))
        self.pqd.set("vid_suimvi_last", int(round(last)))

    def _fps_cb(self, text) -> None:
        """Change the frame rate, holding the cycle's span in frames.

        The cycle end points are frames you picked off the video, so they do
        not move when the capture rate is corrected -- the beat frequency does.
        """
        if self._updating:
            return
        try:
            value = float(str(text).strip())
        except (TypeError, ValueError):
            self.status = "not a number: {!r}".format(text)
            self.refresh()
            return
        if value <= 0:
            self.status = "fps must be positive"
            self.refresh()
            return

        span = self.model.period_frames        # frames, before the change
        self.pqd.set("vid_fps", value)
        if span > 0:
            self.pqd.set("cbf", value / span)
        self.model.rebuild()
        self._sync_cycle_span()
        self.status = "fps = {:g}  ->  cbf {:.4f} Hz over {:.2f} frames".format(
            value, self.model.cbf, span)
        self.refresh()

    def _on_level(self, _event=None) -> None:
        """Rotate so the surface the cilia grow from runs horizontally."""
        if self.no_images:
            self.status = "import images first to level the surface"
            self.refresh()
            return
        try:
            with self._busy("levelling") as report:
                info = level_rotation(self.video, self.pqd, progress=report)
        except Exception as exc:
            self.status = "could not level: {}".format(exc)
            self.refresh()
            return

        if info.get("method") == "none":
            self.status = ("no flat surface and no clear cilia band -- "
                           "rotation left alone")
            self.refresh()
            return

        angle = ((info["rotation"] + 180.0) % 360.0) - 180.0
        self.pqd.set("vid_image_rotation", angle)
        self.video.set_rotation(angle)
        self._updating = True
        try:
            self.tb_rot.set_val("{:g}".format(round(angle, 2)))
        finally:
            self._updating = False
        self.status = ("{} was {:+.2f} deg off -> rotation {:+.2f} deg, "
                       "{:+.2f} deg left; re-check the base".format(
                           info.get("method", "surface"), info["tilt_deg"],
                           angle, info["residual_deg"]))
        self.refresh()

    def _rotation_cb(self, text) -> None:
        """Rotate every frame.  This feeds the measurements, not just the view."""
        if self._updating:
            return
        try:
            value = float(str(text).strip())
        except (TypeError, ValueError):
            self.status = "not a number: {!r}".format(text)
            self.refresh()
            return
        value = ((value + 180.0) % 360.0) - 180.0     # keep it in -180..180
        self.pqd.set("vid_image_rotation", value)
        self.video.set_rotation(value)
        self._updating = True
        try:
            self.tb_rot.set_val("{:g}".format(value))
        finally:
            self._updating = False
        self.status = ("image rotation {:g} deg -- the base position is in "
                       "rotated coordinates".format(value))
        self.refresh()

    def _cycle_first_cb(self, text) -> None:
        if self._updating:
            return
        try:
            first = float(str(text).strip())
        except (TypeError, ValueError):
            self.status = "not a number: {!r}".format(text)
            self.refresh()
            return

        # Hold the cycle length and move its start.
        self.pqd.set("vid_frame_main_cycle_start", first)
        self._sync_cycle_span()
        self.status = "main cycle starts at frame {:g}".format(first)
        self.refresh()

    def _cycle_last_cb(self, text) -> None:
        """The last frame of the cycle fixes its length, hence the beat freq."""
        if self._updating:
            return
        try:
            last = float(str(text).strip())
        except (TypeError, ValueError):
            self.status = "not a number: {!r}".format(text)
            self.refresh()
            return

        first = self.model.macysta
        span = last - first
        if span <= 0:
            self.status = ("cycle last ({:g}) must be after cycle first "
                           "({:g})".format(last, first))
            self.refresh()
            return

        fps = self.model.fps
        if fps <= 0:
            self.status = "set fps on the video tab first"
            self.refresh()
            return

        cbf = fps / span
        self.pqd.set("cbf", cbf)
        self.model.rebuild()
        self._sync_cycle_span()
        self.status = "cycle {:g}..{:g} = {:.2f} frames -> cbf {:.4f} Hz".format(
            first, last, span, cbf)
        self.refresh()

    def _make_tab_cb(self, index: int):
        def cb(_event):
            self.mode = index
            self._select_tab(index)
            self.status = "tab: {}".format(self.MODES[index])
            self.refresh()
        return cb

    def _frame_cb(self, val):
        if self._updating:
            return
        lo, hi = self.play_range
        self.frame = int(np.clip(int(val), lo, hi))
        self.refresh()

    def set_nodes(self, n: int) -> int:
        """Choose how many points lie along the cilium, base to tip.

        The specification stores the node step ``dl``, so this writes
        ``dl = length / (nodes - 1)``.  The reconstruction floors a division,
        which can land one node short, so the step is nudged if needed.
        """
        n = max(2, int(n))
        length = self.model.length
        self.pqd.set("dl", length / (n - 1))
        self.model.rebuild()
        if self.model.n_xy != n:
            self.pqd.set("dl", (length / (n - 1)) * (1.0 - 1e-12))
            self.model.rebuild()
        return self.model.n_xy

    def _steps_cb(self, text):
        if self._updating:
            return
        try:
            n = int(float(str(text).strip()))
        except (TypeError, ValueError):
            self.status = "not a number: {!r}".format(text)
            self.refresh()
            return
        if n < 2:
            self.status = "need at least 2 time steps"
            self.refresh()
            return
        self.n_cs = n
        self.status = "export will write {} time steps".format(n)
        self.refresh()

    def _on_export(self, _event=None) -> None:
        """Write the x-z position of every node, at every time step."""
        outdir = self._output_dir()
        if outdir is None:
            self.status = "import images first"
            self.refresh()
            return
        try:
            os.makedirs(outdir, exist_ok=True)
            self.fourier.length = self.model.length
            self.fourier.cbf = self.model.cbf
            series = ShapeSeries(self.fourier, n_cs=self.n_cs,
                                 n_pts=self.model.n_xy)
            # named after the recording, so exports of several recordings
            # into one folder stay apart
            name = sequence_name(self.images_dir) if self.images_dir else "cilium"
            written = [
                series.save_data(os.path.join(outdir, name + "_z01a.data")),
                series.save_csv(os.path.join(outdir, name + "_cilium_shapes.csv")),
                series.save_npz(os.path.join(outdir, name + "_cilium_shapes.npz")),
            ]
        except Exception as exc:
            self.status = "export failed: {}".format(exc)
            self.refresh()
            return

        for path in written:
            print("wrote {}".format(path))
        self.status = ("exported {} nodes x {} time steps (x00/y00 = the "
                       "cycle start, the {} state) -> {}".format(
                           series.n_xy, self.n_cs,
                           "rightmost" if (effective_direction(self.pqd) or 1) > 0
                           else "leftmost",
                           os.path.join("output", name + "_z01a.data")))
        self.refresh()

    def _nodes_cb(self, val):
        if self._updating:
            return
        got = self.set_nodes(int(val))
        self.status = "nodes = {}  (dl = {:.5g} um)".format(got, self.model.dl)
        self.refresh()

    def _length_cb(self, val):
        """Change the length while holding the node count."""
        if self._updating:
            return
        n = self.model.n_xy
        self.pqd.set("length", float(val))
        self.model.rebuild()
        self.set_nodes(n)
        self.status = "length = {:g} um  ({} nodes, dl = {:.5g} um)".format(
            float(val), self.model.n_xy, self.model.dl)
        self.refresh()

    # ---------------------------------------------- Fourier coefficient edit

    def slot_label(self, slot: int) -> str:
        block, mi, _ = self._slots[slot]
        return "{}[n={},m={}]".format(block, self.sel_n, mi + 1)

    def get_coeff(self, slot: int) -> float:
        block, mi, _ = self._slots[slot]
        return float(self.fourier.coeffs[block][mi, self.sel_n])

    def set_coeff(self, slot: int, value: float) -> None:
        block, mi, _ = self._slots[slot]
        self.fourier.coeffs[block][mi, self.sel_n] = float(value)

    def _make_fourier_cb(self, slot: int):
        def cb(val):
            if self._updating:
                return
            self.set_coeff(slot, val)
            self.sel_slot = slot
            self.status = "{} = {:.4f}".format(self.slot_label(slot), float(val))
            self.refresh(sliders=False)
        return cb

    def _n_cb(self, val):
        if self._updating:
            return
        self.sel_n = int(np.clip(int(val), 0, self.fourier.n_modes - 1))
        self.status = "Fourier mode n = {}".format(self.sel_n)
        self.refresh()

    def cycle_slot(self, delta: int) -> None:
        self.sel_slot = (self.sel_slot + delta) % len(self._slots)
        self.status = "selected {}".format(self.slot_label(self.sel_slot))
        self.refresh(sliders=False)

    def cycle_mode_n(self, delta: int) -> None:
        self.sel_n = (self.sel_n + delta) % self.fourier.n_modes
        self.status = "Fourier mode n = {}".format(self.sel_n)
        self.refresh()

    def _build_tabs(self) -> None:
        """One tab per mode, each holding only that mode's controls."""
        self._tab_axes: dict = {}
        self.tab_buttons: dict = {}
        self._spin_buttons: list = []

        n = len(self.MODES)
        span = (self.TAB_X1 - self.TAB_X0) / n
        for i, name in enumerate(self.MODES):
            ax = self.fig.add_axes(
                [self.TAB_X0 + i * span, self.TAB_Y, span * 0.94, self.TAB_H])
            b = Button(ax, name, color="#e8e8e8", hovercolor="#d0e4ff")
            b.label.set_fontsize(7.5)
            b.on_clicked(self._make_tab_cb(i))
            self.tab_buttons[name] = b

        # Header describing the active tab.
        self.ax_panel = self.fig.add_axes(
            [self.PANEL_X, self.PANEL_TOP + 0.030, self.PANEL_W, 0.001])
        self.ax_panel.axis("off")
        self.panel_title = self.ax_panel.text(
            0.0, 0.0, "", va="bottom", ha="left", fontsize=9,
            style="italic", color="#444444")

        v = self.video

        # video -----------------------------------------------------------
        # A placeholder sequence has a single frame; give the slider a nonzero
        # span so matplotlib does not warn about identical limits.
        self.sl_frame = self._panel_slider(
            "video", 0, "frame", v.first, max(v.last, v.first + 1), self.frame,
            valfmt="%d", valstep=1)
        self.sl_frame.on_changed(self._frame_cb)

        # Typed entry for the frame range and the capture settings.
        self.tb_first = self._spin_textbox(
            "video", 0.180, 0.206, 0.058, "first frame ",
            "{:d}".format(int(self.pqd.getf("vid_first_frame_to_load", 0))),
            1, fmt="{:.0f}")
        self.tb_first.on_submit(self._tb_number_cb(
            "vid_first_frame_to_load", int, after=self._reload_video))

        self.tb_last = self._spin_textbox(
            "video", 0.372, 0.206, 0.058, "last frame ",
            "{:d}".format(int(self.pqd.getf("vid_last_frame_to_load", 0))),
            1, fmt="{:.0f}")
        self.tb_last.on_submit(self._tb_number_cb(
            "vid_last_frame_to_load", int, after=self._reload_video))

        self.tb_ppum = self._spin_textbox(
            "video", 0.150, 0.176, 0.042, "ppum ",
            "{:g}".format(self.pqd.getf("vid_ppum", 1.0)), 0.1)
        self.tb_ppum.on_submit(self._tb_number_cb("vid_ppum", float, " px/um"))

        self.tb_fps = self._spin_textbox(
            "video", 0.262, 0.176, 0.042, "fps ",
            "{:g}".format(self.pqd.getf("vid_fps", 300.0)), 10)
        self.tb_fps.on_submit(self._fps_cb)

        self.tb_rot = self._spin_textbox(
            "video", 0.390, 0.176, 0.040, "rotation ",
            "{:g}".format(self.pqd.getf("vid_image_rotation", 0.0)), 1.0)
        self.tb_rot.on_submit(self._rotation_cb)

        ax = self._panel_box("video", 0.452, 0.174, 0.048, 0.028)
        self.btn_level = Button(ax, "Level", color="#e0eaff",
                                hovercolor="#c4d9ff")
        self.btn_level.label.set_fontsize(7.5)
        self.btn_level.on_clicked(self._on_level)

        # Import a sequence of images and read its dimensions.
        ax = self._panel_box("video", 0.105, 0.142, 0.120, 0.030)
        self.btn_import = Button(ax, "Import images...", color="#e0eaff",
                                 hovercolor="#c4d9ff")
        self.btn_import.label.set_fontsize(8)
        self.btn_import.on_clicked(self._on_import)

        self.tb_dir = self._panel_textbox(
            "video", 0.238, 0.143, 0.262, "", self.images_dir, h=0.028)
        self.tb_dir.on_submit(self._on_import_path)

        ax = self._panel_box("video", 0.105, 0.130, 0.395, 0.001)
        ax.axis("off")
        self.import_info = ax.text(
            0.0, 0.0, "", va="top", ha="left", fontsize=8,
            family="monospace", color="#333333", clip_on=False)
        self._describe_sequence()

        self._build_contrast_panel()
        self._build_enhance_tab()
        self._build_mcc_tab()

        # position --------------------------------------------------------
        self.sl_x0 = self._panel_slider(
            "position", 0, "cil x0", 0, self.img_w,
            self.pqd.getf("vid_cil_x0"), valfmt="%.1f")
        self.sl_x0.on_changed(self._scalar_cb("vid_cil_x0", unit=" px"))
        self.sl_y0 = self._panel_slider(
            "position", 1, "cil y0 invy", 0, self.img_h,
            self.pqd.getf("vid_cil_y0_invy"), valfmt="%.1f")
        self.sl_y0.on_changed(self._scalar_cb("vid_cil_y0_invy", unit=" px"))

        self.sl_ori = self._panel_slider(
            "position", 2, "orientation", -180.0, 180.0,
            self.pqd.getf("vid_cil_orientation"), valfmt="%.2f")
        self.sl_ori.on_changed(
            self._scalar_cb("vid_cil_orientation", unit=" deg"))

        ax = self._panel_box("position", 0.105, 0.146, 0.110, 0.032)
        self.btn_locate = Button(ax, "Find base", color="#e0eaff",
                                 hovercolor="#c4d9ff")
        self.btn_locate.label.set_fontsize(8)
        self.btn_locate.on_clicked(self._on_locate)
        # The frame the pattern is drawn in: upright (turned to the fitted
        # fan's own axis, PATTERN_FRAME_KEY) or the surface's.
        ax = self._panel_box("position", 0.232, 0.146, 0.110, 0.032)
        self.btn_upright = Button(ax, self._frame_label(), color="#ededed",
                                  hovercolor="#d6d6d6")
        self.btn_upright.label.set_fontsize(7.5)
        self.btn_upright.on_clicked(self._on_upright)

        ax = self._panel_box("position", 0.105, 0.134, 0.395, 0.001)
        ax.axis("off")
        self.locate_info = ax.text(
            0.0, 0.0, "estimate the base from where the cilia move",
            va="top", ha="left", fontsize=8, family="monospace",
            color="#333333", clip_on=False)

        # cycle -----------------------------------------------------------
        self.sl_macy = self._panel_slider(
            "cycle", 0, "cycle start", v.first, max(v.last, v.first + 1),
            self.pqd.getf("vid_frame_main_cycle_start"), valfmt="%.2f")
        self.sl_macy.on_changed(
            self._scalar_cb("vid_frame_main_cycle_start", unit=" fr",
                            after=self._sync_cycle_span))

        # Typed entry of the main cycle's end points.  The last frame is not
        # stored directly -- it fixes the cycle length, i.e. the beat frequency.
        self.tb_cyc_first = self._spin_textbox(
            "cycle", 0.180, 0.202, 0.058, "cycle first ",
            "{:g}".format(self.model.macysta), 1)
        self.tb_cyc_first.on_submit(self._cycle_first_cb)

        self.tb_cyc_last = self._spin_textbox(
            "cycle", 0.372, 0.202, 0.058, "cycle last ",
            "{:g}".format(self.model.macysta + self.model.period_frames), 1)
        self.tb_cyc_last.on_submit(self._cycle_last_cb)

        # cbf is not an input any more: it follows from the cycle end points
        # and the frame rate, so it is shown read-only.
        self.cbf_readout = self._readonly_box(
            "cycle", 0.180, 0.168, 0.090, "cbf (Hz) ")

        # Which way the effective stroke sweeps (EFFECTIVE_STROKE_KEY):
        # right, left or auto; Find cycle and Auto fit follow it.
        ax = self._panel_box("cycle", 0.330, 0.165, 0.139, 0.028)
        self.btn_effdir = Button(ax, self._effdir_label(), color="#ededed",
                                 hovercolor="#d6d6d6")
        self.btn_effdir.label.set_fontsize(7.5)
        self.btn_effdir.on_clicked(lambda _e: self._on_effdir_toggle())

        # Live derived values, rewritten on every refresh.
        ax = self._panel_box("cycle", 0.105, 0.140, 0.395, 0.001)
        ax.axis("off")
        self.cycle_info = ax.text(
            0.0, 0.0, "", va="top", ha="left", fontsize=8,
            family="monospace", color="#333333", clip_on=False)

        ax = self._panel_box("cycle", 0.105, 0.098, 0.110, 0.030)
        self.btn_cycle = Button(ax, "Find cycle", color="#e0eaff",
                                hovercolor="#c4d9ff")
        self.btn_cycle.label.set_fontsize(8)
        self.btn_cycle.on_clicked(self._on_find_cycle)

        ax = self._panel_box("cycle", 0.232, 0.098, 0.110, 0.030)
        self.btn_play_main = Button(ax, "Play main cycle", color="#d8f0d8",
                                    hovercolor="#b8e4b8")
        self.btn_play_main.label.set_fontsize(7.5)
        self.btn_play_main.on_clicked(lambda e: self.play_span("main"))

        ax = self._panel_box("cycle", 0.359, 0.098, 0.110, 0.030)
        self.btn_play_all = Button(ax, "Play all frames", color="#d8f0d8",
                                   hovercolor="#b8e4b8")
        self.btn_play_all.label.set_fontsize(7.5)
        self.btn_play_all.on_clicked(lambda e: self.play_span("all"))

        # Detection result -- kept separate so the refresh does not clear it.
        ax = self._panel_box("cycle", 0.105, 0.086, 0.395, 0.001)
        ax.axis("off")
        self.cycle_found = ax.text(
            0.0, 0.0, "estimate the beat period from the video",
            va="top", ha="left", fontsize=8, family="monospace",
            color="#333333", clip_on=False)

        # length ----------------------------------------------------------
        self.sl_len = self._panel_slider(
            "length", 0, "length", 0.5, 20.0, self.pqd.getf("length", 6.0),
            valfmt="%.3f")
        self.sl_len.on_changed(self._length_cb)

        # The file stores the integration step dl, but nodes are what you
        # actually care about, so the control is in nodes and dl is derived.
        self.sl_nodes = self._panel_slider(
            "length", 1, "nodes", 11, 2001, self.model.n_xy,
            valfmt="%d", valstep=1)
        self.sl_nodes.on_changed(self._nodes_cb)

        ax = self._panel_box("length", 0.105, 0.196, 0.395, 0.001)
        ax.axis("off")
        self.nodes_info = ax.text(
            0.0, 0.0, "", va="top", ha="left", fontsize=8,
            family="monospace", color="#333333", clip_on=False)

        self.tb_steps = self._spin_textbox(
            "length", 0.200, 0.136, 0.058, "time steps ", str(self.n_cs),
            1, fmt="{:.0f}")
        self.tb_steps.on_submit(self._steps_cb)

        ax = self._panel_box("length", 0.105, 0.098, 0.110, 0.030)
        self.btn_len = Button(ax, "Find length", color="#e0eaff",
                              hovercolor="#c4d9ff")
        self.btn_len.label.set_fontsize(8)
        self.btn_len.on_clicked(self._on_find_length)

        ax = self._panel_box("length", 0.232, 0.098, 0.110, 0.030)
        self.btn_tip = Button(ax, "Click tip", color="#ffe9b8",
                              hovercolor="#ffd781")
        self.btn_tip.label.set_fontsize(8)
        self.btn_tip.on_clicked(self._arm_tip_click)

        ax = self._panel_box("length", 0.105, 0.084, 0.395, 0.001)
        ax.axis("off")
        self.length_info = ax.text(
            0.0, 0.0, "'Click tip' then click the cilium tip on the image",
            va="top", ha="left", fontsize=8, family="monospace",
            color="#333333", clip_on=False)

        # params ----------------------------------------------------------
        # one Fourier mode n at a time, its 12 coefficients, x left z right
        self.fourier_sliders = []
        self.sl_n = self._panel_slider(
            "params", 0, "mode n", 0, max(self.fourier.n_modes - 1, 1),
            self.sel_n, valfmt="%d", valstep=1)
        self.sl_n.on_changed(self._n_cb)

        self._slots = fourier_slots(self.fourier.n_space)
        per_col = 2 * self.fourier.n_space
        cols = (0.150, 0.360)
        width = 0.135
        top = self.PANEL_TOP - 0.026
        for slot, (block, mi, col) in enumerate(self._slots):
            row = slot % per_col
            ax = self._panel_box("params", cols[col], top - row * 0.026,
                                 width, 0.018)
            s = Slider(ax, "{} m{}".format(block, mi + 1),
                       FOURIER_RANGE[0], FOURIER_RANGE[1],
                       valinit=float(np.clip(
                           self.fourier.coeffs[block][mi, self.sel_n],
                           *FOURIER_RANGE)),
                       valfmt="%.4f")
            s.label.set_fontsize(7.5)
            s.valtext.set_fontsize(7.5)
            s.drawon = False
            s.on_changed(self._make_fourier_cb(slot))
            self.fourier_sliders.append(s)

        ax = self._panel_box("params", 0.105, 0.012, 0.115, 0.028)
        self.btn_tracefit = Button(ax, "Auto fit", color="#e0eaff",
                                   hovercolor="#c4d9ff")
        self.btn_tracefit.label.set_fontsize(8)
        self.btn_tracefit.on_clicked(self._on_tracefit)
        # Auto fit polishes the beat already there (fourier_autofit,
        # ``current``); this one starts over from the traces and the
        # template, as the first Auto fit does.
        ax = self._panel_box("params", 0.105, 0.046, 0.115, 0.024)
        self.btn_freshfit = Button(ax, "Fresh fit", color="#ededed",
                                   hovercolor="#d6d6d6")
        self.btn_freshfit.label.set_fontsize(7.5)
        self.btn_freshfit.on_clicked(lambda _e: self._on_tracefit(fresh=True))

        # The fit's readout (energy, ranking, runner-ups, cycle start, frame,
        # sweep...) goes to the terminal: drawn on the figure it ran over the
        # sliders and the buttons at the bottom.  The HUD keeps the one-line
        # status and the amplitude.
        self.tracefit_info = _Readout("Auto fit")

        self._select_tab(self.mode)

    def _frame_label(self) -> str:
        return "frame: {}".format(pattern_frame_setting(self.pqd))

    def _turn_frame(self, delta_deg: float) -> None:
        """Turn the model frame by ``delta_deg`` with the beat in it, so
        that nothing moves on the frames: the orientation changes and the
        coefficients are turned by the same angle (rotate_fourier_frame)."""
        coeffs = rotate_fourier_frame(self.fourier.coeffs, delta_deg)
        self._fourier_coeffs = coeffs
        self.fourier.set_coeffs(coeffs)
        ori = self.pqd.getf("vid_cil_orientation") + delta_deg
        ori = ((ori + 180.0) % 360.0) - 180.0
        self.pqd.set("vid_cil_orientation", round(ori, 3))
        turn = self.pqd.getf(PATTERN_TURN_KEY, 0.0) + delta_deg
        self.pqd.set(PATTERN_TURN_KEY, round(((turn + 180.0) % 360.0) - 180.0, 3))
        self.model.rebuild()

    def _turn_to_fan_axis(self) -> str:
        """Turn the frame so the fitted fan's axis is the vertical; a line
        for the readout.

        The orientation is normally the surface's: x runs along the cell
        surface, z is its normal, and a beat that leans -- cilia lying to
        one side for most of the beat, as on CO121 -- is drawn leaning.
        Turned to the beat's own axis the fan stands symmetric about the
        vertical and the surface is tilted by the lean in the pattern plot
        and the export.
        """
        lean = fan_lean(self.fourier)
        axis = lean["axis_deg"]
        if abs(axis) < 0.5:
            return ("frame: upright already (fan axis {:+.1f} deg from the "
                    "frame's normal)".format(axis))
        self._turn_frame(-axis)
        return ("frame: upright -- turned {:.1f} deg: the fan leaned {:.1f} deg "
                "{} of the frame's normal (tips {:+.0f}..{:+.0f} deg); "
                "orientation now {:+.2f} deg, the surface tilted by that "
                "much in the pattern".format(
                    abs(axis), abs(axis), "right" if axis > 0 else "left",
                    lean["lo_deg"], lean["hi_deg"],
                    self.pqd.getf("vid_cil_orientation")))

    def _turn_to_surface(self) -> str:
        """Turn the frame back to the surface's orientation (undoing the
        turns recorded in PATTERN_TURN_KEY); a line for the readout."""
        turn = self.pqd.getf(PATTERN_TURN_KEY, 0.0)
        if abs(turn) < 0.05:
            return "frame: surface (orientation {:+.2f} deg)".format(
                self.pqd.getf("vid_cil_orientation"))
        self._turn_frame(-turn)
        self.pqd.set(PATTERN_TURN_KEY, 0.0)
        return ("frame: surface -- turned {:+.1f} deg back to the surface's "
                "orientation {:+.2f} deg".format(
                    -turn, self.pqd.getf("vid_cil_orientation")))

    def _on_upright(self, _event=None) -> None:
        """Toggle the pattern frame (upright / surface) and apply it to the
        beat on the params tab, if one is fitted."""
        cur = pattern_frame_setting(self.pqd)
        nxt = "surface" if cur == "upright" else "upright"
        self.pqd.set(PATTERN_FRAME_KEY, nxt)
        self.btn_upright.label.set_text(self._frame_label())
        if self._current_beat() is None:
            self.status = ("pattern frame: {} (applies when a beat is fitted; "
                           "Auto fit ends in this frame)".format(nxt))
            self.refresh()
            return
        line = (self._turn_to_fan_axis() if nxt == "upright"
                else self._turn_to_surface())
        # the cycle start is the extreme in this frame
        line += "; " + self._align_cycle_start()
        self.status = line + "; Reload (r) undoes"
        self.refresh()

    def _effdir_label(self) -> str:
        return "effective stroke: {}".format(effective_stroke_setting(self.pqd))

    def _on_effdir_toggle(self) -> None:
        """Cycle the effective-stroke setting: right -> left -> auto."""
        cur = effective_stroke_setting(self.pqd)
        nxt = EFFECTIVE_STROKE_CHOICES[
            (EFFECTIVE_STROKE_CHOICES.index(cur) + 1) % len(EFFECTIVE_STROKE_CHOICES)]
        self.pqd.set(EFFECTIVE_STROKE_KEY, nxt)
        self.btn_effdir.label.set_text(self._effdir_label())
        self.status = {
            "right": "effective stroke sweeps left to right, the recovery "
                     "stroke right to left: Find cycle and Auto fit hold to it",
            "left": "effective stroke sweeps right to left, the recovery "
                    "stroke left to right: Find cycle and Auto fit hold to it",
            "auto": "effective stroke: whichever sweep is the faster, judged "
                    "after the fit (the frames alone decide)"}[nxt]
        self.refresh()

    def _select_tab(self, index: int) -> None:
        """Show only the active tab's controls, and highlight its button."""
        active = self.MODES[index]
        for name, axes in self._tab_axes.items():
            for ax in axes:
                ax.set_visible(name == active)

        for name, b in self.tab_buttons.items():
            on = (name == active)
            b.color = "#7aa7ff" if on else "#e8e8e8"
            b.ax.set_facecolor(b.color)
            b.label.set_fontweight("bold" if on else "normal")
        self.panel_title.set_text("{}  --  {}".format(active, self.MODE_BLURB[active]))

    def _sync_sliders(self) -> None:
        """Push the current state back into every slider."""
        self._updating = True
        try:
            self._set(self.sl_frame, self.frame)
            self._set(self.sl_mcc_frame, self.frame)
            self._set(self.sl_x0, self.pqd.getf("vid_cil_x0"))
            self._set(self.sl_y0, self.pqd.getf("vid_cil_y0_invy"))
            self._set(self.sl_ori, self.pqd.getf("vid_cil_orientation"))
            self._set(self.sl_macy, self.pqd.getf("vid_frame_main_cycle_start"))
            self._set(self.sl_len, self.pqd.getf("length", 6.0))
            self._set(self.sl_nodes, self.model.n_xy)
            # dl is the integration step, so coarsening it does perturb the
            # shape a little; the error is roughly proportional to dl.
            self.nodes_info.set_text(
                "{} nodes over {:g} um  ->  dl = {:.5g} um\n"
                "also the integration step: ~{:.3f} um shape error".format(
                    self.model.n_xy, self.model.length, self.model.dl,
                    0.9 * self.model.dl))
            self._set(self.sl_n, self.sel_n)
            for slot, s in enumerate(self.fourier_sliders):
                self._set(s, self.get_coeff(slot))

            # Keep the video tab's text boxes showing the live values --
            # except one being typed into, whose text is the user's.
            self._set_box(self.tb_first,
                          "{:d}".format(int(self.pqd.getf("vid_first_frame_to_load", 0))))
            self._set_box(self.tb_last,
                          "{:d}".format(int(self.pqd.getf("vid_last_frame_to_load", 0))))
            self._set_box(self.tb_ppum, "{:g}".format(self.pqd.getf("vid_ppum", 1.0)))
            self._set_box(self.tb_fps, "{:g}".format(self.pqd.getf("vid_fps", 300.0)))
            self._set_box(self.tb_rot,
                          "{:g}".format(self.pqd.getf("vid_image_rotation", 0.0)))
            self._set_box(self.tb_steps, str(self.n_cs))

            # Main cycle end points, and what they imply.
            first = self.model.macysta
            span = self.model.period_frames
            self._set_box(self.tb_cyc_first, "{:g}".format(round(first, 4)))
            self._set_box(self.tb_cyc_last, "{:g}".format(round(first + span, 4)))
            self.cbf_readout.set_text("{:.4f}".format(self.model.cbf))
            self.cycle_info.set_text(
                "cbf = fps/(last-first) = {:g}/{:.2f} fr    period {:.2f} ms".format(
                    self.model.fps, span,
                    1000.0 / self.model.cbf if self.model.cbf else float("nan")))
        finally:
            self._updating = False

    # -------------------------------------------------- importing a sequence

    def _pick_folder(self):
        """Open a native folder chooser, if the toolkit is available."""
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as exc:
            self.status = ("no folder chooser available ({}); type the path in "
                           "the box instead".format(exc))
            self.refresh()
            return None

        title = "Select the folder of video frames"
        start = self.images_dir if os.path.isdir(self.images_dir) else os.getcwd()

        # Reuse the figure's own Tk window when running on the Tk backend;
        # creating a second Tk root while one is live can misbehave.
        parent = getattr(getattr(self.fig.canvas, "manager", None), "window", None)
        if parent is not None and type(parent).__module__.startswith("tkinter"):
            return filedialog.askdirectory(parent=parent, title=title,
                                           initialdir=start) or None

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        try:
            return filedialog.askdirectory(title=title, initialdir=start) or None
        finally:
            root.destroy()

    def _find_trace_csv(self):
        roots = []
        if self.images_dir:
            roots.append(self.images_dir)
        if self.pqd_path:
            here = os.path.dirname(os.path.abspath(self.pqd_path))
            roots += [here, os.path.dirname(here)]
        for root in roots:
            candidate = os.path.join(root, "cilium_trace_px.csv")
            if os.path.exists(candidate):
                return candidate
        # Also the four-column real-unit traces the reference script uses.
        for root in roots:
            hits = sorted(glob.glob(os.path.join(root, "trace_realunits*.csv")))
            hits += sorted(glob.glob(os.path.join(root, "*_trace.csv")))
            if hits:
                return hits[0]
        return None

    def _fit_fourier_to_image(self, fresh: bool = False) -> None:
        """Auto fit for the Fourier waveform.

        Three candidates, one judge.  The cilium is traced frame by frame
        and a beat fitted through the traces (the reference method); it is
        traced again on the phase-averaged frames, where everything that
        does not repeat with the beat has blurred away; and the beat is
        fitted to the frames themselves, jointly over the whole record,
        from those two fits and from the template beat at several phase
        offsets (image_fit_beat).  The image energy -- how well each beat
        lies along a dark filament in every frame at its own phase -- picks
        the winner.  On a sparse recording the trace fit usually wins or
        the joint fit polishes it; on a dense carpet, where the traces
        hop between neighbours and the trace fit is a fan of scribbles,
        the joint fit is the only one of the three that stays on one beat.
        """
        # A hand-traced file, if there is one, is a far better starting point
        # than anything the automatic walk produces: the walk returns nearly
        # straight spokes, and no fitting method recovers a recovery stroke
        # from those.
        traced_file = self._find_trace_csv()
        if traced_file:
            self._fit_fourier_from_file(traced_file)
            return

        self.tracefit_info.set_text("tracing the cilium in every frame...")
        self.fig.canvas.draw_idle()
        t_start = time.time()
        try:
            with self._busy("tracing and fitting") as report:
                res = fourier_autofit(self.video, self.model,
                                      max_traces=getattr(self, "_max_traces", 120),
                                      progress=report,
                                      current=None if fresh else self._current_beat())
        except Exception as exc:
            self.status = "fourier auto fit failed: {}".format(exc)
            self.tracefit_info.set_text("failed: {}".format(exc))
            self.refresh()
            return
        beat, period_note, info = res["beat"], res["period_note"], res["info"]
        trace_fit, trace_err = res["trace_fit"], res["trace_err"]
        n_used, n_down, n_traces = res["n_used"], res["n_down"], res["n_traces"]
        resid_trace, avg_fit, avg_err = res["resid_trace"], res["avg_fit"], res["avg_err"]
        resid_avg, pol, sync, envelope = res["resid_avg"], res["pol"], res["sync"], res["envelope"]
        joint, scorer, scores = res["joint"], res["scorer"], res["scores"]
        chosen, energy, coeffs = res["chosen"], res["energy"], res["coeffs"]

        # The reference form is a unit-length beat scaled by the cilium
        # length, so the traced length sets L_c and the shape itself is
        # independent of it -- a length error no longer distorts the waveform.
        self._fourier_coeffs = coeffs
        self.fourier.set_coeffs(coeffs)
        self.fourier.length = self.model.length
        self._slots = fourier_slots(self.fourier.n_space)
        # the frame first, the cycle start in it
        frame_line = (self._turn_to_fan_axis()
                      if pattern_frame_setting(self.pqd) == "upright" else "")
        aligned = self._align_cycle_start()
        if frame_line:
            aligned = frame_line + "\n" + aligned
        sw = sweep_report(self.fourier)
        sweep_line = (
            "sweep: one stroke each way, no reversal within the strokes"
            if sw["reversals"] == 0 and sw["excess"] < 0.01 else
            "sweep: NOTE reverses {} times within the strokes (back-and-forth "
            "{:.0f}% of the range)".format(sw["reversals"], 100 * sw["excess"]))
        sweep_line += "; effective stroke: {} (setting on the cycle tab)".format(
            effective_stroke_setting(self.pqd))
        if pattern_frame_setting(self.pqd) != "upright":
            lean = fan_lean(self.fourier)
            sweep_line += ("\nlean: the fan's axis is {:.0f} deg {} of the "
                           "surface normal (tips {:+.0f}..{:+.0f} deg); "
                           "frame: surface (position tab)".format(
                               abs(lean["axis_deg"]),
                               "right" if lean["axis_deg"] > 0 else "left",
                               lean["lo_deg"], lean["hi_deg"]))
        x0 = self.pqd.getf("vid_cil_x0")
        here = float(np.interp(x0, beat["columns"], beat["corr"]))

        ranking = "; ".join("{} {:+.3f}".format(k, v[0]) for k, v in
                            sorted(scores.items(), key=lambda kv: -kv[1][0]))
        if res.get("kept"):
            what = ("previous fit polished" if chosen == "previous fit polished"
                    else "previous fit kept")
            chosen = ("{}: nothing new beat it by {:.2f} "
                      "(best new start {} {:+.3f})".format(
                          what, AUTOFIT_KEEP_MARGIN, res["best_new"],
                          scores[res["best_new"]][0])
                      if res.get("best_new") else what)
        if trace_fit is None:
            trace_line = "frame-by-frame traces unusable: {}".format(trace_err)
        else:
            trace_line = ("{} of {} traces used ({} down-weighted as outliers); "
                          "trace residual {:.3f} um".format(
                              n_used, n_traces, n_down, resid_trace))
        if avg_fit is None:
            trace_line += "\nphase-averaged traces unusable: {}".format(avg_err)
        else:
            trace_line += "; phase-averaged residual {:.3f} um".format(resid_avg)
        # A third of the traces thrown out, or a trace residual of half a
        # micron, means the frame-by-frame tracer could not keep to one
        # cilium -- the carpet is dense here.  The joint fit then follows
        # the texture of the bundle at this spot; the pattern is the
        # best-supported beat, not a measurement of one isolated cilium.
        dense_note = ""
        if trace_fit is not None and (n_down > 0.3 * max(n_used, 1)
                                      or resid_trace > 0.5):
            dense_note = ("\nNOTE dense carpet: the frame-by-frame traces hop "
                          "between neighbours here, so the beat is fitted to "
                          "the frames jointly; for a single cilium choose a "
                          "sparser spot or crop one out")
        pol_line = {-1.0: "dark cilia", 1.0: "bright cilia",
                    None: "polarity from the whole band"}.get(
            pol["sign"], "cilia dark or bright (skew {:+.2f}; by fifths {}), "
            "so lines of either sign count".format(
                pol["skew"], " ".join("{:+.2f}".format(k) for k in pol["skews"])))
        lawn_line = ("lawn in step (phase spread {:.2f} cycles): held to the "
                     "band's height, {:.1f}-{:.1f} um over the beat".format(
                         sync["spread"], float(np.min(envelope)),
                         float(np.max(envelope)))
                     if envelope is not None else
                     "not a lawn in step (phase spread {:.2f} cycles)".format(
                         sync["spread"]) if np.isfinite(sync["spread"])
                     else "lawn not judged")
        spread_line = ("beats within 0.03 of the best energy differ from it "
                       "by up to {:.2f} um".format(joint["spread_um"])
                       if joint["spread_um"] > 0 else
                       "no other start came within 0.03 of the best energy")
        # Which frames the beat was fitted to: the enhancement in force,
        # so that a pattern can be traced back to the settings behind it.
        enh_now = describe_enhance(enhance_from_pqd(self.pqd))
        self.tracefit_info.set_text(
            "beat fitted to the frames (energy {:+.3f}, {} frames at {} px/um, "
            "{}; {}; enhance: {})\n"
            "chosen: {};  ranking: {}\n"
            "{}\n"
            "{}; {}\n"
            "length left at {:.2f} um; the ridge fades at about {}\n"
            "{}\n"
            "{}\n"
            "beat repeats {:.2f} here, best {:.2f} at x={}{}{}\n"
            "{:.0f} s on {}\n"
            "check the overlay; Reload (r) to undo; Auto fit again polishes this "
            "beat, Fresh fit starts over".format(
                energy, joint["n_frames"], "{:.1f}".format(scorer.ppum),
                "frame - mean" if joint["mode"] == "dev" else "raw frames",
                pol_line, enh_now, chosen, ranking, trace_line, lawn_line,
                spread_line, self.model.length,
                "{:.1f} um".format(info["fade_um"]) if info.get("fade_um")
                else "-",
                aligned, sweep_line, here, beat["best_corr"], beat["best_column"],
                ("\n" + period_note) if period_note else "", dense_note,
                time.time() - t_start, describe_compute()))
        self.status = "fourier auto fit: {} (energy {:+.3f})".format(
            chosen, energy)
        self.refresh()

    def _align_cycle_start(self) -> str:
        """Move the main cycle's start to where the recovery stroke begins.

        Find cycle can only guess at this before anything has been fitted,
        and on a crowded recording its guess landed mid-way through the
        recovery.  After the fit the cilium's own motion is known: the end
        of its faster sweep is the end-effective position, and axwa's main
        cycle starts there.  The start frame moves, and the coefficients are
        rotated by the same phase so that what is drawn does not change --
        only which instant is called phase zero.  Returns a line for the
        readout.
        """
        # in the frame the pattern is drawn and exported in: the rightmost
        # state of the upright frame is not quite the surface frame's
        try:
            marks = recovery_start_phase(self.fourier,
                                         prefer=effective_direction(self.pqd))
        except ValueError as exc:
            return "cycle start left alone: {}".format(exc)
        shift = marks["phase_end_effective"]
        by = _decided_by_text(marks)
        if marks.get("after_fast", 0.0) >= 0.03:
            by += ("; phase 0 is the extreme, which the cilium drifts on to "
                   "for {:.0f}% of the beat after its fast sweep".format(
                       100 * marks["after_fast"]))
        period = self.model.period_frames
        if period <= 0:
            return "cycle start left alone: no period"
        old_start = self.model.macysta
        # Any beat is as good as any other, so of the equivalent starts take
        # the one nearest the old start -- moving 15 to 48 when 14.5 was the
        # same phase is churn -- preferring those that keep the whole main
        # cycle inside the record.
        options = [old_start + (shift + k) * period for k in range(-3, 4)]
        inside = [o for o in options
                  if self.video.first <= o <= self.video.last - period]
        start = min(inside or options, key=lambda o: abs(o - old_start))
        if abs(start - old_start) < 0.5:
            return ("cycle start {:.1f} already at the recovery stroke "
                    "(effective sweeps {}, fast for {:.0f}% of the beat, by {})"
                    .format(old_start,
                            "right" if marks["direction"] > 0 else "left",
                            100 * marks["duration_effective"], by))
        coeffs = shift_fourier_phase(self.fourier.coeffs, shift)
        self._fourier_coeffs = coeffs
        self.fourier.set_coeffs(coeffs)
        self.pqd.set("vid_frame_main_cycle_start", float(round(start, 2)))
        self.model.rebuild()
        self._sync_cycle_span()
        return ("cycle start moved {:g} -> {:.1f}: phase 0 is now the end of "
                "the effective stroke, the {} state (sweeps {}, fast for "
                "{:.0f}% of the beat, by {})".format(
                    old_start, start,
                    "rightmost" if marks["direction"] > 0 else "leftmost",
                    "right" if marks["direction"] > 0 else "left",
                    100 * marks["duration_effective"], by))

    def _fit_fourier_from_file(self, path: str) -> None:
        """Fit the beat from a traced file, the way the reference script does."""
        self.tracefit_info.set_text("fitting to {}...".format(
            os.path.basename(path)))
        self.fig.canvas.draw_idle()
        try:
            with self._busy("fitting the traced beat") as report:
                report(0.2, "reading the trace")
                _, curves, units = read_any_trace(path)
                aligned, theta = align_beat_frame(curves)
                # The transform has to run over exactly one beat, or the
                # harmonics are fitted to a record that does not repeat.
                tips = resample_unit_length(aligned, 20)[0][:, -1]
                per = int(round(0.5 * (autocorr_period(tips[:, 0])
                                       + autocorr_period(tips[:, 1]))))
                report(0.6, "fitting the beat")
                fit = fit_beat_reference(aligned, n_modes=FOURIER_DEFAULT_MODES,
                                         num_locs=20, n_space=N_FOURIER_M,
                                         period_samples=per)
                report(1.0, "done")
        except Exception as exc:
            self.status = "trace fit failed: {}".format(exc)
            self.tracefit_info.set_text("failed: {}".format(exc))
            self.refresh()
            return

        scale = self.model.ppum if units == "pixels" else 1.0
        length = fit["mean_length"] / max(scale, 1e-9)
        self._fourier_coeffs = fit["coeffs"]
        self.fourier.set_coeffs(fit["coeffs"])
        self.pqd.set("length", float(length))
        self.model.rebuild()
        self.fourier.length = self.model.length
        self._slots = fourier_slots(self.fourier.n_space)

        wave = FourierWave(fit["coeffs"], length=1.0, cbf=1.0)
        s = np.linspace(0.0, 1.0, 80)
        turn = [abs(curve_turning(np.column_stack(wave.position(s, i / 24.0))))
                for i in range(24)]
        self.tracefit_info.set_text(
            "fitted to {} ({})\n"
            "{} traced times, {} time modes, s^1..s^{}\n"
            "one beat spans {} of them\n"
            "length {:.2f} um from the traces\n"
            "residual {:.3f} of a cilium length\n"
            "bending {:.0f} deg median, {:.0f} at its most bent\n"
            "Reload (r) to undo".format(
                os.path.basename(path), units, len(curves), fit["n_modes"],
                N_FOURIER_M, fit["per_period"], length, fit["resid"],
                np.median(turn), max(turn)))
        self.status = "fitted the traced beat: residual {:.3f} of a length" \
            .format(fit["resid"])
        self.refresh()

    def _targets_in_model_frame(self, curves):
        """Traced pixel curves expressed in the model's own microns."""
        return curves_to_model_frame(curves, self.model)

    def _current_beat(self):
        """The beat on the params tab when it is a fitted or loaded one --
        not the built-in template a fresh file starts from -- for Auto fit
        to polish rather than replace."""
        try:
            coeffs = self.fourier.coeffs
            for key in FOURIER_KEYS:
                a = np.asarray(coeffs[key], dtype=float)
                d = np.asarray(FOURIER_DEFAULT[key], dtype=float)
                if a.shape != d.shape or not np.allclose(a, d, atol=1e-9):
                    return {k: np.asarray(coeffs[k], dtype=float).copy()
                            for k in FOURIER_KEYS}
        except Exception:
            return None
        return None

    def _on_tracefit(self, _event=None, fresh: bool = False) -> None:
        """Auto fit: the beat, from the frames (Fresh fit: from scratch)."""
        if self.no_images:
            self.status = "import images first"
            self.tracefit_info.set_text(
                "no images yet -- press 'Import images...' on the video tab")
            self.refresh()
            return
        self._fit_fourier_to_image(fresh=fresh)

    def _on_find_length(self, _event=None) -> None:
        if self.no_images:
            self.status = "import images first to measure the length"
            self.refresh()
            return
        self.length_info.set_text("searching...")
        self.fig.canvas.draw_idle()
        try:
            with self._busy("measuring the length") as report:
                res = estimate_length(self.video, self.model, progress=report)
        except Exception as exc:
            self.status = "could not estimate the length: {}".format(exc)
            self.length_info.set_text("failed: {}".format(exc))
            self.refresh()
            return

        old = self.model.length
        nodes = self.model.n_xy
        self.pqd.set("length", res["length"])
        self.model.rebuild()
        self.set_nodes(nodes)          # keep the node count, as elsewhere

        gap = res["base_above_surface_um"]
        warn = ""
        if abs(gap) > 0.5:
            warn = ("\nWARNING base sits {:.1f} um {} the cell surface;\n"
                    "this measures outwards from the base, so fix it first"
                    .format(abs(gap), "above" if gap > 0 else "below"))
        self.length_info.set_text(
            "{:.2f} um  (was {:.2f});  plausible range {:.1f} .. {:.1f}\n"
            "measured from the base at row {:.0f}, surface at {:.0f}{}".format(
                res["length"], old, res["low"], res["high"],
                res["base_row"], res["surface_row"], warn))
        self.status = "length -> {:.2f} um".format(res["length"])
        self.refresh()

    def _arm_tip_click(self, _event=None) -> None:
        self._await_tip = True
        self.length_info.set_text(
            "click the cilium tip on the image\n"
            "(step to the most extended frame first)")
        self.status = "waiting for a click on the cilium tip"
        self.refresh()

    def _tip_click(self, event) -> bool:
        """Set the length from a click on the image.  True if consumed."""
        if not getattr(self, "_await_tip", False):
            return False
        if event.inaxes is not self.ax_img or event.xdata is None:
            return False
        self._await_tip = False

        phase = self.model.phase_of_frame(self.frame)
        keep = self.model.length
        # The beat's shape scales with the length, so the length that puts
        # the tip as far from the base as the click is the current one
        # scaled by that ratio of distances.
        x, y = self.model.shape_at(phase)
        px, py = self.model.to_pixels(x, y, image_h=self.img_h)
        r_tip = float(np.hypot(px[-1] - px[0], py[-1] - py[0]))
        r_click = float(np.hypot(event.xdata - px[0], event.ydata - py[0]))
        ppum = self.model.ppum or 1.0
        if r_tip <= 0 or r_click / ppum < self.model.dl * 2:
            self.length_info.set_text("that click is at the base -- try again")
            self.status = "click was too close to the base"
            self.refresh()
            return True
        new_len = float(keep * r_click / r_tip)
        # how far the click sits from where the tip then is
        f = r_click / r_tip
        off = float(np.hypot(px[0] + (px[-1] - px[0]) * f - event.xdata,
                             py[0] + (py[-1] - py[0]) * f - event.ydata) / ppum)

        self.pqd.set("length", new_len)
        self.model.rebuild()
        self.length_info.set_text(
            "{:.2f} um  (was {:.2f})\n"
            "the length that brings the tip as far out as the click;\n"
            "the click was {:.2f} um off the tip's direction".format(
                new_len, keep, off))
        self.status = "length -> {:.2f} um from the clicked tip".format(new_len)
        self.refresh()
        return True

    def _on_find_cycle(self, _event=None) -> None:
        """Estimate the beat period and where the main cycle starts."""
        if self.no_images:
            self.status = "import images first to find the cycle"
            self.refresh()
            return
        self.cycle_found.set_text("searching...")
        self.fig.canvas.draw_idle()
        try:
            with self._busy("finding the cycle") as report:
                res = detect_cycle(self.video, model=self.model,
                                   progress=report)
        except Exception as exc:
            self.status = "could not find the cycle: {}".format(exc)
            self.cycle_found.set_text("failed: {}".format(exc))
            self.refresh()
            return

        old_start = self.model.macysta
        old_cbf = self.model.cbf
        period = res["period"]

        # Snap to the detected landmark nearest the current start, so this
        # refines the choice rather than jumping to a different beat.
        cands = res["candidates"] or [int(round(old_start))]
        start = min(cands, key=lambda f: abs(f - old_start))
        # A fitted beat places the start better than the traced tip (on a
        # dense lawn the traces hop, and on CO121 they put the start a
        # third of a beat into the recovery): with one on the params tab
        # the period is taken from here and the start from the beat.
        fitted = self._current_beat() is not None
        if fitted:
            start = float(old_start)

        fps = self.model.fps
        self.pqd.set("vid_frame_main_cycle_start", float(start))
        if fps > 0 and period > 0:
            self.pqd.set("cbf", fps / period)
        self.model.rebuild()
        self._sync_cycle_span()

        self._cycle_candidates = cands
        if fitted:
            where = ("start from the fitted beat on the params tab: "
                     + self._align_cycle_start())
            start = self.model.macysta
            flip = ("the traced tip would have put it at frame {}; Fresh fit "
                    "on the params tab starts the beat over".format(
                        min(cands, key=lambda f: abs(f - start))))
        elif res.get("source") == "traced tip":
            where = ("effective stroke sweeps {}; fast for {:.0f}% of the "
                     "period; decided by {} (x{:.1f})".format(
                         "right" if res["direction"] > 0 else "left",
                         100 * res["duration_effective"], res["decided_by"],
                         res["speed_ratio"]))
            if res["decided_by"] == "setting":
                where = ("effective stroke sweeps {}; fast for {:.0f}% of the "
                         "period; decided by {}".format(
                             "right" if res["direction"] > 0 else "left",
                             100 * res["duration_effective"],
                             _decided_by_text(res)))
            if res.get("after_fast", 0.0) >= 0.03:
                where += ("; the cycle starts at the extreme, which the "
                          "cilium drifts on to for {:.0f}% of the beat after "
                          "its fast sweep".format(100 * res["after_fast"]))
            other = res.get("other_candidates") or []
            flip = ("if the strokes are the other way round, the start "
                    "is at frame {}".format(
                        min(other, key=lambda f: abs(f - start))
                        if other else "?"))
        else:
            where = ("from the carpet centroid ({}); Auto fit will move the "
                     "start to the recovery stroke".format(
                         "traces unusable: " + res["trace_error"]
                         if res.get("trace_error") else "locate the base first"))
            flip = "other end-effective frames: {}".format(
                ", ".join(str(c) for c in cands[:6]) or "none")
        source = res.get("period_source", "whole field")
        pg = res.get("period_global", period)
        if source == "around the base" and abs(pg - period) > 0.03 * period:
            source += " (whole field says {:.1f})".format(pg)
        self.cycle_found.set_text(
            "period {:.2f} fr over {:.1f} cycles -> cbf {:.3f} Hz "
            "(was {:.3f}); measured {}\n"
            "cycle {} .. {:.1f}  recovery -> effective  (start was {:g})\n"
            "{}\n{}".format(
                period, res["n_cycles"], self.model.cbf, old_cbf, source,
                start, start + period, old_start, where, flip))
        self.status = "cycle {} .. {:.1f}, cbf {:.3f} Hz".format(
            start, start + period, self.model.cbf)
        self.refresh()

    # ------------------------------------------- MCC: particle tracking

    def _build_mcc_tab(self) -> None:
        """Particle tracking.  Calibration and frames come from the video tab.

        There is deliberately no folder, um/px or fps control here: the
        sequence, its scale and its frame rate are already set up on the video
        tab, and duplicating them would let the two disagree.
        """
        self.mcc = {"seed": None, "traj": None, "vel": None}
        self._await_mcc_pick = False

        self.sl_mcc_frame = self._panel_slider(
            "MCC", 0, "frame", self.video.first,
            max(self.video.last, self.video.first + 1), self.frame,
            valfmt="%d", valstep=1)
        self.sl_mcc_frame.on_changed(self._mcc_frame_cb)

        specs = [("Pick particle", 0.140, 0.105, self._arm_mcc_pick, "#ffe9b8"),
                 ("Track", 0.257, 0.085, self._mcc_track, "#e0eaff"),
                 ("Export", 0.352, 0.085, self._mcc_export, "#d8f0d8")]
        self._mcc_buttons = []
        for label, x, w, cb, colour in specs:
            ax = self._panel_box("MCC", x, 0.196, w, 0.030)
            btn = Button(ax, label, color=colour, hovercolor="#c4d9ff")
            btn.label.set_fontsize(8)
            btn.on_clicked(cb)
            self._mcc_buttons.append(btn)

        ax = self._panel_box("MCC", 0.105, 0.182, 0.395, 0.001)
        ax.axis("off")
        self.mcc_info = ax.text(
            0.0, 0.0, "", va="top", ha="left", fontsize=8,
            family="monospace", color="#333333", clip_on=False)

        (self.line_mcc,) = self.ax_img.plot(
            [], [], "-", color="#00ccff", linewidth=1.2, zorder=7)
        (self.line_mcc_seed,) = self.ax_img.plot(
            [], [], marker="+", color="#ff3355", markersize=11, mew=2,
            linestyle="none", zorder=8)
        (self.line_mcc_now,) = self.ax_img.plot(
            [], [], marker="o", color="#ffee00", markersize=7,
            markerfacecolor="none", linestyle="none", zorder=8)
        self._mcc_message("press 'Pick particle', then click it on the image")

    @property
    def mcc_pixel_size(self) -> float:
        """Microns per pixel, from the video tab's ppum (pixels per micron)."""
        ppum = self.model.ppum
        return (1.0 / ppum) if ppum > 0 else 1.0

    @property
    def mcc_fps(self) -> float:
        return self.model.fps

    def _mcc_message(self, text: str) -> None:
        self.mcc_info.set_text(text)

    def _mcc_frame_cb(self, val):
        if self._updating:
            return
        self.frame = int(np.clip(int(val), self.video.first, self.video.last))
        self.refresh()

    def _arm_mcc_pick(self, _event=None) -> None:
        if self.no_images:
            self.status = "import images first to pick a particle"
            self.refresh()
            return
        self._await_mcc_pick = True
        self.status = "click the particle on the image"
        self._mcc_message("click the particle on the image")
        self.refresh()

    def _mcc_click(self, event) -> bool:
        """Set the seed from a click.  True if the click was consumed."""
        if not getattr(self, "_await_mcc_pick", False):
            return False
        if event.inaxes is not self.ax_img or event.xdata is None:
            return False
        self._await_mcc_pick = False

        seed = (float(event.xdata), float(event.ydata))
        self.mcc["seed"] = seed
        self.mcc["traj"] = None
        self.mcc["vel"] = None
        self.pqd.set("mcc_seed_x", seed[0])
        self.pqd.set("mcc_seed_y", seed[1])
        self.status = "seed at ({:.1f}, {:.1f}) -- press Track".format(*seed)
        self._mcc_message(
            "seed ({:.1f}, {:.1f})\npress Track to follow it".format(*seed))
        self.refresh()
        return True

    def _mcc_track(self, _event=None) -> None:
        if self.no_images:
            self.status = "import images first to track"
            self.refresh()
            return
        if self.mcc["seed"] is None:
            self.status = "pick a particle first"
            self.refresh()
            return

        self._mcc_message("tracking...")
        self.fig.canvas.draw_idle()
        try:
            with self._busy("tracking the particle") as report:
                traj = track_particle(self.video, self.mcc["seed"],
                                      max_disp=MCC_MAX_DISP_PX, progress=report)
            self.mcc["traj"] = traj
            self._mcc_recompute()
        except Exception as exc:
            self.status = "tracking failed: {}".format(exc)
            self._mcc_message("tracking failed: {}".format(exc))
            self.refresh()
            return

        vel = self.mcc["vel"]
        self.status = "mean speed {:.1f} um/s over {} frames".format(
            vel["mean_speed_umps"], vel["n_tracked"])
        self.refresh()

    def _mcc_recompute(self) -> None:
        """Velocities from the existing track, at the video tab's calibration."""
        vel = particle_velocity(self.mcc["traj"], self.mcc_pixel_size,
                                self.mcc_fps)
        self.mcc["vel"] = vel
        lost = vel["n_frames"] - vel["n_tracked"]
        self._mcc_message(
            "{:.4f} um/px (ppum {:g})   {:g} fps   max jump {:g} px\n"
            "tracked {}/{} frames (frame {}..{}){}\n"
            "path {:.2f} um in {:.1f} ms;  net {:.2f} um\n"
            "average speed  {:.1f} um/s   (path / time)\n"
            "average velocity {:.1f} um/s   (net / time)\n"
            "instantaneous  {:.1f} +/- {:.1f}   range {:.1f}..{:.1f} um/s".format(
                self.mcc_pixel_size, self.model.ppum, self.mcc_fps,
                MCC_MAX_DISP_PX,
                vel["n_tracked"], vel["n_frames"], vel["first_frame"],
                vel["last_frame"],
                "  (lost {} -- often it left the frame)".format(lost) if lost else "",
                vel["total_path_um"], vel["total_time_s"] * 1000,
                vel["net_disp_um"], vel["mean_speed_umps"],
                vel["net_velocity_umps"], vel["mean_inst_umps"],
                vel["std_inst_umps"], vel["min_inst_umps"], vel["max_inst_umps"]))

    def _output_dir(self):
        """Where exports go: beside the specification, or beside the images."""
        anchor = self.pqd_path or (self.images_dir or None)
        if not anchor:
            return None
        base = os.path.dirname(os.path.abspath(anchor)) if self.pqd_path \
            else os.path.abspath(anchor)
        return os.path.join(base, "output")

    def _mcc_export(self, _event=None) -> None:
        vel = self.mcc.get("vel")
        if vel is None:
            self.status = "nothing to export -- press Track first"
            self.refresh()
            return
        outdir = self._output_dir()
        if outdir is None:
            self.status = "import images first"
            self.refresh()
            return
        try:
            os.makedirs(outdir, exist_ok=True)
            csv = save_particle_csv(os.path.join(outdir, "particle_trajectory.csv"),
                                    vel)
            png = self._mcc_plot(os.path.join(outdir, "particle_velocity.png"), vel)
        except Exception as exc:
            self.status = "export failed: {}".format(exc)
            self.refresh()
            return
        for p in (csv, png):
            print("wrote {}".format(p))
        self.status = "exported {} and {}".format(
            os.path.basename(csv), os.path.basename(png))
        self.refresh()

    def _mcc_plot(self, path: str, vel: dict) -> str:
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
        sc = axs[0].scatter(vel["x_um"], vel["y_um"], c=vel["time_s"] * 1000,
                            cmap="viridis", s=15)
        axs[0].plot(vel["x_um"], vel["y_um"], "-", color="gray", alpha=0.4, lw=1)
        axs[0].invert_yaxis()
        axs[0].set_xlabel("x (um)")
        axs[0].set_ylabel("y (um)")
        axs[0].set_title("Particle trajectory")
        axs[0].set_aspect("equal")
        fig.colorbar(sc, ax=axs[0], label="time (ms)")

        axs[1].plot(vel["time_s"][1:] * 1000, vel["inst_umps"], "-o", ms=3, lw=1)
        axs[1].axhline(vel["mean_inst_umps"], color="red", linestyle="--",
                       label="mean = {:.1f} um/s".format(vel["mean_inst_umps"]))
        axs[1].set_xlabel("time (ms)")
        axs[1].set_ylabel("instantaneous speed (um/s)")
        axs[1].set_title("Speed vs time")
        axs[1].legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    # -------------------------------------------------- contrast (display)

    # ---------------------------------------------------------- enhance

    def _build_enhance_tab(self) -> None:
        """Frame enhancement that feeds the measurements.

        Everything here changes the frames every step reads -- the
        surface, the base, the period, the traces, the fit -- and the
        overlay is drawn on the enhanced frame, so what is seen is what was
        measured.  The contrast settings on the video tab, by contrast,
        change the picture only.
        """
        enh = enhance_from_pqd(self.pqd)
        ax = self._panel_box("enhance", 0.105, 0.112, 0.140, 0.124)
        ax.set_facecolor("none")
        for sp in ax.spines.values():
            sp.set_visible(False)
        self._enh_checks = ["background", "bgmedian", "flatten", "invert",
                            "deflicker", "deblock"]
        self.chk_enhance = CheckButtons(
            ax, ["Background removal", "  as median (Z Project)",
                 "Flatten illumination", "Invert", "Deflicker", "JPEG deblock"],
            [bool(enh["background"]), bool(enh["bgmedian"]), enh["flatten"] > 0,
             bool(enh["invert"]), bool(enh["deflicker"]), bool(enh["deblock"])])
        for lb in self.chk_enhance.labels:
            lb.set_fontsize(7.5)
        self.chk_enhance.on_clicked(lambda _l: self._apply_enhance())

        col1, col2, wb = 0.322, 0.490, 0.040
        self.tb_enh_denoise = self._spin_textbox(
            "enhance", col1, 0.206, wb, "denoise sigma px ",
            "{:g}".format(enh["denoise"]), 0.5)
        self.tb_enh_line = self._spin_textbox(
            "enhance", col1, 0.178, wb, "line denoise px ",
            "{:g}".format(enh["linesmooth"]), 0.5)
        self.tb_enh_deblur = self._spin_textbox(
            "enhance", col1, 0.150, wb, "deblur sigma px ",
            "{:g}".format(enh["deblur"]), 0.25)
        self.tb_enh_sharpen = self._spin_textbox(
            "enhance", col1, 0.122, wb, "sharpen ",
            "{:g}".format(enh["sharpen"]), 0.5)
        self.tb_enh_flatten = self._spin_textbox(
            "enhance", col2, 0.206, wb, "flatten window px ",
            "{:g}".format(enh["flatten"] or 40), 10, fmt="{:.0f}")
        self.tb_enh_contrast = self._spin_textbox(
            "enhance", col2, 0.178, wb, "contrast limit ",
            "{:g}".format(enh["contrast"]), 0.5)
        self.tb_enh_up = self._spin_textbox(
            "enhance", col2, 0.150, wb, "upsample x ",
            "{:d}".format(enh["upsample"]), 1, fmt="{:.0f}")
        for tb in (self.tb_enh_denoise, self.tb_enh_line, self.tb_enh_deblur,
                   self.tb_enh_sharpen, self.tb_enh_flatten,
                   self.tb_enh_contrast, self.tb_enh_up):
            tb.on_submit(lambda _t: self._apply_enhance())
        ax = self._panel_box("enhance", col2 - 0.068, 0.122, 0.068 + wb + 0.015,
                             0.024)
        self._enh_upmethod = int(enh["upmethod"])
        self.btn_enh_upmethod = Button(ax, self._upmethod_label(),
                                       color="#ededed", hovercolor="#d6d6d6")
        self.btn_enh_upmethod.label.set_fontsize(7.5)
        self.btn_enh_upmethod.on_clicked(self._toggle_upmethod)

        # Fiji's tools for thin structures, one row: the line filter
        # (Tubeness / Frangi), the top-hat, the FFT bandpass (its small
        # cutoff is the vid_enh_bp_small setting, 2 px).
        y3, h3 = 0.090, 0.022
        self.tb_enh_lines = self._spin_textbox(
            "enhance", 0.190, y3, wb, "line filter px ",
            "{:g}".format(enh["lines"]), 0.5, h=h3)
        self.tb_enh_tophat = self._spin_textbox(
            "enhance", col1, y3, wb, "top-hat px ",
            "{:g}".format(enh["tophat"]), 0.5, h=h3)
        self.tb_enh_bandpass = self._spin_textbox(
            "enhance", col2, y3, wb, "bandpass px ",
            "{:g}".format(enh["bandpass"]), 5, h=h3)
        for tb in (self.tb_enh_lines, self.tb_enh_tophat, self.tb_enh_bandpass):
            try:
                tb.label.set_fontsize(8)
                tb.text_disp.set_fontsize(8)
            except Exception:
                pass
            tb.on_submit(lambda _t: self._apply_enhance())

        y4 = 0.062
        ax = self._panel_box("enhance", 0.118, y4, 0.070, 0.024)
        self.btn_enh_reset = Button(ax, "Reset", color="#ededed",
                                    hovercolor="#d6d6d6")
        self.btn_enh_reset.label.set_fontsize(8)
        self.btn_enh_reset.on_clicked(self._reset_enhance)
        ax = self._panel_box("enhance", 0.196, y4, 0.090, 0.024)
        self.btn_enh_auto = Button(ax, "Auto", color="#e0eaff",
                                   hovercolor="#c4d9ff")
        self.btn_enh_auto.label.set_fontsize(8)
        self.btn_enh_auto.on_clicked(self._on_auto_enhance)

        # A neural network from a file (see OnnxEnhancer): its path, a
        # chooser, and the pixel range it expects.
        self.tb_enh_model = self._panel_textbox(
            "enhance", 0.345, y4, 0.110, "AI model ", enh["model"], h=0.024)
        try:
            self.tb_enh_model.text_disp.set_fontsize(7)
        except Exception:
            pass
        self.tb_enh_model.on_submit(lambda _t: self._apply_enhance())
        ax = self._panel_box("enhance", 0.459, y4, 0.046, 0.024)
        self.btn_enh_model = Button(ax, "Browse", color="#ededed",
                                    hovercolor="#d6d6d6")
        self.btn_enh_model.label.set_fontsize(7.5)
        self.btn_enh_model.on_clicked(self._browse_model)
        self._enh_modelrange = int(enh["modelrange"])
        ax = self._panel_box("enhance", 0.509, y4, 0.046, 0.024)
        self.btn_enh_range = Button(ax, self._range_label(), color="#ededed",
                                    hovercolor="#d6d6d6")
        self.btn_enh_range.label.set_fontsize(7)
        self.btn_enh_range.on_clicked(self._toggle_range)

        ax = self._panel_box("enhance", 0.105, 0.004, 0.450, 0.054)
        ax.axis("off")
        self.enhance_info = ax.text(
            0.0, 1.0, "", va="top", ha="left", fontsize=7.0,
            family="monospace", color="#333333", clip_on=False)
        self._describe_enhance()

    def _upmethod_label(self) -> str:
        return "upsampling: {}".format(
            "Lanczos" if self._enh_upmethod else "bilinear")

    def _toggle_upmethod(self, _event=None) -> None:
        self._enh_upmethod = 1 - int(self._enh_upmethod)
        self.btn_enh_upmethod.label.set_text(self._upmethod_label())
        self._apply_enhance()

    def _range_label(self) -> str:
        return "range " + MODEL_RANGES[int(self._enh_modelrange)]

    def _toggle_range(self, _event=None) -> None:
        self._enh_modelrange = (int(self._enh_modelrange) + 1) % len(MODEL_RANGES)
        self.btn_enh_range.label.set_text(self._range_label())
        if clean_model_path(self.tb_enh_model.text):
            self._apply_enhance()
        else:
            self.refresh()

    def _browse_model(self, _event=None) -> None:
        start = os.path.dirname(resolve_model_path(self.tb_enh_model.text,
                                                   self.images_dir)) \
            if clean_model_path(self.tb_enh_model.text) else None
        chosen = choose_file("Select an ONNX model", initial=start)
        if chosen is None:
            self.status = ("no file chooser here: type the model's path into "
                           "the AI model box")
            self.refresh()
            return
        self._updating = True
        try:
            self.tb_enh_model.set_val(chosen)
        finally:
            self._updating = False
        self._apply_enhance()

    def _read_enhance_widgets(self) -> dict:
        st = self.chk_enhance.get_status()
        on = dict(zip(self._enh_checks, st))
        settings = enhance_defaults()
        settings.update({"background": int(on["background"]),
                         "bgmedian": int(on["bgmedian"]),
                         "invert": int(on["invert"]),
                         "deflicker": int(on["deflicker"]),
                         "deblock": int(on["deblock"])})

        def num(tb, default):
            try:
                return float(str(tb.text).strip())
            except (TypeError, ValueError):
                return default

        window = num(self.tb_enh_flatten, 40.0)
        settings["flatten"] = int(max(3, window)) if on["flatten"] else 0
        settings["denoise"] = max(0.0, num(self.tb_enh_denoise, 0.0))
        settings["linesmooth"] = float(np.clip(num(self.tb_enh_line, 0.0), 0.0, 8.0))
        settings["deblur"] = float(np.clip(num(self.tb_enh_deblur, 0.0), 0.0, 8.0))
        settings["sharpen"] = max(0.0, num(self.tb_enh_sharpen, 0.0))
        settings["contrast"] = float(np.clip(num(self.tb_enh_contrast, 0.0), 0.0, 20.0))
        settings["upsample"] = int(np.clip(round(num(self.tb_enh_up, 1)), 1, 4))
        settings["upmethod"] = int(self._enh_upmethod)
        settings["model"] = clean_model_path(self.tb_enh_model.text)
        settings["modelrange"] = int(self._enh_modelrange)
        settings["lines"] = float(np.clip(num(self.tb_enh_lines, 0.0), 0.0, 12.0))
        settings["tophat"] = float(np.clip(num(self.tb_enh_tophat, 0.0), 0.0, 12.0))
        settings["bandpass"] = float(np.clip(num(self.tb_enh_bandpass, 0.0), 0.0, 400.0))
        settings["bpsmall"] = float(self.pqd.getf("vid_enh_bp_small", 2.0))
        return settings

    def _set_enhance_widgets(self, settings: dict) -> None:
        """Show ``settings`` in the controls without applying them."""
        full = enhance_defaults()
        full.update(settings)
        self._updating = True
        try:
            st = list(self.chk_enhance.get_status())
            want = [bool(full["background"]), bool(full["bgmedian"]),
                    full["flatten"] > 0, bool(full["invert"]),
                    bool(full["deflicker"]), bool(full["deblock"])]
            for i in range(len(want)):
                if st[i] != want[i]:
                    self.chk_enhance.set_active(i)
            self.tb_enh_lines.set_val("{:g}".format(full["lines"]))
            self.tb_enh_tophat.set_val("{:g}".format(full["tophat"]))
            self.tb_enh_bandpass.set_val("{:g}".format(full["bandpass"]))
            self.tb_enh_denoise.set_val("{:g}".format(full["denoise"]))
            self.tb_enh_line.set_val("{:g}".format(full["linesmooth"]))
            self.tb_enh_deblur.set_val("{:g}".format(full["deblur"]))
            self.tb_enh_sharpen.set_val("{:g}".format(full["sharpen"]))
            self.tb_enh_contrast.set_val("{:g}".format(full["contrast"]))
            if full["flatten"]:
                self.tb_enh_flatten.set_val("{:d}".format(int(full["flatten"])))
            self.tb_enh_up.set_val("{:d}".format(int(full["upsample"])))
            self._enh_upmethod = int(full["upmethod"])
            self.btn_enh_upmethod.label.set_text(self._upmethod_label())
            self.tb_enh_model.set_val(clean_model_path(full.get("model")))
            self._enh_modelrange = int(full.get("modelrange") or 0)
            self.btn_enh_range.label.set_text(self._range_label())
        finally:
            self._updating = False

    def _describe_enhance(self) -> None:
        enh = enhance_from_pqd(self.pqd)
        now = describe_enhance(enh)
        note = ""
        if clean_model_path(enh.get("model")):
            try:
                note = self.video.model_note()
            except Exception as exc:
                note = "NOT USED: {}".format(exc)
        self.enhance_info.set_text(
            "these change the frames every measurement reads (the video tab's\n"
            "contrast changes the picture only); saved in the specification.\n"
            "now: {}{}".format("nothing applied" if now == "none" else now,
                               "\nmodel: " + note if note else ""))

    def _apply_enhance(self, settings=None) -> None:
        if self._updating:
            return
        if self.no_images:
            self.status = "import images first"
            self.refresh()
            return
        full = enhance_defaults()
        full.update(settings or self._read_enhance_widgets())
        settings = full
        old_up = self.model.upsample
        enhance_to_pqd(self.pqd, settings)
        # The base is a pixel position: scale it with the frame.
        ratio = float(settings["upsample"]) / float(old_up)
        if ratio != 1.0:
            self.pqd.set("vid_cil_x0", self.pqd.getf("vid_cil_x0") * ratio)
            self.pqd.set("vid_cil_y0_invy",
                         self.pqd.getf("vid_cil_y0_invy") * ratio)
        if not self._reload_video():
            return
        self.model.rebuild()
        self._describe_enhance()
        self.status = "frames " + (
            "unchanged" if enhance_is_default(settings) else "enhanced") + \
            " for the measurements; re-run Find base / Auto fit"
        if clean_model_path(settings.get("model")):
            note = self.video.model_note()
            if note.startswith("NOT USED"):
                self.status = "AI model {}".format(note)
            else:
                self.status = "AI model in use: {}; re-run Find base / Auto fit".format(note)
        self.refresh()

    def _on_auto_enhance(self, _event=None) -> None:
        """Set the enhancement from the frames; nothing else changes.

        The base, the cycle and the beat are this tab's neighbours' business
        and are left exactly as they are.  When they have been set, they
        serve to *judge* the candidate settings (auto_enhance: traces and
        beats fitted on copies, in the background); when they have not,
        the settings are derived from the frames alone (measure_recording,
        recommend_enhance).  Either way the controls are set and the frames
        enhanced, and the status says what to do next.
        """
        if self.no_images:
            self.status = "import images first"
            self.refresh()
            return
        bx, by = self.pqd.getf("vid_cil_x0"), self.pqd.getf("vid_cil_y0_invy")
        centre = (abs(bx - self.img_w / 2.0) < 1.0
                  and abs(by - self.img_h / 2.0) < 1.0)
        outside = not (0 <= bx < self.img_w and 0 <= by < self.img_h)
        have_base = not (centre or outside)
        have_cycle = abs(self.pqd.getf("cbf", 15.0) - 15.0) > 1e-9
        if not (have_base and have_cycle):
            self._auto_enhance_from_frames(have_base, have_cycle)
            return
        self.enhance_info.set_text("trying the settings on the tracker...")
        self.fig.canvas.draw_idle()
        try:
            with self._busy("auto enhance") as report:
                res = auto_enhance(self.video, self.pqd, progress=report)
        except Exception as exc:
            self.status = "auto enhance failed: {}".format(exc)
            self.enhance_info.set_text("failed: {}".format(exc))
            self.refresh()
            return
        best = res["best"]
        self._set_enhance_widgets(best["settings"])
        self._apply_enhance(best["settings"])
        self._stretch_view_for(best["settings"])
        lines = auto_enhance_table(res)
        print("\n".join(lines))
        shown = lines[:4]
        if len(lines) > 4:
            shown.append(" ... {} more in the terminal".format(len(lines) - 4))
        self.enhance_info.set_text("\n".join(shown))
        again = ("now re-run Find cycle (the flicker correction changes it), "
                 "then Auto fit on the params tab"
                 if best["settings"].get("deflicker")
                 else "now Auto fit on the params tab")
        pat = res.get("pattern") or {}
        verdict = pat.get("verdict")
        if verdict is not None:
            self.status = ("auto enhance: {} -- the setting whose pattern agrees "
                           "best with the others' (they differ by {:.2f} um); {}"
                           .format(best["label"], verdict["spread_um"], again))
        elif res.get("unjudged"):
            self.status = ("auto enhance: the tracer cannot judge settings on "
                           "this recording (best {:.2f} um); frames left as "
                           "recorded; {}".format(res["leader"]["score_um"], again))
        else:
            self.status = "auto enhance: {} (traces {:.3f} um from the beat); {}".format(
                best["label"], best["score_um"], again)
        self.refresh()

    def _stretch_view_for(self, settings: dict) -> None:
        """Background removal, flattening and the bandpass leave the frames
        as small deviations around mid-grey -- right for the measurements,
        but a flat grey picture to look at.  Tick Bytescaling on the video
        tab (display only; no measurement reads it) so the cilia show."""
        flat = (settings.get("background") or settings.get("flatten")
                or settings.get("bandpass"))
        if not flat or self.contrast.get("bytescale"):
            return
        self.contrast["bytescale"] = True
        try:
            self._updating = True
            if not self.chk_bytescale.get_status()[0]:
                self.chk_bytescale.set_active(0)
        except Exception:
            pass
        finally:
            self._updating = False

    def _auto_enhance_from_frames(self, have_base: bool, have_cycle: bool) -> None:
        """Auto without a base or a cycle to judge by: the settings the
        measurements call for, across the whole band above the tissue."""
        self.enhance_info.set_text("measuring the frames...")
        self.fig.canvas.draw_idle()
        try:
            with self._busy("measuring the frames"):
                import copy
                pq_raw = copy.deepcopy(self.pqd)
                enhance_to_pqd(pq_raw, enhance_defaults())
                raw = VideoFrames.from_pqd(pq_raw, self.video.directory)
                defects = measure_acquisition(raw)
                fix = {}
                if defects["flicker"] > FLICKER_FIX_LEVEL:
                    fix["deflicker"] = 1
                if defects["jpeg"] and defects["blockiness"] > BLOCKING_FIX_LEVEL:
                    fix["deblock"] = 1
                meas = measure_recording(raw, WaveModel(self.pqd), whole=True)
                settings = measured_candidates(meas, fix, lawn=False)[0]
        except Exception as exc:
            self.status = "auto enhance failed: {}".format(exc)
            self.enhance_info.set_text("failed: {}".format(exc))
            self.refresh()
            return
        self._set_enhance_widgets(settings)
        self._apply_enhance(settings)
        self._stretch_view_for(settings)
        missing = " and ".join(what for what, have in (("the base (position tab)", have_base),
                                                       ("the cycle (cycle tab)", have_cycle))
                               if not have)
        lines = [describe_measurements(meas),
                 "set from the frames alone: {}".format(describe_enhance(settings)),
                 "to have the settings judged by the pattern they give, set {} "
                 "and press Auto again".format(missing)]
        print("\n".join(lines))
        self.enhance_info.set_text("\n".join(lines))
        self.status = ("auto enhance: {} from the frames alone; set {} on "
                       "their tabs, then Auto fit on the params tab".format(
                           describe_enhance(settings), missing))
        self.refresh()

    def _reset_enhance(self, _event=None) -> None:
        self._set_enhance_widgets(enhance_defaults())
        self._apply_enhance(enhance_defaults())

    # ------------------------------------------------------------- crop

    def crop_description(self) -> str:
        crop = crop_from_pqd(self.pqd)
        if crop is None:
            return "no crop: the whole frame"
        return "crop {}x{} px at ({}, {}) of the levelled frame".format(
            crop[2] - crop[0], crop[3] - crop[1], crop[0], crop[1])

    def _arm_crop(self, _event=None) -> None:
        if self.no_images:
            self.status = "import images first"
            self.refresh()
            return
        self._crop_arm = True
        self.status = ("drag a rectangle on the image to crop to it "
                       "(Level first if the base line is tilted)")
        self.refresh()

    def _crop_press(self, event) -> bool:
        if not self._crop_arm or event.inaxes is not self.ax_img \
                or event.xdata is None:
            return False
        self._crop_drag = (float(event.xdata), float(event.ydata))
        from matplotlib.patches import Rectangle
        if self._crop_patch is not None:
            try:
                self._crop_patch.remove()
            except Exception:
                pass
        self._crop_patch = Rectangle(self._crop_drag, 0, 0, fill=False,
                                     edgecolor="#ffcc00", lw=1.6, ls="--")
        self.ax_img.add_patch(self._crop_patch)
        return True

    def _crop_motion(self, event) -> None:
        if self._crop_drag is None or self._crop_patch is None:
            return
        if event.inaxes is not self.ax_img or event.xdata is None:
            return
        x0, y0 = self._crop_drag
        self._crop_patch.set_bounds(min(x0, event.xdata), min(y0, event.ydata),
                                    abs(event.xdata - x0), abs(event.ydata - y0))
        self.fig.canvas.draw_idle()

    def _crop_release(self, event) -> None:
        if self._crop_drag is None:
            return
        x0, y0 = self._crop_drag
        self._crop_drag = None
        self._crop_arm = False
        if self._crop_patch is not None:
            try:
                self._crop_patch.remove()
            except Exception:
                pass
            self._crop_patch = None
        if event.inaxes is not self.ax_img or event.xdata is None:
            self.status = "crop cancelled"
            self.refresh()
            return
        rect = (x0, y0, float(event.xdata), float(event.ydata))
        if abs(rect[2] - rect[0]) < 8 or abs(rect[3] - rect[1]) < 8:
            self.status = "crop cancelled: that rectangle is too small"
            self.refresh()
            return
        self._apply_crop(rect)

    def _apply_crop(self, rect) -> None:
        """Crop to ``rect`` (in the currently displayed frame), or clear.

        The rectangle is stored in the levelled full frame, so a crop of a
        crop composes, and the base is carried across: a base inside the new
        region keeps its place in the image, one outside it moves to the
        region's centre.  Everything downstream then sees only the region.
        """
        old = crop_from_pqd(self.pqd)
        offx, offy = (old[0], old[1]) if old else (0, 0)
        up = float(self.model.upsample)      # displayed px per source px
        bx = self.pqd.getf("vid_cil_x0") / up
        by = (self.img_h - self.pqd.getf("vid_cil_y0_invy")) / up
        full_bx, full_by = bx + offx, by + offy
        if rect is None:
            new = None
        else:
            x0, y0, x1, y1 = [v / up for v in rect]
            new = crop_tuple((x0 + offx, y0 + offy, x1 + offx, y1 + offy))
            if new is not None:
                new = clip_crop(new, self.video.source_shape)
        for key, val in zip(CROP_KEYS, new if new else (-1, -1, -1, -1)):
            self.pqd.set(key, int(val))
        if not self._reload_video():
            for key, val in zip(CROP_KEYS, old if old else (-1, -1, -1, -1)):
                self.pqd.set(key, int(val))
            self._reload_video()
            return
        nx, ny = (new[0], new[1]) if new else (0, 0)
        bx_new, by_new = (full_bx - nx) * up, (full_by - ny) * up
        if not (0 <= bx_new < self.img_w and 0 <= by_new < self.img_h):
            bx_new, by_new = self.img_w / 2.0, self.img_h / 2.0
        self.pqd.set("vid_cil_x0", float(bx_new))
        self.pqd.set("vid_cil_y0_invy", float(self.img_h - by_new))
        self.model.rebuild()
        self.status = self.crop_description() + (
            "" if new is None else " -- every step now works inside it")
        self.refresh()

    def _build_contrast_panel(self) -> None:
        """Contrast settings for the displayed image only."""
        ax = self._panel_box("video", 0.105, 0.012, 0.395, 0.068)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("none")
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("#999999")
        ax.text(0.5, 1.0, " Contrast Settings ", transform=ax.transAxes,
                ha="center", va="center", fontsize=8.5,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

        ax = self._panel_box("video", 0.118, 0.045, 0.098, 0.026)
        ax.set_facecolor("none")
        for sp in ax.spines.values():
            sp.set_visible(False)
        self.chk_bytescale = CheckButtons(ax, ["Bytescaling"],
                                          [self.contrast["bytescale"]])
        for lb in self.chk_bytescale.labels:
            lb.set_fontsize(8)
        self.chk_bytescale.on_clicked(self._bytescale_cb)

        self.tb_gamma = self._spinbox(
            "video", 0.230, 0.018, 0.042, "Gamma Correction ", "gamma")
        self.tb_sharp = self._spinbox(
            "video", 0.408, 0.047, 0.042, "Sharpness ", "sharpness")
        self.tb_bright = self._spinbox(
            "video", 0.408, 0.018, 0.042, "Brightness ", "brightness")

        # The crop: the part of the frame every step is allowed to see.
        # Unlike the settings above it feeds the measurements, not just the
        # view -- Find base, Find cycle, Find length and Auto fit all work on
        # the cropped frames, and the base and the green cilium live inside.
        ax = self._panel_box("video", 0.232, 0.046, 0.046, 0.024)
        self.btn_crop = Button(ax, "Crop", color="#ffe9b8", hovercolor="#ffd781")
        self.btn_crop.label.set_fontsize(8)
        self.btn_crop.on_clicked(self._arm_crop)
        ax = self._panel_box("video", 0.282, 0.046, 0.046, 0.024)
        self.btn_uncrop = Button(ax, "Clear", color="#ededed", hovercolor="#d6d6d6")
        self.btn_uncrop.label.set_fontsize(8)
        self.btn_uncrop.on_clicked(lambda _e: self._apply_crop(None))
        self._crop_arm = False
        self._crop_drag = None
        self._crop_patch = None
        self.fig.canvas.mpl_connect("motion_notify_event", self._crop_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._crop_release)

    def _spinbox(self, tab, x, y, w, label, key, h=0.022):
        """A text box with up/down arrows -- matplotlib has no spin control."""
        tb = self._panel_textbox(tab, x, y, w, label,
                                 "{:g}".format(self.contrast[key]), h=h)
        tb.on_submit(self._contrast_cb(key))

        self._arrow_pair(tab, x, y, w, h,
                         lambda d, k=key: self._bump_contrast(k, d),
                         CONTRAST_STEP[key])
        return tb

    def _bump_contrast(self, key: str, delta: float) -> None:
        """Arrow click: step the value, clamping at the end of its range.

        Clamping rather than refusing matters: with a step of 0.2 and a floor
        of 0.05, refusing would strand the value at 0.2 and the floor could
        never be reached.
        """
        lo, hi = CONTRAST_LIMIT[key]
        value = round(min(max(self.contrast[key] + delta, lo), hi), 6)
        if value == self.contrast[key]:
            self.status = "{} is already at {:g}".format(key, value)
            self.refresh()
            return
        self.contrast[key] = value
        self._updating = True
        try:
            self._contrast_box(key).set_val("{:g}".format(value))
        finally:
            self._updating = False
        self.status = "{} = {:g}  (display only)".format(key, value)
        self.refresh()

    def _bytescale_cb(self, _label) -> None:
        if self._updating:
            return
        # Read the widget rather than toggling our own copy, so the tick and
        # the setting cannot drift apart.
        self.contrast["bytescale"] = bool(self.chk_bytescale.get_status()[0])
        self.status = "bytescaling {}".format(
            "on" if self.contrast["bytescale"] else "off")
        self.refresh()

    def _contrast_box(self, key: str):
        return {"gamma": self.tb_gamma, "sharpness": self.tb_sharp,
                "brightness": self.tb_bright}[key]

    def _reject_contrast(self, key: str, message: str) -> None:
        """Report a bad entry and put the last good value back in the box."""
        self.status = message
        self._updating = True
        try:
            self._contrast_box(key).set_val("{:g}".format(self.contrast[key]))
        finally:
            self._updating = False
        self.refresh()

    def _contrast_cb(self, key: str):
        def cb(text):
            if self._updating:
                return
            try:
                value = float(str(text).strip())
            except (TypeError, ValueError):
                self._reject_contrast(key, "not a number: {!r}".format(text))
                return
            lo, hi = CONTRAST_LIMIT[key]
            if not (lo <= value <= hi):
                self._reject_contrast(
                    key, "{} must be between {:g} and {:g}".format(key, lo, hi))
                return
            self.contrast[key] = value
            self.status = "{} = {:g}  (display only)".format(key, value)
            self.refresh()
        return cb

    def display_frame(self, frame: int):
        """The frame as shown, with the contrast settings applied."""
        return apply_contrast(self.video.get(frame), **self.contrast)

    def _on_locate(self, _event=None) -> None:
        """Estimate the base position from the observed motion."""
        if self.no_images:
            self.status = "import images first to find the base"
            self.refresh()
            return
        self.locate_info.set_text("searching...")
        self.fig.canvas.draw_idle()
        try:
            with self._busy("finding the base") as report:
                res = locate_base(self.video, self.model, progress=report)
        except Exception as exc:
            self.status = "could not locate the base: {}".format(exc)
            self.locate_info.set_text("failed: {}".format(exc))
            self.refresh()
            return

        old_x = self.pqd.getf("vid_cil_x0")
        old_y = self.pqd.getf("vid_cil_y0_invy")
        self.pqd.set("vid_cil_x0", res["x0"])
        self.pqd.set("vid_cil_y0_invy", res["y0_invy"])
        self.model.rebuild()

        # Point the cilium out of the surface it is anchored in.  On a curved
        # or obliquely viewed carpet that is not straight up, and a model stuck
        # pointing up cannot represent the cilium at all.
        ori_note = ""
        try:
            ori = base_orientation(self.video, self.model)
        except Exception:
            ori = None
        if ori is not None and ori["has_surface"]:
            self.pqd.set("vid_cil_orientation", round(ori["orientation"], 3))
            self.pqd.set(PATTERN_TURN_KEY, 0.0)
            self.model.rebuild()
            ori_note = ("surface tilts {:+.1f} deg here, so orientation "
                        "{:+.1f} deg\n".format(ori["surface_tilt_deg"],
                                               ori["orientation"]))

        ppum = self.model.ppum or 1.0
        self.locate_info.set_text(
            "was ({:.0f}, {:.0f}) -> now ({}, {})\n"
            "moved {:+.2f}, {:+.2f} um;  surface at row {:.0f}\n"
            "{}"
            "peak flat over {} x {} px -- check it, then Reload (r) to undo".format(
                old_x, old_y, res["x0"], res["y0_invy"],
                (res["x0"] - old_x) / ppum, (res["y0_invy"] - old_y) / ppum,
                res["surface_row"], ori_note,
                res["spread_x"], res["spread_y"]))
        self.status = "base -> ({}, {})".format(res["x0"], res["y0_invy"])
        self.refresh()

    def _on_import(self, _event=None) -> None:
        directory = self._pick_folder()
        if directory:
            self.import_sequence(directory)

    def _on_import_path(self, text: str) -> None:
        if self._updating:
            return
        directory = str(text).strip().strip('"')
        if directory and directory != self.images_dir:
            self.import_sequence(directory)

    def import_sequence(self, directory: str) -> bool:
        """Point the model at a new folder of frames and read its dimensions."""
        with self._busy("scanning images") as report:
            report(0.3, "scanning images")
            try:
                info = scan_image_folder(directory)
            except Exception as exc:
                self.status = "could not read the images: {}".format(
                    str(exc).splitlines()[0])
                self.import_info.set_text(
                    "could not read this folder:\n{}\n"
                    "run:  python ciliograph.py check --images <folder>".format(
                        str(exc).splitlines()[0]))
                self.refresh()
                return False
            report(0.8, "loading")
        if info is None:
            self.status = "no images found in {}".format(directory)
            self.import_info.set_text(
                "no image files found in\n{}".format(directory))
            self.refresh()
            return False

        # A new folder starts the program over: every parameter and the
        # waveform go back to what the fitter opens with, held in memory.
        # Nothing is written -- Save writes ciliograph.json beside these
        # images when asked, and not before.  A specification already there is
        # not loaded either; Reload fetches it if that is what you want, and
        # the status line says it is there.
        saved = _pqd_for_images(directory)
        self.pqd = default_pqd_for_images(info)
        self.pqd_path = os.path.join(os.path.abspath(directory), SPEC_NAME)
        self._fourier_coeffs = None
        self._new_model()
        self._slots = fourier_slots(self.fourier.n_space)
        self.sel_n = 0
        self.sel_slot = 0

        self.no_images = False
        if not self._reload_video(directory):
            return False

        self.frame = self.video.first
        folder = os.path.basename(directory.rstrip("/\\"))
        self.status = "imported {} frames from {}; parameters at their defaults".format(
            info["count"], folder)
        if saved:
            self.status += " -- a saved specification is here: Reload (r) loads it"
        self._describe_sequence(info)
        if self.pqd_path:
            print("using {}".format(self.pqd_path))
        self.refresh()
        return True

    def _describe_sequence(self, info=None) -> None:
        """Report what was found in the frame folder."""
        if info is None:
            info = scan_image_folder(self.images_dir)
        if info is None:
            self.import_info.set_text("no images found in the current folder")
            return

        lines = [
            "{} frames  prefix {!r}".format(info["count"], info["basename"]),
            "size {} x {} px".format(info["width"], info["height"]),
        ]
        if info.get("multipage"):
            lines.insert(1, "{} file(s), multi-page stack".format(
                info["n_files"]))
        if not info["uniform"]:
            lines.append("WARNING mixed sizes: {}".format(
                ", ".join("{}x{}".format(w, h) for w, h in info["sizes"][:4])))
        if info.get("unreadable"):
            lines.append("WARNING {} file(s) would not decode; run "
                         "`check`".format(len(info["unreadable"])))
        if info["ppum_from_dpi"]:
            lines.append("file says {:.0f} dpi -> ppum {:.4f} (suggestion)".format(
                info["dpi"], info["ppum_from_dpi"]))
        else:
            lines.append("no scale tag in the files; set ppum yourself")
        self.import_info.set_text("\n".join(lines))

    def _reload_video(self, directory: str | None = None) -> bool:
        """Rebuild the frame sequence after the folder or range changed."""
        directory = directory or self.images_dir
        try:
            video = VideoFrames.from_pqd(self.pqd, directory)
        except Exception as exc:
            self.status = "could not load frames: {}".format(exc)
            self.refresh()
            return False

        self.video = video
        self.images_dir = directory
        self.img_h, self.img_w = video.shape
        # A base outside the frame -- a file cropped by hand, or one written
        # for a larger image -- cannot be traced from; bring it inside.
        bx = self.pqd.getf("vid_cil_x0", -1.0)
        by = self.pqd.getf("vid_cil_y0_invy", -1.0)
        if not (0 <= bx < self.img_w and 0 <= by < self.img_h):
            self.pqd.set("vid_cil_x0", self.img_w / 2.0)
            self.pqd.set("vid_cil_y0_invy", self.img_h / 2.0)
            print("Note: the base ({:g}, {:g}) lay outside the {}x{} frame; "
                  "moved to the centre -- run Find base".format(
                      bx, by, self.img_w, self.img_h))

        # VideoFrames clamps the range to the images that exist; write the
        # clamped values back so the specification and the boxes agree with reality.
        # Without this, asking for frame 72 of a 72-frame clip would leave 72
        # stored while 71 was loaded, and stepping back down did nothing.
        self.pqd.set("vid_first_frame_to_load", video.first)
        self.pqd.set("vid_last_frame_to_load", video.last)

        self.frame = int(np.clip(self.frame, video.first, video.last))

        # A new sequence may be a different size, so show all of it again.
        self.im_artist.set_extent([0, self.img_w, self.img_h, 0])
        self.ax_img.set_xlim(0, self.img_w)
        self.ax_img.set_ylim(self.img_h, 0)
        self._img_extent = (self.img_w, self.img_h)

        self._retune_ranges()
        self._build_timeline()

        self._updating = True
        try:
            self.tb_first.set_val(str(video.first))
            self.tb_last.set_val(str(video.last))
            if self.tb_dir.text != directory:
                self.tb_dir.set_val(directory)
        except Exception:
            pass
        finally:
            self._updating = False
        # The enhance tab shows what the frames now are: after an import
        # that is nothing, after Reload what was saved.
        if hasattr(self, "chk_enhance"):
            self._set_enhance_widgets(enhance_from_pqd(self.pqd))
            self._describe_enhance()
        return True

    def _retune_ranges(self) -> None:
        """Re-range the sliders whose limits depend on the video."""
        self._updating = True
        try:
            for slider, lo, hi in (
                (self.sl_frame, self.video.first, self.video.last),
                (self.sl_mcc_frame, self.video.first, self.video.last),
                (self.sl_macy, self.video.first, self.video.last),
                (self.sl_x0, 0, self.img_w),
                (self.sl_y0, 0, self.img_h),
            ):
                slider.valmin, slider.valmax = lo, hi
                slider.ax.set_xlim(lo, hi)
        finally:
            self._updating = False

    # ------------------------------------------------------------ playback

    @property
    def play_range(self):
        if self.span == "main":
            first = int(round(self.model.macysta))
            last = int(round(self.model.macysta + self.model.period_frames))
            return (max(first, self.video.first), min(last, self.video.last))
        return (self.video.first, self.video.last)

    def play_span(self, span: str) -> None:
        """Play just the main beat cycle, or the whole clip."""
        self.span = "main" if span == "main" else "all"
        lo, hi = self.play_range
        if not (lo <= self.frame <= hi):
            self.frame = lo
        self.playing = True
        self.status = "playing {} ({}..{})".format(
            "the main cycle" if self.span == "main" else "all frames", lo, hi)
        self.refresh()

    def _on_tick(self):
        if self.playing:
            self.step_frame(1)

    def _toggle_play(self, _event=None):
        self.playing = not self.playing
        self.status = "playing" if self.playing else "paused"
        self.refresh(sliders=False)

    def step_frame(self, delta: int) -> None:
        lo, hi = self.play_range
        span = hi - lo + 1
        self.frame = lo + ((self.frame - lo + delta) % span)
        self.refresh(sliders=False)

    def goto(self, frame: int) -> None:
        lo, hi = self.play_range
        self.frame = int(np.clip(frame, lo, hi))
        self.refresh(sliders=False)

    # -------------------------------------------------------------- modes

    def cycle_mode(self, delta: int) -> None:
        self.mode = (self.mode + delta) % len(self.MODES)
        self._select_tab(self.mode)
        self.status = "tab {} ({})".format(self.mode, self.MODES[self.mode])
        self.refresh()

    def adjust(self, direction: int, fast: bool = False, axis: str = "ik") -> None:
        """Apply an increase/decrease in the current mode."""
        mode = self.MODES[self.mode]
        mult = 10.0 if fast else 1.0

        if mode == "video":
            if axis == "jl":
                self.step_frame(direction)
            return

        if mode == "position":
            # j/l always slides the base sideways; i/k drives whichever of
            # height / rotation is selected (p switches between them).
            if axis == "jl":
                self.pqd.set("vid_cil_x0",
                             self.pqd.getf("vid_cil_x0") + 1.0 * mult * direction)
            elif self.pos_target == "orientation":
                self.pqd.set("vid_cil_orientation",
                             self.pqd.getf("vid_cil_orientation")
                             + 0.5 * mult * direction)
            else:
                self.pqd.set("vid_cil_y0_invy",
                             self.pqd.getf("vid_cil_y0_invy") + 1.0 * mult * direction)
            self.status = "base ({:.1f}, {:.1f})  rotation {:.2f} deg  [i/k: {}]".format(
                self.pqd.getf("vid_cil_x0"), self.pqd.getf("vid_cil_y0_invy"),
                self.pqd.getf("vid_cil_orientation"), self.pos_target)

        elif mode == "cycle":
            self.pqd.set("vid_frame_main_cycle_start",
                         self.pqd.getf("vid_frame_main_cycle_start")
                         + 0.2 * mult * direction)
            self.status = "main cycle start = {:.2f}".format(self.model.macysta)

        elif mode == "length":
            n = self.model.n_xy
            self.pqd.set("length", max(self.model.dl,
                                       self.pqd.getf("length") + 0.05 * mult * direction))
            self.model.rebuild()
            self.set_nodes(n)          # keep the node count, adjust dl
            self.status = "length = {:.3f} um  ({} nodes)".format(
                self.model.length, self.model.n_xy)

        elif mode == "params":
            # j/l steps the Fourier mode n, i/k changes the selected
            # coefficient of that mode.
            if axis == "jl":
                self.cycle_mode_n(direction)
                return
            new = self.get_coeff(self.sel_slot) + 0.005 * mult * direction
            self.set_coeff(self.sel_slot, new)
            self.status = "{} = {:.4f}".format(
                self.slot_label(self.sel_slot), new)
            self.refresh()
            return

        self.refresh(sliders=False)

    # ------------------------------------------------------------ commands

    def _on_key(self, event) -> None:
        k = event.key
        if k is None or self._typing():
            return

        if k == " ":
            self._toggle_play()
        elif k == "a":
            self.cycle_mode(1)
        elif k == "s":
            self.cycle_mode(-1)
        elif k == "i":
            self.adjust(+1, False, "ik")
        elif k == "k":
            self.adjust(-1, False, "ik")
        elif k == "I":
            self.adjust(+1, True, "ik")
        elif k == "K":
            self.adjust(-1, True, "ik")
        elif k == "l":
            self.adjust(+1, False, "jl")
        elif k == "j":
            self.adjust(-1, False, "jl")
        elif k == "L":
            self.adjust(+1, True, "jl")
        elif k == "J":
            self.adjust(-1, True, "jl")
        elif k in (".", "right"):
            self.step_frame(1)
        elif k in (",", "left"):
            self.step_frame(-1)
        elif k == "t":
            self.cycle_mode_n(1)
        elif k == "p":
            if self.MODES[self.mode] == "position":
                self.pos_target = ("orientation"
                                   if self.pos_target == "height" else "height")
                self.status = "i/k now drives the {}".format(self.pos_target)
                self.refresh(sliders=False)
            else:
                self.cycle_slot(1)
        elif k == "o":
            self.show_model = not self.show_model
            self.refresh(sliders=False)
        elif k == "z":
            self.reset_zoom()
        elif k == "w":
            self.save()
        elif k == "r":
            self.reload()
        elif k == "h":
            print(HELP_TEXT)
        elif k == "q":
            plt.close(self.fig)

    # ---------------------------------------------------------------- i/o

    def save(self) -> None:
        if not self.pqd_path:
            self.status = "nothing to save yet -- import images first"
            self.refresh()
            return
        # the beat goes into the specification with everything else
        if self._current_beat() is not None:
            self.pqd.set_coeffs({k: self.fourier.coeffs[k].tolist()
                                 for k in FOURIER_KEYS})
        else:
            self.pqd.set_coeffs(None)
        path = self.pqd.save(self.pqd_path, backup=True)
        print("saved {}".format(path))
        self.status = "saved {}{}".format(
            os.path.basename(path),
            " with the fitted beat" if self.pqd.coeffs is not None
            else " (no beat fitted yet)")
        self.refresh(sliders=False)

    def reload(self) -> None:
        if not self.pqd_path:
            self.status = "nothing to reload yet -- import images first"
            self.refresh()
            return
        if not os.path.exists(self.pqd_path):
            # Parameters carried over to a new folder and not saved yet.
            self.status = "nothing saved at {} yet -- Save first".format(
                os.path.basename(self.pqd_path))
            self.refresh()
            return
        try:
            self.pqd = Spec(self.pqd_path)
        except (ValueError, OSError) as exc:
            self.status = "could not read {}: {}".format(
                os.path.basename(self.pqd_path), exc)
            self.refresh()
            return
        self._fourier_coeffs = None
        self._new_model()
        self._slots = fourier_slots(self.fourier.n_space)
        self.sel_n = min(self.sel_n, self.fourier.n_modes - 1)
        # The saved rotation, crop and enhancement come back with the rest:
        # the frames are rebuilt from the file, not left as they were.
        if not self.no_images:
            self._reload_video()
            self.model.rebuild()
        self.status = "reloaded {}{}".format(
            os.path.basename(self.pqd_path),
            " with its beat" if self.pqd.coeffs is not None
            else " (no beat in it: the template is shown)")
        self.refresh()

    # ------------------------------------------------------------- drawing

    def active_shape(self, phase: float):
        """(x, z) of the beat on the params tab at this phase."""
        return self.fourier.shape_at(phase, n_pts=201)

    # -------------------------------------------------------- progress bar

    def _build_progress(self) -> None:
        """A bar drawn over the image while something slow is running."""
        ax = self.fig.add_axes([0.13, 0.455, 0.34, 0.036], zorder=20)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_facecolor("#ffffff")
        for sp in ax.spines.values():
            sp.set_color("#666666")
        self.ax_progress = ax
        self._progress_fill = ax.barh(0.5, 0.0, height=1.0, left=0.0,
                                      color="#4a90d9", align="center")[0]
        self._progress_text = ax.text(
            0.5, 0.5, "", ha="center", va="center", fontsize=8,
            color="#222222", zorder=21)
        ax.set_visible(False)
        self._progress_last = 0.0

    def _pump(self, force: bool = False) -> None:
        """Repaint mid-operation, throttled so the work is not swamped."""
        now = time.time()
        # A full redraw of this figure costs a few tenths of a second, so
        # repainting eight times a second more than doubled the running
        # time of every long operation; twice a second is plenty for a
        # progress bar.
        if not force and (now - self._progress_last) < 0.45:
            return
        self._progress_last = now
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception:
            pass

    @contextlib.contextmanager
    def _busy(self, label: str):
        """Show the bar for the duration, and yield a progress callback."""
        self.ax_progress.set_visible(True)
        self._progress_fill.set_width(0.0)
        self._progress_text.set_text(label)
        self._pump(force=True)

        def report(fraction, note=None):
            self._progress_fill.set_width(float(np.clip(fraction, 0.0, 1.0)))
            self._progress_text.set_text(
                "{}  {:.0f}%".format(note or label, 100 * float(fraction)))
            self._pump()

        try:
            yield report
        finally:
            self.ax_progress.set_visible(False)
            self._progress_last = 0.0

    def _show_image(self, img, w: int, h: int) -> None:
        """Put a frame on the image axes, preserving any zoom.

        The axis limits are only touched when the frame size actually changes.
        Re-applying them on every refresh would undo the user's zoom each time
        the frame advanced, which made it impossible to play a zoomed-in view.
        """
        self.im_artist.set_data(img)
        if getattr(self, "_img_extent", None) != (w, h):
            self.im_artist.set_extent([0, w, h, 0])
            self.ax_img.set_xlim(0, w)
            self.ax_img.set_ylim(h, 0)
            self._img_extent = (w, h)

    def reset_zoom(self) -> None:
        """Restore the full frame."""
        w, h = self.img_w, self.img_h
        self.im_artist.set_extent([0, w, h, 0])
        self.ax_img.set_xlim(0, w)
        self.ax_img.set_ylim(h, 0)
        self._img_extent = (w, h)
        self.status = "zoom reset to the whole frame"
        self.refresh()

    def _mcc_active(self) -> bool:
        return self.MODES[self.mode] == "MCC"

    def _draw_mcc(self) -> None:
        """Show the trajectory over the sequence loaded on the video tab."""
        frame = self.frame
        self._show_image(self.display_frame(frame), self.img_w, self.img_h)

        # Velocities follow the video tab's calibration, so re-derive in case
        # ppum or fps changed since the track was made.
        if self.mcc.get("traj") is not None:
            try:
                self._mcc_recompute()
            except ValueError:
                pass

        for artist in (self.line_model, self.line_base):
            artist.set_data([], [])

        traj = self.mcc.get("traj")
        if traj is not None:
            ok = ~np.isnan(traj[:, 1])
            self.line_mcc.set_data(traj[ok, 1], traj[ok, 2])
            here = traj[traj[:, 0] == frame]
            if len(here) and not np.isnan(here[0, 1]):
                self.line_mcc_now.set_data([here[0, 1]], [here[0, 2]])
            else:
                self.line_mcc_now.set_data([], [])
        else:
            self.line_mcc.set_data([], [])
            self.line_mcc_now.set_data([], [])

        seed = self.mcc.get("seed")
        if seed is not None:
            self.line_mcc_seed.set_data([seed[0]], [seed[1]])
        else:
            self.line_mcc_seed.set_data([], [])

    def refresh(self, sliders: bool = True) -> None:
        if self._mcc_active():
            self._draw_mcc()
            self._sync_sliders()
            self.hud.set_text(self._mcc_hud())
            self._fit_hud_font()
            self.fig.canvas.draw_idle()
            return

        for artist in (self.line_mcc, self.line_mcc_seed, self.line_mcc_now):
            artist.set_data([], [])

        phase = self.model.phase_of_frame(self.frame)

        self._show_image(self.display_frame(self.frame), self.img_w, self.img_h)
        if getattr(self, "btn_effdir", None) is not None:
            self.btn_effdir.label.set_text(self._effdir_label())
        if getattr(self, "btn_upright", None) is not None:
            self.btn_upright.label.set_text(self._frame_label())

        # The analytic waveform always tracks L_c and the beat frequency.
        self.fourier.length = self.model.length
        self.fourier.cbf = self.model.cbf

        main = self.active_shape(phase)

        if self.show_model:
            px, py = self.model.to_pixels(main[0], main[1], image_h=self.img_h)
            self.line_model.set_data(px, py)
            self.line_base.set_data([px[0]], [py[0]])
        else:
            self.line_model.set_data([], [])
            self.line_base.set_data([], [])

        n_show = len(self.cycle_lines)
        for i, artist in enumerate(self.cycle_lines):
            sx, sy = self.active_shape(i / n_show)
            artist.set_data(sx, sy)
        nx, ny = main
        self.line_now.set_data(nx, ny)

        lim = max(6.1, self.model.length * 1.05)
        self.ax_shape.set_xlim(-lim, lim)
        self.ax_shape.set_ylim(-0.3, self.model.length * 1.25)

        self._refresh_timeline()

        # Always resync: a keyboard change in any tab must move that tab's
        # slider too.  The _updating guard stops this echoing back.
        self._sync_sliders()

        self.hud.set_text(self._hud_text(phase))
        self._fit_hud_font()
        self.fig.canvas.draw_idle()

    def _mcc_hud(self) -> str:
        rows = [
            "tab  {:>2}  MCC   particle tracking".format(self.mode),
            "frames  {} .. {}   ({} x {} px)".format(
                self.video.first, self.video.last, self.img_w, self.img_h),
            "showing {}".format(self.frame),
            "",
            "from the video tab:",
            "{:.4f} um/px (ppum {:g})   {:g} fps".format(
                self.mcc_pixel_size, self.model.ppum, self.mcc_fps),
            "",
        ]
        seed = self.mcc.get("seed")
        rows.append("seed    {}".format(
            "({:.1f}, {:.1f})".format(*seed) if seed else "not picked"))

        vel = self.mcc.get("vel")
        if vel is None:
            rows += ["", "no track yet"]
        else:
            rows += [
                "tracked {} of {} frames".format(vel["n_tracked"], vel["n_frames"]),
                "",
                "path      {:8.2f} um".format(vel["total_path_um"]),
                "net       {:8.2f} um".format(vel["net_disp_um"]),
                "time      {:8.2f} ms".format(vel["total_time_s"] * 1000),
                "",
                "avg speed    {:7.1f} um/s".format(vel["mean_speed_umps"]),
                "avg velocity {:7.1f} um/s".format(vel["net_velocity_umps"]),
                "inst  {:.1f} +/- {:.1f} um/s".format(
                    vel["mean_inst_umps"], vel["std_inst_umps"]),
                "inst range {:.1f} .. {:.1f}".format(
                    vel["min_inst_umps"], vel["max_inst_umps"]),
            ]
        rows += ["", "* {}".format(self.status)]
        return "\n".join(rows)

    def _hud_text(self, phase: float) -> str:
        lo, hi = self.play_range
        rows = [
            "tab  {:>2}  {}".format(self.mode, self.MODES[self.mode]),
            "frame   {:>5}   of {}..{} [{}]".format(self.frame, lo, hi, self.span),
            "phase   {:>7.4f}   period {:.2f} fr".format(
                phase, self.model.period_frames),
            "",
        ]

        # The length is held at L at every phase; what varies is how far
        # the raw cubic strayed from it here, which is worth seeing
        # because it is the fit's own error at this phase.
        raw = self.fourier.raw_arclength(phase)
        rows += [
            "Fourier mode n = {} of 0..{}".format(
                self.sel_n, self.fourier.n_modes - 1),
            "selected {}".format(self.slot_label(self.sel_slot)),
            "length {:.2f} um  (fit rescaled x{:.3f} here)".format(
                self.fourier.length, 1.0 / raw if raw > 0 else 1.0),
            "",
        ]
        for slot, (block, mi, _) in enumerate(self._slots):
            mark = ">" if slot == self.sel_slot else " "
            rows.append("{} {:<3} m{}  {:>10.4f}".format(
                mark, block, mi + 1, self.get_coeff(slot)))
        rows += [
            "",
            "ppum {:.3f}   cbf {:.4f} Hz   len {:.2f} um".format(
                self.model.ppum, self.model.cbf, self.model.length),
            self._amplitude_line(),
            "base ({:.1f}, {:.1f})  ori {:.2f} deg".format(
                self.pqd.getf("vid_cil_x0"),
                self.pqd.getf("vid_cil_y0_invy"),
                self.pqd.getf("vid_cil_orientation")),
            "* {}".format(self.status),
        ]
        return "\n".join(rows)

    def beat_amplitude(self, n_phases: int = 48) -> dict:
        """The fitted beat's amplitude: how far the tip sweeps sideways
        (max - min of its x over the beat, in the pattern's frame), the tip's
        angular span and its height range.  Cached on the beat."""
        try:
            key = (float(self.model.length),
                   hashlib.md5(b"".join(np.ascontiguousarray(
                       self.fourier.coeffs[k], dtype=float).tobytes()
                       for k in FOURIER_KEYS)).hexdigest())
        except Exception:
            key = None
        cache = getattr(self, "_amp_cache", None)
        if key is not None and cache is not None and cache[0] == key:
            return cache[1]
        tips = []
        for t in np.arange(n_phases) / float(n_phases):
            x, z = self.active_shape(t)
            tips.append((float(x[-1]), float(z[-1])))
        tips = np.array(tips)
        ang = np.degrees(np.arctan2(tips[:, 0], tips[:, 1]))
        out = {"amplitude_um": float(tips[:, 0].max() - tips[:, 0].min()),
               "tip_x": (float(tips[:, 0].min()), float(tips[:, 0].max())),
               "tip_z": (float(tips[:, 1].min()), float(tips[:, 1].max())),
               "fan_deg": float(ang.max() - ang.min())}
        if key is not None:
            self._amp_cache = (key, out)
        return out

    def _amplitude_line(self) -> str:
        try:
            a = self.beat_amplitude()
        except Exception:
            return "amplitude -"
        return ("amplitude {:.2f} um  (tip x {:+.2f}..{:+.2f}, fan {:.0f} deg, "
                "height {:.1f}-{:.1f} um)".format(
                    a["amplitude_um"], a["tip_x"][0], a["tip_x"][1],
                    a["fan_deg"], a["tip_z"][0], a["tip_z"][1]))

    def _fit_hud_font(self) -> None:
        """Size the readout's font so its lines fit the panel above the
        buttons: at a small window the 9 pt text ran under them."""
        try:
            n_lines = max(1, self.hud.get_text().count("\n") + 1)
            height_pt = self.ax_hud.get_position().height * \
                self.fig.get_size_inches()[1] * 72.0
            size = min(9.0, max(5.5, (height_pt - 4.0) / (n_lines * 1.15)))
            if abs(size - self.hud.get_fontsize()) > 0.05:
                self.hud.set_fontsize(size)
                self.fig.canvas.draw_idle()
        except Exception:
            pass

    def _on_help(self, _event=None) -> None:
        """Open the help as a page in the browser; the terminal gets the
        text as well, and the page's path when no browser opens."""
        import tempfile
        import webbrowser
        print(HELP_TEXT)
        path = os.path.join(tempfile.gettempdir(), "ciliograph_help.html")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(help_html())
            import pathlib
            opened = webbrowser.open(pathlib.Path(path).as_uri())
        except Exception as exc:
            opened = False
            print("could not write or open the help page: {}".format(exc))
        self.status = ("help opened in the browser (and printed in the terminal)"
                       if opened else
                       "help printed in the terminal; the page is at {}".format(path))
        self.refresh(sliders=False)

    def show(self) -> None:
        print(HELP_TEXT)
        plt.show()


def launch_gui(pqd_path: str, images_dir: str, fps_play: float = 12.0,
               fourier_coeffs: dict | None = None,
               n_modes: int | None = FOURIER_DEFAULT_MODES,
               n_space: int | None = None) -> CiliographGui:
    """Open the fitter window."""
    matplotlib = _load_pyplot(interactive=True)
    if matplotlib.get_backend().lower() == "agg":
        raise RuntimeError(
            "matplotlib is using the non-interactive 'Agg' backend, so no window\n"
            "can be opened.  Install an interactive backend, e.g.\n"
            "    pip install PyQt5\n"
            "or run with a display available (DISPLAY set / X forwarding on).")
    gui = CiliographGui(pqd_path, images_dir, fps_play=fps_play,
                        fourier_coeffs=fourier_coeffs,
                        n_modes=n_modes, n_space=n_space)
    gui.show()
    return gui


# =========================================================================
# Command line
# =========================================================================

ROOT = _app_dir()
# No default dataset: the image folder is chosen every run, so that opening
# the program can never silently reuse the last set of images.
DEFAULT_OUTDIR = os.path.join(ROOT, "output")

# Remembers the last folder only to position the dialog, never to skip it.
# The old name is still read, so renaming the program does not quietly forget
# where you were last working.
_LAST_FOLDER_FILE = os.path.join(ROOT, ".ciliograph_last_folder")
_LAST_FOLDER_OLD = os.path.join(ROOT, ".axwa_last_folder")


def _remember_folder(path: str) -> None:
    try:
        with open(_LAST_FOLDER_FILE, "w", encoding="utf-8") as f:
            f.write(os.path.abspath(path))
    except OSError:
        pass


def _recall_folder():
    for candidate in (_LAST_FOLDER_FILE, _LAST_FOLDER_OLD):
        try:
            with open(candidate, encoding="utf-8") as f:
                path = f.read().strip()
        except OSError:
            continue
        if os.path.isdir(path):
            return path
    return None


def read_axwam(run_dir: str) -> dict:
    """Read the first *.axwam run configuration found in `run_dir`."""
    out: dict = {}
    candidates = sorted(glob.glob(os.path.join(run_dir, "*.axwam")))
    if not candidates:
        return out
    with open(candidates[0], encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("!END_OF_DATA"):
                break
            if not line.startswith("!"):
                continue
            parts = line[1:].split(None, 1)
            if len(parts) == 2:
                out[parts[0]] = parts[1].strip()
    out["_axwam_path"] = candidates[0]
    return out


def resolve_paths(args, prompt: bool = True, create: bool = True):
    """Work out where the frames and the specification file live.

    There is no default dataset.  If neither --images nor --run-dir is given,
    the image folder is chosen interactively.  A folder with no specification
    gets a fresh one written beside it for the command-line tools, which need
    a file to read; with ``create`` off -- the fitter -- the path is returned
    unwritten and the fitter starts from defaults in memory instead.
    """
    run_dir = os.path.abspath(args.run_dir) if getattr(args, "run_dir", None) else None
    images_dir = os.path.abspath(args.images) if getattr(args, "images", None) else None

    if images_dir is None and run_dir is not None:
        images_dir = os.path.join(run_dir, "input_images")

    if images_dir is None:
        if not prompt:
            raise SystemExit(
                "no images given: pass --images <folder> (or --run-dir <folder>)")
        chosen = choose_folder("Select the folder of images", _recall_folder())
        if not chosen:
            raise SystemExit(
                "no folder chosen.\n"
                "Pass --images <folder> to skip the dialog, or --run-dir "
                "<folder> for a run laid out with data1/ and input_images/.")
        images_dir = os.path.abspath(chosen)

    _remember_folder(images_dir)

    if getattr(args, "spec", None):
        return os.path.abspath(args.spec), images_dir

    # A run folder laid out the old way keeps working.
    if run_dir is not None:
        cfg = read_axwam(run_dir)
        rel = cfg.get("file_wapa", os.path.join("data1", "wapa.pqd"))
        candidate = os.path.join(run_dir, rel)
        if os.path.exists(candidate):
            return candidate, images_dir

    found = _pqd_for_images(images_dir)
    if found:
        return found, images_dir

    # Nothing to inherit -- start this image set from the defaults.
    fresh = os.path.join(images_dir, SPEC_NAME)
    if not create:
        return fresh, images_dir
    try:
        create_spec_for_images(images_dir, fresh)
    except (ValueError, OSError) as exc:
        raise SystemExit("{}\nChoose a folder that contains the image "
                         "sequence itself.".format(exc))
    print("no specification for these images; wrote a fresh one at\n  {}".format(fresh))
    return fresh, images_dir


def add_common(p):
    p.add_argument("--run-dir", default=None,
                   help="a run folder laid out with data1/ and input_images/; "
                        "without it (or --images) you are asked to choose a "
                        "folder of images")
    p.add_argument("--spec", "--pqd", dest="spec", default=None,
                   help="explicit path to the specification file "
                        "(ciliograph.json; a wapa.pqd of earlier versions is "
                        "read too; overrides --run-dir)")
    p.add_argument("--images", default=None,
                   help="explicit path to the frames folder")
    add_compute(p)


def add_compute(p):
    p.add_argument("--device", choices=("auto", "cpu", "gpu"), default=None,
                   help="where the heavy steps run: auto uses a CUDA GPU when "
                        "CuPy finds one and every CPU core otherwise "
                        "(default: auto, or $CILIOGRAPH_DEVICE)")
    p.add_argument("--workers", type=int, default=None,
                   help="CPU worker processes (default: every core, or "
                        "$CILIOGRAPH_WORKERS); 1 runs on a single core")


def cmd_fit(args) -> int:
    hint = dependency_hint()
    if hint:
        # Said up front, because the symptom downstream is only "could not
        # read the images", which points at the folder rather than the box.
        print("warning:" + hint, file=sys.stderr)
    # The fitter can open with nothing loaded; the images are then chosen with
    # the "Import images..." button, so no dialog is forced at startup.
    if not (args.images or args.run_dir or args.spec):
        pqd_path, images_dir = None, None
        print("no images given -- opening empty; "
              "use 'Import images...' on the video tab")
    else:
        # The fitter writes nothing on opening: a folder without a specification
        # starts from the defaults in memory, and Save creates the file.
        pqd_path, images_dir = resolve_paths(args, create=False)
        if images_dir and not os.path.isdir(images_dir):
            print("error: images folder not found at {}".format(images_dir),
                  file=sys.stderr)
            return 2
        if args.spec and not os.path.exists(pqd_path):
            print("error: specification file not found at {}".format(pqd_path),
                  file=sys.stderr)
            return 2
    try:
        coeffs = load_fourier_coeffs(args.coeffs) if args.coeffs else None
    except (ValueError, OSError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    launch_gui(pqd_path, images_dir, fps_play=args.play_fps,
               fourier_coeffs=coeffs, n_modes=args.modes, n_space=args.space)
    return 0


def cmd_shapes(args) -> int:
    pqd_path, _ = resolve_paths(args)
    if not os.path.exists(pqd_path):
        print("error: specification file not found at {}".format(pqd_path),
              file=sys.stderr)
        return 2
    try:
        run_shapes(pqd_path, outdir=args.outdir, n_cs=args.n_cs,
                   split=args.split, figtext=args.figtext,
                   make_plots=not args.no_plots,
                   coeffs_path=args.coeffs, n_pts=args.n_pts,
                   equal_arclength=not args.equal_s, n_modes=args.modes,
                   n_space=args.space)
    except (ValueError, OSError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    return 0


def cmd_info(args) -> int:
    pqd_path, images_dir = resolve_paths(args)
    if not os.path.exists(pqd_path):
        print("error: specification file not found at {}".format(pqd_path),
              file=sys.stderr)
        return 2

    pqd = Spec(pqd_path)
    model = WaveModel(pqd)

    print("file        : {}".format(pqd_path))
    print("length      : {} um   dl {}   -> {} points".format(
        model.length, model.dl, model.n_xy))
    print("cbf         : {} Hz".format(model.cbf))
    print("fps         : {}   -> {:.3f} frames per beat".format(
        model.fps, model.period_frames))
    print("ppum        : {} px/um".format(model.ppum))
    print("frames      : {} .. {}   main cycle starts at {}".format(
        int(pqd.getf("vid_first_frame_to_load")),
        int(pqd.getf("vid_last_frame_to_load")), model.macysta))
    print("base        : ({}, {}) invy   orientation {} deg".format(
        pqd.getf("vid_cil_x0"), pqd.getf("vid_cil_y0_invy"),
        pqd.getf("vid_cil_orientation")))
    print("compute     : {}".format(describe_compute()))
    if pqd.coeffs is not None:
        w = model.wave
        print("beat        : fitted, {} Fourier modes, s^1..s^{}".format(
            w.n_modes, w.n_space))
        for key in FOURIER_KEYS:
            for mi, row in enumerate(pqd.coeffs[key]):
                print("  {} m{}  {}".format(key, mi + 1, "  ".join(
                    "{: .4f}".format(v) for v in row)))
    else:
        print("beat        : none fitted yet (the published template stands in)")

    if os.path.isdir(images_dir):
        try:
            v = VideoFrames.from_pqd(pqd, images_dir)
            print()
            print("images      : {} frames in {}, {}x{} px".format(
                len(v), images_dir, v.shape[1], v.shape[0]))
        except Exception as exc:
            print("images      : could not read ({})".format(exc))
    return 0


def cmd_selftest(args) -> int:
    """Check the model, the specification file and the fitters."""
    import hashlib
    import tempfile

    scratch = tempfile.TemporaryDirectory()
    if getattr(args, "spec", None) or getattr(args, "images", None) \
            or getattr(args, "run_dir", None):
        pqd_path, _ = resolve_paths(args)
        if not os.path.exists(pqd_path):
            print("error: specification file not found at {}".format(pqd_path),
                  file=sys.stderr)
            return 2
    else:
        # No data needed: the checks run on a default specification.
        pqd_path = os.path.join(scratch.name, SPEC_NAME)
        Spec.defaults(length=6.0, cbf=15.0).save(pqd_path, backup=False)
        print("no --spec/--images given: checking a default specification")

    ok = True

    # 1. Round-trip fidelity of the specification file: what is read is
    #    what is written.  A wapa.pqd of the earlier versions is read too,
    #    and its migrated copy must round-trip the same way.
    original = open(pqd_path, "rb").read().replace(b"\r\n", b"\n")
    p = Spec(pqd_path)
    legacy = not original.lstrip().startswith(b"{")
    if legacy:
        migrated = os.path.join(scratch.name, "migrated.json")
        p.save(migrated, backup=False)
        original = open(migrated, "rb").read().replace(b"\r\n", b"\n")
        p = Spec(migrated)
    same = p.dumps().encode("utf-8") == original
    print("round-trip byte-identical : {} ({} parameters{}{})".format(
        "OK" if same else "FAILED", len(p.scalars),
        ", fitted beat" if p.coeffs is not None else ", no beat yet",
        "; migrated from wapa.pqd" if legacy else ""))
    ok &= same

    # 2. An edit changes exactly one line of the file.
    p2 = Spec(pqd_path)
    p2.set("length", p2.getf("length", 6.0) + 0.05)
    changed = [i for i, (a, b) in enumerate(
        zip(original.decode().split("\n"), p2.dumps().split("\n"))) if a != b]
    good = len(changed) == 1
    print("one edit -> one line change: {} ({} line(s))".format(
        "OK" if good else "FAILED", len(changed)))
    ok &= good

    # 2b. The earlier format is still read: the text of a wapa.pqd (the
    #     AXWA key lines) parses to the parameters, and the JSON written
    #     from them reads back equal.  Nothing writes a wapa.pqd -- not even
    #     this check: the text stays in memory.
    old_dir = os.path.join(scratch.name, "old")
    os.makedirs(old_dir)
    legacy_text = ("!vid_image_filename\tim\n!vid_image_w\t640\n!vid_image_h\t480\n"
                   "!vid_first_frame_to_load\t0\n!vid_last_frame_to_load\t100\n"
                   "!vid_frame_main_cycle_start\t17\n!vid_ppum\t11.76\n"
                   "!vid_fps\t300\n!vid_cil_x0\t377\n!vid_cil_y0_invy\t63\n"
                   "!vid_cil_orientation\t-5.85\n!all_continuous\t1\n"
                   "!constant_n\t0\n!length\t5.5\n!dl\t0.01\n!cbf\t8.9419\n"
                   "!phasef_ert\t0.5\n!phasef_eet\t0.9\n!phasef_eft\t0\n"
                   "!param_series\t0.1\t0.2\n!end\n")
    m_old = Spec.from_text(legacy_text)
    m_old.set_coeffs(FOURIER_DEFAULT)
    m_old.save(os.path.join(old_dir, SPEC_NAME), backup=False)
    m_new = Spec(os.path.join(old_dir, SPEC_NAME))
    want = {"vid_ppum": 11.76, "vid_cil_x0": 377.0, "cbf": 8.9419,
            "length": 5.5, "vid_frame_main_cycle_start": 17.0}
    good = (all(abs(m_old.getf(k) - v) < 1e-9 for k, v in want.items())
            and m_old.scalars == m_new.scalars
            and m_old.coeffs is not None and m_new.coeffs == m_old.coeffs
            and all(np.allclose(FourierWave(m_new.coeffs).coeffs[k],
                                FOURIER_DEFAULT[k]) for k in FOURIER_KEYS)
            and "all_continuous" not in m_old.scalars
            and "param_series" not in m_old.scalars)
    print("earlier format still read  : {} ({} parameters from the AXWA key "
          "lines; written back as {} only)".format(
              "OK" if good else "FAILED", len(m_old.scalars), SPEC_NAME))
    ok &= good

    # 3. The shape at a phase is finite, starts at the base and has the
    #    node count the specification asks for.
    m = WaveModel(p)
    x, y = m.shape_at(0.25)
    good = (len(x) == m.n_xy and x[0] == 0.0 and y[0] == 0.0
            and np.all(np.isfinite(x)) and np.all(np.isfinite(y)))
    print("shape integration          : {} ({} points, tip at {:.2f}, {:.2f})".format(
        "OK" if good else "FAILED", len(x), x[-1], y[-1]))
    ok &= good

    # 4. The arc length is fixed at every phase of the cycle and the nodes
    #    sit at equal arc length.
    totals, spread = [], 0.0
    for t in np.linspace(0.0, 1.0, 25, endpoint=False):
        seg = np.hypot(*[np.diff(a) for a in m.shape_at(t)])
        totals.append(seg.sum())
        spread = max(spread, seg.max() - seg.min())
    totals = np.array(totals)
    good = (abs(totals - m.length).max() < 1e-3 * m.length and spread < 1e-4)
    print("arc length fixed           : {} ({:.4f}..{:.4f} um over the cycle "
          "for L = {:g}, node spacing spread {:.1e})".format(
              "OK" if good else "FAILED", totals.min(), totals.max(),
              m.length, spread))
    ok &= good

    # 5. The beat is periodic: the shape at phase 1 is the shape at phase 0.
    a = np.array(m.shape_at(0.0))
    b = np.array(m.shape_at(1.0))
    err = np.abs(a - b).max()
    good = err < 1e-9
    print("cycle closes on itself     : {} (gap {:.1e} um)".format(
        "OK" if good else "FAILED", err))
    ok &= good

    # 6. The node export.  A default specification (L 6, 14 steps, 201
    #    nodes) must give the z01a.data it has always given -- byte for
    #    byte -- and a reference file, if one is supplied, is compared
    #    against the export of this specification.
    with tempfile.TemporaryDirectory() as td:
        ref_spec = os.path.join(td, "ref.json")
        Spec.defaults(length=6.0, cbf=15.0).save(ref_spec, backup=False)
        run_shapes(ref_spec, outdir=td, n_cs=14, make_plots=False, verbose=False)
        digest = hashlib.md5(open(os.path.join(td, "z01a.data"), "rb").read()).hexdigest()
        good = digest == Z01A_TEMPLATE_MD5
        print("shape export unchanged     : {} (template beat, L 6, 14 steps: "
              "z01a.data md5 {}{})".format(
                  "OK" if good else "FAILED", digest[:10],
                  "" if good else ", expected " + Z01A_TEMPLATE_MD5[:10]))
        ok &= good

        if args.reference:
            run_shapes(pqd_path, outdir=td, n_cs=args.n_cs, make_plots=False,
                       verbose=False)
            produced = os.path.join(td, "z01a.data")
            if not os.path.exists(args.reference):
                print("reference comparison       : SKIPPED (no such file: {})".format(
                    args.reference))
            else:
                a = open(produced, "rb").read()
                b = open(args.reference, "rb").read().replace(b"\r\n", b"\n")
                same = a == b
                print("matches reference z01a.data: {} ({})".format(
                    "OK" if same else "FAILED", os.path.basename(args.reference)))
                ok &= same

    # 7. The GUI wrappers must all accept the arguments the CLI forwards.
    #    Constructing CiliographGui directly in a test skips launch_gui, so a
    #    signature drift between them only shows up when a window is opened.
    import inspect
    forwarded = {"fps_play", "fourier_coeffs", "n_modes", "n_space"}
    launch_args = set(inspect.signature(launch_gui).parameters)
    gui_args = set(inspect.signature(CiliographGui.__init__).parameters)
    missing = sorted((forwarded - launch_args) | (forwarded - gui_args))
    good = not missing
    print("GUI wrappers accept CLI args : {}{}".format(
        "OK" if good else "FAILED", "" if good else " missing " + ", ".join(missing)))
    ok &= good

    # 7b. Every ``args.<name>`` a command reads must be an option its parser
    #     defines.  The fitter once read args.pqd after the option had become
    #     --spec, and the program could not start -- a slip no import or
    #     syntax check sees.
    import ast as _ast
    try:
        if getattr(sys, "frozen", False):
            raise FileNotFoundError("SKIP")
        tree = _ast.parse(open(os.path.abspath(__file__), encoding="utf-8").read())
        parser = build_parser()
        subs = next(a for a in parser._actions
                    if isinstance(a, argparse._SubParsersAction))
        bad = []
        for name, sp in subs.choices.items():
            func = sp.get_default("func")
            fn = next((n for n in tree.body if isinstance(n, _ast.FunctionDef)
                       and func is not None and n.name == func.__name__), None)
            if fn is None:
                bad.append("{}: no function".format(name))
                continue
            used = {t.attr for t in _ast.walk(fn)
                    if isinstance(t, _ast.Attribute)
                    and isinstance(t.value, _ast.Name) and t.value.id == "args"}
            have = {a.dest for a in sp._actions} | {"func", "command"}
            bad += ["{}: args.{}".format(name, u) for u in sorted(used - have)]
        good = not bad
        note = ("{} commands".format(len(subs.choices)) if good
                else "unknown " + ", ".join(bad))
    except Exception as exc:
        if str(exc) == "SKIP":
            good, note = True, "skipped: checked when the .exe was built"
        else:
            good, note = False, "could not check ({})".format(exc)
    print("CLI options match commands : {} ({})".format(
        "OK" if good else "FAILED", note))
    ok &= good

    # 8. Fourier model sanity, on the same L and cbf as this file.
    wave = FourierWave.from_spec(p)
    base = max(float(np.abs(np.array(wave.position(0.0, ph))).max())
               for ph in np.linspace(0, 1, 25))
    lengths = np.array([np.hypot(*[np.diff(a) for a in
                                   wave.shape_at(ph, n_pts=401)]).sum()
                        for ph in np.linspace(0, 1, 120, endpoint=False)])
    seg = np.hypot(*[np.diff(a) for a in wave.shape_at(0.3, n_pts=201)])
    good = (base == 0.0
            and abs(lengths.mean() - wave.length) < 1e-3 * wave.length
            and (lengths.max() - lengths.min()) < 1e-3 * wave.length
            and (seg.max() - seg.min()) < 1e-4)
    print("fourier model              : {} (base {:.1e}, length {:.4f} um, "
          "varying by {:.1e} over the cycle, node spacing spread {:.1e})"
          .format("OK" if good else "FAILED", base, lengths.mean(),
                  lengths.max() - lengths.min(), seg.max() - seg.min()))
    ok &= good

    # 9. The stroke test agrees with the literature's phase convention: the
    #    published table has its phase zero at the end-effective position,
    #    and rotating the coefficients must not change a single shape.
    ref = FourierWave(length=wave.length)
    marks = recovery_start_phase(ref)
    off = min(marks["phase_end_effective"], 1 - marks["phase_end_effective"])
    fast_end = marks["phase_fast_end"]
    shifted = FourierWave(shift_fourier_phase(ref.coeffs, 0.37),
                          length=wave.length)
    drift = max(float(np.abs(np.concatenate(ref.shape_at(t + 0.37))
                             - np.concatenate(shifted.shape_at(t))).max())
                for t in np.linspace(0, 1, 13))
    # ... and turning the model frame (rotate_fourier_frame) is exact too:
    # the shapes, turned back, coincide, and the fan's axis reads that
    # much further to the right in the turned frame.
    turn = 17.0
    turned = FourierWave(rotate_fourier_frame(ref.coeffs, turn), length=wave.length)
    a = np.deg2rad(turn)
    rot_err = 0.0
    for t in np.linspace(0, 1, 7, endpoint=False):
        x0, z0 = ref.shape_at(t)
        x1, z1 = turned.shape_at(t)
        # a point (x, z) of the old frame reads (x cos a + z sin a,
        # -x sin a + z cos a) in the frame turned by a
        rot_err = max(rot_err, float(np.abs(x0 * np.cos(a) + z0 * np.sin(a) - x1).max()),
                      float(np.abs(-x0 * np.sin(a) + z0 * np.cos(a) - z1).max()))
    lean0, lean1 = fan_lean(ref), fan_lean(turned)
    lean_err = abs((lean1["axis_deg"] - lean0["axis_deg"]) - turn)
    good = (off < 0.03 and 0.70 < fast_end < 0.80 and drift < 1e-9
            and rot_err < 1e-9 and lean_err < 0.5)
    print("stroke landmark            : {} (published table: recovery starts at "
          "phase {:.3f}, its fast sweep ends at {:.3f}; phase shift exact to "
          "{:.1e} um; frame turned 17 deg: shapes exact to {:.1e} um, fan axis "
          "{:+.1f} -> {:+.1f} deg)".format(
              "OK" if good else "FAILED", marks["phase_end_effective"], fast_end,
              drift, rot_err, lean0["axis_deg"], lean1["axis_deg"]))
    ok &= good

    # 10. The synthetic renderer: a tiny carpet whose truth reads back, and
    #     whose reference cilium's written centreline matches the table.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tr = render_carpet(tmp, n_frames=3, n_cilia=8, ppum=15.0, fps=100.0,
                           spacing=3.0, masks=True)
        _, cur = load_trace_xyt(os.path.join(tmp, "truth_track.csv"))
        wave_t = FourierWave(load_fourier_coeffs(os.path.join(tmp, "truth_coeffs.txt")),
                             length=tr["length_um"], cbf=tr["cbf_hz"])
        ph0 = tr["reference_phase_at_frame0"]
        x, z = wave_t.shape_at(ph0, n_pts=len(cur[0]))
        drift = float(np.abs(np.column_stack([x, z]) - np.asarray(cur[0])).max())
        frame = _load_image(os.path.join(tmp, tr["frames"][0]))
        # the mask: white along the reference centreline, black elsewhere
        # (the empty margin), and thin
        mask = np.asarray(_load_image(os.path.join(tmp, "masks", tr["frames"][0])),
                          dtype=float) / 255.0
        bx, by = tr["reference_base_px"]
        c = np.asarray(cur[0])
        px = np.clip(np.rint(bx + c[:, 0] * tr["ppum"]).astype(int), 0, mask.shape[1] - 1)
        py = np.clip(np.rint(by - c[:, 1] * tr["ppum"]).astype(int), 0, mask.shape[0] - 1)
        on_line = float(np.median(mask[py, px]))
        margin_max = float(mask[:6, :6].max())
        cover = float((mask > 0.5).mean())
        good = (drift < 1e-3 and frame.shape == (tr["height_px"], tr["width_px"])
                and float(np.asarray(frame).max()) > 200
                and float(np.asarray(frame).min()) < 60
                and mask.shape == frame.shape and on_line > 0.9
                and margin_max == 0.0 and cover < 0.1)
    print("synthetic carpet           : {} (3 frames rendered, truth centreline "
          "matches the table to {:.1e} um; mask {:.2f} on the line, covering "
          "{:.1f}% of the frame)".format("OK" if good else "FAILED", drift,
                                         on_line, 100 * cover))
    ok &= good

    # 10a. Burnt-in graphics: a black corner triangle painted into every
    #      frame of a rendered lawn is found as a marker, left out of
    #      ``valid``, and the surface comes out where it does without it
    #      (CO121: a 66-pixel triangle set the dark threshold, no tissue
    #      was dark enough, and Auto fit found "no periodic motion").
    with tempfile.TemporaryDirectory() as tmp:
        tr = render_carpet(tmp, n_frames=40, n_cilia=12, ppum=12.0, fps=300.0,
                           cbf=15.0, spacing=0.5, wavelength=1e5, blur=1.5,
                           noise=0.03, jitter=0.03, scatter=1.0, bright=0.5,
                           body=0.35, seed=2)
        info = scan_image_folder(tmp)
        pqd_s = default_pqd_for_images(info)
        video_c = VideoFrames.from_pqd(pqd_s, tmp)
        _, mean_c = motion_and_mean(video_c)
        surf_clean = detect_surface(static_mean(video_c, mean_c), valid=video_c.valid)
        from PIL import Image as _Image
        for name in tr["frames"]:
            path = os.path.join(tmp, name)
            a = np.asarray(_load_image(path)).astype(np.float32)
            if a.ndim == 3:
                a = a[..., :3].mean(axis=2)
            yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
            tri = (yy + (a.shape[1] - 1 - xx)) < 12          # top-right corner
            a[tri] = 0.0
            _Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(path)
        video_m = VideoFrames.from_pqd(pqd_s, tmp)
        mk = video_m.markers()
        n_tri = int(tri.sum())
        _, mean_m = motion_and_mean(video_m)
        surf_naive = detect_surface(static_mean(video_m, mean_m))
        surf_marked = detect_surface(static_mean(video_m, mean_m), valid=video_m.valid)
        off_naive = float(np.median(np.abs(surf_naive - surf_clean)))
        off_marked = float(np.median(np.abs(surf_marked - surf_clean)))
        # (a pixel the noise clips to white in every frame is constant and
        # extreme too -- scattered and harmless, and rare over 40 frames)
        n_extra = int((mk & ~tri).sum()) if mk is not None else -1
        good = (mk is not None and bool(mk[tri].all()) and bool(mk[0, -1])
                and n_extra <= 0.01 * tri.size and not video_m.valid[0, -1]
                and off_marked <= 1.0 and off_naive >= 1.0)
    print("burnt-in marker            : {} (a {}-pixel black corner found as a "
          "marker, {} clipped pixels besides; surface off by {:.1f} rows with "
          "the corner counted as image, {:.1f} with it left out)".format(
              "OK" if good else "FAILED", n_tri, n_extra, off_naive, off_marked))
    ok &= good

    # 11. The joint image fit: on a small rendered carpet the analytic
    #     gradient of the energy agrees with finite differences, and the
    #     true beat, phase-aligned, scores better than the same beat half a
    #     cycle out of phase.
    with tempfile.TemporaryDirectory() as tmp:
        tr = render_carpet(tmp, n_frames=24, n_cilia=6, ppum=12.0, fps=240.0,
                           cbf=10.0, spacing=3.0, wavelength=12.0)
        info = scan_image_folder(tmp)
        pqd_s = default_pqd_for_images(info)
        pqd_s.set("vid_ppum", tr["ppum"])
        pqd_s.set("vid_fps", tr["fps"])
        pqd_s.set("cbf", tr["cbf_hz"])
        pqd_s.set("length", tr["length_um"])
        pqd_s.set("vid_cil_x0", tr["reference_base_px"][0])
        pqd_s.set("vid_cil_y0_invy", tr["height_px"] - tr["reference_base_px"][1])
        video_s = VideoFrames.from_pqd(pqd_s, tmp)
        model_s = WaveModel(pqd_s)
        bif = BeatImageFit(video_s, model_s, sigma=2.0, n_frames=24)
        truth_vec = _fourier_vec(load_fourier_coeffs(
            os.path.join(tmp, "truth_coeffs.txt")))
        e_al, v_al, _ = bif.best_phase_shift(truth_vec, n=48)
        e_off = bif.energy(bif.normalise(_fourier_vec(shift_fourier_phase(
            _fourier_dict(v_al), 0.5))))
        e0, parts, g = bif.evaluate(v_al, grad=True)
        idx = np.where(bif.free_mask())[0]
        rng = np.random.default_rng(3)
        agree = []
        for pidx in rng.choice(idx, 8, replace=False):
            hh = 1e-5
            vp = v_al.copy()
            vp[pidx] += hh
            vm = v_al.copy()
            vm[pidx] -= hh
            fd = (bif.energy(vp) - bif.energy(vm)) / (2 * hh)
            agree.append(abs(fd - g[pidx]) <= 0.25 * max(abs(fd), abs(g[pidx]), 0.05))
        good = e_al > e_off and e_al > 0 and sum(agree) >= 6
    print("joint image fit            : {} (aligned truth scores {:+.2f}, half a "
          "cycle off {:+.2f}; gradient agrees with finite differences on {} of 8 "
          "coefficients)".format("OK" if good else "FAILED", e_al, e_off, sum(agree)))
    ok &= good

    # 11a. The sweep penalty: the published beat has one stroke each way
    #      and sweeps right, so it costs nothing with the setting "right"
    #      and something with "left"; a beat with back-and-forth in it is
    #      penalised; the gradient of both terms matches finite differences
    #      exactly (they are functions of the beat alone, including the
    #      way the equal-arc-length samples slide: arc_slide).
    with tempfile.TemporaryDirectory() as tmp:
        tr = render_carpet(tmp, n_frames=12, n_cilia=3, ppum=10.0, fps=120.0,
                           cbf=10.0, spacing=3.0, wavelength=12.0)
        info = scan_image_folder(tmp)
        pqd_s = default_pqd_for_images(info)
        pqd_s.set("vid_ppum", tr["ppum"])
        pqd_s.set("vid_fps", tr["fps"])
        pqd_s.set("cbf", tr["cbf_hz"])
        pqd_s.set("length", tr["length_um"])
        pqd_s.set("vid_cil_x0", tr["reference_base_px"][0])
        pqd_s.set("vid_cil_y0_invy", tr["height_px"] - tr["reference_base_px"][1])
        video_s = VideoFrames.from_pqd(pqd_s, tmp)
        model_s = WaveModel(pqd_s)
        bif = BeatImageFit(video_s, model_s, sigma=2.0, n_frames=12,
                           lam_mono=1.0, direction=1)
        v_t = bif.normalise(_fourier_vec(FOURIER_DEFAULT))
        ex_r, sh_r = bif.sweep_terms(v_t)
        bif.direction = -1
        ex_l, sh_l = bif.sweep_terms(v_t)
        # the same beat with a wobble: a high harmonic added to x
        rng = np.random.default_rng(7)
        v_w = v_t.copy()
        idx = np.where(bif.free_mask())[0]
        v_w[idx] += 0.15 * np.abs(v_t[idx]).mean() * rng.standard_normal(len(idx))
        v_w = bif.normalise(v_w)
        ex_w, sh_w, g_w = bif.sweep_terms(v_w, grad=True)
        agree = []
        for pidx in rng.choice(idx, 6, replace=False):
            hh = 1e-5
            vp, vm = v_w.copy(), v_w.copy()
            vp[pidx] += hh
            vm[pidx] -= hh
            fd = (sum(bif.sweep_terms(vp)) - sum(bif.sweep_terms(vm))) / (2 * hh)
            agree.append(abs(fd - g_w[pidx]) <= 1e-3 * max(abs(fd), abs(g_w[pidx]), 0.01))
        rep = sweep_report(FourierWave(FOURIER_DEFAULT, length=6.0, cbf=10.0))
        marks_l = recovery_start_phase(FourierWave(FOURIER_DEFAULT, length=6.0,
                                                   cbf=10.0), prefer=-1)
        good = (ex_r == 0.0 and sh_r == 0.0 and ex_l == 0.0 and sh_l > 0.1
                and ex_w > 0.01 and sum(agree) == len(agree)
                and rep["excess"] == 0.0 and rep["reversals"] == 0
                and marks_l["direction"] == -1 and marks_l["decided_by"] == "setting")
    print("sweep penalty              : {} (published beat: excess {:.3f}, "
          "shortfall {:.3f} sweeping right, {:.3f} if held to sweep left; "
          "wobbled beat: excess {:.3f}; gradient matches finite differences "
          "on {} of {})".format("OK" if good else "FAILED", ex_r, sh_r, sh_l,
                                ex_w, sum(agree), len(agree)))
    ok &= good

    # 11b. The lawn: on a rendered lawn beating in step, half its cilia
    #      bright and half dark, the polarity is judged "either", the lawn
    #      is judged in step, the band's envelope follows the true height
    #      of the beat, and the energy with the envelope term still has the
    #      right gradient.
    with tempfile.TemporaryDirectory() as tmp:
        tr = render_carpet(tmp, n_frames=40, n_cilia=24, ppum=12.0, fps=300.0,
                           cbf=15.0, spacing=0.5, wavelength=1e5, blur=1.5,
                           noise=0.03, jitter=0.03, scatter=1.0, bright=0.5,
                           body=0.35, seed=5)
        info = scan_image_folder(tmp)
        pqd_s = default_pqd_for_images(info)
        pqd_s.set("vid_ppum", tr["ppum"])
        pqd_s.set("vid_fps", tr["fps"])
        pqd_s.set("cbf", tr["cbf_hz"])
        pqd_s.set("length", tr["length_um"])
        pqd_s.set("vid_cil_x0", tr["reference_base_px"][0])
        pqd_s.set("vid_cil_y0_invy", tr["height_px"] - tr["reference_base_px"][1])
        video_s = VideoFrames.from_pqd(pqd_s, tmp)
        model_s = WaveModel(pqd_s)
        wave_t = FourierWave(load_fourier_coeffs(os.path.join(tmp, "truth_coeffs.txt")),
                             length=tr["length_um"], cbf=tr["cbf_hz"])
        pol = polarity_of(video_s, model_s)
        sync = lawn_synchrony(video_s, model_s)
        env, _bg, _band = band_envelope(video_s, model_s)
        true_h = np.array([wave_t.shape_at((b + 0.5) / len(env), n_pts=200)[1].max()
                           for b in range(len(env))])
        env_err = float(np.sqrt(np.mean((env - true_h) ** 2)))
        bif = BeatImageFit(video_s, model_s, sigma=2.0, n_frames=20, sign=0,
                           envelope=env)
        vec = bif.normalise(_fourier_vec(FOURIER_DEFAULT))
        e0, parts, g = bif.evaluate(vec, grad=True)
        idx = np.where(bif.free_mask())[0]
        rng = np.random.default_rng(5)
        agree = []
        for pidx in rng.choice(idx, 8, replace=False):
            hh = 1e-5
            vp = vec.copy()
            vp[pidx] += hh
            vm = vec.copy()
            vm[pidx] -= hh
            fd = (bif.energy(vp) - bif.energy(vm)) / (2 * hh)
            agree.append(abs(fd - g[pidx]) <= 0.25 * max(abs(fd), abs(g[pidx]), 0.05))
        good = (pol["sign"] == 0.0 and sync["synchronous"] and env_err < 0.6
                and parts["env"] > 0 and sum(agree) >= 6)
    print("lawn judgement             : {} (polarity {}, skew {:+.2f}; phase spread "
          "{:.2f} cycles; envelope within {:.2f} um of the true height; gradient "
          "with the envelope term agrees on {} of 8)".format(
              "OK" if good else "FAILED",
              {-1.0: "dark", 1.0: "bright"}.get(pol["sign"], "either"),
              pol["skew"], sync["spread"], env_err, sum(agree)))
    ok &= good

    # 12. The compute backend: the tracer gives the same curves spread over
    #     the CPU cores, or on the GPU, as on one core; on a GPU the period
    #     search agrees with the CPU's to float32 rounding.
    rng = np.random.default_rng(11)
    jobs = []
    for i in range(4):
        img = (0.3 * rng.random((40, 44))).astype(np.float32)
        img[:, 20 + i:22 + i] += 1.0
        jobs.append(((img, (21.0 + i, 36.0), 28.0,
                      np.deg2rad([-100.0, -80.0])),
                     dict(row_limit=np.full(44, 37.0), len_prior=28.0)))
    single = [trace_minimal_path(*a, **k)[0] for a, k in jobs]
    spread = trace_many_minimal_paths(jobs)
    good = all(np.array_equal(x, y) for x, y in zip(single, spread))
    where = describe_compute()
    cp = _gpu()
    if cp is not None:
        if not _GPU_KERNELS.get("minpath_ok", False):
            where += "; GPU tracing disabled, it disagreed with the CPU"
        F = rng.standard_normal((30, 16, 18)).astype(np.float32)
        norm = float(np.mean(F ** 2))
        taus = [6.0, 7.35, 11.5]
        on_gpu = _lag_dissimilarity(F, norm)(taus)
        on_cpu = []
        for t in taus:
            f0 = int(np.floor(t))
            frac = t - f0
            b = (1 - frac) * F[f0:-1] + frac * F[f0 + 1:] if frac else F[f0:]
            on_cpu.append(float(np.mean((F[:len(b)] - b) ** 2)) / norm)
        lag_ok = np.allclose(on_gpu, on_cpu, rtol=1e-4)
        where += "; GPU period search {}".format(
            "agrees" if lag_ok else "DISAGREES")
        good &= bool(lag_ok)
    print("compute backend            : {} ({}; traced curves identical to a "
          "single-core run)".format("OK" if good else "FAILED", where))
    ok &= good

    # 13. The enhancement tools do what they say on a test pattern: blurred,
    #     noisy curved lines (a cilium's profile), a frame-to-frame lamp
    #     jitter on top of a slow swing, and a JPEG-compressed frame.
    good, notes = enhancement_selfcheck()
    print("enhancement tools          : {} ({})".format(
        "OK" if good else "FAILED", notes))
    ok &= good

    # 14. A neural network from a file: a small blur network is written in
    #     the ONNX format, run through onnxruntime tile by tile, and must
    #     reproduce the same blur computed directly; the model's path must
    #     survive a specification round trip; a variant that refuses small
    #     pieces must be found out and still give the blur.  Skipped
    #     without onnxruntime.
    try:
        from importlib import util as _iu
        have_ort = _iu.find_spec("onnxruntime") is not None
    except Exception:
        have_ort = False
    verdict = onnxruntime_probe() if have_ort else None
    if not have_ort:
        print("AI model step              : SKIPPED (onnxruntime is not installed; "
              "pip install onnxruntime to use .onnx models)")
    elif verdict["ok"] is False:
        print("AI model step              : FAILED ({})".format(verdict["error"]))
        ok = False
    else:
        with tempfile.TemporaryDirectory() as tmp:
            mp = write_blur_onnx(os.path.join(tmp, "blur.onnx"), sigma=1.0)
            rng = np.random.default_rng(2)
            img = (rng.random((150, 170)) * 255).astype(np.float32)
            xk = np.arange(-2, 3)
            gk = np.exp(-xk * xk / 2.0)
            k2 = np.outer(gk, gk)
            k2 /= k2.sum()
            padded = np.pad(img, 2, mode="reflect")
            ref = np.zeros_like(img)
            for i in range(5):
                for j in range(5):
                    ref += k2[i, j] * padded[i:i + 150, j:j + 170]
            try:
                net = OnnxEnhancer(mp)
                net.tile = 64                      # several tiles, blended
                out = enhance_frame(img, dict(enhance_defaults(), model=mp),
                                    refs={"ai": net})
                err = float(np.abs(out - ref).max())
                p_rt = default_pqd_for_images(
                    {"width": 8, "height": 8, "basename": "x", "count": 1})
                enhance_to_pqd(p_rt, dict(enhance_defaults(), model=mp))
                back = enhance_from_pqd(Spec.from_text(p_rt.dumps()))
                # A model that refuses pieces under 200 px, like an exported
                # NAFNet under 384: found, and the small frame mirrored out.
                mp2 = write_blur_onnx(os.path.join(tmp, "blur_min.onnx"),
                                      sigma=1.0, min_side=200)
                net2 = OnnxEnhancer(mp2)
                out2 = enhance_frame(img, dict(enhance_defaults(), model=mp2),
                                     refs={"ai": net2})
                err2 = float(np.abs(out2 - ref).max())
                good = (err < 0.01 and back["model"] == mp and net.scale == 1
                        and err2 < 0.01 and net2.min_side == 256)
                note = ("{}; blur reproduced to {:.4f} grey levels over {} "
                        "tiles; path round-trips; a model refusing pieces under "
                        "200 px ran on {} px ones, blur to {:.4f}".format(
                            net.describe(), err, "4 x 4", net2.min_side, err2))
            except Exception as exc:
                good = False
                note = "failed: {}".format(exc)
        print("AI model step              : {} ({})".format(
            "OK" if good else "FAILED", note))
        ok &= good

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


def enhancement_selfcheck():
    """(ok, notes) for the classical enhancement tools on test patterns."""
    import io
    from PIL import Image
    rng = np.random.default_rng(1)
    h, w = 120, 160
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    clean = np.full((h, w), 200.0)
    for c0, curv in ((50, 0.004), (80, 0.006), (120, -0.003)):
        clean -= 80 * np.exp(-0.5 * ((xx - (c0 + curv * (yy - 60) ** 2))
                                      / 0.8) ** 2)
    blurred = gaussian_blur(clean, 2.0)
    noisy = (blurred + rng.normal(0, 4.0, clean.shape)).astype(np.float32)
    rows = np.arange(30, 90)
    xl = (80 + 0.006 * (rows - 60) ** 2).round().astype(int)

    def depth(img):
        return float(0.5 * (img[rows, xl - 6].mean() + img[rows, xl + 6].mean())
                     - img[rows, xl].mean())

    def noise(img):
        return float(np.std((np.asarray(img) - blurred)[5:25, 125:155]))

    d0, n0 = depth(noisy), noise(noisy)
    db = deblur_gaussian(noisy, 2.0)
    ls = line_smooth(noisy, 2.0)
    gb = gaussian_blur(noisy, 1.0)
    deblur_ok = depth(db) > 1.3 * d0
    line_ok = depth(ls) > 0.9 * d0 and noise(ls) < 0.4 * n0 \
        and depth(ls) > depth(gb)
    # flicker: a slow swing (kept) plus frame-to-frame jitter (removed)
    t = np.arange(60)
    swing = 3.0 * np.sin(2 * np.pi * t / 30.0)
    jitter = rng.normal(0, 2.0, len(t))
    stack = np.array([clean * (1 + (s_ + j) / 200.0) for s_, j in zip(swing, jitter)],
                     dtype=np.float32)
    mean, still = flicker_reference(stack)
    params = flicker_params(stack, mean, still)
    fixed = np.array([deflicker_frame(a, params, i) for i, a in enumerate(stack)])
    flick_ok = (frame_flicker(fixed) < 0.35 * frame_flicker(stack)
                and np.std(_running_mean(fixed[:, still].mean(axis=1), 5))
                > 0.5 * np.std(_running_mean(stack[:, still].mean(axis=1), 5)))
    # JPEG blocks
    buf = io.BytesIO()
    Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), "L").save(
        buf, "JPEG", quality=60)
    buf.seek(0)
    with Image.open(buf) as im:
        q = [int(v) for v in im.quantization[0]]
        jpg = np.asarray(im.convert("L"), dtype=np.float32)
    deb = jpeg_deblock(jpg, q, step=2)
    block_ok = jpeg_blockiness([deb]) < jpeg_blockiness([jpg])
    ref = clahe_reference([noisy], 2.0)
    clahe_ok = bool(np.all(np.diff(ref["luts"], axis=2) >= -1e-3))
    ramp = (xx * 0.5 + yy * 0.25).astype(np.float32)
    lz = resample_lanczos(ramp, 2.0)
    lanczos_ok = float(np.abs(lz - resample_frame(ramp, 2.0))[8:-8, 8:-8].max()) < 0.05
    # Fiji's tools.  The line filter: the lines drawn dark, the empty
    # corner left white.  Top-hat and bandpass on the pattern with a
    # brightness ramp across it: the ramp goes, the lines keep most of
    # their depth.  The median background: a line that sweeps across a
    # static scene leaves no smear in the median, and does in the mean.
    flat = (slice(5, 25), slice(125, 155))
    ld = line_drawing(noisy, 2.5)
    lines_ok = float(ld[rows, xl].mean()) < 60 and float(ld[flat].mean()) > 230
    tilted = (noisy + ramp - ramp.mean()).astype(np.float32)
    left, right = (slice(5, 25), slice(5, 35)), (slice(95, 115), slice(125, 155))
    th = tophat_drawing(tilted, 5.0)       # the pattern's lines are 7 px wide
    bp = bandpass_filter(tilted, 2.0, 40.0)
    ramp_in = abs(float(tilted[right].mean() - tilted[left].mean()))
    tophat_ok = (depth(th) > 0.6 * d0
                 and abs(float(th[right].mean() - th[left].mean())) < 0.1 * ramp_in)
    bandpass_ok = (depth(bp) > 0.7 * d0 and noise(bp) < n0
                   and abs(float(bp[right].mean() - bp[left].mean())) < 0.1 * ramp_in)
    scene = blurred.astype(np.float32)
    sweep = []
    for i in range(20):
        fr = scene.copy()
        col = 20 + 6 * i
        fr[:, col:col + 2] -= 80.0
        sweep.append(fr)
    med_bg = np.median(np.stack(sweep), axis=0)
    mean_bg = np.mean(np.stack(sweep), axis=0)
    median_ok = (float(np.abs(med_bg - scene).max()) < 1.0
                 and float(np.abs(mean_bg - scene).max()) > 3.0)
    good = all((deblur_ok, line_ok, flick_ok, block_ok, clahe_ok, lanczos_ok,
                lines_ok, tophat_ok, bandpass_ok, median_ok))
    notes = ("deblur lifts line contrast x{:.2f}; line denoise keeps {:.0f}% of "
             "it at {:.0f}% of the noise (Gaussian keeps {:.0f}%); deflicker "
             "jitter {:.2f} -> {:.2f}; deblock {:.2f} -> {:.2f}{}; line filter "
             "draws the lines at {:.0f} on white {:.0f}; top-hat keeps {:.0f}% "
             "and bandpass {:.0f}% of the depth under a ramp; median background "
             "off by {:.2f} (mean {:.1f}){}".format(
                 depth(db) / d0, 100 * depth(ls) / d0, 100 * noise(ls) / n0,
                 100 * depth(gb) / d0, frame_flicker(stack),
                 frame_flicker(fixed), jpeg_blockiness([jpg]),
                 jpeg_blockiness([deb]),
                 "" if (clahe_ok and lanczos_ok) else "; CLAHE/Lanczos FAILED",
                 float(ld[rows, xl].mean()), float(ld[flat].mean()),
                 100 * depth(th) / d0, 100 * depth(bp) / d0,
                 float(np.abs(med_bg - scene).max()),
                 float(np.abs(mean_bg - scene).max()),
                 "" if (lines_ok and tophat_ok and bandpass_ok and median_ok)
                 else "; Fiji tools FAILED"))
    return good, notes


def cmd_cycle(args) -> int:
    """Estimate the beat period and the main cycle's first/last frame."""
    pqd_path, images_dir = resolve_paths(args)
    for path, what in ((pqd_path, "specification file"), (images_dir, "images folder")):
        if not os.path.exists(path):
            print("error: {} not found at {}".format(what, path), file=sys.stderr)
            return 2

    pqd = Spec(pqd_path)
    model = WaveModel(pqd)
    video = VideoFrames.from_pqd(pqd, images_dir)

    res = detect_cycle(video, model=model)
    period = res["period"]
    fps = model.fps
    cbf = fps / period if period > 0 else float("nan")

    old_start = model.macysta
    cands = res["candidates"] or [int(round(old_start))]
    start = min(cands, key=lambda f: abs(f - old_start))

    print("frames        : {} .. {}   ({:.1f} beat cycles in the clip)".format(
        video.first, video.last, res["n_cycles"]))
    print()
    print("detected period : {:.2f} frames at {:g} fps -> cbf {:.4f} Hz".format(
        period, fps, cbf))
    print("current in file : {:.2f} frames               -> cbf {:.4f} Hz".format(
        model.period_frames, model.cbf))
    print("difference      : {:+.2f} frames ({:+.1f}%)".format(
        period - model.period_frames,
        100 * (period - model.period_frames) / model.period_frames
        if model.period_frames else float("nan")))
    print()
    print("the main cycle runs recovery -> effective, so both end points sit")
    print("at the end-effective position, one beat apart.")
    print()
    print("cycle start candidates (end-effective, i.e. recovery begins):")
    print("  {}".format(", ".join(str(c) for c in cands)))
    if res.get("source") == "traced tip":
        print("  from the traced cilium: the effective stroke sweeps {}; the "
              "tip is fast for {:.0f}% of the period; decided by {} "
              "(ratio {:.2f})".format(
                  "right" if res["direction"] > 0 else "left",
                  100 * res["duration_effective"], res["decided_by"],
                  res["speed_ratio"]))
        print("  the other turning point is at frames {}".format(
            ", ".join(str(c) for c in res.get("other_candidates", []))))
    else:
        print("  from the carpet centroid{} -- locate the base first for a "
              "landmark on the cilium itself".format(
                  " (trace failed: {})".format(res["trace_error"])
                  if res.get("trace_error") else ""))
    print("nearest the current start ({:g}): {}".format(old_start, start))
    print()
    print("suggested main cycle: first {}  last {:.1f}".format(
        start, start + period))

    if args.apply:
        pqd.set("vid_frame_main_cycle_start", float(start))
        if period > 0 and fps > 0:
            pqd.set("cbf", cbf)
        pqd.set("vid_suimvi_first", int(round(start)))
        pqd.set("vid_suimvi_last", int(round(start + period)))
        written = pqd.save(pqd_path, backup=True)
        print()
        print("applied to {}{}".format(
            written, " (a .bak of the previous file was made)"
            if pqd.last_backup else ""))
    else:
        print()
        print("not applied -- add --apply to write it into the specification")
    return 0


def cmd_length(args) -> int:
    """Estimate the cilium length from how far the beating reaches."""
    pqd_path, images_dir = resolve_paths(args)
    for path, what in ((pqd_path, "specification file"), (images_dir, "images folder")):
        if not os.path.exists(path):
            print("error: {} not found at {}".format(what, path), file=sys.stderr)
            return 2

    pqd = Spec(pqd_path)
    model = WaveModel(pqd)
    video = VideoFrames.from_pqd(pqd, images_dir)

    res = estimate_length(video, model)
    print("estimated length : {:.2f} um".format(res["length"]))
    print("current in file  : {:.2f} um".format(res["current"]))
    print("plausible range  : {:.1f} .. {:.1f} um".format(res["low"], res["high"]))
    print()
    print("measured outwards from the base at ({:.0f}, row {:.0f}); the cell "
          "surface".format(res["base_x0"], res["base_row"]))
    print("in that column is at row {:.0f}, i.e. the base is {:.2f} um {} it."
          .format(res["surface_row"], abs(res["base_above_surface_um"]),
                  "above" if res["base_above_surface_um"] > 0 else "below"))
    if abs(res["base_above_surface_um"]) > 0.5:
        print("A base off the surface shifts this estimate directly -- run "
              "`locate` first.")
    print()
    print("This is measured from how far the beating reaches, and it is only")
    print("weakly constrained: neighbouring cilia touch and the contrast is")
    print("low, so a single cilium cannot be segmented reliably. Confirm it")
    print("against the image -- 'Click tip' in the GUI length tab is better.")

    if args.apply:
        pqd.set("length", res["length"])
        written = pqd.save(pqd_path, backup=True)
        print()
        print("applied to {}{}".format(
            written, " (a .bak of the previous file was made)"
            if pqd.last_backup else ""))
    else:
        print()
        print("not applied -- add --apply to write it into the specification")
    return 0


def cmd_check(args) -> int:
    """Read every frame and report any that fail to decode."""
    # Deliberately does not go through resolve_paths: this is the command you
    # reach for when a folder will not load, so it must not first try to build
    # a specification from the very images that are failing.
    images_dir = args.images or (
        os.path.join(args.run_dir, "input_images") if args.run_dir else None)
    if images_dir is None:
        chosen = choose_folder("Select the folder of images", _recall_folder())
        if not chosen:
            print("no folder chosen; pass --images <folder>", file=sys.stderr)
            return 2
        images_dir = chosen
    images_dir = os.path.abspath(images_dir)
    if not os.path.isdir(images_dir):
        print("error: no such folder: {}".format(images_dir), file=sys.stderr)
        return 2

    print("checking {}".format(images_dir))
    try:
        res = check_images(images_dir, basename=args.prefix or "")
    except Exception as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2

    print("{} frame(s) found".format(res["n_frames"]))
    if not res["bad"]:
        print("all frames decoded cleanly")
        v = res["video"]
        img = np.asarray(v.get(v.first))
        print("first frame: {} {}, values {} .. {}".format(
            img.dtype, img.shape, img.min(), img.max()))
        if getattr(args, "model", None):
            # The model on one frame, outside the window: a network that
            # takes the program down does it here, where the console shows
            # what happened, and the time and memory it needs are printed.
            path = resolve_model_path(args.model, images_dir)
            print("checking onnxruntime and the model in a separate process...")
            if os.path.isfile(path):
                verdict = onnxruntime_probe(path)
                if verdict["ok"] is False:
                    print("error: {}".format(verdict["error"]), file=sys.stderr)
                    return 2
                if verdict["ok"]:
                    print("onnxruntime {} loads the model safely".format(
                        verdict["version"]))
            print("loading model {}".format(path))
            try:
                net = OnnxEnhancer(path, value_range=int(
                    getattr(args, "model_range", 0) or 0),
                    threads=_available_cpus())
            except Exception as exc:
                print("error: {}".format(exc), file=sys.stderr)
                return 2
            print("model: {}".format(net.describe()))
            grey = img[..., :3].mean(axis=2) if img.ndim == 3 else img
            t0 = time.time()
            out = net.run(np.asarray(grey, dtype=np.float32))
            dt = time.time() - t0
            try:
                import resource
                peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
                mem = ", peak memory {:.0f} MB".format(peak)
            except Exception:
                mem = ""
            print("one frame through the model: {:.1f} s, output {} values "
                  "{:.0f} .. {:.0f}{}".format(dt, out.shape, out.min(),
                                              out.max(), mem))
            print("all {} frames would take about {:.0f} min the first time; "
                  "outputs are then kept in {}".format(
                      res["n_frames"], dt * res["n_frames"] / 60.0,
                      model_cache_dir()))
        return 0

    print()
    print("{} frame(s) could not be read:".format(len(res["bad"])))
    for f, path, page, why in res["bad"][:20]:
        print("  frame {:>4}  {}{}\n            {}".format(
            f, os.path.basename(path),
            " page {}".format(page) if page else "", why))
    if len(res["bad"]) > 20:
        print("  ... and {} more".format(len(res["bad"]) - 20))
    print()
    hint = dependency_hint([why for _, _, _, why in res["bad"]])
    if hint:
        print("The frames are probably fine; a package is missing:" + hint)
    else:
        print("Pillow could not decode these even with its fallbacks. Re-saving")
        print("them uncompressed, or as PNG, is usually the quickest way out.")
    return 1


def read_any_trace(path: str):
    """Read a traced beat from either supported file format.

    Four columns (index, x, y, time) in real units, as the reference MATLAB
    script expects, or the six-column cilium_trace_px.csv written by
    trace_cilium_colored.py.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        head = fh.readline()
    if "x_px" in head or "s_index" in head:
        steps, curves, times_ms = load_trace_csv(path)
        return np.asarray(times_ms, dtype=float), curves, "pixels"
    times, curves = load_trace_xyt(path)
    return times, curves, "real units"


def cmd_beatfit(args) -> int:
    """Fit a beat pattern to a traced centreline file, the reference way."""
    path = args.trace
    if not os.path.exists(path):
        print("error: no such trace file: {}".format(path), file=sys.stderr)
        return 2
    try:
        times, curves, units = read_any_trace(path)
    except Exception as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2

    print("trace       : {} ({})".format(path, units))
    print("times       : {}  ({}..{} points each)".format(
        len(curves), min(len(c) for c in curves), max(len(c) for c in curves)))

    aligned, theta = align_beat_frame(curves)
    print("beat frame  : base direction {:+.1f} deg".format(
        np.rad2deg(theta + np.pi / 2)))

    per = args.per_period
    if per is None:
        samples, _ = resample_unit_length(aligned, args.locs)
        per = int(round(0.5 * (autocorr_period(samples[:, -1, 0])
                               + autocorr_period(samples[:, -1, 1]))))
        print("period      : {} samples (from the tip autocorrelation)".format(per))
    else:
        print("period      : {} samples (given)".format(per))

    modes = args.modes
    if modes is None:
        modes = choose_reference_modes(aligned, num_locs=args.locs,
                                       n_space=args.space)
        if modes:
            print("modes       : {} (chosen by held-out error)".format(modes))
    try:
        fit = fit_beat_reference(aligned, n_modes=modes, num_locs=args.locs,
                                 n_space=args.space, period_samples=per)
    except ValueError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2

    print()
    print("fitted      : {} time modes, s^1..s^{}, {} nodes".format(
        fit["n_modes"], args.space, args.locs))
    print("lengths     : {:.3f} .. {:.3f}, mean {:.3f}".format(
        fit["lengths"].min(), fit["lengths"].max(), fit["mean_length"]))
    print("residual    : {:.4f} of a cilium length = {:.4f} in trace units"
          .format(fit["resid"], fit["resid_um"]))

    wave = FourierWave(fit["coeffs"], length=1.0, cbf=1.0)
    s = np.linspace(0.0, 1.0, 80)
    turn = [abs(curve_turning(np.column_stack(wave.position(s, i / 24.0))))
            for i in range(24)]
    print("bending     : {:.0f} deg median, {:.0f} deg at its most bent"
          .format(np.median(turn), max(turn)))

    out = args.out or (os.path.splitext(path)[0] + "_coeffs.txt")
    save_fourier_coeffs(out, fit["coeffs"])
    print()
    print("coefficients written to {}".format(out))
    print("they describe a unit-length beat, so multiply by the cilium length")
    return 0


def autocorr_period(x) -> int:
    """Period of a series, in samples, from the first autocorrelation peak."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    acf = np.array([float(np.dot(x[:len(x) - n], x[n:])) for n in range(len(x))])
    if acf[0] <= 0:
        return len(x)
    acf = acf / acf[0]
    for n in range(1, len(acf) - 1):
        if acf[n - 1] < acf[n] and acf[n + 1] < acf[n]:
            return n
    return len(x)


def cmd_track(args) -> int:
    """Follow one cilium through every frame and write where it goes."""
    pqd_path, images_dir = resolve_paths(args)
    if not os.path.isdir(images_dir):
        print("error: images folder not found at {}".format(images_dir),
              file=sys.stderr)
        return 2
    pqd = Spec(pqd_path)
    video = VideoFrames.from_pqd(pqd, images_dir)
    model = WaveModel(pqd)
    x_base = (float(args.x) if args.x is not None
              else pqd.getf("vid_cil_x0", video.shape[1] / 2.0))
    length_px = max(6.0, model.length * model.ppum)
    print("images      : {} ({} frames, {}x{}{})".format(
        images_dir, len(video), video.shape[1], video.shape[0],
        "" if video.crop is None else ", cropped to {}x{} at ({}, {})".format(
            video.crop[2] - video.crop[0], video.crop[3] - video.crop[1],
            video.crop[0], video.crop[1])))
    print("cilium      : base column {:.0f}, {:.2f} um = {:.0f} px".format(
        x_base, model.length, length_px))

    last = [-1.0]

    def progress(frac, message):
        # Every frame would be a few hundred lines of it.
        if frac >= last[0] + 0.05 or frac >= 1.0 > last[0]:
            last[0] = frac
            print("  {:3.0f}%  {}".format(100.0 * frac, message), flush=True)

    run = track_cilium(video, x_base, length_px, ppum=model.ppum,
                       progress=progress)
    frames = run["frames"]
    curves_px = [run["curves"][f] for f in frames]
    fps = model.fps or 1.0
    times = [f / fps * 1000.0 for f in frames]
    lengths = run["lengths"] / max(model.ppum, 1e-9)
    print()
    print("traced      : {} frames, {} tracer, {} map".format(
        len(frames), "minimal path",
        "darkness" if run["drawn"] else "mean-subtracted"))
    print("arc length  : {:.2f} .. {:.2f} um, median {:.2f}".format(
        lengths.min(), lengths.max(), float(np.median(lengths))))
    tips = np.array([c[-1] for c in curves_px], dtype=float)
    hop = np.hypot(*np.diff(tips, axis=0).T) / max(model.ppum, 1e-9)
    print("tip motion  : {:.3f} um between frames typically, {:.3f} at most"
          .format(float(np.median(hop)), float(hop.max())))

    out_dir = args.outdir or (os.path.dirname(pqd_path) or ".")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, args.stem)
    csv_path = save_trace_xyt(stem + "_track.csv", curves_px, times,
                              model.ppum)
    print()
    print("centrelines written to {}".format(csv_path))

    # The waveform, fitted the reference way to the curves just traced.
    um = []
    for curve in curves_px:
        pts = np.asarray(curve, dtype=float)
        um.append(np.column_stack([(pts[:, 0] - pts[0, 0]) / model.ppum,
                                   (pts[0, 1] - pts[:, 1]) / model.ppum]))
    aligned, theta = align_beat_frame(um)
    # The period, from the signed sideways position of the tip.  Averaging
    # that with the along-beat coordinate, as the reference script does, is
    # wrong here: aligning takes the across-beat coordinate as a distance
    # rather than a signed offset, so its series is folded and its
    # autocorrelation answered 16 frames against the true 82, and the mean of
    # the two put the period at 60 and the residual up by half.
    per = args.per_period
    if per is None:
        cbf = pqd.getf("cbf", 0.0)
        if cbf > 0:
            per = int(round(fps / cbf))
    if per is None:
        per = autocorr_period(np.array([c[-1, 0] for c in um]))
    modes = args.modes
    if modes is None:
        modes = choose_reference_modes(aligned, num_locs=args.locs,
                                       n_space=args.space)
    try:
        fit = fit_beat_reference(aligned, n_modes=modes, num_locs=args.locs,
                                 n_space=args.space, period_samples=per)
    except ValueError as exc:
        print("could not fit a waveform: {}".format(exc), file=sys.stderr)
        return 0
    print()
    print("beat frame  : base direction {:+.1f} deg".format(
        np.rad2deg(theta + np.pi / 2)))
    print("period      : {} frames = {:.1f} Hz".format(
        per, fps / max(per, 1)))
    print("fitted      : {} time modes, s^1..s^{}, {} nodes".format(
        fit["n_modes"], args.space, args.locs))
    print("residual    : {:.4f} of a length = {:.4f} um".format(
        fit["resid"], fit["resid_um"]))
    wave = FourierWave(fit["coeffs"], length=1.0, cbf=1.0)
    turn = [abs(curve_turning(np.column_stack(wave.position(
        np.linspace(0.0, 1.0, 80), i / 24.0)))) for i in range(24)]
    print("bending     : {:.0f} deg median, {:.0f} deg at its most bent"
          .format(np.median(turn), max(turn)))
    coeff_path = save_fourier_coeffs(stem + "_coeffs.txt", fit["coeffs"])
    print("coefficients written to {}".format(coeff_path))

    if not args.no_figure:
        try:
            _load_pyplot(interactive=False)
        except Exception as exc:
            print("no figure: {}".format(exc))
            return 0
        n_show = min(int(args.figure_frames), len(frames))
        pick = np.linspace(0, len(frames) - 1, n_show).astype(int)
        cols = min(6, n_show)
        rows = int(np.ceil(n_show / cols))
        c0, r0, c1, r1 = run["window"]
        fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 3.0 * rows),
                                 squeeze=False)
        for ax_, j in zip(axes.ravel(), pick):
            f = frames[j]
            img = np.asarray(video.get(f), dtype=float)
            if img.ndim == 3:
                img = img[..., :3].mean(axis=2)
            pts = curves_px[j]
            ax_.imshow(img, cmap="gray")
            ax_.plot(pts[:, 0], pts[:, 1], "-", color="#39ff88", lw=1.4)
            ax_.plot([pts[0, 0]], [pts[0, 1]], "o", color="#ff3030", ms=4)
            ax_.set_xlim(c0, c1)
            ax_.set_ylim(r1, r0)
            ax_.set_title("frame {}".format(f), fontsize=8)
            ax_.set_xticks([])
            ax_.set_yticks([])
        for ax_ in axes.ravel()[n_show:]:
            ax_.axis("off")
        fig.tight_layout()
        fig_path = stem + "_track.png"
        fig.savefig(fig_path, dpi=110)
        plt.close(fig)
        print("overlay written to {}".format(fig_path))
    return 0


# ---------------------------------------------------------------------------
# Synthetic carpets with exact ground truth, and validation against them
# ---------------------------------------------------------------------------
#
# The only way to know what the pipeline gets right is to run it on frames
# whose answer is known exactly.  render_carpet draws a metachronal carpet
# from a Fourier table -- the published one, or any coefficient file -- and
# writes the truth beside the frames: the table itself, every parameter, and
# the reference cilium's centreline in every frame.  validate_folder then runs
# the same steps the buttons run and scores each against that truth.


def _splat_polyline(canvas, pts, radius: float):
    """Paint a polyline of half-width ``radius`` (pixels) into a 0/1 canvas."""
    h, w = canvas.shape
    seg = np.diff(pts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    n_sub = np.maximum(1, np.ceil(seg_len / 0.5).astype(int))
    xs, ys = [], []
    for (x0, y0), (dx, dy), n in zip(pts[:-1], seg, n_sub):
        t = np.arange(n) / float(n)
        xs.append(x0 + t * dx)
        ys.append(y0 + t * dy)
    xs.append(pts[-1:, 0])
    ys.append(pts[-1:, 1])
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    r = int(np.ceil(radius))
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
    keep = (ox * ox + oy * oy) <= radius * radius + 1e-9
    ox, oy = ox[keep], oy[keep]
    px = (np.rint(xs).astype(int)[:, None] + ox[None, :]).ravel()
    py = (np.rint(ys).astype(int)[:, None] + oy[None, :]).ravel()
    ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    canvas[py[ok], px[ok]] = 1.0


def render_carpet(out_dir: str, coeffs: dict | None = None,
                  length: float = 6.0, ppum: float = 40.0, cbf: float = 10.0,
                  fps: float = 1000.0, n_frames: int = 300, n_cilia: int = 40,
                  spacing: float = 0.3, wavelength: float = 8.0,
                  wave_direction: int = 1, line_width: float = 2.0,
                  margin: float = 1.0, wall_margin: float = 0.6,
                  blur: float = 0.0, noise: float = 0.0, seed: int = 0,
                  basename: str = "syn", supersample: int = 4,
                  jitter: float = 0.0, scatter: float = 0.0,
                  bright: float = 0.0, body: float = 0.0,
                  masks: bool = False, mask_width: float = 2.0,
                  progress=None) -> dict:
    """Render a metachronal carpet of identical cilia and write its truth.

    ``masks`` also writes, in a ``masks`` subfolder, a picture of every
    frame's answer: the centreline of every cilium, ``mask_width`` pixels
    wide, white on black (0..255, anti-aliased like the frame), whatever
    the cilium's polarity, blur or noise -- the target for training a
    network to find cilia (train_cilia_net.py).

    ``jitter`` gives every cilium but the reference one a fixed random
    phase offset (standard deviation, in cycles) and ``scatter`` moves its
    base by a random fraction of the spacing: a lawn seen side-on, where
    cilia behind one another project to random positions and do not beat
    in perfect step, has texture instead of a solid block.  ``bright`` is
    the fraction of cilia drawn bright on a mid-grey field instead of dark
    on white, as bright-field optics show cilia dark or bright with the
    focus, so that the polarity is ambiguous; ``body`` darkens the
    tissue below the wall by that fraction of full scale, as a real cell
    layer is dark, so that the surface can be found as it is on a
    recording.

    Every cilium beats the same pattern; cilium i at base x_i lags by
    x_i / wavelength of a cycle (``wave_direction`` -1 reverses the travel).
    Frames are 8-bit greyscale PNGs, ink dark on white, drawn ``supersample``
    times oversize and averaged down so the lines are anti-aliased the way a
    plotting library draws them.  ``blur`` (pixels) and ``noise`` (fraction
    of full scale, Gaussian) add optics and camera if wanted; both zero gives
    a clean drawing.

    Written beside the frames: ``truth_coeffs.txt`` (the table, in the file
    format the fitter reads and writes), ``truth.json`` (every parameter and
    the base of every cilium), and ``truth_track.csv`` (the reference
    cilium's centreline in every frame, in the four-column form
    ``track`` writes, so the two can be compared directly).  The reference
    cilium is the middle one.
    """
    import json
    wave = FourierWave(coeffs or FOURIER_DEFAULT, length=length, cbf=cbf)
    os.makedirs(out_dir, exist_ok=True)
    bases_um = margin + length + np.arange(n_cilia) * spacing
    width_um = bases_um[-1] + length + margin
    height_um = length + margin + wall_margin
    w_px = int(round(width_um * ppum))
    h_px = int(round(height_um * ppum))
    wall_row = (length + margin) * ppum          # y down; base sits here
    ref = n_cilia // 2
    period_frames = fps / cbf
    ss = int(max(1, supersample))
    rng = np.random.default_rng(seed)
    phase_off = np.zeros(n_cilia)
    draw_at = bases_um.copy()
    if jitter > 0 or scatter > 0:
        phase_off = rng.normal(0.0, float(jitter), n_cilia)
        draw_at = bases_um + rng.uniform(-0.5, 0.5, n_cilia) * float(scatter) * spacing
        phase_off[ref] = 0.0
        draw_at[ref] = bases_um[ref]
    is_bright = np.zeros(n_cilia, dtype=bool)
    if bright > 0:
        is_bright = rng.random(n_cilia) < float(bright)

    n_pts = int(max(60, min(400, round(length * ppum * 0.6))))
    track_rows = []
    frame_names = []
    mask_dir = os.path.join(out_dir, "masks")
    if masks:
        os.makedirs(mask_dir, exist_ok=True)
    for f in range(n_frames):
        if progress is not None and f % 10 == 0:
            progress(f / float(n_frames), "rendering frame {}".format(f))
        canvas = np.zeros((h_px * ss, w_px * ss), dtype=np.float32)
        canvas_b = np.zeros_like(canvas) if bright > 0 else None
        canvas_m = np.zeros_like(canvas) if masks else None
        t_cycle = f / period_frames
        for i, xb in enumerate(draw_at):
            phase = (t_cycle - wave_direction * bases_um[i] / wavelength
                     + phase_off[i]) % 1.0
            x, z = wave.shape_at(phase, n_pts=n_pts)
            px = (xb + x) * ppum
            py = wall_row - z * ppum
            pts = np.column_stack([px, py]) * ss
            _splat_polyline(canvas_b if is_bright[i] else canvas, pts,
                            0.5 * line_width * ss)
            if canvas_m is not None:
                _splat_polyline(canvas_m, pts, 0.5 * mask_width * ss)
            if i == ref:
                t_ms = f / fps * 1000.0
                for k, (xx, zz) in enumerate(zip(x, z)):
                    track_rows.append((k + 1, xx, zz, t_ms))
        cover = canvas.reshape(h_px, ss, w_px, ss).mean(axis=(1, 3))
        img = 1.0 - cover
        if canvas_b is not None:
            cover_b = canvas_b.reshape(h_px, ss, w_px, ss).mean(axis=(1, 3))
            img = 0.6 - 0.5 * cover + 0.4 * cover_b
        if body > 0:
            img[int(round(wall_row)):, :] -= float(body)
        if blur > 0:
            img = gaussian_blur(img, blur)
        if noise > 0:
            img = img + rng.normal(0.0, noise, img.shape)
        img8 = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
        name = "{}{:04d}.png".format(basename, f)
        from PIL import Image
        Image.fromarray(img8, mode="L").save(os.path.join(out_dir, name))
        frame_names.append(name)
        if canvas_m is not None:
            cover_m = canvas_m.reshape(h_px, ss, w_px, ss).mean(axis=(1, 3))
            m8 = np.clip(np.rint(cover_m * 255.0), 0, 255).astype(np.uint8)
            Image.fromarray(m8, mode="L").save(os.path.join(mask_dir, name))

    save_fourier_coeffs(os.path.join(out_dir, "truth_coeffs.txt"), wave.coeffs)
    with open(os.path.join(out_dir, "truth_track.csv"), "w",
              encoding="utf-8") as fh:
        fh.write("index,x_um,y_um,time_ms\n")
        for k, xx, zz, t_ms in track_rows:
            fh.write("{:d},{:.5f},{:.5f},{:.4f}\n".format(k, xx, zz, t_ms))
    truth = {
        "length_um": length, "ppum": ppum, "cbf_hz": cbf, "fps": fps,
        "period_frames": period_frames, "n_frames": n_frames,
        "n_cilia": n_cilia, "spacing_um": spacing,
        "wavelength_um": wavelength, "wave_direction": wave_direction,
        "line_width_px": line_width, "blur_px": blur, "noise": noise,
        "jitter_cycles": jitter, "scatter": scatter,
        "bright_fraction": bright, "body": body,
        "masks": "masks" if masks else "", "mask_width_px": mask_width,
        "width_px": w_px, "height_px": h_px, "wall_row": wall_row,
        "bases_x_px": [float(b * ppum) for b in bases_um],
        "reference_cilium": ref,
        "reference_base_px": [float(bases_um[ref] * ppum), float(wall_row)],
        "reference_phase_at_frame0": float(
            (-wave_direction * bases_um[ref] / wavelength) % 1.0),
        "coefficients": "truth_coeffs.txt",
        "reference_track": "truth_track.csv",
        "basename": basename,
        "enter_in_ciliograph": {
            "ppum": ppum, "fps": fps,
            "vid_cil_x0": float(bases_um[ref] * ppum),
            "vid_cil_y0_invy": float(h_px - wall_row),
        },
    }
    with open(os.path.join(out_dir, "truth.json"), "w", encoding="utf-8") as fh:
        json.dump(truth, fh, indent=2)
    with open(os.path.join(out_dir, "README_truth.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(
            "Synthetic carpet rendered by ciliograph synth.\n\n"
            "Enter on the video tab:  ppum {:g}   fps {:g}\n"
            "The reference cilium (the {}th of {}) has its base at pixel "
            "({:.1f}, {:.1f}); Find base should land there.\n"
            "True length {:g} um, true cbf {:g} Hz ({:.2f} frames per beat).\n"
            "truth_coeffs.txt is the exact beat; truth_track.csv its "
            "centreline in every frame.\n"
            "Score a run with:  python ciliograph.py validate --images {}\n"
            .format(ppum, fps, ref + 1, n_cilia, bases_um[ref] * ppum,
                    wall_row, length, cbf, period_frames, out_dir))
    truth["frames"] = frame_names
    return truth


def true_beat_shapes(truth_dir: str, n_phases: int = 48, n_pts: int = 20):
    """The exact beat of a synthetic folder as shapes, one per phase."""
    import json
    with open(os.path.join(truth_dir, "truth.json"), encoding="utf-8") as fh:
        truth = json.load(fh)
    wave = FourierWave(load_fourier_coeffs(
        os.path.join(truth_dir, truth["coefficients"])),
        length=truth["length_um"], cbf=truth["cbf_hz"])
    shapes = [np.column_stack(wave.shape_at(t, n_pts=n_pts))
              for t in np.arange(n_phases) / n_phases]
    return truth, wave, shapes


def compare_beats(fit_wave, true_wave, n_phases: int = 48, n_pts: int = 20):
    """RMS distance between two beats after aligning their phase origins.

    Both are sampled at ``n_phases`` phases and ``n_pts`` equal-arc-length
    nodes; the fitted beat is shifted in phase to the offset that minimises
    the mean RMS, since where phase zero falls is a convention.  Returns the
    mean RMS, the per-phase RMS at that shift, and the shift.
    """
    def sample(w, shift):
        return np.array([np.column_stack(w.shape_at((t + shift) % 1.0,
                                                    n_pts=n_pts))
                         for t in np.arange(n_phases) / n_phases])
    truth = sample(true_wave, 0.0)
    best = None
    for k in range(n_phases):
        fit = sample(fit_wave, k / float(n_phases))
        per = np.sqrt(np.mean(np.sum((fit - truth) ** 2, axis=2), axis=1))
        if best is None or per.mean() < best[0]:
            best = (float(per.mean()), per, k / float(n_phases))
    return best


def validate_folder(images_dir: str, max_traces: int = 120, out_dir=None,
                    assume_period: bool = False, verbose=print,
                    enhance: dict | None = None):
    """Run the fitter's steps on a synthetic folder and score them.

    Everything the buttons do, in order -- Find base, Find cycle, Find
    length, Auto fit -- with ppum and fps taken from the truth (they are
    typed in, not estimated), and each answer compared with the exact one.
    Also traces the reference cilium in a sample of frames and measures how
    far each trace lies from the true centreline, which is the tracker's
    own score before any fitting.
    """
    import json
    truth_path = os.path.join(images_dir, "truth.json")
    if not os.path.exists(truth_path):
        raise FileNotFoundError("no truth.json in {}; validate needs a folder "
                                "written by `synth`".format(images_dir))
    with open(truth_path, encoding="utf-8") as fh:
        truth = json.load(fh)
    true_wave = FourierWave(load_fourier_coeffs(
        os.path.join(images_dir, truth["coefficients"])),
        length=truth["length_um"], cbf=truth["cbf_hz"])
    # The true centreline of *any* cilium follows from the table and the
    # wave: cilium i lags by x_i / wavelength.  Find base may legitimately
    # settle on a neighbour of the reference cilium -- on an identical
    # carpet every cilium is the same measurement -- so the tracker is
    # scored against the cilium it actually followed.
    def true_curve(cilium: int, frame: int, n_pts: int = 60):
        x_um = truth["bases_x_px"][cilium] / truth["ppum"]
        phase = (frame / truth["period_frames"]
                 - truth["wave_direction"] * x_um / truth["wavelength_um"]) % 1.0
        return np.column_stack(true_wave.shape_at(phase, n_pts=n_pts))

    import matplotlib
    matplotlib.use("Agg")
    info = scan_image_folder(images_dir)
    pqd = default_pqd_for_images(info)
    pqd.set("vid_ppum", "{:g}".format(truth["ppum"]))
    pqd.set("vid_fps", "{:g}".format(truth["fps"]))
    pqd.set("length", "6.0")
    # An enhancement to score against the truth: every step below then
    # reads the enhanced frames, as it would in the fitter.
    enh = enhance_defaults()
    enh.update(enhance or {})
    up = int(np.clip(int(enh.get("upsample") or 1), 1, 4))
    enh["upsample"] = up
    enhance_to_pqd(pqd, enh)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp_pqd = os.path.join(out_dir or images_dir, "validate_spec.json")
    pqd.save(tmp_pqd, backup=False)
    gui = CiliographGui(tmp_pqd, images_dir)
    report = {"enhance": describe_enhance(enh)}

    def say(line):
        if verbose:
            verbose(line)

    if not enhance_is_default(enh):
        say("Enhance      : {}".format(describe_enhance(enh)))
    t0 = time.time()
    gui._on_locate()
    # positions in the frames as rendered, whatever the upsampling
    bx = gui.pqd.getf("vid_cil_x0") / up
    by = (gui.model.image_h - gui.pqd.getf("vid_cil_y0_invy")) / up
    tx, ty = truth["reference_base_px"]
    # Any base is a valid choice on an identical carpet; score the distance
    # to the nearest true base, and separately note which cilium it is.
    bases = np.array(truth["bases_x_px"])
    nearest = int(np.argmin(np.abs(bases - bx)))
    report["base_error_px"] = float(np.hypot(bx - bases[nearest],
                                             by - truth["wall_row"]))
    report["base_cilium"] = nearest
    say("Find base    : ({:.1f}, {:.1f}) -> nearest true base {:.1f} px "
        "away (cilium {} of {}; wall row {:.0f}, found {:.1f})   [{:.0f}s]"
        .format(bx, by, report["base_error_px"], nearest + 1,
                truth["n_cilia"], truth["wall_row"], by, time.time() - t0))

    t0 = time.time()
    if assume_period:
        gui.pqd.set("cbf", "{:g}".format(truth["cbf_hz"]))
        gui.model.rebuild()
        say("Find cycle   : skipped, period taken from the truth")
    else:
        gui._on_find_cycle()
    per = gui.model.period_frames
    report["period_error_frames"] = float(per - truth["period_frames"])
    report["cbf_error_pct"] = float(100.0 * (gui.model.cbf - truth["cbf_hz"])
                                    / truth["cbf_hz"])
    say("Find cycle   : period {:.2f} frames (true {:.2f}), cbf {:.3f} Hz "
        "(true {:g}, {:+.1f}%)   [{:.0f}s]".format(
            per, truth["period_frames"], gui.model.cbf, truth["cbf_hz"],
            report["cbf_error_pct"], time.time() - t0))

    t0 = time.time()
    gui._on_find_length()
    report["length_um"] = gui.pqd.getf("length")
    report["length_error_pct"] = float(100.0 * (report["length_um"]
                                                - truth["length_um"])
                                       / truth["length_um"])
    say("Find length  : {:.2f} um (true {:g}, {:+.1f}%)   [{:.0f}s]".format(
        report["length_um"], truth["length_um"], report["length_error_pct"],
        time.time() - t0))

    # The tracker on its own: the chosen cilium in a sample of frames.
    t0 = time.time()
    n_true = truth["n_frames"]
    sample = list(range(gui.video.first, gui.video.first
                        + min(n_true, int(np.ceil(per)) + 1),
                        max(1, int(np.ceil(per)) // 24)))
    steps, curves, times_ms = auto_trace(gui.video, gui.model, frames=sample,
                                         max_traces=len(sample))
    ppum = truth["ppum"] * up
    per_frame = []
    for curve, t_ms in zip(curves, times_ms):
        f = int(round(t_ms * truth["fps"] / 1000.0))
        if f >= n_true:
            continue
        tr = true_curve(nearest, f)                            # um, y up
        got = np.asarray(curve, dtype=float)
        got_um = np.column_stack([(got[:, 0] - got[0, 0]) / ppum,
                                  (got[0, 1] - got[:, 1]) / ppum])
        a = _resample_curve(got_um, 20)[0]
        b = _resample_curve(tr, 20)[0]
        per_frame.append(float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))))
    per_frame = np.array(per_frame)
    report["track_rms_um_median"] = float(np.median(per_frame))
    report["track_rms_um_90pct"] = float(np.percentile(per_frame, 90))
    report["track_on_target_frac"] = float(np.mean(per_frame < 0.1 * truth["length_um"]))
    say("Tracking     : {} frames traced; RMS to the true centreline median "
        "{:.2f} um, 90th pct {:.2f} um; within a tenth of a length in {:.0f}% "
        "of frames   [{:.0f}s]".format(
            len(per_frame), report["track_rms_um_median"],
            report["track_rms_um_90pct"],
            100 * report["track_on_target_frac"], time.time() - t0))

    t0 = time.time()
    gui._max_traces = int(max_traces)
    # The truth is in the surface frame (x along the wall); score there.
    gui.pqd.set(PATTERN_FRAME_KEY, "surface")
    gui._on_tracefit()
    fit_wave = FourierWave(gui.fourier.coeffs, length=gui.model.length,
                           cbf=gui.model.cbf)
    mean_rms, per_phase, shift = compare_beats(fit_wave, true_wave)
    report["beat_rms_um"] = mean_rms
    report["beat_rms_um_worst"] = float(per_phase.max())
    tips_fit = np.array([[fit_wave.shape_at(t, n_pts=60)[0][-1],
                          fit_wave.shape_at(t, n_pts=60)[1][-1]]
                         for t in np.arange(96) / 96.0])
    tips_true = np.array([[true_wave.shape_at(t, n_pts=60)[0][-1],
                           true_wave.shape_at(t, n_pts=60)[1][-1]]
                          for t in np.arange(96) / 96.0])
    report["tip_x_range_fit"] = [float(tips_fit[:, 0].min()), float(tips_fit[:, 0].max())]
    report["tip_x_range_true"] = [float(tips_true[:, 0].min()), float(tips_true[:, 0].max())]
    report["tip_z_range_fit"] = [float(tips_fit[:, 1].min()), float(tips_fit[:, 1].max())]
    report["tip_z_range_true"] = [float(tips_true[:, 1].min()), float(tips_true[:, 1].max())]
    say("Auto fit     : {}   [{:.0f}s]".format(gui.status, time.time() - t0))
    say("Beat vs truth: RMS {:.3f} um averaged over the cycle (worst phase "
        "{:.3f}); tip sweep x {:.2f}..{:.2f} (true {:.2f}..{:.2f}), "
        "z {:.2f}..{:.2f} (true {:.2f}..{:.2f})".format(
            mean_rms, per_phase.max(), *report["tip_x_range_fit"],
            *report["tip_x_range_true"], *report["tip_z_range_fit"],
            *report["tip_z_range_true"]))

    # The figure: fitted fan over the true fan, and the per-phase error.
    plt = None
    try:
        from matplotlib import pyplot as plt
    except Exception:
        pass
    fig_path = None
    if plt is not None:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
        for t in np.arange(14) / 14.0:
            x, z = true_wave.shape_at(t, n_pts=120)
            axes[0].plot(x, z, "-", color="0.55", lw=2.2, alpha=0.6)
            x, z = fit_wave.shape_at((t + shift) % 1.0, n_pts=120)
            axes[0].plot(x, z, "-", color="#e74c3c", lw=1.3)
        axes[0].set_aspect("equal")
        axes[0].set_title("grey: true beat   red: Auto fit (phase-aligned)")
        axes[0].set_xlabel("x / um"); axes[0].set_ylabel("z / um")
        axes[1].plot(np.arange(len(per_phase)) / float(len(per_phase)),
                     per_phase, "o-", ms=3)
        axes[1].set_xlabel("phase"); axes[1].set_ylabel("RMS / um")
        axes[1].set_title("shape error by phase (mean {:.3f} um)".format(mean_rms))
        axes[1].set_ylim(0, max(0.5, per_phase.max() * 1.15))
        axes[2].plot(tips_true[:, 0], tips_true[:, 1], "-", color="0.55", lw=2.2, label="true tip path")
        axes[2].plot(tips_fit[:, 0], tips_fit[:, 1], "-", color="#e74c3c", lw=1.3, label="fitted tip path")
        axes[2].set_aspect("equal"); axes[2].legend(fontsize=8)
        axes[2].set_title("tip over one beat"); axes[2].set_xlabel("x / um")
        fig.tight_layout()
        fig_path = os.path.join(out_dir or images_dir, "validation.png")
        fig.savefig(fig_path, dpi=105)
        plt.close(fig)
        say("figure       : {}".format(fig_path))
    report["figure"] = fig_path
    with open(os.path.join(out_dir or images_dir, "validation.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    try:
        os.remove(tmp_pqd)
    except OSError:
        pass
    return report


def cilia_are_bright(video) -> bool:
    """Do the cilia show as bright spikes rather than dark dips?  Then invert.

    Judged the way auto_trace judges polarity: by the skew of each moving
    pixel's intensity over time.  A dark cilium sweeping past a pixel makes
    brief dips below that pixel's mean (negative skew); a fluorescent one
    makes brief spikes (positive).  Comparing moving pixels with still ones
    instead does not work -- the still pixels include the dark tissue, so
    the moving band always looked "brighter" and a bright-field recording
    was inverted.
    """
    motion, mean = motion_and_mean(video)
    band = motion >= np.percentile(motion, 90)
    if band.sum() < 20:
        return False
    n = video.last - video.first + 1
    step = max(1, n // 24)
    dev = []
    for f in range(video.first, video.last + 1, step):
        img = np.asarray(video.get(f), dtype=np.float32)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)
        dev.append((img - mean)[band])
    dev = np.concatenate(dev)
    sd = float(np.std(dev))
    if sd <= 0:
        return False
    skew = float(np.mean((dev / sd) ** 3))
    return skew > 0.05


def _score_enhance_candidate(pqd, directory, cand, sample, motion_raw):
    """One row of auto_enhance's table: trace and fit under one setting."""
    import copy
    pq = copy.deepcopy(pqd)
    enhance_to_pqd(pq, cand)
    up_old = int(np.clip(round(pqd.getf("vid_enh_upsample", 1.0)), 1, 4))
    ratio = int(cand.get("upsample", 1)) / float(up_old)
    if ratio != 1.0:
        pq.set("vid_cil_x0", pqd.getf("vid_cil_x0") * ratio)
        pq.set("vid_cil_y0_invy", pqd.getf("vid_cil_y0_invy") * ratio)
    v2 = VideoFrames.from_pqd(pq, directory)
    m2 = WaveModel(pq)
    row = {"settings": cand, "label": describe_enhance(cand)}
    try:
        _, curves, times = auto_trace(v2, m2, frames=sample,
                                      max_traces=len(sample))
        phases = np.array([m2.phase_of_frame(t * m2.fps / 1000.0)
                           for t in times])
        keep = usable_traces(curves)
        if keep.sum() < 2 * FOURIER_DEFAULT_MODES + 2:
            raise ValueError("only {} usable traces".format(keep.sum()))
        targets = curves_to_model_frame(
            [c for c, k in zip(curves, keep) if k], m2)
        fit = fit_beat_at_phases(targets, phases[keep],
                                 n_modes=FOURIER_DEFAULT_MODES,
                                 num_locs=20, n_space=N_FOURIER_M)
        err = fit["resid_per_trace"] * fit["mean_length"]     # microns
        # Support in the raw motion map, at the raw scale.
        sup = []
        for c in curves:
            pts = np.asarray(c, dtype=float) / float(cand.get("upsample", 1))
            ii = np.clip(np.rint(pts[:, 1]).astype(int), 0,
                         motion_raw.shape[0] - 1)
            jj = np.clip(np.rint(pts[:, 0]).astype(int), 0,
                         motion_raw.shape[1] - 1)
            sup.append(float(motion_raw[ii, jj].mean()))
        row.update({"median_um": float(np.median(err)),
                    "p90_um": float(np.percentile(err, 90)),
                    "downweighted": int(fit["n_downweighted"]),
                    "n_traces": int(keep.sum()),
                    "support": float(np.median(sup))})
    except Exception as exc:
        row.update({"error": str(exc).splitlines()[0]})
    return row


def _enhance_candidate_job(job):
    """auto_enhance's per-candidate work, for a worker process."""
    pqd, directory, cand, sample, motion_raw, threads = job
    _WORKER_THREADS["n"] = int(threads)
    return _score_enhance_candidate(pqd, directory, cand, sample, motion_raw)


# Auto enhance's pattern round: Auto fit under each of a few settings, and
# the setting whose beat agrees best with the others' is taken.  Each
# setting's fit errs in its own way while the beat they all look for is the
# same, so a beat's distance from the rest tracks its distance from the
# truth: on three rendered carpets the ordering by this distance matched the
# ordering by the true error, and the choice was the best setting or within
# 0.003 um of it (lawn: 0.238 um chosen, plain frames 0.337; six cilia:
# 0.045 against 0.043; three cilia: 0.039).  Settings within this tolerance
# of the closest count as equal and the simplest of them is taken.
AUTO_PATTERN_TRACES = 60
AUTO_PATTERN_TOL = 0.03
AUTO_PATTERN_MAX = 8
AUTO_PATTERN_EXTRA = ({"background": 1, "flatten": 40},
                      {"background": 1, "denoise": 0.7},
                      {"bandpass": 40.0}, {"lines": 2.5})


def _fit_pattern_candidate(pqd, directory, cand, max_traces):
    """One setting of the pattern round: Auto fit under it (fourier_autofit),
    the beat it gives and what the fit says about it."""
    import copy
    pq = copy.deepcopy(pqd)
    enhance_to_pqd(pq, cand)
    up_old = int(np.clip(round(pqd.getf("vid_enh_upsample", 1.0)), 1, 4))
    ratio = int(cand.get("upsample", 1)) / float(up_old)
    if ratio != 1.0:
        pq.set("vid_cil_x0", pqd.getf("vid_cil_x0") * ratio)
        pq.set("vid_cil_y0_invy", pqd.getf("vid_cil_y0_invy") * ratio)
    v2 = VideoFrames.from_pqd(pq, directory)
    m2 = WaveModel(pq)
    row = {"settings": cand, "label": describe_enhance(cand)}
    try:
        res = fourier_autofit(v2, m2, max_traces=int(max_traces))
        row.update({"coeffs": res["coeffs"], "energy": float(res["energy"]),
                    "chosen": res["chosen"],
                    "runner_um": float(res["joint"]["spread_um"]),
                    "length": float(m2.length), "cbf": float(m2.cbf)})
    except Exception as exc:
        row["error"] = str(exc).splitlines()[0]
    return row


def _pattern_candidate_job(job):
    pqd, directory, cand, max_traces, threads = job
    _WORKER_THREADS["n"] = int(threads)
    return _fit_pattern_candidate(pqd, directory, cand, max_traces)


def pattern_consensus(rows, tol: float = AUTO_PATTERN_TOL) -> dict | None:
    """The pattern round's judgement: every beat's typical distance from
    the other settings' beats (compare_beats, phase-aligned; the median
    over five or more settings, the mean over fewer), the closest as the
    choice -- the simplest of those within ``tol`` of it -- and the median
    pairwise distance as the spread.  None with fewer than two."""
    ok = [r for r in rows if "coeffs" in r]
    if len(ok) < 2:
        if len(ok) == 1:
            ok[0]["mean_um"] = 0.0
            return {"best": ok[0], "spread_um": float("nan"), "n": 1}
        return None
    waves = [FourierWave(r["coeffs"], length=r["length"], cbf=r["cbf"])
             for r in ok]
    n = len(ok)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = float(compare_beats(waves[i], waves[j])[0])
    for i, r in enumerate(ok):
        others = np.delete(dist[i], i)
        # the median of the distances from four or more others: a couple
        # of outlying settings (a drawing on a lawn) then do not pull
        # everyone's score
        r["mean_um"] = float(np.median(others) if n >= 5 else others.mean())
    d_min = min(r["mean_um"] for r in ok)
    tied = [r for r in ok if r["mean_um"] <= d_min * (1.0 + tol) + 0.01]
    best = min(tied, key=lambda r: (enhance_cost(r["settings"]), r["mean_um"]))
    return {"best": best, "spread_um": float(np.median(dist[np.triu_indices(n, 1)])),
            "n": n}


# The frames measured, and a value for every control derived from them
# (measure_recording, recommend_enhance).  The derived settings are
# candidates like any other: the pattern round decides whether they beat
# the plain frames, and the winner's values fill the controls.
MEASURE_SCALES = (0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.2)


def measure_recording(video, model, n_frames: int = 24,
                      whole: bool = False) -> dict:
    """What the frames are like around the cilium, as recorded: the noise,
    the cilia's width and contrast, how uneven the light is, how much
    static texture there is, the grey levels used, the polarity, and the
    cilium's length in pixels.  ``video`` is the recording without
    enhancement; the window is the one the fit uses, above the tissue --
    or, with ``whole`` (no base chosen yet), the band above the tissue
    across the whole frame."""
    frames = spread_frames(video.first, video.last, n_frames)
    _prefetch(video, frames)
    raws = []
    for f in frames:
        a = np.asarray(video.get(f), dtype=np.float32)
        raws.append(a[..., :3].mean(axis=2) if a.ndim == 3 else a)
    stack = np.stack(raws)
    mean = stack.mean(axis=0)
    h, w = mean.shape
    pqd = model.pqd
    ppum = float(model.ppum or 1.0)
    L_px = float(model.length) * ppum
    x0 = float(pqd.getf("vid_cil_x0"))
    y0 = float(model.image_h - pqd.getf("vid_cil_y0_invy"))
    try:
        valid = np.asarray(video.valid, dtype=bool)
        if valid.shape != mean.shape:
            valid = np.ones(mean.shape, dtype=bool)
    except Exception:
        valid = np.ones(mean.shape, dtype=bool)
    surface = detect_surface(mean, valid=valid)
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    if whole:
        band = ((rows < surface[None, :] - 2)
                & (rows >= surface[None, :] - 1.3 * L_px) & valid)
        if band.sum() < 64:                     # no tissue found: the top
            band = (rows < 0.6 * h) & valid
    else:
        c0 = int(np.clip(x0 - 1.5 * L_px, 0, w - 8))
        c1 = int(np.clip(x0 + 1.5 * L_px + 1, c0 + 8, w))
        r0 = int(np.clip(y0 - 1.3 * L_px, 0, h - 8))
        band = ((cols >= c0) & (cols < c1) & (rows >= r0)
                & (rows < surface[None, :] - 2) & valid)
        if band.sum() < 64:
            band = np.zeros_like(band)
            band[max(0, int(y0) - 40):max(1, int(y0)), c0:c1] = True
            band &= valid
    out = {"ppum": ppum, "length_px": L_px, "n_frames": len(frames),
           "band_px": int(band.sum()), "whole": bool(whole)}
    # noise: the temporal spread of the stillest quarter of the band's
    # pixels (through which no cilium sweeps)
    tstd = stack.std(axis=0)
    still_cut = float(np.percentile(tstd[band], 25)) if band.any() else 0.0
    still = band & (tstd <= still_cut)
    out["noise"] = float(np.median(tstd[still])) if still.any() else float(np.median(tstd[band]))
    # polarity, as the fit judges it -- or, without a base, by the skew of
    # frame - mean over the whole band
    try:
        if whole:
            dev = (stack - mean[None])[:, band]
            sd = float(np.std(dev))
            skew = float(np.mean((dev / sd) ** 3)) if sd > 0 else 0.0
            clear = abs(skew) >= POLARITY_MIN_SKEW
            pol = {"sign": (1.0 if skew > 0 else -1.0) if clear else 0.0,
                   "clear": clear, "skew": skew, "skews": [skew]}
        else:
            pol = polarity_of(video, model)
    except Exception:
        pol = {"sign": 0.0, "clear": False, "skew": float("nan"), "skews": []}
    out["polarity"] = pol
    sign = -1.0 if pol["sign"] in (0.0, -1.0) else 1.0   # -1 dark, +1 bright
    # the cilia's width: at the strongest thin ridges among the moving
    # pixels of a few frames, the profile across the ridge, and its width
    # at half depth between the shoulders either side
    picks = raws[::max(1, len(raws) // 6)][:6]
    moving = band & (tstd >= np.percentile(tstd[band], 60)) if band.any() else band
    widths = []
    for a in picks:
        img = a if sign < 0 else 255.0 - a
        resp = ridge_response_at(img, 1.0)
        hrr, hrc, hcc = gauss_hess(img, 1.0)
        cand = np.where(moving, resp, -np.inf)
        cand[:8, :] = cand[-8:, :] = -np.inf
        cand[:, :8] = cand[:, -8:] = -np.inf
        if not np.isfinite(cand).any():
            continue
        flat = cand.ravel()
        top = np.argpartition(flat, -40)[-40:]
        for idx in top:
            if not np.isfinite(flat[idx]):
                continue
            r, c = divmod(int(idx), w)
            # the across direction: the Hessian's major eigenvector
            ang = 0.5 * np.arctan2(2.0 * hrc[r, c], hrr[r, c] - hcc[r, c])
            dy, dx = np.cos(ang), np.sin(ang)
            t = np.arange(-8, 9)
            rr = np.clip(np.rint(r + t * dy).astype(int), 0, h - 1)
            cc = np.clip(np.rint(c + t * dx).astype(int), 0, w - 1)
            prof = img[rr, cc]
            centre = 8
            shoulder = 0.5 * (prof[:centre].max() + prof[centre + 1:].max())
            depth = shoulder - prof[centre]
            if depth <= 0:
                continue
            half = prof[centre] + 0.5 * depth
            left = centre
            while left > 0 and prof[left - 1] < half:
                left -= 1
            right = centre
            while right < len(prof) - 1 and prof[right + 1] < half:
                right += 1
            widths.append(right - left + 1)
    out["width_px"] = float(np.clip(np.median(widths), 1.5, 8.0)) if widths else 2.5
    out["width_n"] = len(widths)
    # contrast: how far the darkest (brightest) band pixels sit from the
    # band's typical level, per frame
    depths = []
    for a in picks:
        v = a[band]
        depths.append(float(np.median(v) - np.percentile(v, 2)) if sign < 0
                      else float(np.percentile(v, 98) - np.median(v)))
    out["contrast"] = float(np.median(depths))
    out["snr"] = out["contrast"] / max(out["noise"], 1e-6)
    # illumination: the slow part of the mean across the band, with the
    # tissue (and the padding) replaced by the band's level first so that
    # the dark body below does not read as a gradient
    level = float(np.median(mean[band])) if band.any() else float(np.median(mean))
    filled = np.where((rows < surface[None, :] - 2) & valid, mean, level)
    smooth = gaussian_blur(filled, max(4.0, 0.5 * L_px))
    sm = smooth[band]
    out["illumination_range"] = float(np.percentile(sm, 98) - np.percentile(sm, 2))
    # static texture: what the mean keeps beyond the slow light
    tex = (mean - smooth)[band]
    out["static_texture"] = float(1.4826 * np.median(np.abs(tex - np.median(tex))))
    # grey levels used
    lo, hi = np.percentile(stack[:, band] if band.any() else stack, [1, 99])
    out["levels_lo"], out["levels_hi"] = float(lo), float(hi)
    return out


def recommend_enhance(meas: dict, fix=None, lawn: bool = False) -> dict:
    """A value for every control, from the measurements (measure_recording):
    the frames as the tracer and the fit would like them.  ``fix`` is what
    the acquisition needs anyway (deflicker, deblock); ``lawn`` says the
    cilia beat in step, where the line filter loses the bright half and is
    left out."""
    s = enhance_defaults()
    s.update(fix or {})
    L = float(meas.get("length_px") or 60.0)
    width = float(meas.get("width_px") or 2.5)
    noise = float(meas.get("noise") or 0.0)
    snr = float(meas.get("snr") or 99.0)
    pol = meas.get("polarity") or {}
    if pol.get("sign") == 1.0 and pol.get("clear"):
        s["invert"] = 1
    # static texture (the tissue's edge, debris) beyond the noise: take the
    # background out; a lawn that sweeps through most pixels: the median
    if float(meas.get("static_texture") or 0.0) > 1.5 * max(noise, 0.5):
        s["background"] = 1
        s["bgmedian"] = int(bool(lawn))
    # uneven light beyond the noise: flatten over about half a length, so
    # the cilia themselves are not flattened away
    window = int(np.clip(round(0.6 * L), 20, 120))
    if float(meas.get("illumination_range") or 0.0) > max(2.0 * noise, 4.0):
        s["flatten"] = window
    # noise against the cilia's contrast: a small blur when it is not
    # negligible, the line denoise when it is bad
    if snr < 6.0:
        s["denoise"] = 0.7
    if snr < 3.0:
        s["denoise"] = 0.0
        s["linesmooth"] = 1.5
    # a recording that uses a sliver of the grey scale: local contrast
    if float(meas.get("levels_hi", 255.0)) - float(meas.get("levels_lo", 0.0)) < 60.0:
        s["contrast"] = 2.0
    # a cilium of under forty pixels: the tracer's filters are sized in
    # pixels, so give it twice the pixels (Lanczos)
    if L < 40.0:
        s["upsample"] = 2
        s["upmethod"] = 1
    # the thin-structure tools, sized to the cilia (the pattern round
    # decides whether to use them; see auto_enhance)
    s["bpsmall"] = float(max(1.0, round(width / 2.0)))
    s["_bandpass"] = float(window)
    s["_tophat"] = float(round(width / 2.0) + 1.0)
    s["_lines"] = 0.0 if lawn else float(round(width * 2.0) / 2.0)
    return s


def measured_candidates(meas: dict, fix=None, lawn: bool = False) -> list:
    """The measured settings as Auto candidates: the classical set, then
    the same with each thin-structure tool (bandpass in place of flatten
    and denoise, the top-hat, the line filter)."""
    rec = recommend_enhance(meas, fix, lawn)
    bandpass, tophat, lines = rec.pop("_bandpass"), rec.pop("_tophat"), rec.pop("_lines")
    out = [dict(rec)]
    bp = dict(rec)
    bp.update({"bandpass": bandpass, "flatten": 0, "denoise": 0.0})
    out.append(bp)
    th = dict(rec)
    th.update({"tophat": tophat, "denoise": 0.0, "linesmooth": 0.0})
    out.append(th)
    if lines > 0:
        ln = dict(rec)
        ln.update({"lines": lines, "denoise": 0.0, "linesmooth": 0.0,
                   "contrast": 0.0})
        out.append(ln)
    return out


def describe_measurements(meas: dict) -> str:
    pol = meas.get("polarity") or {}
    polarity = {-1.0: "dark", 1.0: "bright"}.get(pol.get("sign"), "dark or bright")
    return ("measured: cilia {:.1f} px wide, {} on the field, contrast {:.0f} "
            "grey levels over noise {:.1f} (SNR {:.0f}); light uneven by {:.0f} "
            "levels; static texture {:.1f}; levels {:.0f}..{:.0f} used; "
            "cilium {:.0f} px long".format(
                meas.get("width_px", 0.0), polarity, meas.get("contrast", 0.0),
                meas.get("noise", 0.0), meas.get("snr", 0.0),
                meas.get("illumination_range", 0.0),
                meas.get("static_texture", 0.0), meas.get("levels_lo", 0.0),
                meas.get("levels_hi", 0.0), meas.get("length_px", 0.0)))


# The refinement round: around the tracer round's leader, each control
# with a number in it is tried at these values, one control at a time, and
# a value is taken only when it beats the leader by more than the noise
# (the spread over the base moved a pixel either way) -- the smallest such
# value.  Values are then confirmed together.
AUTO_REFINE_GRID = {
    "denoise": (0.0, 0.5, 0.7, 1.0, 1.4),
    "linesmooth": (0.0, 1.0, 1.5, 2.5),
    "deblur": (0.0, 1.0, 1.5, 2.0),
    "sharpen": (0.0, 0.5, 1.0),
    "contrast": (0.0, 1.0, 2.0, 3.0),
}


def auto_enhance(video, pqd, n_frames: int = 30, progress=None,
                 candidates=None, pattern: bool = True, refine: bool = True):
    """Choose the enhancement settings by how well one cilium can be tracked.

    Each candidate setting is tried on the tracker itself: the cilium at the
    recorded base is traced in ``n_frames`` frames spread over the record,
    the beat is fitted to those traces at their exact phases, and the
    candidate is scored by how far the individual traces sit from that beat
    -- the median distance, with the 90th percentile deciding ties.  A
    tracer that hops between neighbours or jitters on noise scores badly; one
    that follows a single smooth periodic cilium scores well.  Traces are
    also required to lie on moving pixels of the raw frames, so a setting
    under which the tracer draws smooth nonsense through empty space is
    rejected rather than rewarded.

    Needs the base and the period to be set already (Find base, Find cycle);
    the caller sees to that.  Returns the ranked table and the winner.

    ``pattern``: the pattern round follows -- Auto fit under the plain
    frames, the tracer round's leaders and a few settings that scored
    best on rendered carpets, and the setting whose beat agrees best with
    the others' is the winner (pattern_consensus).  It costs a few Auto
    fits, side by side; the tracer round alone is quick but judges only
    whether one cilium can be followed, which on a lawn none can.
    """
    import copy
    # The frames as recorded -- no enhancement -- for the motion map that
    # traces must stay on, and for the two acquisition defects that are
    # measured rather than guessed: lamp flicker and JPEG blocking.
    pq_raw = copy.deepcopy(pqd)
    enhance_to_pqd(pq_raw, enhance_defaults())
    raw = VideoFrames.from_pqd(pq_raw, video.directory)
    defects = measure_acquisition(raw)
    # Polarity is left alone: the walk tracer judges it for itself from the
    # same skew, and phase-contrast recordings show cilia as both bright
    # halo and dark core, which inverting does not simplify.  Invert stays
    # a manual switch for genuinely bright-on-dark data.
    base = enhance_defaults()
    base["invert"] = int(pqd.getf("vid_enh_invert", 0))
    # Flicker is corrected in every candidate but "none" when it is
    # measured: its worst effect is on Find cycle, which this score does
    # not see (on a flickering test carpet with known truth the period came
    # out doubled without it and right with it, and the fitted beat's error
    # fell from 1.84 to 1.25 um).  Deblocking has to earn its place in the
    # table like any other setting.
    fix = {}
    model_raw = WaveModel(pqd)
    period = float(getattr(model_raw, "period_frames", 0.0) or 0.0)
    if defects["flicker"] > FLICKER_FIX_LEVEL and (
            period <= 0 or period >= 2.5 * FLICKER_WINDOW):
        fix["deflicker"] = 1
    try:
        lawn = lawn_synchrony(raw, model_raw)
    except Exception:
        lawn = {"spread": float("nan"), "synchronous": False}
    # The frames measured, and a value for every control derived from them
    # (measure_recording, recommend_enhance): candidates in both rounds.
    try:
        meas = measure_recording(raw, model_raw)
        measured = measured_candidates(meas, fix, bool(lawn.get("synchronous")))
    except Exception as exc:
        meas = {"error": str(exc).splitlines()[0]}
        measured = []
    if candidates is None:
        bf = {"background": 1, "flatten": 40}
        grid = [{}, {"background": 1}, {"flatten": 40},
                dict(bf),
                {"background": 1, "denoise": 0.7},
                dict(bf, denoise=0.7),
                {"background": 1, "sharpen": 0.8},
                dict(bf, linesmooth=1.5),
                dict(bf, deblur=1.5),
                dict(bf, deblur=2.5),
                # Fiji's thin-structure tools (see tophat_drawing etc.)
                {"background": 1, "bgmedian": 1},
                dict(bf, tophat=3.0),
                {"lines": 2.5},
                dict(bf, lines=2.5)]
        if defects["jpeg"] and defects["blockiness"] > BLOCKING_FIX_LEVEL:
            grid += [{"deblock": 1}, dict(bf, deblock=1),
                     dict(bf, deblock=1, denoise=0.7)]
        # A network named on the enhance tab is tried on its own and with
        # the two usual companions; the other candidates run without it.
        model = clean_model_path(pqd.get("vid_enh_model"))
        if model:
            try:
                note = video.model_note()
            except Exception as exc:
                note = "NOT USED: {}".format(exc)
            if note.startswith("NOT USED"):
                model = ""             # a model that cannot run is not tried
        if model:
            base["modelrange"] = int(pqd.getf("vid_enh_model_range", 0))
            grid += [{"model": model}, dict(model=model, background=1),
                     dict(model=model, flatten=40), dict(bf, model=model)]
        candidates = []
        for i, g in enumerate(grid):
            c = dict(base)
            if i > 0:
                c.update(fix)           # "none" stays the frames as recorded
            c.update(g)
            candidates.append(c)
        if fix:
            # and the correction on its own, to show what it does
            candidates.insert(1, dict(base, **fix))
        labels = {describe_enhance(c) for c in candidates}
        for c in measured:             # the measured settings, when new
            if describe_enhance(c) not in labels:
                candidates.append(c)
                labels.add(describe_enhance(c))
    motion_raw, _ = motion_and_mean(raw)
    motion_raw = motion_raw / max(float(np.percentile(motion_raw, 95)), 1e-9)

    frames_all = list(range(video.first, video.last + 1))
    stride = max(1, int(np.ceil(len(frames_all) / float(n_frames))))
    sample = frames_all[::stride]

    # Every candidate is scored on its own -- its own enhanced copy of the
    # frames, its own traces and fit -- so the candidates run side by side,
    # one worker process each, as many at once as the cores and the free
    # memory allow; the cores left over prepare frames inside each worker.
    # Each row is exactly what the one-at-a-time loop produced.
    n_cores = compute_info()["workers"]
    h, w = video.shape
    up_max = max(int(c.get("upsample") or 1) for c in candidates)
    n_vid = video.last - video.first + 1
    per_job = n_vid * h * w * up_max ** 2 * 8 * 1.5 + 3e8
    n_proc = int(max(1, min(n_cores, len(candidates),
                            0.75 * available_memory() // per_job)))
    if not _processes_allowed() or _in_worker_process():
        # Threads would only queue for the interpreter lock here: the
        # tracer is Python-heavy.  One candidate at a time, all threads on
        # its frames.
        n_proc = 1
    inner = max(1, n_cores // n_proc)

    def run(jobs, lo, hi, what):
        def report(frac):
            if progress is not None:
                progress(lo + (hi - lo) * frac, "{} {} of {} ({} at a time)"
                         .format(what, int(round(frac * len(jobs))),
                                 len(jobs), n_proc))
        report(0.0)
        if n_proc == 1:
            out = []
            for i, job in enumerate(jobs):
                out.append(_score_enhance_candidate(*job[:5]))
                report((i + 1) / float(len(jobs)))
            return out
        # The slow settings start first, so none is left running alone at
        # the end while the other cores wait.
        order = sorted(range(len(jobs)),
                       key=lambda i: -enhance_cost(jobs[i][2]))
        done = parallel_map(_enhance_candidate_job, [jobs[i] for i in order],
                            progress=report, workers=n_proc)
        out = [None] * len(jobs)
        for i, r in zip(order, done):
            out[i] = r
        return out

    # progress: the tracer round, its re-scoring, the refinement, the pattern round
    t_hi, r_hi, f_hi = (0.35, 0.45, 0.6) if pattern else (0.6, 0.75, 1.0)
    if not refine:
        f_hi = r_hi
    rows = run([(pqd, video.directory, cand, sample, motion_raw, inner)
                for cand in candidates], 0.0, t_hi, "tried")
    scored = [r for r in rows if "median_um" in r]
    if not scored:
        raise ValueError("no setting produced enough traces to score")
    best_support = max(r["support"] for r in scored)
    for r in scored:
        r["rejected"] = r["support"] < 0.6 * best_support
    # Score: the median and the 90th percentile together.  The median alone
    # rewards a setting that is slightly better on most frames while
    # hopping on the rest -- exactly the failure being selected against --
    # and the tail alone is noisy on thirty frames.
    for r in scored:
        r["score_um"] = 0.5 * (r["median_um"] + r["p90_um"])
        r["scores"] = [r["score_um"]]
    ok = [r for r in scored if not r["rejected"]] or scored
    ok.sort(key=lambda r: r["score_um"])

    # The score is noisy.  The walk tracer chains each frame to the one
    # before, so one hop onto a neighbour changes many traces at once:
    # moving the base by a single pixel moved the score of the plain frames
    # of input_images4 from 0.66 to 0.85 and 1.01 um, and of input_images1
    # by 0.05.  So the leaders, and the plain frames for reference, are
    # scored again with the base a pixel to either side, and ranked by the
    # mean of the three; the spread is reported, and a difference smaller
    # than it is not a difference.
    defaults = enhance_defaults()
    plain = [r for r in scored if all(
        r["settings"].get(n, d) == d for n, d in defaults.items()
        if n != "invert")]
    top = ok[:AUTO_ENHANCE_RESCORE]
    top += [r for r in plain if r not in top]
    up_now = int(np.clip(round(pqd.getf("vid_enh_upsample", 1.0)), 1, 4))
    jobs, owners = [], []
    for r in top:
        for dx in (-1.0, 1.0):
            pq = copy.deepcopy(pqd)
            pq.set("vid_cil_x0", pqd.getf("vid_cil_x0") + dx * up_now)
            jobs.append((pq, video.directory, r["settings"], sample,
                         motion_raw, inner))
            owners.append(r)
    for r, extra in zip(owners, run(jobs, t_hi, r_hi, "re-scored")):
        if "median_um" in extra:
            r["scores"].append(0.5 * (extra["median_um"] + extra["p90_um"]))
    for r in top:
        r["score_um"] = float(np.mean(r["scores"]))
        r["spread_um"] = 0.5 * float(np.max(r["scores"]) - np.min(r["scores"]))
    final = [r for r in top if not r.get("rejected")] or top
    final.sort(key=lambda r: r["score_um"])
    lead = final[0]
    # Of the settings that the leader does not beat by more than the noise,
    # the simplest is taken: a filter has to earn its place, and every one
    # is paid for again each time a frame is read.  The frames as recorded
    # are the simplest of all.
    tied = [r for r in final
            if r["score_um"] - lead["score_um"]
            < max(lead.get("spread_um", 0.0), r.get("spread_um", 0.0))]
    best = min(tied, key=lambda r: (enhance_cost(r["settings"]), r["score_um"]))
    # The score is the tracer's: how far its traces sit from one beat.
    # Where it cannot follow one cilium under any setting -- a dense lawn,
    # where the traces land on neighbours whatever the frames look like --
    # every score is a failure and the ranking is noise.  Then no setting
    # is applied, and the readout says why: Auto fit's joint fit measures
    # the lawn without the traces, and its readout judges the beat.
    n_failed = sum(1 for r in rows if "median_um" not in r)
    unjudged = (best["score_um"] >= AUTO_ENHANCE_MAX_SCORE
                or n_failed > len(rows) // 2)
    if unjudged:
        plain_row = next((r for r in plain), None)
        if plain_row is not None:
            best = plain_row
    out = {"rows": rows, "best": best, "sample_frames": sample,
           "defects": defects, "within_noise": best is not lead,
           "leader": lead, "unjudged": unjudged, "n_failed": n_failed,
           "lawn": lawn, "tracer_best": best, "measurements": meas}

    # ---- the refinement round: the controls with a number in them, one at
    # a time around the leader (see AUTO_REFINE_GRID).
    refined = None
    if refine and not unjudged and lead.get("settings") is not None:
        f_mid = r_hi + 0.7 * (f_hi - r_hi)
        length_px = float(meas.get("length_px") or 60.0) if isinstance(meas, dict) else 60.0
        start = dict(lead["settings"])
        grid = {k: list(v) for k, v in AUTO_REFINE_GRID.items()}
        if start.get("flatten"):
            grid["flatten"] = sorted({int(np.clip(round(f * length_px), 15, 160))
                                      for f in (0.4, 0.6, 0.8, 1.2)})
        if start.get("bandpass"):
            bp0 = float(start["bandpass"])
            grid["bandpass"] = [round(bp0 * f, 1) for f in (0.5, 0.75, 1.5, 2.0)]
        jobs3, owners3 = [], []
        for name, values in grid.items():
            for value in values:
                if value == start.get(name, 0):
                    continue
                c = dict(start)
                c[name] = value
                if name == "denoise" and value:
                    c["linesmooth"] = 0.0
                if name == "linesmooth" and value:
                    c["denoise"] = 0.0
                jobs3.append((pqd, video.directory, c, sample, motion_raw, inner))
                owners3.append((name, value))
        rows3 = run(jobs3, r_hi, f_mid, "refined") if jobs3 else []
        noise = max(float(lead.get("spread_um", 0.0)), 0.02)
        tried = []
        changes = {}
        for (name, value), r3 in zip(owners3, rows3):
            sc = (0.5 * (r3["median_um"] + r3["p90_um"])
                  if "median_um" in r3 else float("nan"))
            tried.append({"name": name, "value": value, "score_um": sc,
                          "error": r3.get("error"),
                          "rejected": bool(r3.get("support", 1.0) < 0.6 * best_support)})
        for name in grid:
            cands3 = [t for t in tried if t["name"] == name and np.isfinite(t["score_um"])
                      and not t["rejected"]]
            if not cands3:
                continue
            winner = min(cands3, key=lambda t: t["score_um"])
            # a change has to beat the start by more than the noise, and the
            # smallest value that does is taken
            good = [t for t in cands3 if t["score_um"] < lead["score_um"] - noise]
            if good:
                pick = min(good, key=lambda t: (abs(float(t["value"])), t["score_um"]))
                changes[name] = pick["value"]
            winner["best_of_control"] = True
        # The changes are added one at a time, the biggest gain first, and
        # each is confirmed the way the leaders were -- the base a pixel
        # either side -- against the setting so far: two changes that each
        # help alone need not help together.
        confirmed = []
        current, cur_score = dict(start), float(lead["score_um"])
        gains = sorted(changes.items(), key=lambda kv: min(
            t["score_um"] for t in tried if t["name"] == kv[0]
            and t["value"] == kv[1]))
        n_steps = max(1, len(gains))
        for k, (name, value) in enumerate(gains):
            cand = dict(current)
            cand[name] = value
            if name == "denoise" and value:
                cand["linesmooth"] = 0.0
            if name == "linesmooth" and value:
                cand["denoise"] = 0.0
            jobs4 = [(pqd, video.directory, cand, sample, motion_raw, inner)]
            for dx in (-1.0, 1.0):
                pq = copy.deepcopy(pqd)
                pq.set("vid_cil_x0", pqd.getf("vid_cil_x0") + dx * up_now)
                jobs4.append((pq, video.directory, cand, sample, motion_raw, inner))
            lo4 = f_mid + (f_hi - f_mid) * k / n_steps
            hi4 = f_mid + (f_hi - f_mid) * (k + 1) / n_steps
            rows4 = run(jobs4, lo4, hi4, "confirming")
            scores4 = [0.5 * (r4["median_um"] + r4["p90_um"]) for r4 in rows4
                       if "median_um" in r4]
            if not scores4:
                continue
            score4 = float(np.mean(scores4))
            spread4 = 0.5 * float(np.max(scores4) - np.min(scores4))
            if score4 < cur_score - noise:
                row = dict(rows4[0]) if "median_um" in rows4[0] else {"settings": cand}
                row.update({"settings": cand, "label": describe_enhance(cand),
                            "score_um": score4, "spread_um": spread4,
                            "scores": scores4, "rejected": False, "refined": True})
                refined, current, cur_score = row, cand, score4
                confirmed.append((name, value, score4, spread4))
            else:
                confirmed.append((name, value, score4, None))
        out["refine"] = {"start": lead, "tried": tried, "changes": changes,
                         "noise_um": noise, "refined": refined,
                         "confirmed": confirmed}
        if refined is not None:
            rows.append(refined)
            if not pattern:
                out["best"] = refined
                out["tracer_best"] = refined
                out["within_noise"] = False
    if not pattern:
        return out

    # ---- the pattern round (see AUTO_PATTERN_TRACES).  The plain frames,
    # the refined and the measured settings, the tracer round's leaders,
    # and the settings that scored best on rendered carpets; each fitted
    # the way the Auto fit button fits.
    pool = [r["settings"] for r in plain]
    if refined is not None:
        pool.append(refined["settings"])
    pool += list(measured)
    pool += [r["settings"] for r in final if not r.get("rejected")][:2]
    for g in AUTO_PATTERN_EXTRA:
        c = dict(base)
        c.update(fix)
        c.update(g)
        pool.append(c)
    cands2, seen = [], set()
    for c in pool:
        lab = describe_enhance(c)
        if lab not in seen:
            seen.add(lab)
            cands2.append(c)
    cands2 = cands2[:AUTO_PATTERN_MAX]
    jobs2 = [(pqd, video.directory, c, AUTO_PATTERN_TRACES, inner) for c in cands2]

    def report2(frac):
        if progress is not None:
            progress(f_hi + (1.0 - f_hi) * frac,
                     "pattern round: Auto fit under {} of {} settings ({} at a time)"
                     .format(int(round(frac * len(jobs2))), len(jobs2), n_proc))
    report2(0.0)
    if n_proc == 1:
        rows2 = []
        for i, job in enumerate(jobs2):
            rows2.append(_fit_pattern_candidate(*job[:4]))
            report2((i + 1) / float(len(jobs2)))
    else:
        order = sorted(range(len(jobs2)), key=lambda i: -enhance_cost(jobs2[i][2]))
        done = parallel_map(_pattern_candidate_job, [jobs2[i] for i in order],
                            progress=report2, workers=n_proc)
        rows2 = [None] * len(jobs2)
        for i, r in zip(order, done):
            rows2[i] = r
    verdict = pattern_consensus(rows2)
    out["pattern"] = {"rows": rows2, "verdict": verdict,
                      "traces": AUTO_PATTERN_TRACES}
    if verdict is not None:
        out["best"] = verdict["best"]
        out["within_noise"] = False
    return out


# A best Auto enhance score at or above this (um; the traces' distance from
# the one beat fitted through them) means the tracer followed no single
# cilium under any setting, and the ranking says nothing: input_images3, a
# dense lawn, scored 1.33 under its best setting with half the settings
# failing outright, where single cilia score 0.25 to 0.45.
AUTO_ENHANCE_MAX_SCORE = 1.0


def enhance_cost(settings: dict) -> float:
    """How much an enhancement does, and costs: one per step, three for the
    slow ones (line denoise, deblocking), two for upsampling."""
    s = settings
    cost = 0.0
    for name in ("background", "flatten", "denoise", "sharpen", "deblur",
                 "contrast", "deflicker", "bandpass"):
        cost += 1.0 if s.get(name) else 0.0
    cost += 2.0 if s.get("tophat") else 0.0
    cost += 2.0 if s.get("lines") else 0.0
    cost += 3.0 if s.get("linesmooth") else 0.0
    cost += 3.0 if s.get("deblock") else 0.0
    cost += 2.0 if int(s.get("upsample") or 1) > 1 else 0.0
    cost += 4.0 if clean_model_path(s.get("model")) else 0.0
    return cost


# Leaders of Auto enhance's first round that are scored again with the base
# moved a pixel either way (see auto_enhance).
AUTO_ENHANCE_RESCORE = 3


def auto_enhance_table(res) -> list:
    """auto_enhance's result as readable lines, best first."""
    d = res.get("defects") or {}
    lines = ["flicker {:.1f} grey levels{}; {}".format(
        d.get("flicker", 0.0),
        " -> deflicker" if d.get("flicker", 0.0) > FLICKER_FIX_LEVEL else "",
        ("JPEG blocking {:.2f}{}".format(
            d.get("blockiness", 1.0),
            " -> deblock tried" if d.get("blockiness", 1.0) > BLOCKING_FIX_LEVEL
            else "") if d.get("jpeg") else "not JPEG"))]
    meas = res.get("measurements") or {}
    if "width_px" in meas:
        lines.append(describe_measurements(meas))
    elif meas.get("error"):
        lines.append("measurements failed: {}".format(meas["error"]))
    pat = res.get("pattern")
    if pat and pat.get("verdict"):
        # The pattern round decided; it goes first, the tracer round after.
        v = pat["verdict"]
        best = v["best"]
        lines.append("pattern round: Auto fit under each setting ({} traces); a "
                     "beat's mean distance from the other settings' beats, um -- "
                     "a wrong setting lands far from the rest; the simplest "
                     "within {:.0f}% of the closest is taken:".format(
                         pat["traces"], 100 * AUTO_PATTERN_TOL))
        for r in sorted([r for r in pat["rows"] if "coeffs" in r],
                        key=lambda r: r["mean_um"]):
            lines.append(" {:<34s} {:.3f}   energy {:+.2f}, runners-up {}{}".format(
                short_enhance(r["settings"]), r["mean_um"], r["energy"],
                "{:.2f} um".format(r["runner_um"]) if r["runner_um"] > 0
                else "none", "  <- chosen" if r is best else ""))
        for r in pat["rows"]:
            if "coeffs" not in r:
                lines.append(" {:<34s} failed: {}".format(
                    short_enhance(r["settings"]), r.get("error", "?")))
        lines.append(" the patterns from these settings differ by {:.2f} um "
                     "(median); then the tracer round:".format(v["spread_um"]))
    best = res["best"]
    ref = res.get("refine")
    if ref:
        start = ref["start"]
        names = {"denoise": "denoise sigma", "linesmooth": "line denoise",
                 "deblur": "deblur sigma", "sharpen": "sharpen",
                 "contrast": "contrast limit", "flatten": "flatten window",
                 "bandpass": "bandpass"}
        lines.append("refinement round: each number around the leader {} ({:.3f} "
                     "+- {:.3f}); a value is taken when it beats the leader by "
                     "more than the noise:".format(
                         short_enhance(start["settings"]), start["score_um"],
                         ref["noise_um"]))
        for name in names:
            tried = [t for t in ref["tried"] if t["name"] == name]
            if not tried:
                continue
            cur = start["settings"].get(name, 0)
            cells = ["{:g}: {}".format(
                t["value"], "failed" if t["error"] else
                ("{:.3f}".format(t["score_um"]) + (" rejected" if t["rejected"] else "")))
                for t in tried]
            verdict = ("-> {:g}".format(ref["changes"][name]) if name in ref["changes"]
                       else "kept {:g}".format(cur))
            lines.append(" {:<15s} {}   {}".format(names[name], ", ".join(cells), verdict))
        for name, value, score, spread in ref.get("confirmed", []):
            lines.append(" + {} {:g}: {:.3f}{} -- {}".format(
                names.get(name, name), value, score,
                " +- {:.3f}".format(spread) if spread is not None else "",
                "kept" if spread is not None else
                "not beyond the noise with the others, dropped"))
        if ref.get("refined") is not None:
            r = ref["refined"]
            lines.append(" refined: {} {:.3f} +- {:.3f}{}".format(
                short_enhance(r["settings"]), r["score_um"], r["spread_um"],
                "" if pat else "  <- chosen"))
        elif not ref["changes"]:
            lines.append(" no value beat the leader by more than the noise")
    at = len(lines) + 1                       # notes go right after this line
    lines.append("score in um = (median + 90th pct) / 2 of the traces' "
                 "distance from the beat; +- over base -1/0/+1 px:")
    tracer_best = res.get("tracer_best", best)

    def key(r):
        # the re-scored leaders first, best first; then the rest
        return (0 if "spread_um" in r else 1, r.get("score_um", 9e9))

    for r in sorted(res["rows"], key=key):
        name = short_enhance(r["settings"])
        if "median_um" not in r:
            lines.append(" {:<34s} failed: {}".format(name, r["error"]))
            continue
        val = ("{:.3f} +- {:.3f}".format(r["score_um"], r["spread_um"])
               if "spread_um" in r else "{:.3f}".format(r["score_um"]))
        mark = (" <- chosen" if r is best else
                " <- the tracer's choice" if (r is tracer_best and pat) else "")
        lines.append(" {:<34s} {:<15s}{}{}".format(
            name, val, mark,
            "  rejected: off the moving pixels" if r.get("rejected") else ""))
    lawn = res.get("lawn") or {}
    if lawn.get("synchronous"):
        lines.insert(at, " lawn beating in step (phase spread {:.2f}): the tracer "
                         "cannot follow one cilium here, so its scores are weak "
                         "evidence{}".format(
                             lawn.get("spread", float("nan")),
                             "; the pattern round judges the beat" if pat
                             else "; Auto fit's readout judges the beat"))
    if res.get("unjudged"):
        lines.insert(at, " the tracer followed no single cilium under any setting "
                         "({} of {} failed, best {:.2f} um){}".format(
                             res.get("n_failed", 0), len(res["rows"]),
                             res["leader"]["score_um"],
                             "" if pat else ": nothing applied; a dense lawn is "
                             "measured by Auto fit's joint fit, not by these "
                             "traces"))
    elif res.get("within_noise"):
        lines.insert(at, " {} scored best, but {} is within the noise of it "
                         "and simpler: chosen".format(
                             short_enhance(res["leader"]["settings"]),
                             short_enhance(best["settings"])))
    return lines


def short_enhance(settings: dict) -> str:
    """describe_enhance, abbreviated for a table."""
    s = settings
    parts = [p for p, on in (
        ("deblock", s.get("deblock")), ("deflicker", s.get("deflicker")),
        ("inv", s.get("invert")),
        ("bgmed" if s.get("bgmedian") else "bg", s.get("background")),
        ("bp{:g}-{:g}".format(float(s.get("bpsmall") or 0),
                              float(s.get("bandpass") or 0)), s.get("bandpass")),
        ("flat{}".format(int(s.get("flatten") or 0)), s.get("flatten")),
        ("tophat{:g}".format(float(s.get("tophat") or 0)), s.get("tophat")),
        ("blur{:g}".format(float(s.get("denoise") or 0)), s.get("denoise")),
        ("line{:g}".format(float(s.get("linesmooth") or 0)), s.get("linesmooth")),
        ("deblur{:g}".format(float(s.get("deblur") or 0)), s.get("deblur")),
        ("sharp{:g}".format(float(s.get("sharpen") or 0)), s.get("sharpen")),
        ("lines{:g}".format(float(s.get("lines") or 0)), s.get("lines")),
        ("clahe{:g}".format(float(s.get("contrast") or 0)), s.get("contrast")),
        ("x{}".format(int(s.get("upsample") or 1)),
         int(s.get("upsample") or 1) > 1),
        ("ai", clean_model_path(s.get("model")))) if on]
    return "+".join(parts) if parts else "none"


# Above these an acquisition defect is corrected in every Auto enhance
# candidate (but "none").  Flicker as frame-to-frame jitter in grey levels:
# input_images4 jitters by 2.5, the steady recordings here by 0.15 or
# less.  Blocking as the ratio of edge strength on the 8-px JPEG grid to
# elsewhere: input_images2 1.83, lossless and 100 %-quality files 1.00 to
# 1.05.
FLICKER_FIX_LEVEL = 1.0
BLOCKING_FIX_LEVEL = 1.10


def measure_acquisition(video, n_max: int = 40) -> dict:
    """Lamp flicker and JPEG blocking in a recording, as recorded.

    Flicker is measured on the frames as delivered (unenhanced), blocking
    on the decoded files themselves, before any rotation moves the block
    grid off the pixel grid."""
    # Flicker is frame-to-frame, so it needs consecutive frames.
    run = list(range(video.first, min(video.last, video.first + 199) + 1))
    _prefetch(video, run)
    grey = []
    for f in run:
        a = np.asarray(video.get(f), dtype=np.float32)
        grey.append(a[..., :3].mean(axis=2) if a.ndim == 3 else a)
    try:
        valid = video._raw_valid()
        if valid.shape != grey[0].shape:
            valid = None
    except Exception:
        valid = None
    out = {"flicker": frame_flicker(grey, valid), "blockiness": 1.0,
           "jpeg": False}
    frames = list(range(video.first, video.last + 1))
    frames = frames[::max(1, len(frames) // 12)][:12]
    paths = [video.index[f] for f in frames] if hasattr(video, "index") else []
    if paths and jpeg_quant_table(paths[0][0]) is not None:
        files = []
        for path, page in paths:
            a = np.asarray(_load_image(path, page), dtype=np.float32)
            files.append(a[..., :3].mean(axis=2) if a.ndim == 3 else a)
        out["jpeg"] = True
        out["blockiness"] = jpeg_blockiness(files)
    return out


def describe_enhance(settings: dict) -> str:
    parts = []
    if settings.get("deblock"):
        parts.append("deblock")
    if settings.get("deflicker"):
        parts.append("deflicker")
    if settings.get("invert"):
        parts.append("invert")
    if settings.get("background"):
        parts.append("background (median)" if settings.get("bgmedian")
                     else "background")
    if settings.get("bandpass"):
        parts.append("bandpass {:g}-{:g}".format(
            float(settings.get("bpsmall") or 0.0), float(settings["bandpass"])))
    if settings.get("flatten"):
        parts.append("flatten {}".format(int(settings["flatten"])))
    if settings.get("tophat"):
        parts.append("top-hat {:g}".format(settings["tophat"]))
    if settings.get("denoise"):
        parts.append("denoise {:g}".format(settings["denoise"]))
    if settings.get("linesmooth"):
        parts.append("line denoise {:g}".format(settings["linesmooth"]))
    if settings.get("deblur"):
        parts.append("deblur {:g}".format(settings["deblur"]))
    if settings.get("sharpen"):
        parts.append("sharpen {:g}".format(settings["sharpen"]))
    if settings.get("lines"):
        parts.append("lines {:g}".format(settings["lines"]))
    if settings.get("contrast"):
        parts.append("contrast {:g}".format(settings["contrast"]))
    if int(settings.get("upsample") or 1) > 1:
        parts.append("x{}{}".format(int(settings["upsample"]),
                                    " Lanczos" if settings.get("upmethod") else ""))
    model = clean_model_path(settings.get("model"))
    if model:
        parts.append("model {}".format(os.path.basename(model)))
    return " + ".join(parts) if parts else "none"


def cmd_synth(args) -> int:
    """Render a synthetic carpet with exact ground truth."""
    coeffs = load_fourier_coeffs(args.coeffs) if args.coeffs else None
    last = [-1.0]

    def progress(frac, message):
        if frac >= last[0] + 0.1:
            last[0] = frac
            print("  {:3.0f}%  {}".format(100 * frac, message), flush=True)

    t0 = time.time()
    truth = render_carpet(
        args.out, coeffs=coeffs, length=args.length, ppum=args.ppum,
        cbf=args.cbf, fps=args.fps, n_frames=args.frames, n_cilia=args.cilia,
        spacing=args.spacing, wavelength=args.wavelength,
        wave_direction=-1 if args.leftward else 1,
        line_width=args.line_width, blur=args.blur, noise=args.noise,
        seed=args.seed, jitter=args.jitter, scatter=args.scatter,
        bright=args.bright, body=args.body, masks=bool(args.masks),
        progress=progress)
    print("rendered {} frames of {}x{} px in {:.0f}s -> {}".format(
        truth["n_frames"], truth["width_px"], truth["height_px"],
        time.time() - t0, args.out))
    if args.masks:
        print("  every cilium's centreline, {:g} px wide, in {}".format(
            truth["mask_width_px"], os.path.join(args.out, "masks")))
    print("  {} cilia, {:g} um apart, wavelength {:g} um; line width {:g} px; "
          "blur {:g} px; noise {:g}".format(
              truth["n_cilia"], truth["spacing_um"], truth["wavelength_um"],
              truth["line_width_px"], truth["blur_px"], truth["noise"]))
    print("  reference cilium base at ({:.1f}, {:.1f}); enter ppum {:g}, fps {:g}"
          .format(*truth["reference_base_px"], truth["ppum"], truth["fps"]))
    print("  truth written: truth.json, truth_coeffs.txt, truth_track.csv")
    print("  score a run with:  python ciliograph.py validate --images {}"
          .format(args.out))
    return 0


def parse_enhance_arg(text) -> dict:
    """"background=1,linesmooth=1.5" -> {"background": 1, "linesmooth": 1.5}."""
    out = {}
    for item in str(text or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        name, _, value = item.partition("=")
        name = name.strip().lower()
        if name not in ENHANCE_KEYS:
            raise ValueError("unknown enhancement {!r}; known: {}".format(
                name, ", ".join(ENHANCE_KEYS)))
        default = ENHANCE_KEYS[name][1]
        if isinstance(default, str):
            out[name] = clean_model_path(value)
            continue
        v = float(value) if value.strip() else 1.0
        out[name] = int(round(v)) if isinstance(default, int) else v
    return out


def cmd_validate(args) -> int:
    """Run the pipeline on a synthetic folder and score it against the truth."""
    images_dir = os.path.abspath(args.images)
    try:
        enhance = parse_enhance_arg(args.enhance)
        if getattr(args, "model", None):
            enhance["model"] = clean_model_path(args.model)
        report = validate_folder(images_dir, max_traces=args.max_traces,
                                 out_dir=args.outdir,
                                 assume_period=args.assume_period,
                                 enhance=enhance)
    except (FileNotFoundError, ValueError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    print()
    print("summary: base off by {:.1f} px; cbf {:+.1f}%; length {:+.1f}%; "
          "tracking median {:.2f} um; beat RMS {:.3f} um".format(
              report["base_error_px"], report["cbf_error_pct"],
              report["length_error_pct"], report["track_rms_um_median"],
              report["beat_rms_um"]))
    return 0


def cmd_beat(args) -> int:
    """Report how periodic the beat is, column by column."""
    pqd_path, images_dir = resolve_paths(args)
    if not os.path.isdir(images_dir):
        print("error: images folder not found at {}".format(images_dir),
              file=sys.stderr)
        return 2

    pqd = Spec(pqd_path) if os.path.exists(pqd_path) else None
    video = (VideoFrames.from_pqd(pqd, images_dir) if pqd is not None
             else VideoFrames(images_dir))
    res = beat_strength(video)
    fps = pqd.getf("vid_fps", 300.0) if pqd is not None else 300.0

    print("frames        : {}".format(len(video)))
    print("period        : {:.0f} frames = {:.2f} Hz at {:g} fps".format(
        res["period"], fps / max(res["period"], 1e-9), fps))
    print()
    print("how well the frames repeat with the beat (1.0 = perfect):")
    print("  best        : {:.3f} at column x = {}".format(
        res["best_corr"], res["best_column"]))
    print("  median      : {:.3f} across the field".format(res["median_corr"]))
    if pqd is not None:
        x0 = pqd.getf("vid_cil_x0")
        here = float(np.interp(x0, res["columns"], res["corr"]))
        print("  at your base: {:.3f} (x = {:g})".format(here, x0))
        if here < res["best_corr"] - 0.08:
            print()
            print("the beat repeats better at x = {}; a waveform fitted at a".
                  format(res["best_column"]))
            print("weakly repeating base is poorly constrained whatever the")
            print("fitting method.")
    print()
    print("this is the height of the self-correlation peak above its own")
    print("baseline: a clean side-on view of one cilium measures about 1.1,")
    print("a dense ciliary carpet 0.4 to 0.6.")

    order = np.argsort(res["corr"])[::-1][:8]
    print()
    print("strongest columns:")
    for k in order:
        print("  x = {:>4}   {:.3f} at lag {:.0f} frames".format(
            res["columns"][k], res["corr"][k], res["lag"][k]))
    return 0


def cmd_level(args) -> int:
    """Find the rotation that makes the cilia base line horizontal."""
    pqd_path, images_dir = resolve_paths(args)
    for path, what in ((pqd_path, "specification file"), (images_dir, "images folder")):
        if not os.path.exists(path):
            print("error: {} not found at {}".format(what, path), file=sys.stderr)
            return 2

    pqd = Spec(pqd_path)
    video = VideoFrames.from_pqd(pqd, images_dir)
    try:
        info = level_rotation(video, pqd)
    except ValueError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2

    angle = ((info["rotation"] + 180.0) % 360.0) - 180.0
    if info.get("method") == "none":
        print("no flat surface to fit, and the moving region is too round to")
        print("give a direction.  Rotation left unchanged.")
        return 1

    if info.get("method") == "cilia band":
        print("no flat surface here, so the direction of the cilia band was")
        print("used instead ({} pixels, elongation {:.1f})".format(
            info["n_columns"], info.get("elongation", float("nan"))))
    else:
        print("surface fitted over {} columns (scatter {:.1f} px)".format(
            info["n_columns"], info["fit_rms_px"]))
    print("current rotation : {:+.2f} deg".format(info["current_rotation"]))
    print("surface tilt now : {:+.2f} deg".format(info["tilt_deg"]))
    print()
    print("rotation to level it: {:+.2f} deg   "
          "(converged in {} step{}, {:+.2f} deg left)".format(
              angle, info["iterations"], "" if info["iterations"] == 1 else "s",
              info["residual_deg"]))

    if args.apply:
        pqd.set("vid_image_rotation", angle)
        written = pqd.save(pqd_path, backup=True)
        video.set_rotation(angle)
        after = measure_surface_tilt(video)["tilt_deg"]
        print()
        print("applied; the surface now sits at {:+.2f} deg".format(after))
        print("the base position is in rotated coordinates -- re-run `locate`")
        print("applied to {}{}".format(
            written, " (a .bak of the previous file was made)"
            if pqd.last_backup else ""))
    else:
        print()
        print("not applied -- add --apply to write it into the specification")
    return 0


def cmd_locate(args) -> int:
    """Estimate where the cilium base sits on the image."""
    pqd_path, images_dir = resolve_paths(args)
    for path, what in ((pqd_path, "specification file"), (images_dir, "images folder")):
        if not os.path.exists(path):
            print("error: {} not found at {}".format(what, path), file=sys.stderr)
            return 2

    pqd = Spec(pqd_path)
    model = WaveModel(pqd)
    video = VideoFrames.from_pqd(pqd, images_dir)

    res = locate_base(video, model, step=args.step)

    cur_x = pqd.getf("vid_cil_x0")
    cur_y = pqd.getf("vid_cil_y0_invy")
    ppum = model.ppum or 1.0

    print("frames        : {} .. {}  ({}x{} px)".format(
        video.first, video.last, res["image_w"], res["image_h"]))
    print("cell surface  : row {:.0f} at x = {}".format(
        res["surface_row"], res["x0"]))
    print()
    print("estimated base: x0 = {}   y0_invy = {}".format(
        res["x0"], res["y0_invy"]))
    print("current base  : x0 = {:g}   y0_invy = {:g}".format(cur_x, cur_y))
    print("difference    : {:+.0f} px in x, {:+.0f} px in y  "
          "({:+.2f}, {:+.2f} um)".format(
              res["x0"] - cur_x, (res["image_h"] - res["y0_invy"])
              - (res["image_h"] - cur_y),
              (res["x0"] - cur_x) / ppum, (cur_y - res["y0_invy"]) / ppum))
    print()
    print("confidence    : peak is flat over {} px in x, {} px in y".format(
        res["spread_x"], res["spread_y"]))

    if args.plot:
        _load_pyplot(interactive=False)
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
        axs[0].imshow(res["motion"], cmap="inferno")
        axs[0].plot(range(res["image_w"]), res["surface"], "c-", lw=1,
                    label="cell surface")
        axs[0].plot([cur_x], [res["image_h"] - cur_y], "c+", ms=14, mew=2,
                    label="current")
        axs[0].plot([res["x0"]], [res["y0_row"]], "wx", ms=12, mew=2,
                    label="estimated")
        axs[0].legend(fontsize=8)
        axs[0].set_xticks([]); axs[0].set_yticks([])
        axs[0].set_title("motion map (temporal std)")
        im = axs[1].imshow(res["score_map"], cmap="viridis", aspect="auto",
                           extent=[res["xs"][0], res["xs"][-1],
                                   res["ys"][-1], res["ys"][0]])
        axs[1].plot([cur_x], [res["image_h"] - cur_y], "c+", ms=14, mew=2)
        axs[1].plot([res["x0"]], [res["y0_row"]], "wx", ms=12, mew=2)
        axs[1].set_title("match between swept model and motion")
        fig.colorbar(im, ax=axs[1])
        fig.tight_layout()
        out = args.plot if isinstance(args.plot, str) else "locate_base.png"
        fig.savefig(out, dpi=90)
        plt.close(fig)
        print("wrote {}".format(out))

    try:
        ori = base_orientation(video, model)
        print()
        print("surface at the base tilts {:+.1f} deg (fit scatter {:.1f} px),"
              .format(ori["surface_tilt_deg"], ori["scatter_px"]))
        print("so the cilium should leave it at orientation {:+.1f} deg "
              "(now {:g})".format(ori["orientation"], ori["current"]))
        if not ori["has_surface"]:
            print("but there is no tissue in that column, so this is unusable")
            ori = None
    except Exception as exc:
        print("could not measure the surface direction: {}".format(exc))
        ori = None

    if args.apply:
        pqd.set("vid_cil_x0", res["x0"])
        pqd.set("vid_cil_y0_invy", res["y0_invy"])
        if ori is not None:
            pqd.set("vid_cil_orientation", round(ori["orientation"], 3))
        written = pqd.save(pqd_path, backup=True)
        print()
        print("applied to {}{}".format(
            written, " (a .bak of the previous file was made)"
            if pqd.last_backup else ""))
    else:
        print()
        print("not applied -- add --apply to write it into the specification")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ciliograph",
        description="Convert videos of beating cilia to digital ciliary models.")
    p.add_argument("--version", action="version",
                   version="Ciliograph {}".format(__version__))
    sub = p.add_subparsers(dest="command")

    f = sub.add_parser("fit", help="open the interactive fitter")
    add_common(f)
    f.add_argument("--play-fps", type=float, default=12.0,
                   help="playback speed of the frame viewer (default: %(default)s)")
    f.add_argument("--modes", type=int, default=FOURIER_DEFAULT_MODES,
                   help="number of Fourier modes n to use (default: %(default)s)")
    f.add_argument("--coeffs", default=None,
                   help="start from this Fourier coefficient text file instead "
                        "of the beat in the specification")
    f.add_argument("--space", type=int, default=None,
                   help="powers of s to use, s^1..s^SPACE "
                        "(default: 3, Blake's cubic)")
    f.set_defaults(func=cmd_fit)

    s = sub.add_parser("shapes", help="generate cilium shapes over the beat cycle")
    add_common(s)
    s.add_argument("--n-cs", type=int, default=14,
                   help="number of time steps per beat cycle (default: %(default)s)")
    s.add_argument("--outdir", default=DEFAULT_OUTDIR,
                   help="where to write results (default: %(default)s)")
    s.add_argument("--split", type=int, default=None,
                   help="index splitting recovery from effective stroke "
                        "(default: derived from !phasef_ert)")
    s.add_argument("--figtext", default=None, help="annotation text for the figures")
    s.add_argument("--no-plots", action="store_true", help="skip the PNG figures")
    s.add_argument("--modes", type=int, default=FOURIER_DEFAULT_MODES,
                   help="number of Fourier modes n to use (default: %(default)s)")
    s.add_argument("--coeffs", default=None,
                   help="Fourier coefficient text file; omit to use the beat in "
                        "the specification (or the template before a fit)")
    s.add_argument("--n-pts", type=int, default=None,
                   help="points along the cilium (default: 201)")
    s.add_argument("--equal-s", action="store_true",
                   help="space nodes equally in s instead of equally along the "
                        "arc length")
    s.add_argument("--space", type=int, default=None,
                   help="powers of s to use, s^1..s^SPACE "
                        "(default: 3, Blake's cubic)")
    s.set_defaults(func=cmd_shapes)

    i = sub.add_parser("info", help="summarise a specification file")
    add_common(i)
    i.set_defaults(func=cmd_info)

    cy = sub.add_parser("cycle",
                        help="estimate the beat period and main cycle start")
    add_common(cy)
    cy.add_argument("--apply", action="store_true",
                    help="write the estimate into the specification (start and cbf)")
    cy.set_defaults(func=cmd_cycle)

    ln = sub.add_parser("length",
                        help="estimate the cilium length from the beating")
    add_common(ln)
    ln.add_argument("--apply", action="store_true",
                    help="write the estimate into the specification")
    ln.set_defaults(func=cmd_length)

    ck = sub.add_parser("check",
                        help="read every frame and report any that fail")
    add_common(ck)
    ck.add_argument("--prefix", default=None,
                    help="only files starting with this (default: all images)")
    ck.add_argument("--model", default=None,
                    help="also load this .onnx model and run it on the first "
                         "frame, reporting time and memory")
    ck.add_argument("--model-range", type=int, default=0, dest="model_range",
                    help="pixel range the model expects: 0 = 0..1, 1 = 0..255, "
                         "2 = -1..1")
    ck.set_defaults(func=cmd_check)

    lv = sub.add_parser("level",
                        help="find the rotation that makes the cilia base "
                             "line horizontal")
    add_common(lv)
    lv.add_argument("--apply", action="store_true",
                    help="write the rotation into the specification")
    lv.set_defaults(func=cmd_level)

    bf = sub.add_parser("beatfit",
                        help="fit a beat pattern to a traced centreline file")
    bf.add_argument("trace", help="trace file: index,x,y,time in real units, "
                                  "or cilium_trace_px.csv")
    bf.add_argument("--modes", type=int, default=None,
                    help="time modes to keep (default: chosen by held-out error)")
    bf.add_argument("--space", type=int, default=N_FOURIER_M,
                    help="highest power of s (default: %(default)s, as Blake "
                         "and Ishikawa use)")
    bf.add_argument("--locs", type=int, default=20,
                    help="nodes along the cilium (default: %(default)s)")
    bf.add_argument("--per-period", type=int, default=None,
                    help="samples per beat (default: from the autocorrelation)")
    bf.add_argument("--out", default=None,
                    help="where to write the coefficients")
    bf.set_defaults(func=cmd_beatfit)

    tr = sub.add_parser("track",
                        help="follow one cilium through every frame")
    add_common(tr)
    tr.add_argument("--x", type=float, default=None,
                    help="column of the cilium base (default: from the specification)")
    tr.add_argument("--outdir", default=None,
                    help="where to write the outputs (default: beside the specification)")
    tr.add_argument("--stem", default="cilium",
                    help="stem for the output file names")
    tr.add_argument("--locs", type=int, default=20,
                    help="nodes along the cilium for the waveform fit")
    tr.add_argument("--space", type=int, default=3,
                    help="highest power of s in the waveform fit")
    tr.add_argument("--modes", type=int, default=None,
                    help="time modes (default: chosen by held-out error)")
    tr.add_argument("--per-period", type=int, default=None,
                    dest="per_period",
                    help="frames per beat (default: from the autocorrelation)")
    tr.add_argument("--figure-frames", type=int, default=12,
                    dest="figure_frames", help="frames to show in the overlay")
    tr.add_argument("--no-figure", action="store_true", dest="no_figure",
                    help="skip the overlay figure")
    tr.set_defaults(func=cmd_track)

    sy = sub.add_parser("synth",
                        help="render a synthetic carpet with exact ground truth")
    sy.add_argument("--out", required=True, help="folder to write the frames into")
    sy.add_argument("--coeffs", default=None,
                    help="Fourier coefficient file for the beat (default: the built-in table)")
    sy.add_argument("--length", type=float, default=6.0, help="cilium length, um")
    sy.add_argument("--ppum", type=float, default=40.0, help="pixels per um")
    sy.add_argument("--cbf", type=float, default=10.0, help="beat frequency, Hz")
    sy.add_argument("--fps", type=float, default=1000.0, help="frames per second")
    sy.add_argument("--frames", type=int, default=300, help="number of frames")
    sy.add_argument("--cilia", type=int, default=40, help="number of cilia")
    sy.add_argument("--spacing", type=float, default=0.3, help="distance between bases, um")
    sy.add_argument("--wavelength", type=float, default=8.0,
                    help="metachronal wavelength, um")
    sy.add_argument("--leftward", action="store_true",
                    help="wave travels towards -x instead of +x")
    sy.add_argument("--line-width", type=float, default=2.0, dest="line_width",
                    help="drawn width of a cilium, px")
    sy.add_argument("--blur", type=float, default=0.0, help="Gaussian blur, px")
    sy.add_argument("--noise", type=float, default=0.0,
                    help="Gaussian noise as a fraction of full scale")
    sy.add_argument("--seed", type=int, default=0)
    sy.add_argument("--jitter", type=float, default=0.0,
                    help="random phase offset per cilium, sd in cycles "
                         "(a lawn that does not beat in perfect step)")
    sy.add_argument("--scatter", type=float, default=0.0,
                    help="random shift of each base, as a fraction of the "
                         "spacing (cilia behind one another project to "
                         "random positions)")
    sy.add_argument("--bright", type=float, default=0.0,
                    help="fraction of cilia drawn bright on a grey field "
                         "instead of dark on white (0.5: polarity ambiguous)")
    sy.add_argument("--body", type=float, default=0.0,
                    help="darken the tissue below the wall by this fraction "
                         "of full scale (0.3: like a cell layer)")
    sy.add_argument("--masks", action="store_true",
                    help="also write masks/<frame>.png: every cilium's "
                         "centreline, 2 px wide, white on black -- the answer "
                         "a network is trained to give (train_cilia_net.py)")
    sy.set_defaults(func=cmd_synth)

    va = sub.add_parser("validate",
                        help="run the pipeline on a synth folder and score it")
    va.add_argument("--images", required=True, help="folder written by synth")
    va.add_argument("--outdir", default=None,
                    help="where to write validation.png/json (default: the folder)")
    va.add_argument("--max-traces", type=int, default=120, dest="max_traces",
                    help="frames Auto fit may trace (fewer is faster)")
    va.add_argument("--assume-period", action="store_true", dest="assume_period",
                    help="skip Find cycle and take the period from the truth")
    va.add_argument("--enhance", default="",
                    help="enhancement to apply first, e.g. "
                         "background=1,linesmooth=1.5,deblur=2 (names as in "
                         "the specification without vid_enh_)")
    va.add_argument("--model", default=None,
                    help="an .onnx image model to run on the frames first "
                         "(needs onnxruntime); add modelrange=1 to --enhance "
                         "for a model that expects 0..255, 2 for -1..1")
    add_compute(va)
    va.set_defaults(func=cmd_validate)

    bt = sub.add_parser("beat",
                        help="report how periodic the beat is, and where")
    add_common(bt)
    bt.set_defaults(func=cmd_beat)

    lo = sub.add_parser("locate",
                        help="estimate where the cilium base sits on the image")
    add_common(lo)
    lo.add_argument("--apply", action="store_true",
                    help="write the estimate into the specification")
    lo.add_argument("--plot", nargs="?", const="locate_base.png", default=None,
                    help="save a diagnostic image (default: locate_base.png)")
    lo.add_argument("--step", type=int, default=2,
                    help="search grid step in pixels (default: %(default)s)")
    lo.set_defaults(func=cmd_locate)

    t = sub.add_parser("selftest", help="verify the model against your data")
    add_common(t)
    t.add_argument("--n-cs", type=int, default=14,
                   help="time steps to use for the export check (default: %(default)s)")
    t.add_argument("--reference", default=None,
                   help="a known-good z01a.data to compare against byte for byte")
    t.set_defaults(func=cmd_selftest)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)

    # Bare `python ciliograph.py` opens the fitter, which is what you want
    # almost every time; the other commands are still named explicitly.
    commands = {"fit", "shapes", "info", "cycle", "locate", "level", "length",
                "check", "beatfit", "track", "synth", "validate", "beat",
                "selftest"}
    if not argv or (argv[0] not in commands and argv[0] not in ("-h", "--help",
                                                                "--version")):
        argv = ["fit"] + argv

    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    if getattr(args, "device", None) or getattr(args, "workers", None):
        configure_compute(device=getattr(args, "device", None),
                          workers=getattr(args, "workers", None))
    return args.func(args)


def _ort_probe_main(argv) -> int:
    """``--ort-probe [model]``: import onnxruntime (and open the model) in
    this process -- the standalone .exe's stand-in for ``python -c``."""
    import onnxruntime as ort
    print("ORT-VERSION", ort.__version__, flush=True)
    if argv:
        o = ort.SessionOptions()
        o.log_severity_level = 3
        ort.InferenceSession(argv[0], o, providers=["CPUExecutionProvider"])
        print("ORT-MODEL-OK", flush=True)
    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # worker processes of the standalone .exe
    if len(sys.argv) > 1 and sys.argv[1] == "--ort-probe":
        raise SystemExit(_ort_probe_main(sys.argv[2:]))
    raise SystemExit(main())
