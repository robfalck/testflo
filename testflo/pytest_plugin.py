"""
testflo-pytest: testflo-style MPI test execution as a pytest plugin.

Design
------
This plugin brings testflo's MPI execution model to pytest:

* Tests marked with ``@pytest.mark.parallel(mpi=N)`` (``parallel(N)`` and
  ``parallel(nprocs=N)`` are aliases) or belonging to a unittest.TestCase
  with an ``N_PROCS`` class attribute (testflo syntax) are executed under a
  spawned ``mpirun -n N`` subprocess.

* ``@pytest.mark.parallel(multiprocessing=M)`` declares a test that does
  *not* need MPI but will spin up its own pool of M worker processes.  Such
  tests run in-process on a ``FakeComm``; the marker exists so the plugin
  can account for the cores they use.  The two may be combined:
  ``parallel(mpi=N, multiprocessing=M)`` costs ``N * M`` cores (each rank
  spawns its own pool); a serial test costs 1.  By default a test only
  starts when its core cost fits within the cores available to the machine,
  counting every test already in flight across all pytest-xdist workers
  (serial ones included, since a busy worker is a busy core);
  ``--max-concurrent-cores`` overrides the budget and ``--oversubscribe``
  removes it.

* Inside that subprocess, *every rank* runs the test body naturally.  Plain
  ``assert`` statements work per-rank -- no special ``parallel_assert`` is
  required.  Like testflo's ``mpirun.py``, per-rank results are gathered
  collectively *after* each rank's test protocol has completed, so a failure
  on a subset of ranks cannot desynchronize the result collection itself.

* Rank 0 of the child serializes the per-rank outcomes to a results file.
  The launching (host) pytest process reads that file and synthesizes normal
  pytest TestReports, so parallel tests show up in the terminal, in
  ``-ra`` summaries, and in JUnit XML exactly like ordinary tests, with
  per-rank tracebacks and captured output attached.

The host process never imports ``mpi4py.MPI`` (and therefore never
initializes MPI), mirroring testflo's launcher behavior.  This keeps the
launcher safe to fork subprocesses from and avoids nested-MPI issues.

Execution modes
---------------
1. **Launcher mode** (normal ``pytest`` invocation): parallel tests are run
   via spawned ``mpirun`` subprocesses; serial tests run in place.
2. **Child mode** (``TESTFLO_PYTEST_CHILD=1``, set by the launcher): the test
   runs unmodified on every rank; results are gathered at session finish.
3. **Outer-mpirun mode** (user ran ``mpirun -n N pytest ...`` themselves):
   parallel tests whose nprocs matches the world size run in place; all other
   tests are skipped.
"""

import os
import sys
import json
import time
import shutil
import numbers
import subprocess
import functools
import collections
import collections.abc

import pytest


# ---------------------------------------------------------------------------
# constants / environment flags
# ---------------------------------------------------------------------------

CHILD_FLAG = "TESTFLO_PYTEST_CHILD"
"""Set to '1' in the environment of the spawned mpirun child processes."""

RESULTS_FLAG = "TESTFLO_PYTEST_RESULTS"
"""Path of the JSON results file the child's rank 0 writes."""

DEFAULT_NPROCS = 2
"""nprocs used when ``@pytest.mark.parallel`` is given with no arguments."""


def _is_child():
    return os.environ.get(CHILD_FLAG) == "1"


def _outer_world_size():
    """Determine the MPI world size from launcher-provided environment
    variables *without* importing mpi4py (which would initialize MPI in
    the launcher process).
    """
    for var in ("OMPI_COMM_WORLD_SIZE",      # Open MPI
                "PMI_SIZE",                   # MPICH / hydra / Intel MPI
                "MV2_COMM_WORLD_SIZE",        # MVAPICH2
                "MSMPI_RANK_SIZE"):           # MS-MPI
        val = os.environ.get(var)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
    return 1


def _under_mpi():
    """True when this process is itself an MPI rank: a spawned child, or
    the user ran ``mpirun -n N pytest`` (outer-mpirun mode)."""
    return _is_child() or _outer_world_size() > 1


_MPIRUN_KEY = pytest.StashKey()
"""Path of the mpirun/mpiexec to spawn with (None if none was found);
resolved once in ``pytest_configure``."""


@functools.lru_cache(maxsize=None)
def _mpirun_extra_args(mpirun_exe):
    """Extra args for the detected MPI implementation.

    Open MPI refuses to start more ranks than detected cores unless
    ``--oversubscribe`` is given, which is a common failure mode in CI.
    Detection is done by running ``mpirun --version`` (cheap, cached) rather
    than importing mpi4py in the launcher.
    """
    try:
        out = subprocess.run([mpirun_exe, "--version"], capture_output=True,
                             text=True, timeout=30).stdout.lower()
    except Exception:
        return []
    if "open mpi" in out or "open-mpi" in out or "openrte" in out:
        return ["--oversubscribe"]
    return []


@functools.lru_cache(maxsize=None)
def _have_mpi4py():
    import importlib.util
    return importlib.util.find_spec("mpi4py") is not None


# ---------------------------------------------------------------------------
# marker parsing (mpi-pytest compatible)
# ---------------------------------------------------------------------------

def _as_tuple(arg):
    return tuple(arg) if isinstance(arg, collections.abc.Iterable) else (arg,)


_BAD_MARKER_MSG = (
    "Bad arguments given to parallel marker; expected parallel(N), "
    "parallel(mpi=N), parallel([N1, N2, ...]), parallel(multiprocessing=M) "
    "or parallel(mpi=N, multiprocessing=M)")


def _parse_marker(marker):
    """Return ``(mpi_sizes, multiprocessing)`` requested by a parallel marker.

    ``mpi_sizes`` is a tuple of MPI communicator sizes (more than one entry
    means the test is parametrized over sizes); ``multiprocessing`` is the
    number of worker processes the test itself will spawn (0 if none).

    Accepted forms: bare ``parallel`` (-> ``DEFAULT_NPROCS`` ranks),
    ``parallel(N)``, ``parallel([N1, N2])``, ``parallel(mpi=N)``,
    ``parallel(nprocs=N)`` (back-compat alias for ``mpi``),
    ``parallel(multiprocessing=M)`` and ``parallel(mpi=N, multiprocessing=M)``.
    """
    kwargs = dict(marker.kwargs)
    mp = kwargs.pop("multiprocessing", 0)
    mpi_kw = [k for k in ("mpi", "nprocs") if k in kwargs]

    if kwargs and set(kwargs) - {"mpi", "nprocs"}:
        raise pytest.UsageError(_BAD_MARKER_MSG)
    if len(marker.args) > 1 or (marker.args and mpi_kw) or len(mpi_kw) > 1:
        raise pytest.UsageError(_BAD_MARKER_MSG)

    if marker.args:
        mpi = _as_tuple(marker.args[0])
    elif mpi_kw:
        mpi = _as_tuple(kwargs[mpi_kw[0]])
    elif "multiprocessing" in marker.kwargs:
        mpi = (0,)   # multiprocessing-only: no mpirun
    else:
        mpi = (DEFAULT_NPROCS,)

    if (not isinstance(mp, numbers.Integral) or isinstance(mp, bool)
            or mp < 0):
        raise pytest.UsageError(
            "parallel marker: multiprocessing must be a non-negative int")
    if not mpi or not all(isinstance(n, numbers.Integral) and n >= 0
                          for n in mpi):
        raise pytest.UsageError(_BAD_MARKER_MSG)

    return tuple(int(n) for n in mpi), int(mp)


def _cores(mpi, multiprocessing):
    """Total cores a test needs: each MPI rank spawns its own pool."""
    return max(int(mpi), 1) * max(int(multiprocessing), 1)


ParallelSpec = collections.namedtuple("ParallelSpec",
                                      "mpi multiprocessing cores")
"""What a collected item needs: MPI ranks (<= 1 means no mpirun), worker
processes it spawns itself, and the resulting core cost."""

_SPEC_KEY = pytest.StashKey()
"""``ParallelSpec`` of every collected item."""


def _parallel_spec_for_item(item):
    """Return the ``ParallelSpec`` for a collected test item.

    Resolution order for the MPI size:
      1. ``[nprocs=N]`` parametrization (from a multi-valued parallel marker)
      2. ``@pytest.mark.parallel`` marker
      3. testflo-style ``N_PROCS`` class attribute
    """
    marker = item.get_closest_marker("parallel")
    mp = 0
    if marker is not None:
        mpis, mp = _parse_marker(marker)
        if hasattr(item, "callspec") and "_nprocs" in item.callspec.params:
            mpi = int(item.callspec.params["_nprocs"])
        elif len(mpis) != 1:
            # should have been parametrized away in pytest_generate_tests
            raise pytest.UsageError(
                f"multi-valued parallel marker on {item.nodeid} was not "
                "parametrized; is pytest_generate_tests being blocked?")
        else:
            mpi = mpis[0]
    else:
        n = getattr(getattr(item, "cls", None), "N_PROCS", None)
        mpi = int(n) if n is not None else 1

    return ParallelSpec(mpi, mp, _cores(mpi, mp))


# ---------------------------------------------------------------------------
# hooks common to all modes
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    group = parser.getgroup("testflo", "testflo-style MPI execution")
    group.addoption(
        "--nompi", action="store_true", default=False,
        help="run parallel-marked tests in the current process on a "
             "communicator of size 1 instead of spawning MPI (testflo's "
             "--nompi).")
    group.addoption(
        "--mpirun-exe", action="store", default=None,
        help="path to the mpirun/mpiexec executable to use for spawning "
             "parallel tests (default: found on PATH).")
    group.addoption(
        "--mpi-timeout", action="store", type=float, default=None,
        help="timeout in seconds for each spawned MPI test; guards against "
             "deadlocks from desynchronized collective calls.")
    group.addoption(
        "--max-concurrent-cores", action="store", type=int, default=None,
        metavar="N", dest="max_concurrent_cores",
        help="maximum total core cost of tests in flight at once across "
             "all pytest-xdist workers (default: the number of cores "
             "available to this process). A test's cost is "
             "max(mpi,1)*max(multiprocessing,1), 1 for a serial test; a "
             "test waits until its cost fits within the budget before it "
             "starts, and a test that can never fit is reported as failed.")
    group.addoption(
        "--oversubscribe", action="store_true", default=False,
        help="remove the core budget: run tests regardless of how many "
             "cores are free (same as TESTFLO_PYTEST_OVERSUBSCRIBE=1). "
             "An explicit --max-concurrent-cores still applies.")
    group.addoption(
        "--mpi-concurrent-slots", action="store", type=int, default=None,
        metavar="N", dest="max_concurrent_cores",
        help="deprecated alias for --max-concurrent-cores.")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "parallel(mpi=N, multiprocessing=M): run this test under MPI on N "
        f"processes (default: {DEFAULT_NPROCS}; positional N or nprocs=N are "
        "aliases; a list parametrizes over communicator sizes) and/or "
        "declare that the test spawns M worker processes itself. "
        "multiprocessing-only tests run in-process. The test's core cost "
        "is max(N,1)*max(M,1), used by --max-concurrent-cores.")

    config.stash[_MPIRUN_KEY] = (config.getoption("--mpirun-exe")
                                 or shutil.which("mpirun")
                                 or shutil.which("mpiexec"))

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_cmdline_main(config):
    """Default to ``--dist worksteal`` when xdist is active and the user did
    not set ``--dist`` explicitly.

    A worker that is waiting for its test's cores to become free cannot
    hand the test back to xdist, so it simply waits; with work stealing the
    tests queued behind it are picked up by idle workers instead of being
    held hostage.  (Older xdist without a worksteal scheduler keeps its
    default ``load`` distribution.)

    Uses tryfirst+hookwrapper so config.option.dist is set *before* xdist's
    own tryfirst pytest_cmdline_main body runs and creates the scheduler.
    """
    if config.pluginmanager.hasplugin("xdist") and not _is_child():
        if config.getoption("--dist", default="no") == "no" and _have_worksteal():
            config.option.dist = "worksteal"
    outcome = yield
    return outcome


@functools.lru_cache(maxsize=None)
def _have_worksteal():
    try:
        from xdist.scheduler import WorkStealingScheduling  # noqa: F401
    except ImportError:
        return False
    return True


def pytest_sessionstart(session):
    """Running xdist *inside* an outer mpirun makes no sense (every rank
    would start its own worker cluster); reject it up front.  Note this
    check needs no mpi4py import (see _outer_world_size).
    """
    if _is_child() or _outer_world_size() <= 1:
        return
    try:
        import xdist
    except ImportError:
        return
    if xdist.is_xdist_controller(session) or xdist.is_xdist_worker(session):
        raise pytest.UsageError(
            "pytest-xdist cannot be combined with an outer mpirun launch; "
            "run plain `pytest -n ...` and let testflo-pytest spawn MPI "
            "per-test instead")


def pytest_generate_tests(metafunc):
    """Split multi-valued parallel markers into one test per size.

    ``@pytest.mark.parallel([2, 3])`` becomes ``test[nprocs=2]`` and
    ``test[nprocs=3]``.  (Same behavior as mpi-pytest.)  This must run in
    both launcher and child mode so node IDs match between the two.
    """
    markers = tuple(m for m in getattr(metafunc.function, "pytestmark", ())
                    if m.name == "parallel")
    if not markers:
        return

    marker, = markers
    nprocss, _ = _parse_marker(marker)
    if len(nprocss) > 1:
        metafunc.fixturenames.append("_nprocs")
        metafunc.parametrize("_nprocs", nprocss, ids=lambda n: f"nprocs={n}")


def pytest_collection_modifyitems(config, items):
    """Attach parallel markers to testflo-style ``N_PROCS`` TestCase tests
    so they are visible to ``-m parallel`` selection, stash every item's
    ``ParallelSpec``, and mark MPI tests skipped when MPI cannot be used.

    No special xdist grouping is done: every test, serial or parallel, is
    distributed freely and reserves its core cost from the shared budget
    just before it runs (see ``pytest_runtest_protocol``).
    """
    no_mpi = None
    if not _under_mpi() and not config.getoption("--nompi"):
        if not _have_mpi4py():
            no_mpi = "mpi4py is required to run parallel tests (or use --nompi)"
        elif config.stash[_MPIRUN_KEY] is None:
            no_mpi = ("mpirun/mpiexec was not found in the system path "
                      "(or use --nompi)")

    for item in items:
        if item.get_closest_marker("parallel") is None:
            n = getattr(getattr(item, "cls", None), "N_PROCS", None)
            if n is not None and int(n) > 1:
                item.add_marker(pytest.mark.parallel(nprocs=int(n)))
        spec = _parallel_spec_for_item(item)
        item.stash[_SPEC_KEY] = spec
        if spec.mpi > 1 and no_mpi:
            item.add_marker(pytest.mark.skip(reason=no_mpi))


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

class FakeComm(object):
    """Stand-in for an MPI communicator when running without MPI
    (serial tests, or parallel tests under ``--nompi``)."""

    rank = 0
    size = 1

    def barrier(self):
        pass

    Barrier = barrier

    def bcast(self, obj, root=0):
        return obj

    def gather(self, obj, root=0):
        return [obj]

    def allgather(self, obj):
        return [obj]

    def allreduce(self, obj, op=None):
        return obj


@pytest.fixture
def comm(request):
    """The communicator this test is running on.

    ``MPI.COMM_WORLD`` when running under MPI, or a size-1 ``FakeComm``
    when MPI is not active (serial tests, ``--nompi``).  This lets test
    bodies be written against ``comm.rank`` / ``comm.size`` without
    conditional imports.
    """
    if _under_mpi():
        from mpi4py import MPI
        return MPI.COMM_WORLD
    if "mpi4py.MPI" in sys.modules and not request.config.getoption("--nompi"):
        # MPI is already initialized in this process (e.g. by the code under
        # test); hand out the real COMM_WORLD.  We deliberately never
        # *initialize* MPI in the launcher process ourselves.
        from mpi4py import MPI
        return MPI.COMM_WORLD
    return FakeComm()


@pytest.fixture(autouse=True)
def _mpi_barrier_finalize(request):
    """Barrier at the end of each test when running under MPI, to localize
    tests that are not fully collective (same idea as mpi-pytest)."""
    if _under_mpi():
        from mpi4py import MPI
        request.addfinalizer(MPI.COMM_WORLD.barrier)


# ---------------------------------------------------------------------------
# child mode: run naturally, gather at the end
# ---------------------------------------------------------------------------

_child_results = collections.defaultdict(list)
"""nodeid -> every report pytest produced for it on this rank (setup, call,
teardown, or a failed collection).  Reduction to one outcome per rank
happens in the launcher (``_aggregate_rank_results``)."""


def _serialize_report(report):
    longrepr = report.longrepr
    if longrepr is not None and not isinstance(longrepr, tuple):
        # skips carry a plain (path, lineno, reason) tuple; anything else
        # is a repr object that only needs to survive as text
        longrepr = str(longrepr)
    return {"when": report.when,
            "outcome": report.outcome,
            "longrepr": longrepr,
            "sections": [list(sec) for sec in report.sections],
            "duration": getattr(report, "duration", 0.0),
            "wasxfail": getattr(report, "wasxfail", None)}


@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report):
    if _is_child():
        _child_results[report.nodeid].append(_serialize_report(report))


@pytest.hookimpl(trylast=True)
def pytest_collectreport(report):
    if _is_child() and report.failed:
        _child_results[report.nodeid or "<collection>"].append(
            _serialize_report(report))


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """In child mode, gather per-rank results on COMM_WORLD and have rank 0
    write them to the results file.

    This gather happens *after* every rank has finished its own test
    protocol -- pytest has already caught any assertion errors per-rank --
    which is what makes plain asserts safe here (testflo's model).
    """
    if not _is_child():
        return

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    rank_payload = {"rank": comm.rank, "results": _child_results}
    all_payloads = comm.gather(rank_payload, root=0)

    if comm.rank == 0:
        results_path = os.environ.get(RESULTS_FLAG)
        if results_path:
            with open(results_path, "w") as f:
                json.dump({"nprocs": comm.size, "ranks": all_payloads}, f)


# ---------------------------------------------------------------------------
# launcher mode: spawn mpirun and synthesize reports
# ---------------------------------------------------------------------------

def _should_launch_mpi(item):
    """True if this item should be run via a spawned mpirun subprocess:
    launcher mode, more than one rank requested, MPI not disabled, and the
    item not already marked skip (e.g. because MPI is unavailable -- see
    ``pytest_collection_modifyitems``)."""
    return (not _under_mpi()
            and not item.config.getoption("--nompi")
            and item.stash[_SPEC_KEY].mpi > 1
            and item.get_closest_marker("skip") is None)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    if _is_child():
        return
    world = _outer_world_size()
    if world > 1:
        # outer-mpirun mode: user ran `mpirun -n N pytest ...` themselves
        nprocs = item.stash[_SPEC_KEY].mpi
        if nprocs != world:
            pytest.skip(f"test requires {nprocs} MPI procs but pytest was "
                        f"launched with {world}")


OVERSUBSCRIBE_FLAG = "TESTFLO_PYTEST_OVERSUBSCRIBE"
"""Env var equivalent of ``--oversubscribe``."""


def _available_cores():
    """Cores this process may use: the affinity mask where the OS exposes
    one (containers/CI runners often restrict it), else the logical count.
    """
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _core_budget(config):
    """Resolve the core budget: explicit ``--max-concurrent-cores`` wins;
    otherwise ``--oversubscribe`` / ``TESTFLO_PYTEST_OVERSUBSCRIBE=1`` means
    unlimited (None); otherwise the cores available to this process.
    """
    explicit = config.getoption("max_concurrent_cores")
    if explicit is not None:
        return explicit
    if (config.getoption("--oversubscribe")
            or os.environ.get(OVERSUBSCRIBE_FLAG) == "1"):
        return None
    return _available_cores()


def _acquire_cores(item, cores):
    """Reserve ``cores`` from the core budget.

    Returns ``(limiter, None)`` on success (``limiter`` is None when no
    budget is configured) or ``(None, error_message)`` when the test can
    never fit or the wait timed out.  The caller must ``release()`` the
    limiter when the test is done.
    """
    max_cores = _core_budget(item.config)
    if max_cores is None:
        return None, None
    if cores > max_cores:
        return None, (f"test requires {cores} cores but "
                      f"--max-concurrent-cores={max_cores}")
    limiter = _SlotLimiter(max_cores)
    timeout = item.config.getoption("--mpi-timeout")
    try:
        # bound the wait so a wedged run can't block forever
        limiter.acquire(cores, timeout=(timeout or 3600) * 10)
    except TimeoutError as exc:
        return None, str(exc)
    return limiter, None


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    """Run every test in launcher mode inside a core reservation.

    A serial test costs 1 core, a multiprocessing test its pool size, and
    an MPI test ranks * pool; the reservation is what lets a parallel test
    see the serial tests other xdist workers are busy with.  MPI tests then
    execute entirely inside the spawned ``mpirun`` (fixtures included, so
    nothing is double-executed); everything else runs pytest's normal
    protocol in this process.  Without xdist a 1-core test has nothing to
    contend with and is left to pytest untouched.
    """
    if _under_mpi():
        return None  # child / outer-mpirun: plain pytest protocol
    launch = _should_launch_mpi(item)
    spec = item.stash[_SPEC_KEY]
    # in-process, only the test's own pool counts (no ranks are spawned)
    cores = spec.cores if launch else max(spec.multiprocessing, 1)
    if not launch and cores == 1 and "PYTEST_XDIST_TESTRUNUID" not in os.environ:
        return None

    ihook = item.ihook
    ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)

    limiter, err = _acquire_cores(item, cores)
    if err is not None:
        reports = _failed_reports(item, err)
    else:
        try:
            if launch:
                reports = _run_mpi_item(item)
            else:
                from _pytest.runner import runtestprotocol
                runtestprotocol(item, nextitem=nextitem)  # logs its own
                reports = ()
        finally:
            if limiter is not None:
                limiter.release()
    for rep in reports:
        ihook.pytest_runtest_logreport(report=rep)

    ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True


def _make_report(item, when, outcome, longrepr=None, sections=(),
                 duration=0.0, wasxfail=None):
    from _pytest.reports import TestReport

    kwargs = dict(nodeid=item.nodeid,
                  location=item.location,
                  # {name: 1} form, as pytest's own reports use -- the
                  # marker objects themselves are not serializable by
                  # pytest-xdist's execnet transport
                  keywords={x: 1 for x in item.keywords},
                  outcome=outcome,
                  longrepr=longrepr,
                  when=when,
                  sections=list(sections),
                  duration=duration)
    now = time.time()
    report = TestReport(start=now - duration, stop=now, **kwargs)
    if wasxfail is not None:
        report.wasxfail = wasxfail
    return report


def _reports(item, call_report):
    """The three-phase report list for a test whose whole protocol ran
    elsewhere: setup and teardown trivially passed around ``call_report``."""
    return [_make_report(item, "setup", "passed"),
            call_report,
            _make_report(item, "teardown", "passed")]


def _failed_reports(item, longrepr):
    return _reports(item, _make_report(item, "call", "failed",
                                       longrepr=longrepr))


class _SlotLimiter(object):
    """Cross-process budget of cores in use by running tests.

    pytest-xdist workers are independent processes, so the budget is kept
    in a small JSON state file in the temp directory, keyed by xdist's
    ``PYTEST_XDIST_TESTRUNUID`` (all workers of one run share it; without
    xdist the key falls back to this process's pid, where the limiter is
    trivially correct).

    The state maps ``pid -> cores`` for every current holder, where pid is
    the *pytest worker* holding the reservation (never the mpirun child),
    plus a FIFO of waiting requests so large requests are not starved.
    Mutual exclusion is an OS byte-range lock on the state file itself
    (``fcntl.flock`` / ``msvcrt.locking``): no lockfile is created or
    deleted per cycle, and the OS drops the lock if its holder dies.

    Leak safety: a crash inside a spawned mpirun (segfault, OOM, abort)
    only kills the child; ``subprocess.run`` returns to the worker, which
    releases normally.  If the worker itself dies while holding cores, its
    pid is pruned from the state by the next acquire that finds the budget
    full (``_read_holders(prune=True)``), so a dead worker cannot starve
    the rest of the run.
    """

    POLL = 0.05         # seconds between acquire attempts when full

    def __init__(self, max_slots):
        import tempfile
        self.max_slots = max_slots
        key = os.environ.get("PYTEST_XDIST_TESTRUNUID", str(os.getpid()))
        base = os.path.join(tempfile.gettempdir(), f"testflo_pytest_{key}")
        self.state_path = base + ".slots"

    @staticmethod
    def _pid_alive(pid):
        if sys.platform == "win32":
            # os.kill(pid, 0) only does OpenProcess here, which still
            # succeeds for a process that has *exited* while any handle to
            # it remains open -- e.g. the xdist controller's Popen handle
            # on a crashed worker.  Ask the process object whether it is
            # actually still running.
            import ctypes
            SYNCHRONIZE = 0x00100000
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if not handle:
                return False
            # WaitForSingleObject(0) == WAIT_TIMEOUT (0x102) => still running
            alive = kernel32.WaitForSingleObject(handle, 0) == 0x102
            kernel32.CloseHandle(handle)
            return alive
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    @staticmethod
    def _oslock(f):
        if sys.platform == "win32":
            import msvcrt
            # LK_LOCK only retries once a second, so spin on LK_NBLCK
            while True:
                f.seek(0)
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    return
                except OSError:
                    time.sleep(0.001)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX)

    @staticmethod
    def _osunlock(f):
        if sys.platform == "win32":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_UN)

    def _locked(self, fn):
        """Run ``fn(data) -> result`` with the state file exclusively locked
        and write ``data`` back afterwards; returns ``result``."""
        fd = os.open(self.state_path, os.O_RDWR | os.O_CREAT)
        with os.fdopen(fd, "r+") as f:
            self._oslock(f)
            try:
                f.seek(0)
                try:
                    data = json.loads(f.read() or "{}")
                except ValueError:
                    data = {}
                data.setdefault("holders", {})    # pid -> cores
                data.setdefault("waiting", [])    # [[pid, cores, seq]]
                data.setdefault("seq", 0)
                data.setdefault("hwm", 0)
                result = fn(data)
                f.seek(0)
                f.truncate()
                f.write(json.dumps(data))
                f.flush()
                return result
            finally:
                self._osunlock(f)

    def _prune(self, data):
        """Drop holders and waiters whose process died without releasing."""
        data["holders"] = {pid: n for pid, n in data["holders"].items()
                           if self._pid_alive(int(pid))}
        data["waiting"] = [w for w in data["waiting"]
                           if self._pid_alive(int(w[0]))]

    def acquire(self, nprocs, timeout=None):
        """Block until ``nprocs`` cores are reserved for this process.

        Requests are served in FIFO order, with one relaxation: a request
        may pass the ones queued ahead of it if it still leaves enough free
        cores for the oldest of them.  Serial tests can therefore keep
        flowing around a waiting multi-core test, but never in a way that
        stops it from ever fitting -- without this an 8-rank test on an
        8-core box would wait until the whole serial queue drained.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        me = str(os.getpid())

        def room(data):
            """Whether my request fits now without squeezing out the
            oldest request queued ahead of me."""
            in_use = sum(data["holders"].values())
            mine = next((w[2] for w in data["waiting"] if w[0] == me), None)
            ahead = [w for w in data["waiting"] if mine is None or w[2] < mine]
            need = nprocs + (ahead[0][1] if ahead else 0)
            return in_use + need <= self.max_slots, in_use

        def try_take(data):
            ok, in_use = room(data)
            if not ok:
                # maybe only because someone died; the pid probes are only
                # worth doing when we would otherwise wait
                self._prune(data)
                ok, in_use = room(data)
            if ok:
                self._leave(data, me)
                data["holders"][me] = nprocs
                data["hwm"] = max(data["hwm"], in_use + nprocs)
            elif not any(w[0] == me for w in data["waiting"]):
                data["seq"] += 1
                data["waiting"].append([me, nprocs, data["seq"]])
            return ok

        while True:
            if self._locked(try_take):
                return
            if deadline is not None and time.monotonic() > deadline:
                self._locked(lambda d: self._leave(d, me))
                raise TimeoutError(
                    f"timed out waiting for {nprocs} cores "
                    f"(budget: --max-concurrent-cores={self.max_slots})")
            time.sleep(self.POLL)

    @staticmethod
    def _leave(data, me):
        data["waiting"] = [w for w in data["waiting"] if w[0] != me]

    def release(self):
        me = str(os.getpid())

        self._locked(lambda data: data["holders"].pop(me, None))


def _clean_child_env():
    """Build the environment for the spawned mpirun, free of any MPI
    job-identity state.

    Why this matters: if MPI ends up initialized in the launcher process
    (e.g. a test module does an unguarded top-level ``from mpi4py import
    MPI`` that gets imported during collection), Open MPI's singleton
    ``MPI_Init`` injects job-identity variables (``OMPI_MCA_ess=singleton``,
    ``PMIX_RANK``, ``PMIX_NAMESPACE``, ``PMIX_SERVER_URI*``, ...) into the
    process environment via C-level ``setenv()``.  A nested ``mpirun`` that
    inherits those tries to attach to the parent's singleton PMIx job and
    fails -- silently, on Ubuntu-packaged Open MPI.  This is the "nested
    MPI_Init" limitation documented by mpi-pytest's forking mode.

    Two layers of defense here:

    1. ``os.environ`` is Python's snapshot from interpreter startup and
       never reflects libmpi's later ``setenv()`` calls, so building the
       child env from it (rather than letting the child inherit the live C
       environ) already drops singleton-init pollution.
    2. We additionally scrub any PMIx/PMI/ORTE/PRRTE job-identity variables
       that were present at launcher startup, without touching user MCA
       tuning variables (``OMPI_MCA_*`` other than launch/ess state) or
       ``OMPI_ALLOW_RUN_AS_ROOT*``.

    The primary defense, inherited from testflo's design, is that this
    plugin never initializes MPI in the launcher at all.
    """
    scrub_prefixes = ("PMIX_", "PMI_", "PRTE_", "PRRTE_", "ORTE_",
                      "OMPI_COMM_WORLD_", "OMPI_MCA_ess",
                      "OMPI_MCA_orte", "OMPI_MCA_prte",
                      "OMPI_APP_CTX_", "OMPI_UNIVERSE_")
    return {k: v for k, v in os.environ.items()
            if not k.startswith(scrub_prefixes)}


def _child_nodeid(item):
    """The node ID to select in the child pytest invocation.

    Under ``--dist loadgroup``, xdist rewrites worker node IDs to
    ``<nodeid>@<group>``; that suffix means nothing to a fresh pytest
    session, so strip it.  Only a suffix matching one of the item's actual
    xdist_group names is removed, which keeps legitimate ``@`` characters
    inside parametrization brackets safe.
    """
    nodeid = item.nodeid
    for mark in item.iter_markers("xdist_group"):
        gname = (mark.args[0] if mark.args
                 else mark.kwargs.get("name", "default"))
        suffix = f"@{gname}"
        if nodeid.endswith(suffix):
            return nodeid[:-len(suffix)]
    return nodeid


def _run_mpi_item(item):
    """Spawn ``mpirun -n nprocs python -m pytest <nodeid>`` and convert the
    gathered per-rank results into TestReports for the launcher session.
    """
    import tempfile

    config = item.config
    mpirun = config.stash[_MPIRUN_KEY]
    timeout = config.getoption("--mpi-timeout")

    fd, results_path = tempfile.mkstemp(prefix="testflo_pytest_",
                                        suffix=".json")
    os.close(fd)

    env = _clean_child_env()
    env[CHILD_FLAG] = "1"
    env[RESULTS_FLAG] = results_path
    # match testflo's behavior when testing OpenMDAO-based code
    env.setdefault("OPENMDAO_USE_MPI", "1")

    cmd = ([mpirun] + _mpirun_extra_args(mpirun) +
           ["-n", str(item.stash[_SPEC_KEY].mpi),
            sys.executable, "-m", "pytest",
            _child_nodeid(item),
            "-q", "--no-header", "-p", "no:cacheprovider"])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              env=env, timeout=timeout,
                              cwd=str(config.rootpath))
        data = _read_results(results_path)
    except subprocess.TimeoutExpired:
        return _failed_reports(
            item, f"MPI test timed out after {timeout} s.\n"
                  "A common cause is desynchronized collective calls: a "
                  "rank-local failure occurred before a collective "
                  "operation inside the test body.")
    finally:
        try:
            os.remove(results_path)
        except OSError:
            pass

    if data is None:
        # the child crashed or aborted before results could be gathered
        return _failed_reports(
            item, f"MPI subprocess exited with code {proc.returncode} "
                  "before results could be collected.\n"
                  f"command: {' '.join(cmd)}\n"
                  f"--- stdout ---\n{proc.stdout}\n"
                  f"--- stderr ---\n{proc.stderr}")

    return _reports(item, _aggregate_rank_results(item, data))


def _read_results(path):
    """The child's gathered results, or None if it never wrote them."""
    try:
        with open(path) as f:
            contents = f.read()
        return json.loads(contents) if contents.strip() else None
    except (OSError, json.JSONDecodeError):
        return None


def _aggregate_rank_results(item, data):
    """Combine per-rank outcomes into a single call report, testflo-style:
    any rank failing fails the test, with per-rank tracebacks and captured
    output labeled by rank.
    """
    nprocs = data["nprocs"]
    failures = []      # (rank, longrepr)
    skips = []         # (rank, longrepr)
    sections = []
    duration = 0.0
    wasxfail = None
    n_with_results = 0

    for payload in sorted(data["ranks"], key=lambda p: p["rank"]):
        rank = payload["rank"]
        # one nodeid per child invocation (plus possible collect errors)
        reports = [r for reps in payload["results"].values() for r in reps]
        if not reports:
            continue
        n_with_results += 1
        duration = max(duration, sum(r["duration"] or 0.0 for r in reports))
        for r in reports:
            if r["wasxfail"] is not None:
                wasxfail = r["wasxfail"]
            sections.extend((f"rank {rank}: {name}", content)
                            for name, content in r["sections"] if content)
        failed = [r for r in reports if r["outcome"] == "failed"]
        skipped = [r for r in reports if r["outcome"] == "skipped"]
        if failed:
            failures.append((rank, failed[0]["longrepr"] or "(no traceback)"))
        elif skipped:
            skips.append((rank, skipped[0]["longrepr"] or "skipped"))

    if n_with_results == 0:
        return _make_report(item, "call", "failed",
                            longrepr="MPI child ranks reported no results "
                                     "(test may not have been collected in "
                                     "the subprocess).")

    if failures:
        parts = []
        for rank, longrepr in failures:
            parts.append(f"{'=' * 30} rank {rank} of {nprocs} {'=' * 30}\n"
                         f"{longrepr}")
        ok_ranks = sorted(set(range(nprocs))
                          - {r for r, _ in failures}
                          - {r for r, _ in skips})
        if ok_ranks:
            parts.append(f"(ranks {ok_ranks} passed)")
        return _make_report(item, "call", "failed",
                            longrepr="\n".join(parts),
                            sections=sections, duration=duration)

    if skips and len(skips) >= n_with_results:
        # every reporting rank skipped (or xfailed)
        if wasxfail is not None:
            # xfail: pytest represents this as a skipped call report
            # carrying a `wasxfail` attribute -> terminal shows XFAIL
            return _make_report(item, "call", "skipped",
                                longrepr=str(skips[0][1]), sections=sections,
                                duration=duration, wasxfail=wasxfail)
        reason = skips[0][1]
        if isinstance(reason, (list, tuple)):   # child's (path, lineno, reason)
            reason = reason[2]
        longrepr = (str(item.path), item.location[1], str(reason))
        return _make_report(item, "call", "skipped", longrepr=longrepr,
                            sections=sections, duration=duration)

    return _make_report(item, "call", "passed", sections=sections,
                        duration=duration, wasxfail=wasxfail)
