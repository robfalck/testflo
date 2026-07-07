"""
testflo-pytest: testflo-style MPI test execution as a pytest plugin.

Design
------
This plugin brings testflo's MPI execution model to pytest:

* Tests marked with ``@pytest.mark.parallel(nprocs=N)`` (mpi-pytest syntax) or
  belonging to a unittest.TestCase with an ``N_PROCS`` class attribute
  (testflo syntax) are executed under a spawned ``mpirun -n N`` subprocess.

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
import collections.abc

import pytest


# ---------------------------------------------------------------------------
# constants / environment flags
# ---------------------------------------------------------------------------

CHILD_FLAG = "TESTFLO_PYTEST_CHILD"
"""Set to '1' in the environment of the spawned mpirun child processes."""

RESULTS_FLAG = "TESTFLO_PYTEST_RESULTS"
"""Path of the JSON results file the child's rank 0 writes."""

MAX_NPROCS_FLAG = "TESTFLO_PYTEST_MAX_NPROCS"
"""Optional env var limiting the maximum number of processes per test."""

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


@functools.lru_cache(maxsize=None)
def _find_mpirun():
    exe = os.environ.get("TESTFLO_PYTEST_MPIRUN")
    if exe:
        return exe
    for name in ("mpirun", "mpiexec"):
        found = shutil.which(name)
        if found:
            return found
    return None


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


def _parse_marker_nprocs(marker):
    """Return the tuple of process counts requested by a parallel marker.

    Accepts ``parallel(4)``, ``parallel(nprocs=4)``, ``parallel([2, 3])``,
    ``parallel(nprocs=[2, 3])`` and bare ``parallel`` (-> DEFAULT_NPROCS).
    """
    if len(marker.args) == 1 and not marker.kwargs:
        return _as_tuple(marker.args[0])
    elif len(marker.kwargs) == 1 and not marker.args and "nprocs" in marker.kwargs:
        return _as_tuple(marker.kwargs["nprocs"])
    elif not marker.args and not marker.kwargs:
        return (DEFAULT_NPROCS,)
    raise pytest.UsageError(
        "Bad arguments given to parallel marker; expected parallel(N), "
        "parallel(nprocs=N), or parallel([N1, N2, ...])")


def _nprocs_for_item(item):
    """Number of processes a single collected test item should run on.

    Resolution order:
      1. ``[nprocs=N]`` parametrization (from a multi-valued parallel marker)
      2. ``@pytest.mark.parallel`` marker
      3. testflo-style ``N_PROCS`` class attribute
    """
    if hasattr(item, "callspec") and "_nprocs" in item.callspec.params:
        return int(item.callspec.params["_nprocs"])

    marker = item.get_closest_marker("parallel")
    if marker is not None:
        nprocss = _parse_marker_nprocs(marker)
        if len(nprocss) != 1:
            # should have been parametrized away in pytest_generate_tests
            raise pytest.UsageError(
                f"multi-valued parallel marker on {item.nodeid} was not "
                "parametrized; is pytest_generate_tests being blocked?")
        return int(nprocss[0])

    n = getattr(getattr(item, "cls", None), "N_PROCS", None)
    if n is not None:
        return int(n)

    return 1


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
        "--mpi-concurrent-slots", action="store", type=int, default=None,
        metavar="N",
        help="maximum total number of MPI ranks in flight at once across "
             "all pytest-xdist workers (default: unlimited). A parallel "
             "test waits until its nprocs fit within the budget before its "
             "mpirun is spawned. Useful on resource-limited CI machines.")
    group.addoption(
        "--mpi-workers", action="store", type=int, default=1,
        metavar="N",
        help="number of pytest-xdist workers dedicated to MPI tests "
             "(default: 1). MPI tests are distributed round-robin across N "
             "xdist groups so they run in series within each group. "
             "--dist loadgroup is applied automatically when xdist is active "
             "and --dist has not been set explicitly.")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "parallel(nprocs): run this test under MPI on nprocs processes "
        f"(default: {DEFAULT_NPROCS}). nprocs may be a list to parametrize "
        "over multiple communicator sizes.")

    if config.getoption("--mpirun-exe"):
        os.environ["TESTFLO_PYTEST_MPIRUN"] = config.getoption("--mpirun-exe")
        _find_mpirun.cache_clear()

    if config.pluginmanager.hasplugin("xdist") and not _is_child():
        # Force --dist loadgroup unless the user explicitly chose a dist mode.
        # xdist's ini default for 'dist' is 'no'; treat that as "not set".
        dist_value = config.getoption("--dist", default="no")
        if dist_value == "no":
            config.option.dist = "loadgroup"


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
    nprocss = _parse_marker_nprocs(marker)

    max_nprocs = os.environ.get(MAX_NPROCS_FLAG)
    if max_nprocs is not None:
        max_nprocs = int(max_nprocs)
        for nprocs in nprocss:
            if nprocs > max_nprocs:
                raise pytest.UsageError(
                    f"Requested a parallel test with too many ranks "
                    f"({nprocs} > {MAX_NPROCS_FLAG}={max_nprocs})")

    if len(nprocss) > 1:
        metafunc.fixturenames.append("_nprocs")
        metafunc.parametrize("_nprocs", nprocss, ids=lambda n: f"nprocs={n}")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Attach parallel markers to testflo-style ``N_PROCS`` TestCase tests
    so they are visible to ``-m parallel`` selection, and stash the resolved
    nprocs on every item.

    When pytest-xdist is active, parallel tests are distributed round-robin
    across ``--mpi-workers`` xdist groups (``testflo_mpi_0``,
    ``testflo_mpi_1``, ...), unless the user set their own xdist_group.
    Under ``--dist loadgroup`` (applied automatically when ``--mpi-workers``
    is set) each group is pinned to one worker, so MPI tests run in series
    within each worker while serial tests distribute freely.

    ``tryfirst`` matters: xdist's worker rewrites node IDs to
    ``<nodeid>@<group>`` in its *own* pytest_collection_modifyitems, which
    pluggy runs before ours by default (LIFO registration); the group mark
    must already be attached when that happens.
    """
    xdist_active = config.pluginmanager.hasplugin("xdist")
    mpi_workers = max(1, config.getoption("--mpi-workers"))
    mpi_counter = 0
    for item in items:
        if item.get_closest_marker("parallel") is None:
            n = getattr(getattr(item, "cls", None), "N_PROCS", None)
            if n is not None and int(n) > 1:
                item.add_marker(pytest.mark.parallel(nprocs=int(n)))
        nprocs = _nprocs_for_item(item)
        item.stash[_NPROCS_KEY] = nprocs
        if (xdist_active and nprocs > 1
                and item.get_closest_marker("xdist_group") is None):
            group = f"testflo_mpi_{mpi_counter % mpi_workers}"
            item.add_marker(pytest.mark.xdist_group(group))
            mpi_counter += 1


_NPROCS_KEY = pytest.StashKey()


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
    if _is_child() or _outer_world_size() > 1:
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
    if _is_child() or _outer_world_size() > 1:
        from mpi4py import MPI
        request.addfinalizer(MPI.COMM_WORLD.barrier)


# ---------------------------------------------------------------------------
# child mode: run naturally, gather at the end
# ---------------------------------------------------------------------------

_child_results = {}


def _record_child_report(report):
    """Keep the 'worst' report per nodeid: a failed setup/call/teardown
    beats a pass; a call report beats setup/teardown noise."""
    nodeid = report.nodeid
    entry = {
        "when": report.when,
        "outcome": report.outcome,
        "longrepr": str(report.longrepr) if report.longrepr is not None else None,
        "sections": [list(s) for s in report.sections],
        "duration": getattr(report, "duration", 0.0),
        "wasxfail": getattr(report, "wasxfail", None),
    }
    prev = _child_results.get(nodeid)
    if prev is None:
        _child_results[nodeid] = entry
        return
    # accumulate captured output sections across phases
    entry["sections"] = prev["sections"] + entry["sections"]
    entry["duration"] = prev["duration"] + entry["duration"]
    if prev["outcome"] != "passed" and entry["outcome"] == "passed":
        # keep the failure/skip info, but keep accumulated sections
        prev["sections"] = entry["sections"]
        prev["duration"] = entry["duration"]
        return
    if prev.get("wasxfail") is not None and entry.get("wasxfail") is None:
        entry["wasxfail"] = prev["wasxfail"]
    _child_results[nodeid] = entry


@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report):
    if not _is_child():
        return
    if report.when == "call" or report.outcome != "passed":
        _record_child_report(report)
    elif report.when == "setup" and report.nodeid not in _child_results:
        # remember setup sections so captured fixture output isn't lost
        _record_child_report(report)


@pytest.hookimpl(trylast=True)
def pytest_collectreport(report):
    if not _is_child():
        return
    if report.failed:
        _child_results[report.nodeid or "<collection>"] = {
            "when": "collect",
            "outcome": "failed",
            "longrepr": str(report.longrepr),
            "sections": [],
            "duration": 0.0,
            "wasxfail": None,
        }


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
    """True if this item should be run via a spawned mpirun subprocess."""
    if _is_child() or _outer_world_size() > 1:
        return False
    if item.config.getoption("--nompi"):
        return False
    return item.stash.get(_NPROCS_KEY, 1) > 1


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    nprocs = item.stash.get(_NPROCS_KEY, 1)

    if _is_child():
        return

    world = _outer_world_size()
    if world > 1:
        # outer-mpirun mode: user ran `mpirun -n N pytest ...` themselves
        if nprocs != world:
            pytest.skip(f"test requires {nprocs} MPI procs but pytest was "
                        f"launched with {world}")
        return

    if nprocs > 1 and not item.config.getoption("--nompi"):
        if not _have_mpi4py():
            pytest.skip("mpi4py is required to run parallel tests "
                        "(or use --nompi)")
        if _find_mpirun() is None:
            pytest.skip("mpirun/mpiexec was not found in the system path "
                        "(or use --nompi)")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    """Take over the runtest protocol for parallel tests in launcher mode.

    The test (including its fixtures) executes entirely inside the spawned
    MPI subprocess -- nothing test-related runs in the launcher, so
    fixtures are not double-executed.
    """
    if not _should_launch_mpi(item):
        return None  # normal in-process execution

    ihook = item.ihook
    ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)

    reports = _run_mpi_item(item)
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
    try:
        now = time.time()
        report = TestReport(start=now - duration, stop=now, **kwargs)
    except TypeError:  # pytest < 7.4 has no start/stop
        report = TestReport(**kwargs)
    if wasxfail is not None:
        report.wasxfail = wasxfail
    return report


class _SlotLimiter(object):
    """Cross-process budget of concurrently running MPI ranks.

    pytest-xdist workers are independent processes, so the budget is kept
    in a small JSON state file in the temp directory, keyed by xdist's
    ``PYTEST_XDIST_TESTRUNUID`` (all workers of one run share it; without
    xdist the key falls back to this process's pid, where the limiter is
    trivially correct).

    The state maps ``pid -> nprocs`` for every current holder.  Mutual
    exclusion uses an ``O_CREAT | O_EXCL`` lockfile (portable, no deps).
    On every acquire, holders whose pid is no longer alive are pruned, so
    a worker that dies while holding slots cannot deadlock the rest of
    the run.
    """

    POLL = 0.2          # seconds between acquire attempts
    STALE_LOCK = 30.0   # break a lock this old whose owner pid is dead

    def __init__(self, max_slots):
        import tempfile
        self.max_slots = max_slots
        key = os.environ.get("PYTEST_XDIST_TESTRUNUID", str(os.getpid()))
        base = os.path.join(tempfile.gettempdir(), f"testflo_pytest_{key}")
        self.state_path = base + ".slots"
        self.lock_path = base + ".lock"

    @staticmethod
    def _pid_alive(pid):
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _lock(self):
        while True:
            try:
                fd = os.open(self.lock_path,
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return
            except FileExistsError:
                try:
                    st = os.stat(self.lock_path)
                    with open(self.lock_path) as f:
                        owner = int(f.read().strip() or 0)
                    if (time.time() - st.st_mtime > self.STALE_LOCK
                            and not self._pid_alive(owner)):
                        os.remove(self.lock_path)  # break dead owner's lock
                        continue
                except (OSError, ValueError):
                    pass
                time.sleep(0.01)

    def _unlock(self):
        try:
            os.remove(self.lock_path)
        except OSError:
            pass

    def _read_holders(self):
        try:
            with open(self.state_path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = {"holders": {}, "hwm": 0}
        # prune holders whose process died without releasing
        data["holders"] = {pid: n for pid, n in data["holders"].items()
                           if self._pid_alive(int(pid))}
        return data

    def _write(self, data):
        tmp = self.state_path + f".tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self.state_path)

    def acquire(self, nprocs, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            self._lock()
            try:
                data = self._read_holders()
                in_use = sum(data["holders"].values())
                if in_use + nprocs <= self.max_slots:
                    data["holders"][str(os.getpid())] = nprocs
                    data["hwm"] = max(data.get("hwm", 0), in_use + nprocs)
                    self._write(data)
                    return
            finally:
                self._unlock()
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(
                    f"timed out waiting for {nprocs} MPI slots "
                    f"(budget: {self.max_slots})")
            time.sleep(self.POLL)

    def release(self):
        self._lock()
        try:
            data = self._read_holders()
            data["holders"].pop(str(os.getpid()), None)
            self._write(data)
        finally:
            self._unlock()


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

    nprocs = item.stash[_NPROCS_KEY]
    config = item.config
    mpirun = _find_mpirun()
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
           ["-n", str(nprocs),
            sys.executable, "-m", "pytest",
            _child_nodeid(item),
            "-q", "--no-header", "-p", "no:cacheprovider"])

    setup_report = _make_report(item, "setup", "passed")
    teardown_report = _make_report(item, "teardown", "passed")

    max_slots = config.getoption("--mpi-concurrent-slots")
    limiter = None
    if max_slots is not None:
        if nprocs > max_slots:
            os.remove(results_path)
            call_report = _make_report(
                item, "call", "failed",
                longrepr=f"test requires {nprocs} MPI ranks but "
                         f"--mpi-concurrent-slots={max_slots}")
            return [setup_report, call_report, teardown_report]
        limiter = _SlotLimiter(max_slots)
        try:
            # bound the wait so a wedged run can't block forever
            limiter.acquire(nprocs, timeout=(timeout or 3600) * 10)
        except TimeoutError as exc:
            os.remove(results_path)
            call_report = _make_report(item, "call", "failed",
                                       longrepr=str(exc))
            return [setup_report, call_report, teardown_report]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              env=env, timeout=timeout,
                              cwd=str(config.rootpath))
    except subprocess.TimeoutExpired:
        os.remove(results_path)
        longrepr = (f"MPI test timed out after {timeout} s.\n"
                    "A common cause is desynchronized collective calls: a "
                    "rank-local failure occurred before a collective "
                    "operation inside the test body.")
        call_report = _make_report(item, "call", "failed", longrepr=longrepr)
        return [setup_report, call_report, teardown_report]
    finally:
        if limiter is not None:
            limiter.release()

    try:
        with open(results_path) as f:
            contents = f.read()
        data = json.loads(contents) if contents.strip() else None
    except (OSError, json.JSONDecodeError):
        data = None
    finally:
        try:
            os.remove(results_path)
        except OSError:
            pass

    if data is None:
        # the child crashed or aborted before results could be gathered
        longrepr = (f"MPI subprocess exited with code {proc.returncode} "
                    "before results could be collected.\n"
                    f"command: {' '.join(cmd)}\n"
                    f"--- stdout ---\n{proc.stdout}\n"
                    f"--- stderr ---\n{proc.stderr}")
        call_report = _make_report(item, "call", "failed", longrepr=longrepr)
        return [setup_report, call_report, teardown_report]

    call_report = _aggregate_rank_results(item, data)
    return [setup_report, call_report, teardown_report]


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
        results = payload["results"]
        if not results:
            continue
        n_with_results += 1
        # one nodeid per child invocation (plus possible collect errors)
        for nodeid, res in results.items():
            duration = max(duration, res.get("duration") or 0.0)
            if res.get("wasxfail") is not None:
                wasxfail = res["wasxfail"]
            for name, content in res.get("sections", ()):
                if content:
                    sections.append((f"rank {rank}: {name}", content))
            if res["outcome"] == "failed":
                failures.append((rank, res["longrepr"] or "(no traceback)"))
            elif res["outcome"] == "skipped":
                skips.append((rank, res["longrepr"] or "skipped"))

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
                                longrepr=skips[0][1], sections=sections,
                                duration=duration, wasxfail=wasxfail)
        reason = skips[0][1]
        if reason.startswith("("):  # repr of a (path, lineno, reason) tuple
            try:
                import ast
                reason = ast.literal_eval(reason)[2]
            except Exception:
                pass
        longrepr = (str(item.path), item.location[1], str(reason))
        return _make_report(item, "call", "skipped", longrepr=longrepr,
                            sections=sections, duration=duration)

    return _make_report(item, "call", "passed", sections=sections,
                        duration=duration, wasxfail=wasxfail)
