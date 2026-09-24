"""Tests for testflo-pytest itself.

These use pytest's ``pytester`` fixture to run the plugin end-to-end,
including real mpirun spawning where MPI is available.
"""

import os
import platform
import shutil
import subprocess

import pytest

pytest_plugins = ["pytester"]

HAVE_MPI = (shutil.which("mpirun") or shutil.which("mpiexec")) is not None
try:
    import mpi4py  # noqa: F401
except ImportError:
    HAVE_MPI = False

mpi = pytest.mark.skipif(not HAVE_MPI, reason="requires mpi4py and mpirun")


def _is_mpich_on_macos():
    """Check if running MPICH on macOS."""
    if platform.system() != "Darwin":
        return False
    try:
        output = subprocess.check_output(["mpirun", "--version"], stderr=subprocess.STDOUT, text=True)
        return "HYDRA" in output
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


skip_mpich_macos = pytest.mark.skipif(
    _is_mpich_on_macos(),
    reason="MPICH on macOS has OFI finalization issues with forced process termination"
)


@pytest.fixture(autouse=True)
def _no_default_core_budget(monkeypatch):
    """Lift the default core budget for the plugin's own tests so they don't
    depend on how many cores the CI runner has.  Tests that exercise the
    budget pass an explicit --max-concurrent-cores, which takes precedence;
    the test of the default itself removes this again."""
    monkeypatch.setenv("TESTFLO_PYTEST_OVERSUBSCRIBE", "1")


@mpi
def test_parallel_pass(pytester):
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.mpi(2)
        def test_ok(comm):
            assert comm.size == 2
            vals = comm.allgather(comm.rank)
            assert vals == [0, 1]
        """
    )
    result = pytester.runpytest_subprocess("-v")
    result.assert_outcomes(passed=1)


@mpi
def test_parallel_natural_assert_failure(pytester):
    """A plain assert failing on a single rank fails the whole test and
    the per-rank traceback is reported -- the core testflo behavior."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.mpi(3)
        def test_fail_on_rank_1(comm):
            if comm.rank == 1:
                assert False, "boom on rank 1"
        """
    )
    result = pytester.runpytest_subprocess("-v")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*rank 1 of 3*"])
    result.stdout.fnmatch_lines(["*boom on rank 1*"])


@mpi
def test_parametrized_nprocs(pytester):
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.mpi([2, 3])
        def test_sizes(comm):
            assert comm.allreduce(1) == comm.size
        """
    )
    result = pytester.runpytest_subprocess("-v")
    result.assert_outcomes(passed=2)
    result.stdout.fnmatch_lines(["*nprocs=2*", "*nprocs=3*"])


def test_collection_sorted_by_core_cost(pytester):
    """Expensive tests are moved to the front of the collection (stable
    sort: equal-cost tests keep their file order) so they claim cores
    while the budget is still empty.  Collection-only, so no MPI needed."""
    pytester.makepyfile(
        """
        import pytest

        def test_s1(): pass

        @pytest.mark.multiprocessing(3)
        def test_mp3(): pass

        def test_s2(): pass

        @pytest.mark.mpi(2)
        @pytest.mark.multiprocessing(2)
        def test_mpi2_mp2(): pass

        @pytest.mark.mpi([2, 3])
        def test_sizes(): pass

        @pytest.mark.mpi
        def test_bare(): pass
        """
    )
    result = pytester.runpytest_subprocess("--collect-only", "-q", "--nompi")
    names = [line.split("::")[-1] for line in result.stdout.lines
             if "::" in line]
    assert names == ["test_mpi2_mp2",            # 4
                     "test_mp3", "test_sizes[nprocs=3]",   # 3
                     "test_sizes[nprocs=2]", "test_bare",  # 2
                     "test_s1", "test_s2"]                 # 1


@mpi
def test_n_procs_class_attribute(pytester):
    """testflo-style N_PROCS on a unittest.TestCase triggers MPI spawning."""
    pytester.makepyfile(
        """
        import unittest

        class TestMPI(unittest.TestCase):
            N_PROCS = 2

            def test_size(self):
                from mpi4py import MPI
                self.assertEqual(MPI.COMM_WORLD.size, 2)
        """
    )
    result = pytester.runpytest_subprocess("-v")
    result.assert_outcomes(passed=1)


@mpi
def test_xfail_propagates(pytester):
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.mpi(2)
        @pytest.mark.xfail(reason="known bad")
        def test_xf(comm):
            assert False
        """
    )
    result = pytester.runpytest_subprocess("-ra")
    result.assert_outcomes(xfailed=1)


@mpi
def test_skip_propagates(pytester):
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.mpi(2)
        def test_skipped(comm):
            pytest.skip("not today")
        """
    )
    result = pytester.runpytest_subprocess("-v")
    result.assert_outcomes(skipped=1)


@mpi
def test_captured_output_labeled_by_rank(pytester):
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.mpi(2)
        def test_output(comm):
            print(f"hello from rank {comm.rank}")
            if comm.rank == 0:
                raise RuntimeError("fail so output is shown")
        """
    )
    result = pytester.runpytest_subprocess("-v")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*rank 0: Captured stdout call*"])
    result.stdout.fnmatch_lines(["*hello from rank 0*"])


@mpi
@skip_mpich_macos
def test_mpi_timeout_breaks_deadlock(pytester):
    """A desynchronized collective (the known caveat of the natural-assert
    model) is broken by --mpi-timeout instead of hanging forever."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.mpi(2)
        def test_deadlock(comm):
            if comm.rank == 0:
                comm.barrier()   # rank 1 never arrives
        """
    )
    result = pytester.runpytest_subprocess("--mpi-timeout=5", "-v")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*timed out*"])


try:
    import xdist  # noqa: F401
    HAVE_XDIST = True
except ImportError:
    HAVE_XDIST = False


@mpi
@pytest.mark.skipif(not HAVE_XDIST, reason="requires pytest-xdist")
def test_xdist_mixed_suite(pytester):
    """Serial and parallel tests both work when distributed by xdist:
    each worker spawns its own mpirun, and synthesized per-rank reports
    survive execnet serialization back to the controller."""
    pytester.makepyfile(
        """
        import pytest

        def test_serial():
            assert True

        @pytest.mark.mpi(2)
        def test_par_ok(comm):
            assert comm.allreduce(1) == 2

        @pytest.mark.mpi(2)
        def test_par_fail(comm):
            if comm.rank == 1:
                assert False, "boom on rank 1"

        @pytest.mark.mpi(2)
        def test_par_skip(comm):
            pytest.skip("nope")
        """
    )
    result = pytester.runpytest_subprocess("-n", "2", "-v")
    result.assert_outcomes(passed=2, failed=1, skipped=1)
    result.stdout.fnmatch_lines(["*rank 1 of 2*"])
    result.stdout.fnmatch_lines(["*boom on rank 1*"])


@mpi
@pytest.mark.skipif(not HAVE_XDIST, reason="requires pytest-xdist")
def test_setupstate_reconciled_after_mpi_launch(pytester):
    """A test whose protocol is fully bypassed (here, an MPI launch) must
    not leave stale collectors on pytest's own SetupState stack -- otherwise
    the next item, if collected from a different module, trips pytest's own
    "previous item was not torn down properly" assertion the moment it
    lands on the same xdist worker right after the bypassed item."""
    pytester.makepyfile(test_a="""
        import pytest

        def test_before():
            pass

        @pytest.mark.mpi(2)
        def test_mid(comm):
            assert comm.size == 2
        """, test_b="""
        def test_after():
            pass
        """)
    result = pytester.runpytest_subprocess("-n", "1", "-v")
    result.assert_outcomes(passed=3)
    assert "not torn down properly" not in result.stdout.str()


@pytest.mark.skipif(not HAVE_XDIST, reason="requires pytest-xdist")
def test_setupstate_reconciled_after_core_budget_refusal(pytester):
    """Same bug as above, triggered without needing real MPI: a test whose
    core request can never fit the budget is failed without its protocol
    ever running, and must reconcile the SetupState stack the same way."""
    pytester.makepyfile(test_a="""
        import pytest

        def test_before():
            pass

        @pytest.mark.multiprocessing(4)
        def test_mid(comm):
            pass
        """, test_b="""
        def test_after():
            pass
        """)
    result = pytester.runpytest_subprocess(
        "-n", "1", "--max-concurrent-cores=1", "-v")
    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines(["*requires 4 cores*"])
    assert "not torn down properly" not in result.stdout.str()


@pytest.mark.skipif(not HAVE_XDIST, reason="requires pytest-xdist")
def test_xdist_defaults_to_worksteal(pytester):
    """No xdist grouping is imposed any more; work stealing is the default
    distribution so a worker waiting for cores doesn't block the tests
    queued behind it.  An explicit --dist is left alone."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.parametrize("i", range(4))
        def test_serial(i):
            pass

        @pytest.mark.multiprocessing(2)
        def test_mp(comm):
            pass
        """
    )
    result = pytester.runpytest_subprocess("-n", "2", "-v")
    result.assert_outcomes(passed=5)
    result.stdout.fnmatch_lines(["*WorkStealingScheduling*"])
    result = pytester.runpytest_subprocess("-n", "2", "--dist", "load", "-v")
    result.assert_outcomes(passed=5)
    result.stdout.fnmatch_lines(["*LoadScheduling*"])


@pytest.mark.skipif(not HAVE_XDIST, reason="requires pytest-xdist")
def test_waiting_multicore_test_is_not_starved(pytester):
    """A 3-core test on a 3-core budget with 3 workers churning serial tests
    must get its turn as soon as the running serial tests finish, not after
    the whole serial queue drains: once it is waiting, no new serial test
    may start ahead of it."""
    pytester.makepyfile(
        """
        import multiprocessing, os, time
        import pytest

        LOGDIR = os.path.join(os.path.dirname(__file__), "intervals")

        def _log(name, t0, t1):
            os.makedirs(LOGDIR, exist_ok=True)
            with open(os.path.join(LOGDIR, name), "w") as f:
                f.write(f"{name} {t0} {t1}")

        def burn(x):
            time.sleep(0.3)
            return x

        @pytest.mark.multiprocessing(3)
        def test_pool(comm):
            t0 = time.time()
            with multiprocessing.Pool(3) as pool:
                pool.map(burn, [1, 2, 3])
            _log("pool", t0, time.time())

        @pytest.mark.parametrize("i", range(12))
        def test_serial(i):
            t0 = time.time(); time.sleep(0.2); _log(f"serial{i}", t0, time.time())
        """
    )
    result = pytester.runpytest_subprocess("-n", "3",
                                           "--max-concurrent-cores=3", "-v")
    result.assert_outcomes(passed=13)
    intervals = {}
    for path in (pytester.path / "intervals").iterdir():
        name, t0, t1 = path.read_text().split()
        intervals[name] = (float(t0), float(t1))
    p0, p1 = intervals.pop("pool")
    # nothing overlaps the 3-core test ...
    for name, (s0, s1) in intervals.items():
        assert s1 <= p0 or s0 >= p1, f"{name} overlapped the pool test"
    # ... and it ran early: at most the serial tests already in flight when
    # it started waiting (one per other worker, plus a little race slack)
    # finished before it, the rest were deferred behind it
    before = sum(1 for s0, _ in intervals.values() if s0 < p0)
    assert before <= 4, f"{before} serial tests ran ahead of the pool test"


@mpi
@pytest.mark.skipif(not HAVE_XDIST, reason="requires pytest-xdist")
def test_concurrent_slots_budget(pytester):
    """The deprecated --mpi-concurrent-slots alias still works."""
    pytester.makepyfile(
        """
        import pytest, time

        @pytest.mark.mpi(2)
        def test_a(comm): time.sleep(0.5)

        @pytest.mark.mpi(2)
        def test_b(comm): time.sleep(0.5)

        @pytest.mark.mpi(2)
        def test_c(comm): time.sleep(0.5)
        """
    )
    result = pytester.runpytest_subprocess("-n", "3",
                                           "--mpi-concurrent-slots=2", "-v")
    result.assert_outcomes(passed=3)


def test_nompi_runs_in_process(pytester):
    """--nompi runs parallel tests on a FakeComm of size 1, like testflo."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.mpi(4)
        def test_fake(comm):
            assert comm.size == 1
            assert comm.allgather(comm.rank) == [0]
        """
    )
    result = pytester.runpytest_subprocess("--nompi", "-v")
    result.assert_outcomes(passed=1)


def test_serial_untouched(pytester):
    pytester.makepyfile(
        """
        def test_plain():
            assert True
        """
    )
    result = pytester.runpytest_subprocess("-v")
    result.assert_outcomes(passed=1)


def test_parse_marker_forms():
    from testflo.pytest_plugin import (_parse_mpi_marker, _parse_mp_marker,
                                       _cores, DEFAULT_NPROCS)

    def mpi_mark(*args, **kwargs):
        return pytest.mark.mpi(*args, **kwargs).mark

    def mp_mark(*args, **kwargs):
        return pytest.mark.multiprocessing(*args, **kwargs).mark

    assert _parse_mpi_marker(mpi_mark()) == (DEFAULT_NPROCS,)
    assert _parse_mpi_marker(mpi_mark(4)) == (4,)
    assert _parse_mpi_marker(mpi_mark(nprocs=4)) == (4,)
    assert _parse_mpi_marker(mpi_mark([2, 3])) == (2, 3)
    assert _parse_mpi_marker(mpi_mark(nprocs=[2, 3])) == (2, 3)

    assert _parse_mp_marker(mp_mark(4)) == 4

    assert _cores(4, 0) == 4
    assert _cores(0, 4) == 4
    assert _cores(4, 2) == 8
    assert _cores(0, 0) == 1

    for bad in (mpi_mark(4, nprocs=4), mpi_mark(bogus=2), mpi_mark(mpi="2")):
        with pytest.raises(pytest.UsageError):
            _parse_mpi_marker(bad)

    for bad in (mp_mark(), mp_mark(-1), mp_mark([2, 3]), mp_mark(bogus=2)):
        with pytest.raises(pytest.UsageError):
            _parse_mp_marker(bad)


def test_multiprocessing_only_runs_in_process(pytester):
    """multiprocessing(M) never spawns mpirun: the test runs in this process
    on a FakeComm, but is still selectable with -m multiprocessing."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.multiprocessing(4)
        def test_mp(comm):
            assert comm.size == 1

        def test_serial():
            pass
        """
    )
    result = pytester.runpytest_subprocess("-v")
    result.assert_outcomes(passed=2)
    result = pytester.runpytest_subprocess("-v", "-m", "multiprocessing")
    result.assert_outcomes(passed=1, deselected=1)


def test_bare_multiprocessing_marker_rejected(pytester):
    """multiprocessing has no sensible default pool size, so a bare marker
    (unlike a bare @pytest.mark.mpi) is a usage error, not DEFAULT_NPROCS."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.multiprocessing
        def test_mp(comm):
            pass
        """
    )
    result = pytester.runpytest_subprocess("-v")
    assert result.ret != 0
    result.stderr.fnmatch_lines(["*Bad arguments given to multiprocessing marker*"])


def test_deselect_mpi_keeps_multiprocessing(pytester):
    """The motivating CI use case: `-m 'not mpi'` strips out every test that
    would spawn mpirun while leaving multiprocessing-only tests runnable."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.mpi(2)
        def test_needs_mpi(comm):
            pass

        @pytest.mark.multiprocessing(2)
        def test_needs_pool(comm):
            assert comm.size == 1

        def test_plain():
            pass
        """
    )
    result = pytester.runpytest_subprocess("-v", "-m", "not mpi")
    result.assert_outcomes(passed=2, deselected=1)


def test_core_budget_over_limit_fails(pytester):
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.multiprocessing(4)
        def test_mp(comm):
            pass
        """
    )
    result = pytester.runpytest_subprocess("--max-concurrent-cores=2", "-v")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*requires 4 cores*"])


def test_default_budget_is_available_cores(pytester, monkeypatch):
    """With no flags, a test needing more cores than the machine has is
    refused; --oversubscribe (or the env var) lets it run anyway."""
    from testflo.pytest_plugin import _available_cores
    monkeypatch.delenv("TESTFLO_PYTEST_OVERSUBSCRIBE")
    too_many = _available_cores() + 1
    pytester.makepyfile(
        f"""
        import pytest

        @pytest.mark.multiprocessing({too_many})
        def test_big(comm):
            pass

        @pytest.mark.multiprocessing(1)
        def test_small(comm):
            pass
        """
    )
    result = pytester.runpytest_subprocess("-v")
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines([f"*requires {too_many} cores*"])

    result = pytester.runpytest_subprocess("--oversubscribe", "-v")
    result.assert_outcomes(passed=2)

    monkeypatch.setenv("TESTFLO_PYTEST_OVERSUBSCRIBE", "1")
    result = pytester.runpytest_subprocess("-v")
    result.assert_outcomes(passed=2)

    # an explicit budget beats --oversubscribe
    result = pytester.runpytest_subprocess("--oversubscribe",
                                           "--max-concurrent-cores=1", "-v")
    result.assert_outcomes(passed=1, failed=1)


@mpi
def test_mpi_nprocs_kwarg_alias(pytester):
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.mpi(nprocs=2)
        def test_ok(comm):
            assert comm.size == 2
        """
    )
    result = pytester.runpytest_subprocess("-v")
    result.assert_outcomes(passed=1)


@mpi
def test_mpi_times_multiprocessing_charges_product(pytester):
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.mpi(2)
        @pytest.mark.multiprocessing(2)
        def test_both(comm):
            assert comm.size == 2
        """
    )
    result = pytester.runpytest_subprocess("--max-concurrent-cores=3", "-v")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*requires 4 cores*"])
    result = pytester.runpytest_subprocess("--max-concurrent-cores=4", "-v")
    result.assert_outcomes(passed=1)


# ---------------------------------------------------------------------------
# tests that really use multiprocessing (and MPI + multiprocessing together)
# ---------------------------------------------------------------------------

def test_multiprocessing_pool_in_marked_test(pytester):
    """A multiprocessing-only parallel test can actually spin up a Pool and
    do work in it; no MPI machinery is involved."""
    pytester.makepyfile(
        """
        import multiprocessing
        import pytest

        def square(x):
            return x * x

        @pytest.mark.multiprocessing(3)
        def test_pool(comm):
            assert comm.size == 1
            with multiprocessing.Pool(3) as pool:
                got = pool.map(square, range(10))
            assert got == [x * x for x in range(10)]
        """
    )
    result = pytester.runpytest_subprocess("-v")
    result.assert_outcomes(passed=1)


@pytest.mark.skipif(not HAVE_XDIST, reason="requires pytest-xdist")
def test_core_budget_serializes_pools(pytester):
    """Two Pool-using tests that each cost the whole budget must not run at
    the same time, even though xdist hands them to different workers.

    Each test logs its wall-clock interval; the intervals must not overlap.
    """
    pytester.makepyfile(
        """
        import multiprocessing, os, time
        import pytest

        LOGDIR = os.path.join(os.path.dirname(__file__), "intervals")

        def burn(x):
            time.sleep(0.3)
            return x

        def _run(name):
            t0 = time.time()
            with multiprocessing.Pool(2) as pool:
                assert pool.map(burn, [1, 2]) == [1, 2]
            t1 = time.time()
            os.makedirs(LOGDIR, exist_ok=True)
            with open(os.path.join(LOGDIR, name), "w") as f:
                f.write(f"{name} {t0} {t1}")

        @pytest.mark.multiprocessing(2)
        def test_a(comm): _run("a")

        @pytest.mark.multiprocessing(2)
        def test_b(comm): _run("b")

        @pytest.mark.multiprocessing(2)
        def test_c(comm): _run("c")
        """
    )
    # --dist load so xdist really does try to run them concurrently
    result = pytester.runpytest_subprocess("-n", "3", "--dist", "load",
                                           "--max-concurrent-cores=2", "-v")
    result.assert_outcomes(passed=3)
    intervals = []
    for path in (pytester.path / "intervals").iterdir():
        name, t0, t1 = path.read_text().split()
        intervals.append((float(t0), float(t1), name))
    assert len(intervals) == 3
    intervals.sort()
    for (_, end, a), (start, _, b) in zip(intervals, intervals[1:]):
        assert start >= end, f"{a} and {b} overlapped despite budget of 2"


@pytest.mark.skipif(not HAVE_XDIST, reason="requires pytest-xdist")
def test_serial_tests_count_against_budget(pytester):
    """Serial tests cost one core each.  With a budget of 2, a 2-core pool
    test may not overlap with any serial test running on another worker,
    while the serial tests may overlap each other (1 + 1 <= 2)."""
    pytester.makepyfile(
        """
        import multiprocessing, os, time
        import pytest

        LOGDIR = os.path.join(os.path.dirname(__file__), "intervals")

        def _log(name, t0, t1):
            os.makedirs(LOGDIR, exist_ok=True)
            with open(os.path.join(LOGDIR, name), "w") as f:
                f.write(f"{name} {t0} {t1}")

        def burn(x):
            time.sleep(0.4)
            return x

        @pytest.mark.parametrize("i", range(4))
        def test_serial(i):
            t0 = time.time(); time.sleep(0.4); _log(f"serial{i}", t0, time.time())

        @pytest.mark.multiprocessing(2)
        def test_pool(comm):
            t0 = time.time()
            with multiprocessing.Pool(2) as pool:
                pool.map(burn, [1, 2])
            _log("pool", t0, time.time())
        """
    )
    result = pytester.runpytest_subprocess("-n", "3", "--dist", "load",
                                           "--max-concurrent-cores=2", "-v")
    result.assert_outcomes(passed=5)
    intervals = {}
    for path in (pytester.path / "intervals").iterdir():
        name, t0, t1 = path.read_text().split()
        intervals[name] = (float(t0), float(t1))
    assert len(intervals) == 5
    p0, p1 = intervals.pop("pool")
    for name, (s0, s1) in intervals.items():
        assert s1 <= p0 or s0 >= p1, f"{name} overlapped the 2-core pool test"
    # and at most two serial tests at once (the budget, not the 3 workers)
    events = sorted([(s0, 1) for s0, _ in intervals.values()]
                    + [(s1, -1) for _, s1 in intervals.values()])
    running = peak = 0
    for _, d in events:
        running += d
        peak = max(peak, running)
    assert peak <= 2


def test_combo_under_nompi_charges_only_multiprocessing(pytester):
    """parallel(mpi=4, multiprocessing=2) costs 8 cores normally, but under
    --nompi no ranks are spawned so only the pool's 2 cores are charged."""
    pytester.makepyfile(
        """
        import multiprocessing
        import pytest

        def double(x):
            return 2 * x

        @pytest.mark.mpi(4)
        @pytest.mark.multiprocessing(2)
        def test_combo(comm):
            assert comm.size == 1
            with multiprocessing.Pool(2) as pool:
                assert pool.map(double, [1, 2, 3]) == [2, 4, 6]
        """
    )
    result = pytester.runpytest_subprocess("--nompi",
                                           "--max-concurrent-cores=2", "-v")
    result.assert_outcomes(passed=1)
    result = pytester.runpytest_subprocess("--nompi",
                                           "--max-concurrent-cores=1", "-v")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*requires 2 cores*"])


@mpi
def test_mpi_ranks_each_use_a_pool(pytester):
    """parallel(mpi=2, multiprocessing=2): every MPI rank runs its own Pool
    and the per-rank results are combined collectively."""
    pytester.makepyfile(
        """
        import multiprocessing
        import pytest

        def square(x):
            return x * x

        @pytest.mark.mpi(2)
        @pytest.mark.multiprocessing(2)
        def test_pool_per_rank(comm):
            assert comm.size == 2
            # each rank squares a different slice
            mine = list(range(comm.rank * 5, (comm.rank + 1) * 5))
            with multiprocessing.Pool(2) as pool:
                local = pool.map(square, mine)
            everything = sum(comm.allgather(local), [])
            assert everything == [x * x for x in range(10)]
        """
    )
    result = pytester.runpytest_subprocess("--max-concurrent-cores=4", "-v")
    result.assert_outcomes(passed=1)


@mpi
def test_mpi_sizes_parametrized_with_pool(pytester):
    """A list of MPI sizes still parametrizes when multiprocessing is also
    given; each size runs with its pool and costs size*2 cores."""
    pytester.makepyfile(
        """
        import multiprocessing
        import pytest

        def inc(x):
            return x + 1

        @pytest.mark.mpi([2, 3])
        @pytest.mark.multiprocessing(2)
        def test_sizes(comm):
            with multiprocessing.Pool(2) as pool:
                got = pool.map(inc, [comm.rank] * 2)
            assert got == [comm.rank + 1] * 2
            assert comm.allreduce(1) == comm.size
        """
    )
    result = pytester.runpytest_subprocess("--max-concurrent-cores=6", "-v")
    result.assert_outcomes(passed=2)
    result.stdout.fnmatch_lines(["*nprocs=2*", "*nprocs=3*"])
    # budget of 5 fits the 2-rank case (4 cores) but not the 3-rank (6)
    result = pytester.runpytest_subprocess("--max-concurrent-cores=5", "-v")
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*requires 6 cores*"])


def test_rank_report_aggregation(pytester):
    """The launcher reduces the raw per-phase reports each rank ships to
    one outcome per rank, then one call report per test.  Exercised with
    real pytest reports so it does not need MPI."""
    import json
    from testflo.pytest_plugin import (_serialize_report,
                                       _aggregate_rank_results)
    pytester.makepyfile(
        """
        import pytest

        def test_pass(): print("out"); assert True
        def test_fail(): assert 0, "boom"
        def test_skip(): pytest.skip("nope")
        @pytest.mark.xfail(reason="known")
        def test_xfail(): assert 0
        """
    )
    reprec = pytester.inline_run("-p", "no:cacheprovider", "--oversubscribe")
    by_node = {}
    for rep in reprec.getreports("pytest_runtest_logreport"):
        by_node.setdefault(rep.nodeid, []).append(_serialize_report(rep))
    items = {i.name: i for i in pytester.getitems(
        pytester.path.joinpath("test_rank_report_aggregation.py").read_text())}

    def agg(name, ranks):
        nodeid, = (n for n in by_node if n.endswith("::" + name))
        data = {"nprocs": len(ranks),
                "ranks": [{"rank": r, "results": {nodeid: by_node[nodeid]}}
                          for r in ranks]}
        # the real path goes through the JSON results file
        return _aggregate_rank_results(items[name],
                                       json.loads(json.dumps(data)))

    rep = agg("test_pass", [0, 1])
    assert rep.outcome == "passed"
    assert ("rank 1: Captured stdout call", "out\n") in rep.sections

    rep = agg("test_fail", [0, 1, 2])
    assert rep.outcome == "failed"
    assert "rank 0 of 3" in rep.longrepr and "boom" in rep.longrepr

    rep = agg("test_skip", [0])
    assert rep.outcome == "skipped"
    assert rep.longrepr[2] == "Skipped: nope"   # (path, lineno, reason) form

    rep = agg("test_xfail", [0, 1])
    assert rep.outcome == "skipped" and rep.wasxfail == "known"

    # a rank that failed plus one that passed -> failed, passing rank noted
    nodeid_f, = (n for n in by_node if n.endswith("::test_fail"))
    nodeid_p, = (n for n in by_node if n.endswith("::test_pass"))
    data = {"nprocs": 2, "ranks": [
        {"rank": 0, "results": {nodeid_f: by_node[nodeid_f]}},
        {"rank": 1, "results": {nodeid_f: by_node[nodeid_p]}}]}
    rep = _aggregate_rank_results(items["test_fail"], data)
    assert rep.outcome == "failed" and "(ranks [1] passed)" in rep.longrepr


# ---------------------------------------------------------------------------
# the core tracker itself (in-process; no xdist or manager needed)
# ---------------------------------------------------------------------------

def test_core_tracker_fifo_fairness(monkeypatch):
    """A 3-core request queued behind two 1-core holders on a 4-core budget
    must not be starved: once it is waiting, a new 1-core request may only
    pass it if that still leaves room (it doesn't), so the 1-core request
    waits, the big one runs as soon as the holders release."""
    import threading
    from testflo.pytest_plugin import _CoreTracker

    # pids here are fake; keep the reaper from "cleaning up" after them
    monkeypatch.setattr(_CoreTracker, "REAP_INTERVAL", 3600)
    t = _CoreTracker(4)
    t.acquire(1, 1)
    t.acquire(2, 1)
    order = []

    def big():
        t.acquire(3, 3); order.append("big")

    def small():
        t.acquire(4, 1); order.append("small")

    tb = threading.Thread(target=big); tb.start()
    while not t.stats()["waiting"]:
        pass                                    # big is now queued
    ts = threading.Thread(target=small); ts.start()
    import time; time.sleep(0.2)
    assert order == [], "nothing should have run yet"
    assert t.stats()["waiting"] == [[3, 3], [4, 1]]
    t.release(1); t.release(2)
    tb.join(5); ts.join(5)
    assert order == ["big", "small"]
    assert t.stats()["hwm"] == 4


def test_core_tracker_reaps_dead_workers():
    """Cores held (or awaited) by a worker that died are reclaimed, so a
    crashed worker cannot permanently starve the budget."""
    import os, sys, threading
    from testflo.pytest_plugin import _CoreTracker

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    # keep `proc` (and so its Windows process handle) alive: the reaper
    # must recognise an exited process even while a handle to it is open
    dead = proc.pid
    me = os.getpid()
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        t = _CoreTracker(4)
        t.acquire(dead, 4)
        assert t.reap() == {dead}
        assert t.stats()["holders"] == {}
        t.acquire(me, 2)                         # fits immediately now

        # a dead *waiter* at the head of the queue is dropped too, and the
        # live request behind it proceeds
        t.waiting.append([dead, 4])
        done = []
        th = threading.Thread(
            target=lambda: (t.acquire(sleeper.pid, 2), done.append(1)))
        th.start()
        th.join(0.3)
        assert not done, "should be blocked behind the dead waiter"
        assert t.reap() == {dead}
        th.join(5)
        assert done and t.stats()["holders"] == {me: 2, sleeper.pid: 2}
    finally:
        sleeper.kill()


# ---------------------------------------------------------------------------
# coverage of spawned processes
#
# The plugin runs mpi tests inside a spawned `mpirun`, so the test body never
# executes in the process that reports it.  Coverage of those ranks rides on
# coverage.py's own subprocess mechanism (COVERAGE_PROCESS_CONFIG + the .pth
# file coverage installs); the plugin's job is only to let it through and, if
# nothing else arranged for it, to synthesize it.
# ---------------------------------------------------------------------------

def test_clean_child_env_preserves_coverage_vars(monkeypatch):
    """Scrubbing any of these would silently drop every line that only runs
    under MPI, so the preservation is asserted, not left to chance."""
    from testflo.pytest_plugin import _clean_child_env

    keep = {
        "COVERAGE_PROCESS_CONFIG": ":data:abc",
        "COVERAGE_PROCESS_START": "/tmp/.coveragerc",
        "COVERAGE_FILE": "/tmp/.coverage",
        "COV_CORE_SOURCE": "mypkg",
        "PYTHONPATH": "/somewhere/on/the/path",
    }
    for k, v in keep.items():
        monkeypatch.setenv(k, v)
    # ... while the MPI job-identity scrubbing still happens
    monkeypatch.setenv("PMIX_NAMESPACE", "pollution")
    monkeypatch.setenv("OMPI_MCA_ess", "singleton")

    env = _clean_child_env()

    for k, v in keep.items():
        assert env.get(k) == v, f"{k} must survive into the mpirun child"
    assert "PMIX_NAMESPACE" not in env
    assert "OMPI_MCA_ess" not in env


def test_child_coverage_env_defers_to_existing(monkeypatch):
    """Exactly one mechanism may start coverage in a rank.  If anything has
    already arranged for it, the plugin must not add a second."""
    from testflo.pytest_plugin import _child_coverage_env

    for var in ("COVERAGE_PROCESS_CONFIG", "COVERAGE_PROCESS_START",
                "COV_CORE_SOURCE"):
        monkeypatch.delenv("COVERAGE_PROCESS_CONFIG", raising=False)
        monkeypatch.delenv("COVERAGE_PROCESS_START", raising=False)
        monkeypatch.delenv("COV_CORE_SOURCE", raising=False)
        monkeypatch.setenv(var, "already-set")
        assert _child_coverage_env() == {}, f"{var} already owns tracing"


def test_child_coverage_env_forces_parallel(monkeypatch):
    """The safety net: when coverage is running but nothing arranged for
    subprocesses, synthesize the config -- with ``parallel`` forced on, or
    every rank would write the same data file and clobber the others."""
    coverage = pytest.importorskip("coverage")
    from coverage.config import CoverageConfig
    from testflo.pytest_plugin import _child_coverage_env

    for var in ("COVERAGE_PROCESS_CONFIG", "COVERAGE_PROCESS_START",
                "COV_CORE_SOURCE"):
        monkeypatch.delenv(var, raising=False)

    cov = coverage.Coverage(data_file=None)
    cov.config.parallel = False          # the situation we must correct
    monkeypatch.setattr(coverage.Coverage, "current", staticmethod(lambda: cov))

    env = _child_coverage_env()

    assert set(env) == {"COVERAGE_PROCESS_CONFIG"}
    assert CoverageConfig.deserialize(env["COVERAGE_PROCESS_CONFIG"]).parallel
    # and the live config must not have been mutated as a side effect
    assert cov.config.parallel is False


def test_child_coverage_env_noop_without_coverage(monkeypatch):
    """No coverage running -> nothing to propagate."""
    coverage = pytest.importorskip("coverage")
    from testflo.pytest_plugin import _child_coverage_env

    for var in ("COVERAGE_PROCESS_CONFIG", "COVERAGE_PROCESS_START",
                "COV_CORE_SOURCE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(coverage.Coverage, "current", staticmethod(lambda: None))

    assert _child_coverage_env() == {}


def test_child_pytest_args_neutralize_ini_addopts():
    """The spawned rank must not inherit the project's ini addopts: `-n auto`
    there would give every rank its own xdist cluster, and `--cov=pkg` would
    start a second coverage on top of the .pth-started one."""
    from testflo.pytest_plugin import CHILD_PYTEST_ARGS

    assert "-o" in CHILD_PYTEST_ARGS
    assert "addopts=" in CHILD_PYTEST_ARGS
    i = CHILD_PYTEST_ARGS.index("-o")
    assert CHILD_PYTEST_ARGS[i + 1] == "addopts="
    assert "no:xdist" in CHILD_PYTEST_ARGS


def test_subprocess_child_coverage(pytester):
    """End-to-end: a line that only ever executes in a *spawned subprocess*
    must still be reported as covered.

    This is exactly the mechanism the spawned mpirun ranks rely on -- a rank
    is a plain subprocess that inherits COVERAGE_PROCESS_CONFIG and starts
    tracing from coverage.py's .pth file -- so this guards the MPI path on
    every platform, with or without MPI installed.
    """
    pytest.importorskip("pytest_cov")

    pytester.makepyfile(child_code="""
        def only_in_child(x):
            computed_in_child = x * 2      # must be reported as covered
            return computed_in_child
        """)
    pytester.makepyfile("""
        import subprocess, sys

        def test_spawns_child():
            # stands in for `mpirun -n N python -m pytest <nodeid>`
            subprocess.run(
                [sys.executable, "-c",
                 "import child_code; child_code.only_in_child(21)"],
                check=True)
        """)
    # `patch = subprocess` exports COVERAGE_PROCESS_CONFIG and implies
    # parallel=true. Section headers must not be indented or configparser
    # silently ignores the file ("Remainder of file ignored").
    pytester.path.joinpath(".coveragerc").write_text(
        "[run]\npatch = subprocess\nsource = child_code\n")

    result = pytester.runpytest_subprocess(
        "--cov=child_code", "--cov-report=term-missing")

    result.assert_outcomes(passed=1)
    # the child-only line is covered => 100%, nothing missing
    result.stdout.fnmatch_lines(["*child_code.py*100%*"])


@pytest.mark.skipif(platform.system() == "Windows",
                    reason="multiprocessing pool children do not flush "
                           "coverage on Windows/spawn; see the coverage "
                           "notes in the README")
def test_multiprocessing_pool_coverage(pytester):
    """Pool workers need more than the ranks do: a pool child exits through
    ``os._exit`` (``BaseProcess._bootstrap``), which skips the atexit hook
    coverage saves from, so ``patch = _exit`` is required on top of
    ``patch = subprocess``.  Relevant to the composed
    ``mpi(N)`` + ``multiprocessing(M)`` case, where the pool lives inside a
    rank.
    """
    pytest.importorskip("pytest_cov")

    pytester.makepyfile(pool_code="""
        def only_in_worker(x):
            computed_in_child = x * 2      # must be reported as covered
            return computed_in_child
        """)
    pytester.makepyfile("""
        import multiprocessing
        import pytest
        from pool_code import only_in_worker

        @pytest.mark.multiprocessing(2)
        def test_pool():
            with multiprocessing.Pool(2) as pool:
                assert pool.map(only_in_worker, [1, 2]) == [2, 4]
        """)
    pytester.path.joinpath(".coveragerc").write_text(
        "[run]\npatch = _exit, subprocess\nsource = pool_code\n")

    result = pytester.runpytest_subprocess(
        "--cov=pool_code", "--cov-report=term-missing")

    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*pool_code.py*100%*"])


@mpi
def test_mpi_rank_coverage(pytester):
    """The payoff: a line that only ever executes inside a spawned mpirun
    rank must be reported as covered.  Without the coverage env reaching the
    ranks this silently reports 0% for every MPI-only code path."""
    pytest.importorskip("pytest_cov")

    pytester.makepyfile(rank_code="""
        def only_under_mpi(rank):
            computed_in_rank = rank + 1    # must be reported as covered
            return computed_in_rank
        """)
    pytester.makepyfile("""
        import pytest
        from rank_code import only_under_mpi

        @pytest.mark.mpi(2)
        def test_ranks(comm):
            assert only_under_mpi(comm.rank) == comm.rank + 1
        """)
    pytester.path.joinpath(".coveragerc").write_text(
        "[run]\npatch = subprocess\nsource = rank_code\n")

    result = pytester.runpytest_subprocess(
        "--cov=rank_code", "--cov-report=term-missing")

    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*rank_code.py*100%*"])


@mpi
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="multiprocessing pool children do not flush "
                           "coverage on Windows/spawn")
def test_mpi_rank_with_pool_coverage(pytester):
    """The composed case: mpi(N) + multiprocessing(M), where the measured
    line runs in a pool worker *inside* an mpirun rank -- two process
    boundaries from the session that reports it."""
    pytest.importorskip("pytest_cov")

    pytester.makepyfile(nested_code="""
        def in_pool_in_rank(x):
            deeply_nested = x * 3          # must be reported as covered
            return deeply_nested
        """)
    pytester.makepyfile("""
        import multiprocessing
        import pytest
        from nested_code import in_pool_in_rank

        @pytest.mark.mpi(2)
        @pytest.mark.multiprocessing(2)
        def test_pool_in_rank(comm):
            with multiprocessing.Pool(2) as pool:
                assert pool.map(in_pool_in_rank, [1, 2]) == [3, 6]
        """)
    pytester.path.joinpath(".coveragerc").write_text(
        "[run]\npatch = _exit, subprocess\nsource = nested_code\n")

    result = pytester.runpytest_subprocess(
        "--cov=nested_code", "--cov-report=term-missing")

    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*nested_code.py*100%*"])


def test_child_coverage_env_payload_actually_measures(tmp_path, monkeypatch):
    """Closes the loop on the safety net: the env it synthesizes must really
    make a spawned child record coverage, not merely look plausible.

    Uses a plain subprocess, which is what an mpirun rank is, so this runs
    without MPI.
    """
    coverage = pytest.importorskip("coverage")
    import sys
    from testflo.pytest_plugin import _child_coverage_env

    for var in ("COVERAGE_PROCESS_CONFIG", "COVERAGE_PROCESS_START",
                "COV_CORE_SOURCE"):
        monkeypatch.delenv(var, raising=False)

    target = tmp_path / "child_target.py"
    target.write_text("def run():\n    measured = 1 + 1\n    return measured\n")
    data_file = tmp_path / ".coverage"

    # stand in for the coverage the user's `pytest --cov` has running, with
    # no subprocess support configured -- the case the net exists for
    cov = coverage.Coverage(data_file=str(data_file), source=[str(tmp_path)])
    cov.config.parallel = False
    monkeypatch.setattr(coverage.Coverage, "current", staticmethod(lambda: cov))

    env = dict(os.environ, **_child_coverage_env())
    assert "COVERAGE_PROCESS_CONFIG" in env

    subprocess.run([sys.executable, "-c", "import child_target; child_target.run()"],
                   cwd=tmp_path, env=env, check=True)

    written = list(tmp_path.glob(".coverage.*"))
    assert written, "the child recorded no coverage at all"

    combined = coverage.Coverage(data_file=str(data_file))
    combined.combine()
    data = combined.get_data()
    measured = [f for f in data.measured_files() if "child_target" in f]
    assert measured, f"child_target.py not measured; got {data.measured_files()}"
    assert 2 in data.lines(measured[0]), "the child-only line was not recorded"
