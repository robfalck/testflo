"""Tests for testflo-pytest itself.

These use pytest's ``pytester`` fixture to run the plugin end-to-end,
including real mpirun spawning where MPI is available.
"""

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


@mpi
def test_parallel_pass(pytester):
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.parallel(nprocs=2)
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

        @pytest.mark.parallel(nprocs=3)
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

        @pytest.mark.parallel([2, 3])
        def test_sizes(comm):
            assert comm.allreduce(1) == comm.size
        """
    )
    result = pytester.runpytest_subprocess("-v")
    result.assert_outcomes(passed=2)
    result.stdout.fnmatch_lines(["*nprocs=2*", "*nprocs=3*"])


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

        @pytest.mark.parallel(nprocs=2)
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

        @pytest.mark.parallel(nprocs=2)
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

        @pytest.mark.parallel(nprocs=2)
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

        @pytest.mark.parallel(nprocs=2)
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

        @pytest.mark.parallel(nprocs=2)
        def test_par_ok(comm):
            assert comm.allreduce(1) == 2

        @pytest.mark.parallel(nprocs=2)
        def test_par_fail(comm):
            if comm.rank == 1:
                assert False, "boom on rank 1"

        @pytest.mark.parallel(nprocs=2)
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
def test_xdist_loadgroup_pins_mpi_to_one_worker(pytester):
    """With --mpi-workers=1 (default), loadgroup is applied automatically and
    all MPI tests share one xdist worker while serial tests distribute freely."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.parametrize("i", range(4))
        def test_serial(i):
            pass

        @pytest.mark.parallel(nprocs=2)
        def test_mpi_a(comm): assert comm.size == 2

        @pytest.mark.parallel(nprocs=2)
        def test_mpi_b(comm): assert comm.size == 2
        """
    )
    result = pytester.runpytest_subprocess("-n", "2", "-v")
    result.assert_outcomes(passed=6)
    # loadgroup is applied automatically; verify the scheduler is active
    result.stdout.fnmatch_lines(["*LoadGroupScheduling*"])


@mpi
@pytest.mark.skipif(not HAVE_XDIST, reason="requires pytest-xdist")
def test_mpi_workers_spreads_across_groups(pytester):
    """--mpi-workers=2 distributes MPI tests across two xdist groups so
    both workers handle MPI tests."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.parallel(nprocs=2)
        def test_mpi_a(comm): assert comm.size == 2

        @pytest.mark.parallel(nprocs=2)
        def test_mpi_b(comm): assert comm.size == 2

        @pytest.mark.parallel(nprocs=2)
        def test_mpi_c(comm): assert comm.size == 2

        @pytest.mark.parallel(nprocs=2)
        def test_mpi_d(comm): assert comm.size == 2
        """
    )
    result = pytester.runpytest_subprocess("-n", "2", "--mpi-workers=2", "-v")
    result.assert_outcomes(passed=4)
    result.stdout.fnmatch_lines(["*LoadGroupScheduling*"])


@mpi
@pytest.mark.skipif(not HAVE_XDIST, reason="requires pytest-xdist")
def test_concurrent_slots_budget(pytester, monkeypatch):
    """--mpi-concurrent-slots bounds total in-flight ranks across workers;
    the limiter's high-water mark proves the budget was respected."""
    monkeypatch.setenv("TMPDIR", str(pytester.path))
    pytester.makepyfile(
        """
        import pytest, time

        @pytest.mark.parallel(nprocs=2)
        def test_a(comm): time.sleep(0.5)

        @pytest.mark.parallel(nprocs=2)
        def test_b(comm): time.sleep(0.5)

        @pytest.mark.parallel(nprocs=2)
        def test_c(comm): time.sleep(0.5)
        """
    )
    result = pytester.runpytest_subprocess("-n", "3",
                                           "--mpi-concurrent-slots=2", "-v")
    result.assert_outcomes(passed=3)
    import glob, json
    slot_files = glob.glob(str(pytester.path / "testflo_pytest_*.slots"))
    assert slot_files, "slot state file not found"
    state = json.load(open(slot_files[0]))
    assert state["hwm"] <= 2, f"budget exceeded: hwm={state['hwm']}"
    assert not state["holders"], "slots leaked"


def test_nompi_runs_in_process(pytester):
    """--nompi runs parallel tests on a FakeComm of size 1, like testflo."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.parallel(nprocs=4)
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


def test_max_nprocs_guard(pytester, monkeypatch):
    monkeypatch.setenv("TESTFLO_PYTEST_MAX_NPROCS", "4")
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.parallel(nprocs=8)
        def test_too_big(comm):
            pass
        """
    )
    result = pytester.runpytest_subprocess("-v")
    assert result.ret != 0
    combined = result.stdout.str() + result.stderr.str()
    assert "too many ranks" in combined
