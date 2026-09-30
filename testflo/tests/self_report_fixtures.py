"""Tests in this module intentionally fail, get skipped, or use
@unittest.expectedFailure, purely to validate testflo's own pass/fail/skip
counting logic. The filename deliberately doesn't match testflo's default
'test*.py' discovery pattern, so a plain `testflo .` never picks this module
up on its own -- see testflo/tests/test_self_test.py, which is what actually
runs this (as a subprocess) and asserts the resulting counts are correct.
Don't run this file directly and expect a clean result.
"""

import os

import unittest


class TestfloTestCase(unittest.TestCase):
    def test_fail(self):
        self.fail("This test should fail")

    @unittest.expectedFailure
    def test_expected_fail_good(self):
        self.fail("I expected this")

    @unittest.expectedFailure
    def test_unexpected_success(self):
        pass

    @unittest.skip("skipping 1")
    def test_skip(self):
        self.fail("This test should have been skipped.")


class TestfloTestCaseWFixture(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pid = os.getpid()
        print("setting up %s, pid=%d" % (cls.__name__, cls.pid))

    @classmethod
    def tearDownClass(cls):
        assert os.getpid() == cls.pid
        print("tearing down %s, pid=%d" % (cls.__name__, cls.pid))

    def test_tcase_grouped_fail(self):
        self.fail("failure 2")

    @unittest.expectedFailure
    def test_tcase_grouped_expected_fail(self):
        self.fail("I expected this")

    @unittest.expectedFailure
    def test_tcase_grouped_unexpected_success(self):
        pass

    @unittest.skip("skipping 2")
    def test_tcase_grouped_skip(self):
        pass


@unittest.skip("skipping a whole testcase...")
class SkippedTestCase(unittest.TestCase):
    def test_1(self):
        self.fail("This test should have been skipped.")

    def test_2(self):
        self.fail("This test should have been skipped.")

    def test_3(self):
        self.fail("This test should have been skipped.")

    def test_4(self):
        self.fail("This test should have been skipped.")


class TestfloTestCaseWFixture2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pid = os.getpid()
        print("setting up %s, pid=%d" % (cls.__name__, cls.pid))

    @classmethod
    def tearDownClass(cls):
        assert os.getpid() == cls.pid
        print("tearing down %s, pid=%d" % (cls.__name__, cls.pid))

    def test_tcase_grouped_fail(self):
        self.fail("failure 3")

    @unittest.expectedFailure
    def test_tcase_grouped_expected_fail(self):
        self.fail("I expected this")

    @unittest.expectedFailure
    def test_tcase_grouped_unexpected_success(self):
        pass

    @unittest.skip("skipping 2")
    def test_tcase_grouped_skip(self):
        self.fail("This test should have been skipped.")


@unittest.skip("skipping a whole testcase...")
class SkippedTestCase2(unittest.TestCase):
    def test_1(self):
        self.fail("This test should have been skipped.")

    def test_2(self):
        self.fail("This test should have been skipped.")

    def test_3(self):
        self.fail("This test should have been skipped.")

    def test_4(self):
        self.fail("This test should have been skipped.")


if __name__ == '__main__':
    unittest.main()
