
import os

import unittest

modpid = None

def setUpModule():
    global modpid
    modpid = os.getpid()
    print("\ncalled setUpModule from pid %d\n" % modpid)

def tearDownModule():
    global modpid
    mypid = os.getpid()
    assert mypid == modpid
    print("\ncalled tearDownModule from pid %d\n" % mypid)


class TestfloTestCaseWFixture2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pid = os.getpid()
        print("\nsetting up %s, pid=%d\n" % (cls.__name__, cls.pid))

    @classmethod
    def tearDownClass(cls):
        assert os.getpid() == cls.pid
        print("\ntearing down %s, pid=%d\n" % (cls.__name__, cls.pid))

    def test_tcase_grouped_ok(self):
        assert os.getpid() == self.pid


if __name__ == '__main__':
    unittest.main()
