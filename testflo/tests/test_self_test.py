
import os
import re
import subprocess
import tempfile
import unittest

# Update these if fixtures are added/removed/changed in self_report_fixtures.py.
EXPECTED = {"ran": 20, "passed": 3, "failed": 6, "skipped": 11}


class TestSelfReport(unittest.TestCase):
    def test_self_report_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            outfile = os.path.join(tmp, "report.out")
            proc = subprocess.run(
                ["testflo", "-o", outfile, "testflo.tests.self_report_fixtures"],
                capture_output=True, text=True,
            )
            report = ""
            if os.path.exists(outfile):
                with open(outfile) as f:
                    report = f.read()

        actual = {
            "ran": int(re.search(r"Ran (\d+) tests?", report).group(1)),
            "passed": int(re.search(r"Passed:\s*(\d+)", report).group(1)),
            "failed": int(re.search(r"Failed:\s*(\d+)", report).group(1)),
            "skipped": int(re.search(r"Skipped:\s*(\d+)", report).group(1)),
        }

        self.assertEqual(actual, EXPECTED,
                          "testflo's self-report counting changed:\n%s" % report)
        self.assertEqual(proc.returncode, 1)


if __name__ == '__main__':
    unittest.main()
