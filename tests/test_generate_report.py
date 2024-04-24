"""Example test template."""

import os
import unittest

from aind_ephys_rig_qc.generate_report import generate_qc_report


class TestGenerateReport(unittest.TestCase):
    """Example Test Class"""

    def test_generate_report(self):
        """Example of how to test the truth of a statement."""

        generate_qc_report("", "qc.pdf")
        self.assertTrue(os.path.exists("qc.pdf"))


if __name__ == "__main__":
    unittest.main()
