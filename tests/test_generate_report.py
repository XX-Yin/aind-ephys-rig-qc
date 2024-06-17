"""Example test template."""

import os
import unittest

from aind_ephys_rig_qc.generate_report import generate_qc_report


class TestGenerateReport(unittest.TestCase):
    """Example Test Class"""

    def test_generate_report(self):
        """Example of how to test the truth of a statement."""
        directory = "F:/npOptoRecordings/691894_2023-10-04_18-03-13"
        report_name = "qc.pdf"
        generate_qc_report(directory, report_name)
        self.assertTrue(os.path.exists(os.path.join(directory, report_name)))
