"""Example test template."""

import os
import unittest
from matplotlib.figure import Figure
from aind_ephys_rig_qc.generate_report import generate_qc_report
from aind_ephys_rig_qc.qc_figures import plot_drift


class TestGenerateReport(unittest.TestCase):
    """Example Test Class"""

    def test_generate_report(self):
        """Check if output is pdf."""
        directory = "F:/npOptoRecordings/691894_2023-10-04_18-03-13"
        report_name = "qc.pdf"
        generate_qc_report(directory, report_name)
        self.assertTrue(os.path.exists(os.path.join(directory, report_name)))

    def test_drift(self):
        """Check if output is figure."""
        directory = "F:/npOptoRecordings/691894_2023-10-04_18-03-13"
        stream_name = "ProbeA-AP"
        fig = plot_drift(directory, stream_name)
        self.assertIsInstance(fig, Figure)
