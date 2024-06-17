"""
Generates a PDF report from an Open Ephys data directory
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from open_ephys.analysis import Session

from aind_ephys_rig_qc import __version__ as package_version
from aind_ephys_rig_qc.pdf_utils import PdfReport
from aind_ephys_rig_qc.qc_figures import (
    plot_power_spectrum,
    plot_raw_data,
    plot_drift,
)

from aind_ephys_rig_qc.temporal_alignment import align_timestamps


def generate_qc_report(
    directory,
    report_name="QC.pdf",
    timestamp_alignment_method="local",
    original_timestamp_filename="original_timestamps.npy",
):
    """
    Generates a PDF report from an Open Ephys data directory

    Saves QC.pdf

    Parameters
    ----------
    directory : str
        The path to the Open Ephys data directory
    report_name : str
        The name of the PDF report
    timestamp_alignment_method : str
        The type of alignment to perform
        Option 1: 'local' (default)
        Option 2: 'harp' (extract Harp timestamps from the NIDAQ stream)
        Option 3: 'none' (don't align timestamps)
    original_timestamp_filename : str
        The name of the file for archiving the original timestamps

    """

    pdf = PdfReport("aind-ephys-rig-qc v" + package_version)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", size=12)
    pdf.set_y(30)
    pdf.write(h=12, text=f"Overview of recordings in {directory}")

    pdf.set_font("Helvetica", "", size=10)
    pdf.set_y(45)
    pdf.embed_table(get_stream_info(directory), width=pdf.epw)

    if (
        timestamp_alignment_method == "local"
        or timestamp_alignment_method == "harp"
    ):
        # perform local alignment first in either case
        align_timestamps(
            directory,
            align_timestamps_to=timestamp_alignment_method,
            original_timestamp_filename=original_timestamp_filename,
            pdf=pdf,
        )

        if timestamp_alignment_method == "harp":
            # optionally align to Harp timestamps
            align_timestamps(
                directory,
                align_timestamps_to=timestamp_alignment_method,
                original_timestamp_filename=original_timestamp_filename,
                pdf=pdf,
            )

    create_qc_plots(pdf, directory)

    pdf.output(os.path.join(directory, report_name))


def get_stream_info(directory):
    """
    Get information about the streams in an Open Ephys data directory

    Parameters
    ----------
    directory : str
        The path to the Open Ephys data directory

    Returns
    -------
    pd.DataFrame
        A DataFrame with information about the streams

    """

    session = Session(directory)

    stream_info = {
        "Record Node": [],
        "Rec Idx": [],
        "Exp Idx": [],
        "Stream": [],
        "Duration (s)": [],
        "Channels": [],
    }

    for recordnode in session.recordnodes:

        current_record_node = os.path.basename(recordnode.directory).split(
            "Record Node "
        )[1]

        for recording in recordnode.recordings:

            current_experiment_index = recording.experiment_index
            current_recording_index = recording.recording_index

            for stream in recording.continuous:

                sample_rate = stream.metadata["sample_rate"]
                data_shape = stream.samples.shape
                channel_count = data_shape[1]
                duration = data_shape[0] / sample_rate

                stream_info["Record Node"].append(current_record_node)
                stream_info["Rec Idx"].append(current_recording_index)
                stream_info["Exp Idx"].append(current_experiment_index)
                stream_info["Stream"].append(stream.metadata["stream_name"])
                stream_info["Duration (s)"].append(duration)
                stream_info["Channels"].append(channel_count)

    return pd.DataFrame(data=stream_info)


def get_event_info(events, stream_name):
    """
    Get information about the events in a given stream

    Parameters
    ----------
    events : pd.DataFrame
        A DataFrame with information about the events
    stream_name : str
        The name of the stream to query

    Returns
    -------
    pd.DataFrame
        A DataFrame with information about events for one stream

    """
    event_info = {
        "Line": [],
        "First Time (s)": [],
        "Last Time (s)": [],
        "Event Count": [],
        "Event Rate (Hz)": [],
    }

    events_for_stream = events[events.stream_name == stream_name]

    for line in events_for_stream.line.unique():
        events_for_line = events_for_stream[
            (events_for_stream.line == line) & (events_for_stream.state == 1)
        ]

        frequency = np.mean(np.diff(events_for_line.timestamp))
        first_time = events_for_line.iloc[0].timestamp
        last_time = events_for_line.iloc[-1].timestamp

        event_info["Line"].append(line)
        event_info["First Time (s)"].append(round(first_time, 2))
        event_info["Last Time (s)"].append(round(last_time, 2))
        event_info["Event Count"].append(events_for_line.shape[0])
        event_info["Event Rate (Hz)"].append(round(frequency, 2))

    return pd.DataFrame(data=event_info)


def create_qc_plots(pdf, directory):
    """
    Create QC plots for an Open Ephys data directory
    """

    session = Session(directory)

    for recordnode in session.recordnodes:

        current_record_node = os.path.basename(recordnode.directory).split(
            "Record Node "
        )[1]

        for recording in recordnode.recordings:

            current_experiment_index = recording.experiment_index
            current_recording_index = recording.recording_index

            events = recording.events

            for stream in recording.continuous:

                duration = (
                    stream.samples.shape[0] / stream.metadata["sample_rate"]
                )

                stream_name = stream.metadata["stream_name"]

                pdf.add_page()
                pdf.set_font("Helvetica", "B", size=12)
                pdf.set_y(30)
                pdf.write(h=12, text=f"{stream_name}")
                pdf.set_font("Helvetica", "", size=10)
                pdf.set_y(40)
                pdf.write(h=10, text=f"Record Node: {current_record_node}")
                pdf.set_y(45)
                pdf.write(
                    h=10,
                    text=f"Recording Index: " f"{current_recording_index}",
                )
                pdf.set_y(50)
                pdf.write(
                    h=10,
                    text=f"Experiment Index: " f"{current_experiment_index}",
                )
                pdf.set_y(55)
                pdf.write(
                    h=10,
                    text=f"Source Node: "
                    f"{stream.metadata['source_node_name']}"
                    f" ({stream.metadata['source_node_id']})",
                )
                pdf.set_y(60)
                pdf.write(h=10, text=f"Duration: {duration} s")
                pdf.set_y(65)
                pdf.write(
                    h=10,
                    text=f"Sample Rate: "
                    f"{stream.metadata['sample_rate']} Hz",
                )
                pdf.set_y(70)
                pdf.write(h=10, text=f"Channels: {stream.samples.shape[1]}")

                df = get_event_info(events, stream_name)

                pdf.set_y(80)
                pdf.set_font("Helvetica", "B", size=11)
                pdf.write(h=12, text="Event info")
                pdf.set_y(90)
                pdf.set_font("Helvetica", "", size=10)
                pdf.embed_table(df, width=pdf.epw)

                pdf.set_y(120)
                pdf.embed_figure(
                    plot_raw_data(
                        stream.samples,
                        stream.metadata["sample_rate"],
                        stream_name,
                    )
                )

                pdf.set_y(200)
                pdf.embed_figure(
                    plot_power_spectrum(
                        stream.samples,
                        stream.metadata["sample_rate"],
                        stream_name,
                    )
                )

                if "Probe" in stream_name and "LFP" not in stream_name:
                    pdf.set_y(200)
                    pdf.embed_figure(plot_drift(directory, stream_name))
            # pdf.set_y(200)
            # pdf.embed_figure(
            #     plot_timealign(
            #         recording
            #     )
            # )


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Two input arguments are required:")
        print(" 1. A data directory")
        print(" 2. A JSON parameters file")
    else:
        with open(sys.argv[2], "r",) as f:
            parameters = json.load(f)

        directory = sys.argv[1]

        if not os.path.exists(directory):
            raise ValueError(f"Data directory {directory} does not exist.")

        generate_qc_report(directory, **parameters)
