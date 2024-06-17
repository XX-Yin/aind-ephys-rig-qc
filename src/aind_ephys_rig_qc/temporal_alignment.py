"""
Aligns timestamps across multiple data streams
"""

import json
import os
import sys

from open_ephys.analysis import Session
from harp.clock import decode_harp_clock, align_timestamps_to_anchor_points
from matplotlib.figure import Figure
import numpy as np


def align_timestamps(
    directory,
    align_timestamps_to="local",
    original_timestamp_filename="original_timestamps.npy",
    pdf=None,
):
    """
    Aligns timestamps across multiple Open Ephys data streams

    Parameters
    ----------
    directory : str
        The path to the Open Ephys data directory
    align_timestamps_to : str
        The type of alignment to perform
        Option 1: 'local' (default)
        Option 2: 'harp' (extract Harp timestamps from the NIDAQ stream)
    original_timestamp_filename : str
        The name of the file for archiving the original timestamps
    qc_report : PdfReport
        Report for adding QC figures (optional)
    """

    session = Session(directory)

    local_sync_line = 1
    main_stream_index = 0

    for recordnode in session.recordnodes:

        current_record_node = os.path.basename(recordnode.directory).split(
            "Record Node "
        )[1]

        for recording in recordnode.recordings:

            current_experiment_index = recording.experiment_index
            current_recording_index = recording.recording_index

            if pdf is not None:
                pdf.add_page()
                pdf.set_font("Helvetica", "B", size=12)
                pdf.set_y(30)
                pdf.write(
                    h=12,
                    text=f"Temporal alignment of Record Node {current_record_node}, Experiment {current_experiment_index}, Recording {current_recording_index}",
                )
                fig = Figure(figsize=(10, 4))
                ax1, ax2 = fig.subplots(nrows=1, ncols=2)

            events = recording.events

            main_stream = recording.continuous[main_stream_index]

            main_stream_name = main_stream.metadata["stream_name"]
            main_stream_source_node_id = main_stream.metadata["source_node_id"]
            main_stream_sample_rate = main_stream.metadata["sample_rate"]

            main_stream_events = events[
                (events.stream_name == main_stream_name)
                & (events.processor_id == main_stream_source_node_id)
                & (events.line == local_sync_line)
                & (events.state == 1)
            ]

            main_stream_times = (
                main_stream_events.sample_number.values
                / main_stream_sample_rate
            )
            main_stream_times = (
                main_stream_times - main_stream_times[0]
            )  # start at 0

            print(
                f"Total events for {main_stream_name}: {len(main_stream_events)}"
            )

            for stream_idx, stream in enumerate(recording.continuous):

                if stream_idx != main_stream_index:

                    stream_name = stream.metadata["stream_name"]
                    source_node_id = stream.metadata["source_node_id"]
                    sample_rate = stream.metadata["sample_rate"]

                    events_for_stream = events[
                        (events.stream_name == stream_name)
                        & (events.processor_id == source_node_id)
                        & (events.line == local_sync_line)
                        & (events.state == 1)
                    ]

                    print(
                        f"Total events for {stream_name}: {len(events_for_stream)}"
                    )

                    if pdf is not None:

                        ax1.plot(
                            (
                                np.diff(events_for_stream.timestamp)
                                - np.diff(main_stream_events.timestamp)
                            )
                            * 1000,
                            label=(stream_name),
                            linewidth=0.1,
                        )
                        ax1.set_ylim([-1.5, 1.5])

                    assert len(main_stream_events) == len(events_for_stream)

                    local_stream_times = (
                        events_for_stream.sample_number.values / sample_rate
                    )

                    ts = align_timestamps_to_anchor_points(
                        local_stream_times,
                        local_stream_times,
                        main_stream_times,
                    )

                    # write the new timestamps .npy files

                    if pdf is not None:
                        ax2.plot(
                            (np.diff(ts) - np.diff(main_stream_times)) * 1000,
                            label=stream_name,
                            linewidth=1,
                        )
                        ax2.set_ylim([-1.5, 1.5])

            if pdf is not None:
                ax1.set_title("Original alignment")
                ax1.set_xlabel("Event number")
                ax1.set_ylabel("Time interval (ms)")
                ax1.legend()

                ax2.set_title("After local alignment")
                ax2.set_xlabel("Event number")
                ax2.set_ylabel("Time interval (ms)")
                ax2.legend()

                pdf.set_y(40)
                pdf.embed_figure(fig)


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

        align_timestamps(directory, **parameters)
