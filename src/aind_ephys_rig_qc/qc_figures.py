"""
Generates figures for checking ephys data quality
"""

from matplotlib.figure import Figure
from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import spikeinterface as si
import spikeinterface.extractors as se
from spikeinterface.core.node_pipeline import (
    ExtractDenseWaveforms,
    run_node_pipeline,
)
from spikeinterface.sortingcomponents.peak_detection import (
    DetectPeakLocallyExclusive,
)
from spikeinterface.sortingcomponents.peak_localization import (
    LocalizeCenterOfMass,
)


def plot_raw_data(data, sample_rate, stream_name):
    """
    Plot a snippet of raw data as an image

    Parameters
    ----------
    data : np.ndarray
        The data to plot (samples x channels)
    sample_rate : float
        The sample rate of the data
    stream_name : str
        The name of the stream

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plot
    """

    fig = Figure(figsize=(10, 4))
    ax = fig.subplots()

    ax.imshow(data[:1000, :].T, aspect="auto", cmap="RdBu")
    ax.set_title(f"{stream_name} Raw Data")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Channels")

    return fig


def plot_power_spectrum(data, sample_rate, stream_name):
    """
    Plot the power spectrum of the data

    Parameters
    ----------
    data : np.ndarray
        The data to plot (samples x channels)
    sample_rate : float
        The sample rate of the data
    stream_name : str
        The name of the stream

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plot
    """

    fig = Figure(figsize=(10, 4))
    ax = fig.subplots(2, 1, gridspec_kw={"height_ratios": [2, 1]})

    subset = data[:1000, :]
    p_channel = []
    for i in range(subset.shape[1]):
        f, p = welch(subset[:, i], fs=sample_rate)
        ax[1].plot(f, p)
        p_channel.append(p)

    p_channel = np.array(p_channel)
    extent = [f.min(), f.max(), subset.shape[1] - 1, 0]
    ax[0].imshow(p_channel, extent=extent, aspect="auto", cmap="inferno")
    ax[0].set_ylabel("Channels")
    ax[0].set_title(f"{stream_name} PSD")
    ax[1].set_xlabel("Frequency")
    ax[1].set_ylabel("Power")

    return fig


def plot_timealign(streams):
    """
    Plot the timealignment of the data

    Parameters
    ----------
    data : streams
        The recording streams to plot, including ProbeA-AP. 

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plot
    """

    fig = Figure(figsize=(10, 4))
    ax = fig.subplots(1, 2)

    stream_time = []
    stream_names = []
    """extract time of data streams"""
    for stream_ind in range(len(streams.continuous)):
        stream_time.append(streams.continuous[stream_ind].timestamps)
        stream_names.append(
            streams.continuous[stream_ind].metadata["stream_name"]
        )

    """plot time alignment"""
    ax[0].plot(stream_time[0], label=stream_names[0])
    ax[0].legend()
    ax[0].set_title("Time Alignment_original")
    ax[0].set_xlabel("Samples")
    ax[0].set_ylabel("Time (s)")

    """plot time alignment after alignment"""
    ignore_after_time = stream_time[0][-1] - np.min(stream_time[0])
    # last time in recording - min time

    streams.add_sync_line(
        1,  # TTL line number
        100,  # processor ID
        "ProbeA-AP",  # stream name
        main=True,  # set as the main stream
        ignore_intervals=[(ignore_after_time * 30000, np.inf)],
    )

    streams.add_sync_line(
        1,  # TTL line number
        100,  # processor ID
        "ProbeA-LFP",  # stream name
        ignore_intervals=[(ignore_after_time * 2500, np.inf)],
    )

    streams.add_sync_line(
        1,  # TTL line number
        103,  # processor ID
        "PXIe-6341",  # stream name
        ignore_intervals=[(ignore_after_time * 30000, np.inf)],
    )
    streams.compute_global_timestamps(overwrite=True)
    """extract time of data streams"""
    stream_time_align = []
    for stream_ind in range(len(streams.continuous)):
        stream_time_align.append(streams.continuous[stream_ind].timestamps)

    """plot time alignment"""
    ax[1].plot(stream_time_align[0], label=stream_names[0])
    ax[1].legend()
    ax[1].set_title("Time Alignment_aligned")
    ax[1].set_xlabel("Samples")
    ax[1].set_ylabel("Time (s)")

    return fig


def plot_drift(diretory, stream_name):
    """
    Plot the drift of the data by spike localization

    Parameters
    ----------
    openephys_folder : str
        The path to the OpenEphys data folder

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    visualization_drift_params = {
        "detection": {
            "peak_sign": "neg",
            "detect_threshold": 5,
            "exclude_sweep_ms": 0.1,
        },
        "localization": {
            "ms_before": 0.1,
            "ms_after": 0.3,
            "radius_um": 100.0,
        },
        "n_skip": 30,
        "alpha": 0.15,
        "vmin": -200,
        "vmax": 0,
        "cmap": "Greys_r",
        "figsize": [10, 10],
    }

    # get blocks/experiments and streams info
    si.set_global_job_kwargs(n_jobs=-1)

    # drift
    cmap = plt.get_cmap(visualization_drift_params["cmap"])
    norm = Normalize(
        vmin=visualization_drift_params["vmin"],
        vmax=visualization_drift_params["vmax"],
        clip=True,
    )
    n_skip = visualization_drift_params["n_skip"]
    alpha = visualization_drift_params["alpha"]

    stream_names, stream_ids = se.get_neo_streams("openephys", diretory)
    spike_stream = [
        curr_stream_name
        for curr_stream_name in stream_names
        if stream_name in curr_stream_name
    ][0]

    recording = se.read_openephys(
        folder_path=diretory, stream_name=spike_stream,
    )

    # Here we use the node pipeline implementation
    peak_detector_node = DetectPeakLocallyExclusive(
        recording, **visualization_drift_params["detection"]
    )
    extract_dense_waveforms_node = ExtractDenseWaveforms(
        recording,
        ms_before=visualization_drift_params["localization"]["ms_before"],
        ms_after=visualization_drift_params["localization"]["ms_after"],
        parents=[peak_detector_node],
        return_output=False,
    )
    localize_peaks_node = LocalizeCenterOfMass(
        recording,
        radius_um=visualization_drift_params["localization"]["radius_um"],
        parents=[peak_detector_node, extract_dense_waveforms_node],
    )
    pipeline_nodes = [
        peak_detector_node,
        extract_dense_waveforms_node,
        localize_peaks_node,
    ]
    peaks, peak_locations = run_node_pipeline(
        recording, nodes=pipeline_nodes, job_kwargs=si.get_global_job_kwargs()
    )
    peak_amps = peaks["amplitude"]
    y_locs = recording.get_channel_locations()[:, 1]
    ylim = [np.min(y_locs), np.max(y_locs)]

    fig_drift, axs_drift = plt.subplots(
        ncols=recording.get_num_segments(),
        figsize=visualization_drift_params["figsize"],
    )
    for segment_index in range(recording.get_num_segments()):
        segment_mask = peaks["segment_index"] == segment_index
        x = peaks[segment_mask]["sample_index"] / recording.sampling_frequency
        y = peak_locations[segment_mask]["y"]
        # subsample
        x_sub = x[::n_skip]
        y_sub = y[::n_skip]
        a_sub = peak_amps[::n_skip]
        colors = cmap(norm(a_sub))

    if recording.get_num_segments() == 1:
        ax_drift = axs_drift
    else:
        ax_drift = axs_drift[segment_index]
    ax_drift.scatter(x_sub, y_sub, s=1, c=colors, alpha=alpha)
    ax_drift.set_xlabel("time (s)", fontsize=12)
    ax_drift.set_ylabel("depth ($\\mu$m)", fontsize=12)
    ax_drift.set_xlim(
        0,
        recording.get_num_samples(segment_index=segment_index)
        / recording.sampling_frequency,
    )
    ax_drift.set_ylim(ylim)
    ax_drift.spines["top"].set_visible(False)
    ax_drift.spines["right"].set_visible(False)

    ax_drift.set_title(stream_name)

    return fig_drift
