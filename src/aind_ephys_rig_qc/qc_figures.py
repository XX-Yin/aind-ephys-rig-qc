"""
Generates figures for checking ephys data quality
"""

from matplotlib.figure import Figure
from scipy.signal import welch
import numpy as np


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
    extent = [f.min(), f.max(), 0, subset.shape[1] - 1]
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
        The recording streams to plot

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
    for stream_ind in range(len(stream_time)):
        ax[0].plot(stream_time[stream_ind], label=stream_names[stream_ind])
    ax[0].legend()
    ax[0].set_title("Time Alignment_original")
    ax[0].set_xlabel("Samples")
    ax[0].set_ylabel("Time (s)")

    """plot time alignment after alignment"""
    ignore_after_time = stream_time[0][-1] - np.min(stream_time[0])

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
    for stream_ind in range(len(stream_time)):
        ax[1].plot(
            stream_time_align[stream_ind], label=stream_names[stream_ind]
        )
    ax[1].legend()
    ax[1].set_title("Time Alignment_aligned")
    ax[1].set_xlabel("Samples")
    ax[1].set_ylabel("Time (s)")

    return fig
