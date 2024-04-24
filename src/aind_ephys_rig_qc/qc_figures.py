"""
Generates figures for checking ephys data quality
"""

from matplotlib.figure import Figure
from scipy.signal import welch


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
    ax = fig.subplots()

    subset = data[:1000, :]

    for i in range(subset.shape[1]):
        f, p = welch(subset[:, i], fs=sample_rate)
        ax.plot(f, p)

    ax.set_title(f"{stream_name} PSD")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power")

    return fig
