"""
Aligns timestamps across multiple data streams
"""

import json
import os
import sys

import numpy as np


def align_timestamps(
    directory,
    align_timestamps_to="local",
    original_timestamp_filename="original_timestamps.npy",
):
    """
    Aligns timestamps across multiple streams

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
    """

    return None


def get_num_barcodes(harp_events, delta_time=0.5):
    """
    Returns the number of barcodes

    Parameter
    ---------
    harp_events : pd.DataFrame
        Events dataframe from open ephys tools
    delta_time : float
        The time difference between barcodes

    Returns
    -------
    numbarcodes : int
        The number of barcodes in the recordings
    """
    (splits,) = np.where(np.diff(harp_events.timestamp) > delta_time)
    return len(splits) - 1


def get_barcode(harp_events, index, delta_time=0.5):
    """
    Returns a subset of original DataFrame corresponding to a specific
    barcode

    Parameter
    ---------
    harp_events : pd.DataFrame
        Events dataframe from open ephys tools
    index : int
        The index of the barcode being requested
    delta_time : float
        The time difference between barcodes

    Returns
    -------
    sample_numbers : np.array
        Array of integer sample numbers for each barcode event
    states : np.array
        Array of states (1 or 0) for each barcode event

    """
    (splits,) = np.where(np.diff(harp_events.timestamp) > delta_time)

    barcode = harp_events.iloc[splits[index] + 1:splits[index + 1] + 1]

    return barcode.sample_number.values, barcode.state.values


def convert_barcode_to_time(
    sample_numbers, states, baud_rate=1000.0, sample_rate=30000.0
):
    """
    Converts event sample numbers and states to
    a Harp timestamp in seconds.

    Harp timestamp is encoded as 32 bits, with
    the least significant bit coming first, and 2 bits
    between each byte.
    """

    samples_per_bit = int(sample_rate / baud_rate)
    middle_sample = int(samples_per_bit / 2)

    intervals = np.diff(sample_numbers)

    barcode = np.concatenate(
        [
            np.ones((count,)) * state
            for state, count in zip(states[:-1], intervals)
        ]
    ).astype("int")

    val = np.concatenate(
        [
            np.arange(
                samples_per_bit + middle_sample + samples_per_bit * 10 * i,
                samples_per_bit * 10 * i
                - middle_sample
                + samples_per_bit * 10,
                samples_per_bit,
            )
            for i in range(4)
        ]
    )
    s = np.flip(barcode[val])
    harp_time = s.dot(2 ** np.arange(s.size)[::-1])

    return harp_time


def rescale_times_linear(times_ephys, harp_events, delta_time=0.5):
    """
    Applies a linear rescaling to the ephys timestamps
    based on the HARP timestamps.

    Parameters
    ----------
    times_ephys : np.array
        Array with the ephys timestamps
    harp_events : pd.DataFrame
        Events dataframe from open ephys tools
    delta_time : float
        The time difference between barcodes

    Returns
    -------
    new_times : np.array
        Rescaled ephys timestamps
    """
    splits = np.where(np.diff(harp_events.timestamp) > delta_time)[0]
    last_index = len(splits) - 2

    t1_ephys = (
        harp_events.iloc[splits[0] + 1:splits[1] + 1].iloc[0].timestamp
    )
    t2_ephys = (
        harp_events.iloc[splits[last_index] + 1:splits[last_index + 1] + 1]
        .iloc[0]
        .timestamp
    )

    sample_numbers, states = get_barcode(harp_events, 0)
    t1_harp = convert_barcode_to_time(sample_numbers, states)
    sample_numbers, states = get_barcode(harp_events, last_index)
    t2_harp = convert_barcode_to_time(sample_numbers, states)

    new_times = np.copy(times_ephys)
    scaling = (t2_harp - t1_harp) / (t2_ephys - t1_ephys)
    new_times -= t1_ephys
    new_times *= scaling
    new_times += t1_harp

    return new_times


def compute_ephys_harp_times(
    times_ephys,
    harp_events,
    fs=30_000,
    delta_time=0.5,
    wrong_decoded_delta_time=2,
):
    """
    Computes ephys timestamps assuming that they are uniformly samples
    between barcodes. The times_ephys are only used to get the
    number of samples.

    Parameters
    ----------
    times_ephys : np.array
        Array with the ephys timestamps
    harp_events : pd.DataFrame
        Events dataframe from open ephys tools
    delta_time : float
        The time difference between barcodes
    wrong_decoded_delta_time : float
        The time difference threshold between barcodes to detect a wrong
        decoding and fit the barcode time

    Returns
    -------
    new_times : np.array
        Rescaled ephys timestamps
    """
    sampling_period = 1 / fs

    # compute all barcode times
    num_harp_events = get_num_barcodes(harp_events, delta_time=delta_time)
    timestamps_harp = np.zeros(num_harp_events, dtype="float64")
    for i in range(num_harp_events):
        sample_numbers, states = get_barcode(
            harp_events, i, delta_time=delta_time
        )
        barcode_harp_time = convert_barcode_to_time(sample_numbers, states)
        timestamps_harp[i] = barcode_harp_time

    # fix any wrong decoding
    (wrong_decoded_idxs,) = np.where(
        (np.diff(timestamps_harp)) > wrong_decoded_delta_time
    )
    print(f"Found {len(wrong_decoded_idxs)} badly aligned timestamps")
    timestamps_harp[wrong_decoded_idxs + 1]
    for idx in wrong_decoded_idxs + 1:
        new_ts = (
            timestamps_harp[idx - 1]
            + (timestamps_harp[idx + 1] - timestamps_harp[idx - 1]) / 2
        )
        timestamps_harp[idx] = new_ts

    # get indices of harp clock in ephys timestamps
    (splits,) = np.where(np.diff(harp_events.timestamp) > delta_time)
    harp_clock_indices = np.searchsorted(
        times_ephys, harp_events.timestamp.values[splits[:-1] + 1]
    )

    # compute new ephys times
    times_ephys_aligned = np.zeros_like(times_ephys, dtype="float64")

    # here we use the BARCODE as a metronome and assume that ephys
    # is uniformly sampled in between HARP beats
    for i, (t_harp, harp_idx) in enumerate(
        zip(timestamps_harp, harp_clock_indices)
    ):
        if i == 0:
            first_sample = 0
            num_samples = harp_idx
            # fill in other chunks
            sample_indices = np.arange(num_samples)
            t_start = t_harp - len(sample_indices) / fs
        else:
            first_sample = harp_clock_indices[i - 1]
            num_samples = harp_idx - harp_clock_indices[i - 1]
            t_start = timestamps_harp[i - 1]

        # fill in other chunks
        sample_indices = np.arange(num_samples)
        if i != 0:
            t_start = timestamps_harp[i - 1]
        else:
            t_start = t_harp - len(sample_indices) / fs
        t_stop = t_harp - sampling_period
        times_ephys_aligned[first_sample:harp_idx] = np.linspace(
            t_start, t_stop, num_samples
        )

    # take care of last chunk
    num_samples = len(times_ephys_aligned) - harp_idx
    t_start = times_ephys_aligned[harp_idx - 1] + sampling_period
    t_stop = t_start + num_samples / fs
    times_ephys_aligned[harp_idx:] = np.linspace(t_start, t_stop, num_samples)

    return times_ephys_aligned


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Two input arguments are required:")
        print(" 1. A data directory")
        print(" 2. A JSON parameters file")
    else:
        with open(
            sys.argv[2],
            "r",
        ) as f:
            parameters = json.load(f)

        directory = sys.argv[1]

        if not os.path.exists(directory):
            raise ValueError(f"Data directory {directory} does not exist.")

        align_timestamps(directory, **parameters)
