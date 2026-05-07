import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# --------------------------
# Helper functions
# --------------------------
def load_if_exists(file_path):
    """
    Load a .npy file if it exists; otherwise, print a warning and return None.
    If the file is loaded, print the array length and the first and last timestamp.
    """
    if os.path.exists(file_path):
        print("Loading:", file_path)
        data = np.load(file_path)
        if data is not None:
            print(f"Loaded data length: {len(data)}")
            if len(data) > 0:
                print(f"First timestamp: {data[0]}, Last timestamp: {data[-1]}")
        return data
    else:
        print("File does not exist:", file_path)
        return None

def check_abnormal_timestamps(data, label, context_window=10):
    """
    Check if the timestamp array is strictly increasing.
    If not, print the indices where the timestamp decreases and also print
    the surrounding timestamps (context) for each abnormal point.
    
    Parameters:
        data: 1D array of timestamps.
        label: A label to identify which data array is being checked.
        context_window: How many points before and after the abnormal point to print.
    """
    if data is None:
        print(f"{label}: No data available to check.")
        return None
    if len(data) < 2:
        print(f"{label}: Not enough data points to check monotonicity.")
        return None

    diffs = np.diff(data)
    abnormal_idx = np.where(diffs < 0)[0]
    if abnormal_idx.size > 0:
        abnormal_points = abnormal_idx + 1  # the drop happens at index+1
        print(f"{label}: Abnormal timestamps detected at indices {abnormal_points}.")
        for idx in abnormal_points:
            start = max(0, idx - context_window)
            end = min(len(data), idx + context_window + 1)
            print(f"Context around abnormal index {idx}:")
            for j in range(start, end):
                print(f"  Index {j}: timestamp = {data[j]}")
    else:
        print(f"{label}: Timestamps are monotonically increasing.")
    return abnormal_idx

def plot_timestamp(ax, data, title, max_points=1000):
    """
    Plot a 1D timestamp array on the given axis with a title.
    
    To accelerate plotting, if the number of points exceeds max_points,
    only a subset (every nth point) is plotted.
    
    Abnormal (non-monotonic) points that happen to fall on the sampled
    indices are marked in red.
    """
    if data is not None:
        n_points = len(data)
        if n_points > max_points:
            step = max(1, n_points // max_points)
            indices = np.arange(0, n_points, step)
            sampled_data = data[indices]
        else:
            indices = np.arange(n_points)
            sampled_data = data

        ax.plot(indices, sampled_data, marker='o', linestyle='-', markersize=2, label='Timestamp')
        # Compute abnormal indices on the full data.
        diffs = np.diff(data)
        abnormal_full = np.where(diffs < 0)[0] + 1  # indices in the full data array
        abnormal_in_sample = np.intersect1d(indices, abnormal_full)
        if abnormal_in_sample.size > 0:
            ax.plot(abnormal_in_sample, data[abnormal_in_sample], 'ro', markersize=5, label='Abnormal Timestamp')
            ax.legend()
    else:
        ax.text(0.5, 0.5, "Data not available",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Timestamp")

def analyze_and_plot_timestamps(session_folder, context_window=10, max_points=1000):
    """
    Analyze and plot continuous and event timestamp arrays stored in the given session folder.
    
    This function assumes that the local timestamp files are named 'local_timestamps.npy'
    and the additional files are named 'original_timestamps.npy'.
    
    Parameters:
        session_folder : str
            Path to the session folder.
        context_window : int, optional
            Number of points before and after an abnormal timestamp to print (default is 10).
        max_points : int, optional
            Maximum number of points to plot (downsampling is applied if exceeded; default is 1000).
    """
    # Fixed file name suffixes.
    local_timestamps_name = "local_timestamps"
    original_timestamps_name = "original_timestamps"
    
    # Define probe folders for continuous data.
    probeA_folder = os.path.join(session_folder, "Record Node 103", "experiment1", "recording1", "continuous", "Neuropix-PXI-100.ProbeA")
    probeB_folder = os.path.join(session_folder, "Record Node 103", "experiment1", "recording1", "continuous", "Neuropix-PXI-100.ProbeB")
    
    # --------------------------
    # Continuous timestamps file paths
    # --------------------------
    probeA_timestamps_path = os.path.join(probeA_folder, "timestamps.npy")
    probeB_timestamps_path = os.path.join(probeB_folder, "timestamps.npy")
    
    probeA_local_timestamps_path = os.path.join(probeA_folder, f"{local_timestamps_name}.npy")
    probeB_local_timestamps_path = os.path.join(probeB_folder, f"{local_timestamps_name}.npy")
    
    probeA_original_timestamps_path = os.path.join(probeA_folder, f"{original_timestamps_name}.npy")
    probeB_original_timestamps_path = os.path.join(probeB_folder, f"{original_timestamps_name}.npy")
    
    # --------------------------
    # Event timestamps file paths
    # --------------------------
    probeA_event_timestamps_path = os.path.join(session_folder, "Record Node 103", "experiment1", "recording1", "events", "Neuropix-PXI-100.ProbeA", "TTL", "timestamps.npy")
    probeB_event_timestamps_path = os.path.join(session_folder, "Record Node 103", "experiment1", "recording1", "events", "Neuropix-PXI-100.ProbeB", "TTL", "timestamps.npy")
    
    probeA_event_local_timestamps_path = os.path.join(session_folder, "Record Node 103", "experiment1", "recording1", "events", "Neuropix-PXI-100.ProbeA", "TTL", f"{local_timestamps_name}.npy")
    probeB_event_local_timestamps_path = os.path.join(session_folder, "Record Node 103", "experiment1", "recording1", "events", "Neuropix-PXI-100.ProbeB", "TTL", f"{local_timestamps_name}.npy")
    
    probeA_event_original_timestamps_path = os.path.join(session_folder, "Record Node 103", "experiment1", "recording1", "events", "Neuropix-PXI-100.ProbeA", "TTL", f"{original_timestamps_name}.npy")
    probeB_event_original_timestamps_path = os.path.join(session_folder, "Record Node 103", "experiment1", "recording1", "events", "Neuropix-PXI-100.ProbeB", "TTL", f"{original_timestamps_name}.npy")
    
    # --------------------------
    # Load the timestamp arrays
    # --------------------------
    # Continuous (probe) timestamps
    probeA_timestamps = load_if_exists(probeA_timestamps_path)
    probeB_timestamps = load_if_exists(probeB_timestamps_path)
    
    probeA_local_timestamps = load_if_exists(probeA_local_timestamps_path)
    probeB_local_timestamps = load_if_exists(probeB_local_timestamps_path)
    
    probeA_original_timestamps = load_if_exists(probeA_original_timestamps_path)
    probeB_original_timestamps = load_if_exists(probeB_original_timestamps_path)
    
    # Event timestamps
    probeA_event_timestamps = load_if_exists(probeA_event_timestamps_path)
    probeB_event_timestamps = load_if_exists(probeB_event_timestamps_path)
    
    probeA_event_local_timestamps = load_if_exists(probeA_event_local_timestamps_path)
    probeB_event_local_timestamps = load_if_exists(probeB_event_local_timestamps_path)
    
    probeA_event_original_timestamps = load_if_exists(probeA_event_original_timestamps_path)
    probeB_event_original_timestamps = load_if_exists(probeB_event_original_timestamps_path)
    
    # --------------------------
    # Check for abnormal timestamps (printing context)
    # --------------------------
    print("\n--- Continuous Timestamps Check ---")
    check_abnormal_timestamps(probeA_timestamps, "Probe A Continuous Timestamps", context_window)
    check_abnormal_timestamps(probeA_local_timestamps, "Probe A Continuous local_timestamps", context_window)
    check_abnormal_timestamps(probeA_original_timestamps, "Probe A Continuous original_timestamps", context_window)
    
    check_abnormal_timestamps(probeB_timestamps, "Probe B Continuous Timestamps", context_window)
    check_abnormal_timestamps(probeB_local_timestamps, "Probe B Continuous local_timestamps", context_window)
    check_abnormal_timestamps(probeB_original_timestamps, "Probe B Continuous original_timestamps", context_window)
    
    print("\n--- Event Timestamps Check ---")
    check_abnormal_timestamps(probeA_event_timestamps, "Probe A Event Timestamps", context_window)
    check_abnormal_timestamps(probeA_event_local_timestamps, "Probe A Event local_timestamps", context_window)
    check_abnormal_timestamps(probeA_event_original_timestamps, "Probe A Event original_timestamps", context_window)
    
    check_abnormal_timestamps(probeB_event_timestamps, "Probe B Event Timestamps", context_window)
    check_abnormal_timestamps(probeB_event_local_timestamps, "Probe B Event local_timestamps", context_window)
    check_abnormal_timestamps(probeB_event_original_timestamps, "Probe B Event original_timestamps", context_window)
    
    # --------------------------
    # Plot the timestamp arrays
    # --------------------------
    # Create a grid of subplots:
    # - Rows 0-2: Continuous (probe) timestamps (Standard, local, and original)
    # - Rows 3-5: Event timestamps (Standard, local, and original)
    # Each row has 2 columns: Probe A (left) and Probe B (right)
    fig, axs = plt.subplots(nrows=6, ncols=2, figsize=(12, 18))
    fig.tight_layout(pad=3.0)
    
    # Continuous Data Plots
    plot_timestamp(axs[0, 0], probeA_timestamps, "Probe A Continuous Timestamps", max_points)
    plot_timestamp(axs[0, 1], probeB_timestamps, "Probe B Continuous Timestamps", max_points)
    
    plot_timestamp(axs[1, 0], probeA_local_timestamps, "Probe A Continuous local_timestamps", max_points)
    plot_timestamp(axs[1, 1], probeB_local_timestamps, "Probe B Continuous local_timestamps", max_points)
    
    plot_timestamp(axs[2, 0], probeA_original_timestamps, "Probe A Continuous original_timestamps", max_points)
    plot_timestamp(axs[2, 1], probeB_original_timestamps, "Probe B Continuous original_timestamps", max_points)
    
    # Event Data Plots
    plot_timestamp(axs[3, 0], probeA_event_timestamps, "Probe A Event Timestamps", max_points)
    plot_timestamp(axs[3, 1], probeB_event_timestamps, "Probe B Event Timestamps", max_points)
    
    plot_timestamp(axs[4, 0], probeA_event_local_timestamps, "Probe A Event local_timestamps", max_points)
    plot_timestamp(axs[4, 1], probeB_event_local_timestamps, "Probe B Event local_timestamps", max_points)
    
    plot_timestamp(axs[5, 0], probeA_event_original_timestamps, "Probe A Event original_timestamps", max_points)
    plot_timestamp(axs[5, 1], probeB_event_original_timestamps, "Probe B Event original_timestamps", max_points)
    
    plt.show()

# --------------------------
# Setup logger for each session
# --------------------------
def setup_logger_for_session(session_folder, output_folder):
    """
    Create a logger for a given session with a log file name that includes the session name and a timestamp.
    """
    session_name = os.path.basename(session_folder.rstrip(os.sep))
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(output_folder, f"rig_qc_{session_name}_{timestamp_str}.log")
    logger = logging.getLogger(session_name)
    logger.setLevel(logging.INFO)
    # Clear any previous handlers.
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_filename)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger, log_filename

class LoggerWriter:
    """
    A helper class to redirect stdout and stderr to a logger.
    """
    def __init__(self, level):
        self.level = level
    def write(self, message):
        if message.strip() != "":
            self.level(message.strip())
    def flush(self):
        pass

# --------------------------
# Main block: iterate over sessions and create separate log files
# --------------------------
if __name__ == '__main__':
    output_folder = r"H:\rig_qc"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    session_folders = [
        r'I:\753126\753126_2024-10-14_11-52-34',
        r'I:\753126\753126_2024-10-10_14-41-12',
        r'I:\764787\764787_2024-12-11_15-01-07',
        r'I:\764790\764790_2024-12-19_16-11-28',
        r'X:\769884\769884_2025-01-15_16-12-54',
        r'X:\769884\769884_2025-01-16_18-33-07',
        r'I:\753126\753126_2024-10-10_14-41-12',
        r'I:\753125\753125_2024-10-09_10-50-07'
    ]


    session_folders = [r'I:\764769\764769_2024-12-11_18-21-41',
    r'I:\764769\764769_2024-12-12_16-04-55',
    r'I:\764769\764769_2024-12-13_15-40-48'
    ]

    session_folders = [r'X:\0\0_2025-02-12_11-18-01'
    ]

    for session_folder in session_folders:
        # Set up a separate logger for this session.
        logger, log_filename = setup_logger_for_session(session_folder, output_folder)
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        sys.stdout = LoggerWriter(logger.info)
        sys.stderr = LoggerWriter(logger.error)
        
        print("=" * 80)
        print(f"Processing session folder: {session_folder}")
        analyze_and_plot_timestamps(session_folder, context_window=10, max_points=1000)
        
        # Restore original stdout and stderr.
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
