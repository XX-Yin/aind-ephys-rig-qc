import os
directory = r'X:\764791\764791_2025-01-16_12-50-11'


recordingnodes=['Record Node 103']
experiments=[1,2,3,4,5,6,7,8,9,10]
recordings=['recording1']
types=['continuous','events']
streams=['Neuropix-PXI-100.ProbeA','Neuropix-PXI-100.ProbeB','NI-DAQmx-102.PXIe-6341']

for recordingnode in recordingnodes:
    for experiment in experiments:
        for recording in recordings:
            for type in types:
                for stream in streams:
                    if type=='continuous':
                        original_timestamp_file=os.path.join(directory,recordingnode,'experiment'+str(experiment),recording,'continuous',stream,'original_timestamps.npy')
                        timestamp_file=os.path.join(directory,recordingnode,'experiment'+str(experiment),recording,'continuous',stream,'timestamps.npy')
                        if os.path.exists(original_timestamp_file):
                            os.remove(timestamp_file)
                            os.rename(original_timestamp_file,timestamp_file)
                    elif type=='events':
                        original_timestamp_file=os.path.join(directory,recordingnode,'experiment'+str(experiment),recording,'events',stream,'TTL','original_timestamps.npy')
                        timestamp_file=os.path.join(directory,recordingnode,'experiment'+str(experiment),recording,'events',stream,'TTL','timestamps.npy')
                        if os.path.exists(original_timestamp_file):
                            os.remove(timestamp_file)
                            os.rename(original_timestamp_file,timestamp_file)