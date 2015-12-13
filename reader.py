import numpy as np
import cPickle as pickle
import gzip
import data_io
window_size = 200
import random

def window(wave):
    length = wave.shape[0]
    windows = length // window_size
    remainder = length - windows * window_size
    start = random.randint(0,remainder)
    return wave[start:start + windows * window_size].reshape(windows,window_size)


def batch_and_pad(stream, batch_size=5,mean=0,std=1):
    while True:
        data = [stream.next()[1] for _ in xrange(batch_size)]
        lengths = [x.shape[0] // window_size for x in data]
        max_length = max(lengths)
        buf = np.zeros((max_length, len(data), window_size), dtype=np.float32)
        for i, x in enumerate(data):
            windowed_data = window((x-mean)/std)
            buf[:windowed_data.shape[0], i] = windowed_data
        yield buf, np.array(lengths,dtype=np.int32)


def get_normalise(stream):
    sample_count = 0
    total_amp = 0
    total_amp_sqr = 0
    count = 0
    for name, data in stream:
        sample_count += data.shape[0]
        total_amp += np.sum(data)
        total_amp_sqr += np.sum(data**2)
        count += 1
    mean = total_amp / float(sample_count)
    std = np.sqrt(total_amp_sqr / float(sample_count) - mean)
    return mean, std, count
