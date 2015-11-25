import numpy as np
import cPickle as pickle
import gzip
import data_io
window_size = 200
window_idxs = np.arange(100000).reshape(100000,1) + np.arange(window_size).reshape(1,window_size)

def window(wave):
    return wave[window_idxs[:wave.shape[0] - window_size + 1]]


def batch_and_pad(stream,batch_size=5):
    while True:
        data = [ stream.next()[1] for _ in xrange(batch_size) ]
        lengths = [x.shape[0] - window_size + 1 for x in data]
        max_length = max(lengths)
        buf = np.zeros((len(data),max_length,window_size),dtype=np.int16)
        for i,x in enumerate(data):
            windowed_data = window(x)
            buf[i,:windowed_data.shape[0]] = windowed_data
        yield buf,np.array(lengths)

def get_normalise(stream):
    sample_count = 0
    total_amp = 0
    total_amp_sqr = 0
    for name,data in stream:
        sample_count += data.shape[0]
        total_amp += np.sum(data)
        total_amp_sqr += np.sum(data**2)

    mean = total_amp / float(sample_count)
    std = np.sqrt(total_amp_sqr/float(sample_count) - mean)
    return mean,std


if __name__ == "__main__":
    rand_stream = data_io.random_select_stream(*[
                    data_io.stream_file('data/train.%02d.pklgz'%i)
                    for i in xrange(1,20)
                ])
    mean,std = get_normalise(rand_stream)

    def stream():
        stream = data_io.random_select_stream(*[
                    data_io.stream_file('data/train.%02d.pklgz'%i)
                    for i in xrange(1,20)
                ])
        stream = data_io.buffered_sort(stream,key=lambda x: x[1].shape[0],buffer_items=10)
        batched_stream = batch_and_pad(stream,batch_size=5)
        batched_stream = data_io.buffered_random(batched_stream,buffer_items=10)
        return batched_stream

    for data,lengths in stream():
        print lengths
        print ( data - mean ) / std
