import gzip
import cPickle as pickle
import sys
import numpy as np
from itertools import izip
import random


def splice(stream, left=5, right=5):
    left_buf = right_buf = None
    idxs = np.arange(1000).reshape(1000, 1) + np.arange(left + 1 + right)
    for frames in stream:
        dim = frames.shape[1]
        if left_buf is None:
            left_buf = np.zeros((left, dim), dtype=np.float32)
            right_buf = np.zeros((right, dim), dtype=np.float32)
        length = frames.shape[0]
        if length > idxs.shape[0]:
            idxs = np.arange(length).reshape(length, 1) + np.arange(left + 1 + right)
        frames = np.concatenate([left_buf, frames, right_buf])
        frames = frames[idxs[:length]]
        frames = frames.reshape(length, (left + 1 + right) * dim)
        yield frames


def stream_file(filename, open_method=gzip.open):
    with open_method(filename, 'rb') as fd:
        try:
            while True:
                x = pickle.load(fd)
                yield x
        except EOFError:
            pass


def stream(*filenames, **kwargs):
    gens = [stream_file(f) for f in filenames]
    return zip_streams(*gens, **kwargs)


def random_select_stream(*streams):
    while len(streams) > 0:
        stream_idx = random.randint(0, len(streams) - 1)
        try:
            yield streams[stream_idx].next()
        except StopIteration:
            streams = streams[:stream_idx] + streams[stream_idx + 1:]


def zip_streams(*streams, **kwargs):
    with_name = kwargs.get('with_name', False)
    while True:
        items = [s.next() for s in streams]
        assert(all(x[0] == items[0][0] for x in items))
        result = tuple(x[1] for x in items)

        if with_name:
            result = (items[0][0],) + result
        if len(result) == 1:
            yield result[0]
        else:
            yield result


def buffered_random(stream, buffer_items=256, leak_percent=0.9):
    item_buffer = [None] * buffer_items
    leak_count = int(buffer_items * leak_percent)
    item_count = 0
    for item in stream:
        item_buffer[item_count] = item
        item_count += 1
        if buffer_items == item_count:
            random.shuffle(item_buffer)
            for item in item_buffer[leak_count:]:
                yield item
            item_count = leak_count
    if item_count > 0:
        item_buffer = item_buffer[:item_count]
        random.shuffle(item_buffer)
        for item in item_buffer:
            yield item


def buffered_sort(stream, buffer_items=256, key=lambda x: x[0].shape[0]):
    item_buffer = [None] * buffer_items
    item_count = 0
    for item in stream:
        item_buffer[item_count] = item
        item_count += 1
        if buffer_items == item_count:
            item_buffer.sort(key=key)
            for item in item_buffer:
                yield item
            item_count = 0
    if item_count > 0:
        item_buffer = item_buffer[:item_count]
        item_buffer.sort(key=key)
        for item in item_buffer:
            yield item


def batch_and_pad(stream, batch_size=5):
    buffer_frames = [None] * batch_size
    buffer_phns = [None] * batch_size
    while True:
        print "Accumulating.."
        for i in xrange(batch_size):
            frames, phns = stream.next()
            buffer_frames[i] = frames
            buffer_phns[i] = phns
        print "Done."
        batch_size_curr = i + 1
        batch_frames_length = max(f.shape[0] for f in buffer_frames)
        batch_phns_length = max(p.shape[0] for p in buffer_phns)
        feature_dim = buffer_frames[-1].shape[1]
        batch_frames = np.zeros((batch_size_curr, batch_frames_length, feature_dim), dtype=np.float32)
        batch_phns = np.zeros((batch_size_curr, batch_phns_length), dtype=np.int32)
        batch_frames_mask = np.zeros((batch_size_curr, batch_frames_length), dtype=np.int8)
        batch_phns_mask = np.zeros((batch_size_curr, batch_phns_length), dtype=np.int8)
        for i in xrange(batch_size_curr):
            frames_seq_length = buffer_frames[i].shape[0]
            phns_seq_length = buffer_phns[i].shape[0]
            batch_frames[i, :frames_seq_length] = buffer_frames[i]
            batch_phns[i, :phns_seq_length] = buffer_phns[i]
            batch_frames_mask[i, :frames_seq_length] = 1
            batch_phns_mask[i, :phns_seq_length] = 1

        yield (batch_frames, batch_frames_mask), (batch_phns, batch_phns_mask)
        print "Yielded."

if __name__ == "__main__":
    import time
    stream = stream('data/raw_train.19.pklgz', 'data/raw_train_trans.19.pklgz')
    stream = buffered_sort(stream)
    batched_stream = batch_and_pad(stream)

    for (f, f_mask), (p, p_mask) in batched_stream:
        print p_mask
