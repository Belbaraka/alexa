import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def awgn(fs, duration):
    '''Additive white gaussian noise sampled at fs for a duration of length seconds'''
    mean = 0
    std = 1 
    num_samples = fs * duration
    samples = np.random.normal(mean, std, size=num_samples)
    return samples


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def newRoom_RIR(source_pos, source_signal, Xdim, Ydim, Zdim, mic_pos, fs):
    '''Compute the RIR of a new sound source '''
    
    #Create room
    room = pra.ShoeBox([Xdim,Ydim], fs=fs, max_order = 4, absorption=0.2)
    room.extrude(Zdim)
    
    #Add microphone
    room.add_microphone_array(pra.MicrophoneArray(mic_pos, room.fs))
    
    #Add source
    room.add_source(source_pos, signal=source_signal)
    
    room.image_source_model(use_libroom=False)
    
    room.compute_rir()
    
    return room.rir


def create_training_set(grid, signal_train, Xdim, Ydim, Zdim, mic_pos, fs):
    '''Create training dictionary by computing the Room Impulse Response of the different sound source position in the grid'''
    n_samples, _ = grid.shape
    training_set = dict()
    
    for i,pos in enumerate(tqdm(grid)):
        rir = newRoom_RIR(pos, signal_train, Xdim, Ydim, Zdim, mic_pos, fs)
        training_set[i] = rir
    
    return training_set


def add_zeroPad(signal1, signal2):
    l1 = len(signal1)
    l2 = len(signal2)
    
    if l1 > l2:
        pad_size = l1 - l2
        signal2 = np.pad(signal2, (0, pad_size), 'constant')
    else:
        pad_size = l2 - l1
        signal1 = np.pad(signal1, (0, pad_size), 'constant')
    return signal1, signal2   


def correlate(signal1, signal2):
    
    if len(signal1) != len(signal2):
        signal1, signal2 = add_zeroPad(signal1, signal2)
    
    corr = np.corrcoef(signal1, signal2)
    return corr[0,1]
        

def compute_similarities(rir, train, top=3):
    '''Compute the cross correlation between the test RIR with all the RIR of the training set and select the top matches rir's'''
    
    similarities = []
    for key, val in train.items():
        rir_train = val[0][0]
        similarities.append(correlate(rir, rir_train))
    
    top_array = np.array(similarities).argsort()[-top:][::-1]
    return top_array