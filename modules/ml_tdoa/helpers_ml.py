import numpy as np
import pyroomacoustics as pra
import random, math
from tqdm import tqdm

#### CONSTANTS #####
M = 6 #number of pair of mics used
c = 343.0 #sound speed at 20°C
tdoa_estimate_correction = 3.75

def compute_trueTDOA_estimateTDOA(source_pos, mics_pos, mics_signals, fs, max_tau, M=M):
    
    n_mics, _ = mics_pos.shape

    indexes_1 = np.random.randint(0, n_mics, size=M)
    indexes_2 = np.random.randint(0, n_mics, size=M)
    
    tdoa_gt = compute_M_tdoa(source_pos, mics_pos, indexes_1, indexes_2)
    tdoa_estimates = compute_tdoa_estimates(mics_signals, indexes_1, indexes_2, fs, max_tau, M)
    
    return tdoa_gt, tdoa_estimates


def compute_tdoa_estimates(mic_array, indexes_1, indexes_2, fs, max_tau, M=M):
    '''
    This function computes the ground truth TDOAs and their respictive estimate using 
    the GCC-PHAT method
    '''
    
    tdoa_estimates = []
    for i in tqdm(range(M)):
        tdoa_estimate, _ = gcc_phat(mic_array[indexes_1[i],:], mic_array[indexes_2[i],:], max_tau=max_tau, fs=fs)
        tdoa_estimates.append(tdoa_estimate)
    return tdoa_estimate_correction * np.array(tdoa_estimates)


def compute_tdoa(source_pos, mic1_pos, mic2_pos):
    
    src_to_mic1 = np.linalg.norm(source_pos - mic1_pos)
    src_to_mic2 = np.linalg.norm(source_pos - mic2_pos)
    
    return (src_to_mic1 - src_to_mic2)/c

def compute_M_tdoa(source_pos, mics_pos, indexes_1, indexes_2):
    '''
    This function computes the M TDOAs between randomly selected mics
    '''
    _, dim = mics_pos.shape
    
    assert dim==3, 'mic_array shape must be of the form (number_mics, dimension)'
    #assert n_mics>=M, 'max number of TDOA pair must be less than {n}'.format(n=n_mics)
    
    tdoas = []
    
    for i in range(M):
        tdoa = compute_tdoa(source_pos, mics_pos[indexes_1[i], :], mics_pos[indexes_2[i], :])
        tdoas.append(tdoa)
        
    return np.array(tdoas)

def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    from : https://github.com/xiongyihui/tdoa/blob/master/gcc_phat.py
    '''
    
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)
    
    return tau, cc

def sphere_micsArray(n_mics, room_dim, radius=0.01, randomize=False):
    rnd = 1.
    if randomize:
        rnd = random.random() * n_mics

    points = []
    offset = 2./n_mics
    increment = math.pi * (3. - math.sqrt(5.))

    for i in range(n_mics):
        y = ((i * offset) - 1) + (offset / 2) 
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % n_mics) * increment

        x = math.cos(phi) * r 
        z = math.sin(phi) * r 
        
        norm = np.sqrt(x**2 + y**2 + z**2)
        x /= (1/radius)*norm
        y /= (1/radius)*norm
        z /= (1/radius)*norm
        points.append([x+ room_dim[0]/2, y + room_dim[1]/2 , z + room_dim[2]/2])

    return np.array(points)