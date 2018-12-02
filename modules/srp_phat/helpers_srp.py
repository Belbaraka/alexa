import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.io import wavfile
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import math, random



def sphere_micsArray(n_mics, room_dim, radius=0.1, randomize=False):
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

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan(y/x)
    
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    return x, y, z

def changeOf_axes(posX, posY, posZ, room_dim):
    new_X = posX - room_dim[0]/2 
    new_Y = posY - room_dim[1]/2 
    new_Z = posZ - room_dim[2]/2 

    return new_X, new_Y, new_Z


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

def micAxis_to_orgAxis(x_mic, y_mic, z_mic, room_dim):
    x_src = room_dim[0]/2 + x_mic
    y_src = room_dim[1]/2 + y_mic
    z_src = room_dim[2]/2 + z_mic
    
    return x_src, y_src, z_src