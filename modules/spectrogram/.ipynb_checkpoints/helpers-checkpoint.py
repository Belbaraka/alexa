from scipy.signal import correlate
import numpy as np 
from scipy.io import wavfile
import IPython
import matplotlib.pyplot as plt
import pyroomacoustics as pra




def signals_cross_correlation(signal1, signal2):
    l1 = len(signal1)
    l2 = len(signal2)
    
    #normalize the input arrays
    signal1 = (signal1-signal1.mean()) / signal1.std()
    signal2 = (signal2-signal2.mean()) / signal2.std()
    
    if l1 > l2:
        
        corr = correlate(signal1, signal2)
    else:
        corr = correlate(signal2, signal1)
    
    return corr 

def plot_cross(corr):
    fig = plt.figure()
    fig.set_size_inches(10,4)
    plt.plot(corr)
    plt.title("Correlation")


def compute_spectrogram(fs, signal, low_pass=False):
    
    
    #signal = signal[:,0] # mono audio
    
    #apply the lowpass filter if needed
    if low_pass:
        h_len = 50
        h = np.ones(h_len)
        h /= np.linalg.norm(h)

        
        # stft paramaters 
        fft_len = 512 
        block_size = fft_len - 50 + 1
        hop = block_size // 2
        window = pra.hann(block_size, flag='asymmetric', length='full') 

        stft = pra.realtime.STFT(block_size, hop=hop, analysis_window=window, channels=1)
        stft.set_filter(h, zb=h.shape[0] - 1)  
        processed_audio = np.zeros(signal.shape)
        n = 0
        while  signal.shape[0] - n > hop:
            stft.analysis(signal[n:n+hop,])
            stft.process()  # apply the filter
            processed_audio[n:n+hop,] = stft.synthesis()
            n += hop
            
        fig = plt.figure()
        fig.set_size_inches(20,8)
        plt.subplot(2,1,1)
        spec1, _, _, _ = plt.specgram(signal[:n-hop].astype(np.float32), NFFT=256, Fs=fs, vmin=-20, vmax=30) # dBFS scale
        plt.title('Original Signal', fontsize=22)
        plt.subplot(2,1,2)
        spec2, _, _, _ = plt.specgram(processed_audio[hop:n], NFFT=256, Fs=fs, vmin=-20, vmax=30) # dBFS scale
        plt.title('Lowpass Filtered Signal', fontsize=22)
        return spec1, spec2
    else:
        fig = plt.figure()
        fig.set_size_inches(20,5)
        spec, _, _, _ = plt.specgram(signal.astype(np.float32), NFFT=256, Fs=fs, vmin=-20, vmax=30) # dBFS scale
        plt.title('Spectrogram', fontsize=22)
        return spec
    