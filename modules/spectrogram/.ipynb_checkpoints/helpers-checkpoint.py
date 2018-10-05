from scipy.signal import correlate
import matplotlib.pyplot as plt




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
