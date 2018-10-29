## Sound Source Localization Algorithms


### 1. TDOA-Based Locators
Two step procedure : time delay estimation (TDE) of the speech signals relative to pairs of spatially separated microphones. This data + microphone's positions are then used to generate hyperbolic curves which are then intersected in some optimal sense to arrive at a source location estimate. -> TDE is the key to the effectiveness of localizers

#### Limitations
Inability to accommodate multi-source scenarios. Excessive ambient noise or moderate to high reverberation levels in the acoustic field typically results in poor TDOA figures.

#### Examples
*GCC-PHAT* 



### 2. Steered-Beamformer-Based Locators
Scan the space of interest with an acoustic beam and find a region which produces the highest beam output energy. This method is very simple to implement and was used exclusively in early analog array systems. 

#### Limitations
It suffers from extremely poor spatial resolution, and poor response time. It becomes fairly impractical when a large number of discrete locations are to be scanned, or when continuous spatial resolution is desired

#### Examples
*TODO*



### 3. High-Resolution Spectral-Estimation-Based Locators
Based on a spectral phase correlation matrix derived for all elements in the array. Such a matrix is estimated from the captured data. The spectral estimation-based techniques attempt to perform a “best” statistical fit for source location using the above matrix.

#### Limitations
Although numerous techniques exist, most are limited in application to narrowband signals. These techniques also tend to rely on a high degree of signal statistical stationarity and an absence of reflections and interfering sources. Most applications of the spectral estimation locators occur in the radar domain, and are not general and robust enough to be applicable to wideband acoustic signals. The techniques also tend to be computationally intensive, and are thus not well suited for real-time applications.

#### Examples
*TODO*