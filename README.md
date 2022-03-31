# SpeckleContrastEstimation
The x-ray photon correlation spectroscopy (XPCS) is an x-ray experimental technique 
to probe sample dynamics from the x-ray speckle contrast change.

When one designs experiments with XPCS, one prominent challenge is to estimate the signal level.
In XPCS experiment, there are two kinds of signal level to worry about:
    
1. One needs to determine the photon position from the detector signal
2. One needs to determine the speckle contrast and its variation by calculation the photon correlation.

Because our detector has intrinsic noises, therefore, we need the scattering intensity to 
be strong to dominate the signal. Therefore, we tend to put the detector closer to the sample
to get a stronger signal per pixel.

However, on the other hand, the detector pixel size cannot be reduced infinitely.
Currently, the best we have is roughly 50 um. 
When the detector pixel size is larger than the speckle size,
the speckle contrast drops quickly and we would need more measurement time to accumulate 
the required statistics. 
From this point of view, we tend to move the detector away from the sample 
to increase the available pixel numbers at a specific Q and to resolve the speckles better. 

This repo aims to facilitate the decision process by providing an estimation on both the scattering 
intensity and the speckle contrast for a given sample at any given Q.      
 


## Notice
I have adopted code from Alex, Yanwen, and my previous works.
I do not claim the copyright of this piece of code.

On the other hand, I have not tested programs thoroughly. 
Therefore, I can not guarantee that these programs work.

# Scattering Intensity

The scattering intensity can be written in the following way
