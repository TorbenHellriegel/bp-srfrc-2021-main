.. Speech Recognition for Robot Control documentation master file, created by
   sphinx-quickstart on Wed Jan  5 17:28:55 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Speech Recognition for Robot Control's documentation!
================================================================

This is the documentation for the bachelor project Speech Recognition for Robot Control,
created in the Winter Semester of 2021/22.

Many modern electronic devices are now voice controlled and not all of them need to understand
every word they hear. Cleaning robots for example only need a very small vocabulary and should be
be able to work autonomously without access to a complex online voice recognition program with a
gigantic library of words. Our program aims to accomplish this by recognizing only a few important
keywords and mostly ignoring all other sound inputs like silence or noise.

For our project we have used relatively simple speech recognition methods with few calculations
in order to be more energy efficient. The main calculation steps done by our program are as follows:

First we calculate the STFT (Short-Time Fourier Transform) of the sound input and then estimate
the noise PSD (power spectral density) and speech PSD. With those results, we use VAD (Voice Activity Detection)
to determine if there is speech present that we need to understand. Then we filter the noise out
of the original sound input with a wiener filter. Finally, we use MFCC (Mel-Frequency Cepstral coefficients)
and a DTW Classifier (Dynamic Time Warping) to determine which word has been said.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   gui
   stft
   noisepsd
   vad
   wienerfilter
   mfcc
   dtw

* :ref:`genindex`
