**********************
WienerFilter
**********************


The wiener filter module filters out noise out of an input signal. This module requires a precalculated
noise PSD (power spectral density) in order to work as intended. The functions are designed to calculate
one time frame at a time. This way this module can be used in real time speech processing applications
as well as in batch processing by using a loop.



Class and Functions
===================

.. autoclass:: src.wienerf.wienerfilter.WienerFilter

When you first initialize the wienerfilter the default values are usually sufficient. You may need to give
the number of frequencies of your input signal, If the number is different from the default value of 1024.
The alpha is used during calculations to determine how high the calculations from the last time step
are valued compared to the current time step. Giving different alpha values (between 0 and 1) may give better
end results depending on the input.


.. automethod:: src.wienerf.wienerfilter.WienerFilter.speech_update

This function calculates the speech PSD using the formula:

.. math:: \widehat{\sigma ^2_{s}}(l) = \alpha \lvert \widehat{S}(l-1)\rvert ^2 + (1-\alpha) max(0,\lvert Y(l)\rvert ^2 - \sigma ^2_{s}(l))

The first part of the addition S(l-1) is the calculated clean speech from the last time frame.
If there is no last time frame, it is an array of zeros. The second part is the maximum of 0
and the current speech PSD estimate. Both values are scaled by alpha and 1-alpha respectively
and then added to give a better final speech PSD estimate.



.. automethod:: src.wienerf.wienerfilter.WienerFilter.compute

This function calculates the clean speech using the formula:

.. math:: \widehat{S_{k}}(l) = \frac{\sigma ^2_{S,k}(l)}{\sigma ^2_{S,k}(l) + \sigma ^2_{N,k}(l)} Y_{k}(l)

The division involving the speech and noise PSD is the filter in this equation which is applied
to the input signal Y. the higher the noise PSD for a given frequency is, the lower the value
of the filter becomes and the more of the input is filtered out. This way frequencies containing
mostly noise get filtered out while the frequencies with the speech can pass the filter.
The end result is a denoised version of the original input signal containing the clean speech.



How to use
===========

Step 1: Load all required inputs
---------------------------------

Load the noisy input signal that you want the wiener filter to denoise and the noise PSD,
which needs to be calculated by a different module beforehand


Step 2: Initialize the wiener filter
-------------------------------------

This does sometimes require giving parameters if your input does not match the default values. The default number
of frequencies is set to 1024. If your input signal has a different number, give it instead. The alpha determines
how high the previous time frame is valued when calculating the clean speech for the current one. The default value
for alpha is set to 0.97 which usually returns good results. The higher the alpha (between 1 and 0) the more weight
is placed on the last time frame. If you want to test if different alpha values give better results, you can change
this parameter as well.


Step 3: Calculate the clean speech
-----------------------------------

To calculate the clean speech, you need to iterate over each time frame in the input signal. This is because
the speech PSD, which gets calculated alongside the output of the wiener filter, requires the last time frame
in order to better predict a better result. During the iteration, first call wienerfilter.speech_update()
with the current noisy speech frame and the noise PSD frame. Then call wienerfilter.compute() also with
the current noisy speech frame and the noise PSD frame, as well as the calculated speech PSD. The wiener filter
saves the result of this calculation internally to use for the next time frame.



Use example
============

.. code-block:: python

    # Load all required inputs
    noisy_speech = [] # Put the noisy speech signal containing the complex values of the STFT here.
    noise_psd = [] # Put the power spectral density estimate here.

    # Initialize the wiener filter
    wf = WienerFilter(number_of_frequencies=noisy_speech.shape[0], alpha=0.4)

    # Initialize an empty result array
    clean_speech = np.empty_like(noisy_speech)

    # Calculate the clean speech
    for i in range(noisy_speech.shape[1]):
        speech_psd = wf.speech_update(noisy_speech[:,i], noise_psd[:,i])
        clean_speech[:,i] = wf.compute(noisy_speech[:,i], noise_psd[:,i], speech_psd)

    return clean_speech