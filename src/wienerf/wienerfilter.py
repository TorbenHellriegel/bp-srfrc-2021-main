import numpy as np

EPS = 1e-20

class WienerFilter:

    def __init__(self, number_of_frequencies=1024, alpha=0.97) -> None:
        """
        Constructs all needed Attributes of the WienerFilter Object

        Parameters
        ----------
        number_of_frequencies: int
            The number of frequencies of the signal that is supposed to be procesed by the wiener filter
            Default = 1024

        alpha: float [0-1)
            Determines how high the previous time frame is valued when calculating the next
            Default = 0.97

        Returns
        -------
        None
        """
        if(0 <= alpha < 1):
            self.alpha = alpha
        else:
            raise ValueError('The alpha Parameter should be between 0 <= alpha < 1.')
        self.last_clean_speech = np.zeros(number_of_frequencies, dtype=complex)
        pass

    def speech_update(self, y_frame, sigma_noise):
        """
        A function to calculate the power spectral density (PSD) of a single time-frame of a speech signal.

        Parameters
        ----------
        y_frame: ndarray
            The speech signal frame containing the complex values of the STFT.
            
        sigma_noise: ndarray
            The noise PSD.

        Returns
        -------
        result : ndarray
            The speech PSD.
        """
        last_estimate = compute_spectogram(self.last_clean_speech)
        current_estimate = np.maximum(np.zeros_like(y_frame), compute_spectogram(y_frame) - sigma_noise)
        return  self.alpha * last_estimate + (1 - self.alpha) * current_estimate

    def compute(self, y_frame, sigma_noise, sigma_speech):
        """
        A function to calculate the clean speech from the noisy speech and the noise and speech power spectral density (PSD).

        Parameters
        ----------
        y_frame: ndarray
            The speech signal frame containing the complex values of the STFT.
            
        sigma_noise: ndarray
            The noise PSD.
            
        sigma_speech: ndarray
            The speech PSD.

        Returns
        -------
        result : ndarray
            An array of the clean speech.
        """
        G_l = sigma_speech / (sigma_speech + sigma_noise + EPS)
        self.last_clean_speech = G_l * y_frame
        return self.last_clean_speech


def compute_spectogram(y_frame):
    """
    A function to convert a complex speech signal to its spectogram.

    Parameters
    ----------
    y_frame: ndarray
        The speech signal frame containing the complex values of the STFT.

    Returns
    -------
    result : ndarray
        An array with the spectogram.
    """
    return np.abs(y_frame) ** 2