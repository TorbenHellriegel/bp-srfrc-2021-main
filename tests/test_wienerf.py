from cmath import nan
import numpy as np
import time
import noisereduce as nr
import matplotlib.pyplot as plt
import random
import asyncio
import soundfile
from tqdm import tqdm
from librosa import load, stft, istft

import os.path
import sys

def joinpath(first, second):
    """
    Joins the two paths and normalizes them.

    Parameters
    ----------
    first: str
        The path that will be appended to.
    second: str
        The path that will be appended.
    Returns
    -------
    result : str
        The joined path.
    """
    return os.path.normpath(os.path.join(first, second))

bp_dir = os.path.dirname(os.path.dirname(__file__))
wienerf_path = joinpath(bp_dir,"src")

sys.path.append(wienerf_path)
from wienerf import WienerFilter
from noise import NoisePSD


def test_basic():
    wf = WienerFilter(number_of_frequencies=3, alpha=0.5)
    assert np.all(np.array([0,0,0]) == wf.speech_update(np.array([0,0,0]), np.array([0,0,0])))
    assert np.all(np.array([8,8,8]) == wf.speech_update(np.array([0+5j,0+5j,0-5j]), np.array([9,9,9])))
    assert np.all(np.array([2,2,2]) == wf.compute(np.array([2,2,2]), np.array([0,0,0]), np.array([1,1,1])))
    assert np.all(np.array([10,10,10]) == wf.speech_update(np.array([0-5j,0-5j,0-5j]), np.array([9,9,9])))


def test_perf():
    # speech_signal, speech_sr = load(joinpath(bp_dir, f'resources/Clean Speech Example.wav').replace('\\','/'), sr=None)
    speech_signal, speech_sr = load(joinpath(bp_dir, f'resources/Default/backward/0a2b400e_nohash_0.wav').replace('\\','/'), sr=None)
    speech = stft(speech_signal)

    noise_signal, noise_sr = load(joinpath(bp_dir, f'resources/Default/_background_noise_/running_tap.wav').replace('\\','/'), sr=None)
    noise = stft(noise_signal)

    # Constructing the noisy speech by adding the noise and a slightly offset speech
    speech_length = speech.shape[1]
    offset_speech = np.zeros_like(noise)
    offset_speech[:, 50:50 + speech_length] = speech
    noisy_speech = noise + offset_speech

    noisy_speech = noisy_speech[:, 0:100] # The time limit is supposed to hold true for both short aswell as long signals.
                                          # Comment line above, if using a longer signal.

    norm = np.max(np.abs(noisy_speech))
    noisy_speech = noisy_speech / norm
    speech = speech / norm
    noise = noise / norm

    # Initializing the wiener filter an calculating an example noise PSD
    wf = WienerFilter(number_of_frequencies=noisy_speech.shape[0])
    noise_psd = np.abs(noise) ** 2

    speech_psd = np.empty_like(noisy_speech)
    speech_estimate = np.empty_like(noisy_speech)

    # Measuring the time it takes to filter the input
    start_time = time.time()
    for i in range(noisy_speech.shape[1]):
        speech_psd[:,i] = wf.speech_update(noisy_speech[:,i], noise_psd[:,i])
        speech_estimate[:,i] = wf.compute(noisy_speech[:,i], noise_psd[:,i], speech_psd[:,i])
    end_time = time.time()

    time_elapsed = (end_time - start_time)

    reversed = istft(speech_estimate)

    time_limit = (reversed.shape[0] / noise_sr) * 0.01

    # Should run 100x faster than a real time input.
    assert  time_elapsed <= time_limit


def test_accuracy():
    snr_values = []
    si_sdr_values = []
    si_sdr_values_baseline = []
    si_sdr_values_no_reduction = []

    speech_signal, speech_sr = load(joinpath(bp_dir, f'resources/Clean Speech Example.wav').replace('\\','/'), sr=None)
    noise_signal, noise_sr = load(joinpath(bp_dir, f'resources/Default/_background_noise_/running_tap.wav').replace('\\','/'), sr=None)

    # Testing with different signal to noise ratios
    for i in range(1, 31):
        speech = stft(speech_signal)
        noise = (i / 20) * noise_signal
        noise = stft(noise)


        # Constructing the noisy speech by adding the noise and a slightly offset speech
        speech_length = speech.shape[1]
        offset_speech = np.zeros_like(noise)
        offset_speech[:, 100:100 + speech_length] = speech
        noisy_speech = noise + offset_speech

        norm = np.max(np.abs(noisy_speech))
        noisy_speech = noisy_speech / norm
        speech = speech / norm
        noise = noise / norm

        # Initializing the wiener filter an calculating an example noise PSD
        wf = WienerFilter(number_of_frequencies=noisy_speech.shape[0])
        noise_psd = np.abs(noise) ** 2

        speech_psd = np.zeros_like(noisy_speech)
        speech_estimate = np.zeros_like(noisy_speech)

        # calculating our speech estimate
        for i in range(noisy_speech.shape[1]):
            speech_psd[:,i] = wf.speech_update(noisy_speech[:,i], noise_psd[:,i])
            speech_estimate[:,i] = wf.compute(noisy_speech[:,i], noise_psd[:,i], speech_psd[:,i])

        # calculating the istft to be able to compare the results
        speech_time = istft(speech)
        noise_time = istft(noise[:, 100:100 + speech_length])
        speech_estimate_time = istft(speech_estimate[:, 100:100 + speech_length])

        # calculating the snr and si-sdr values to evaluate our results
        snr_result = snr(speech_time, noise_time)
        si_sdr_result = si_sdr(speech_time, speech_estimate_time)
        si_sdr_result_baseline = si_sdr(speech_time, nr.reduce_noise(y=speech_time+noise_time, sr=speech_sr))
        si_sdr_result_no_reduction = si_sdr(speech_time, speech_time+noise_time)

        assert si_sdr_result > si_sdr_result_baseline
        assert si_sdr_result > si_sdr_result_no_reduction

        if __name__ == "__main__":
            snr_values.append(snr_result)
            si_sdr_values.append(si_sdr_result)
            si_sdr_values_baseline.append(si_sdr_result_baseline)
            si_sdr_values_no_reduction.append(si_sdr_result_no_reduction)


    if __name__ == "__main__":
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.scatter(snr_values, si_sdr_values, label=r"wiener filter")
        ax1.scatter(snr_values, si_sdr_values_baseline, color="lime", label=r"baseline noise reduction")
        ax1.scatter(snr_values, si_sdr_values_no_reduction, color="red", label=r"no noise reduction")
        ax1.legend(loc=2)
        ax1.set_xlabel("snr")
        ax1.set_ylabel("si_sdr")
        plt.show()


def snr(signal, noise):
    """
    Calculate the Signal to Noise Ratio for a desired signal with a given noise in the time domain.

    Parameters
    ----------
    signal: ndarray
        A 1D-Array of the signal in time domain.

    noise: ndarray
        A 1D-Array of the noise in time domain.

    Returns
    -------
    result : number
        The signal to noise ratio.
    """
    ratio = np.sum(signal**2) / np.sum(noise**2)
    return 10 * np.log10(ratio)


def si_sdr(signal, speech_estimate):
    """
    Calculate the scale-aware signal to distortion Ratio for a target signal and an estimate of that signal.

    Parameters
    ----------
    signal: ndarray
        A 1D-Array of the target signal in time domain.

    speech_estimate: ndarray
        A 1D-Array of the estimate in time domain.

    Returns
    -------
    result : number
        The signal to distortion ratio.
    """
    e_target = (np.dot(speech_estimate, signal) / np.sum(signal**2)) * signal
    e_res = e_target - speech_estimate
    return snr(e_target, e_res)


def get_list(txt_path):
    try:
        with open(txt_path, "r") as f:
            data = f.read().splitlines()
            path_list = [a for a in data if(a.endswith('.wav'))]
        return path_list
    except IOError:
        print(f"The file ${txt_path} doesn't exist or could not be opened.")
        print("You need to specifie a .txt file which contains paths and words of .wav files.")
        print("The .txt file should look like this:\n")
        print("D:/Documents/audios/01.wav,Dog\nD:/Documents/audios/02.wav,Cat\nD:/Documents/audios/03.wav,Hat\n")
        exit()


def display():
    speech_signal, speech_sr = load(joinpath(bp_dir, f'resources/Clean Speech Example.wav').replace('\\','/'), sr=None) # TODO load in the examples into the target directory
    noise_signal, noise_sr = load(joinpath(bp_dir, f'resources/Default/_background_noise_/running_tap.wav').replace('\\','/'), sr=None) # TODO load in the examples into the target directory

    speech = stft(speech_signal)
    noise_stft = stft(noise_signal)
    # Constructing the noisy speech by adding the noise and a slightly offset speech
    speech_length = speech.shape[1]
    offset_speech = np.zeros_like(noise_stft)
    offset_speech[:, 100:100 + speech_length] = speech
    noise_stft /= 1
    noisy_speech = noise_stft + offset_speech
    norm = np.max(np.abs(noisy_speech))
    noisy_speech = noisy_speech / norm
    # Initializing the wiener filter an calculating an example noise PSD
    wf = WienerFilter(number_of_frequencies=noisy_speech.shape[0], alpha=0.5)
    wf2 = WienerFilter(number_of_frequencies=noisy_speech.shape[0], alpha=0.9)
    noise_object = NoisePSD(noisy_speech.shape[0])
    noise_psd = np.zeros_like(noisy_speech)
    speech_psd = np.zeros_like(noisy_speech)
    speech_estimate = np.zeros_like(noisy_speech)
    speech_psd09 = np.zeros_like(noisy_speech)
    speech_estimate09 = np.zeros_like(noisy_speech)
    # calculating our speech estimate
    for i in range(noisy_speech.shape[1]):
        noise_psd[:,i] = noise_object.update(noisy_speech[:,i])
        speech_psd[:,i] = wf.speech_update(noisy_speech[:,i], noise_psd[:,i])
        speech_estimate[:,i] = wf.compute(noisy_speech[:,i], noise_psd[:,i], speech_psd[:,i])
        speech_psd09[:,i] = wf2.speech_update(noisy_speech[:,i], noise_psd[:,i])
        speech_estimate09[:,i] = wf2.compute(noisy_speech[:,i], noise_psd[:,i], speech_psd09[:,i])

    speech_time = istft(speech)
    noise_time = istft(noise_stft[:, 100:100 + speech_length])
    print(snr(speech_time, noise_time))
    import librosa
    import librosa.display

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(13,13))
    # D = offset_speech[:, 100:100 + speech.shape[1]]
    # S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    # img = librosa.display.specshow(S_db, y_axis='log', x_axis='time', ax=ax[0])

    D = noisy_speech[:, 100:100 + speech.shape[1]]
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(S_db, y_axis='log', x_axis='time', ax=ax[0])
    ax[0].set(title='Noisy speech')

    D = speech_estimate[:, 100:100 + speech.shape[1]]
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(S_db, y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set(title='Enhanced Speech a=0.5')

    D = speech_estimate09[:, 100:100 + speech.shape[1]]
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, y_axis='log', x_axis='time', ax=ax[2])
    ax[2].set(title='Enhanced Speech a=0.9')
    plt.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()

async def load_signal(path):
    signal, sr = load(path, sr=None)
    return stft(signal), sr

async def start_loading_signals(paths, queu):
    for path in paths:
        await queu.put(asyncio.create_task(load_signal(path)))

async def evaluate():
    alpha_list = [0.97]#[0.5, 0.75, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]
    num_samples = 100        # Max 11005 (Size of the Testing List for the Speech Commands Dataset)
    num_noises = 6          # Max 6 (because there are only 6 noises in _background_noise_/)
    df = np.full((len(alpha_list)+3, num_samples*num_noises), nan)


    speech_paths = random.sample(get_list(joinpath(bp_dir, 'resources/Default/testing_list.txt')),k=num_samples) # A random selection of k voice samples
    for i in range(len(speech_paths)):
        speech_paths[i] = joinpath(bp_dir, f'resources/Default/{speech_paths[i]}').replace('\\','/')
    q_speech = asyncio.Queue(maxsize=16)
    main_task_s = asyncio.create_task(start_loading_signals(speech_paths, q_speech))

    noise_paths = random.sample([f for f in os.listdir(joinpath(bp_dir, 'resources/Default/_background_noise_')) if f.endswith('.wav')],k=num_noises)
    noise_signals = []
    for i in range(len(noise_paths)):
        noise_paths[i] = joinpath(bp_dir, f'resources/Default/_background_noise_/{noise_paths[i]}').replace('\\','/')
        noise_signals.append(await load_signal(noise_paths[i]))
        noise_signals[i] = (noise_signals[i][0][:,:500], noise_signals[i][1])


    for i in tqdm(range(len(speech_paths))):
        task_s = await q_speech.get()
        original_speech, speech_sr = await task_s
        speech_length = original_speech.shape[1]

        for j in range(len(noise_signals)):
            original_noise, noise_sr = noise_signals[j]

            # Constructing the noisy speech by adding the noise and a slightly offset speech
            
            noisy_speech = np.zeros((original_noise.shape[0], max(original_noise.shape[1], speech_length+30)),dtype=complex)
            noisy_speech[:, 30:30 + speech_length] = original_speech
            noisy_speech[:, : original_noise.shape[1]] += original_noise

            norm = np.max(np.abs(noisy_speech)) # TODO Hmmm Dies sieht komisch aus, ergebnisse werden ganz krumm (echt durch 600 Teilen???)
            noisy_speech = noisy_speech / norm
            speech = original_speech / norm
            noise = original_noise / norm

            # Initializing the NoisePSD module
            noise_object = NoisePSD(noisy_speech.shape[0])
            noise_psd = np.zeros_like(noisy_speech)

            for k in range(len(alpha_list)):
                # Initializing the wiener filter
                wf = WienerFilter(number_of_frequencies=noisy_speech.shape[0], alpha=alpha_list[k])
                # noise_psd = np.abs(noise) ** 2 # This would be simulating a perfect noise estiamtion

                speech_psd = np.zeros_like(noisy_speech)
                speech_estimate = np.zeros_like(noisy_speech)

                # calculating our speech estimate
                for l in range(noisy_speech.shape[1]):
                    noise_psd[:,l] = noise_object.update(noisy_speech[:,l])
                    speech_psd[:,l] = wf.speech_update(noisy_speech[:,l], noise_psd[:,l])
                    speech_estimate[:,l] = wf.compute(noisy_speech[:,l], noise_psd[:,l], speech_psd[:,l])

                # calculating the istft to be able to compare the results
                speech_time = istft(speech)
                speech_estimate_time = istft(speech_estimate[:, 30:30 + speech_length])
                # calculating the si-sdr values to evaluate our results
                si_sdr_result = si_sdr(speech_time, speech_estimate_time)
                df[3+k, i*num_noises + j] = si_sdr_result
                # with soundfile.SoundFile(f'../estimated_speech{alpha_list[k]}.wav', 'w', samplerate=speech_sr, channels=1) as s_file:
                #     s_file.write(data=speech_estimate_time)

            # calculating the istft to be able to compare the results
            noise_time = istft(noise[:, 30:30 + speech_length])
            noisy_time = istft(noisy_speech[:, 30:30 + speech_length])
            # with soundfile.SoundFile('../speech.wav', 'w', samplerate=speech_sr, channels=1) as s_file:
            #     s_file.write(data=speech_time)
            # with soundfile.SoundFile('../noisy.wav', 'w', samplerate=speech_sr, channels=1) as s_file:
            #     s_file.write(data=noisy_time)

            # calculating the snr and si-sdr values to evaluate our results
            snr_result = snr(speech_time, noise_time)
            si_sdr_result_no_reduction = si_sdr(speech_time, noisy_time)
            si_sdr_result_baseline = si_sdr(speech_time, nr.reduce_noise(y=noisy_time, sr=speech_sr, n_fft=512))
            df[0, i*num_noises + j] = snr_result
            df[1, i*num_noises + j] = si_sdr_result_no_reduction
            df[2, i*num_noises + j] = si_sdr_result_baseline

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.scatter(df[0], df[1], marker='1', color="red", label=r"no noise reduction", alpha=300/(num_samples*num_noises+300))
    ax1.scatter(df[0], df[2], edgecolors='none', marker='o', color="lime", label=r"baseline noise reduction", alpha=300/(num_samples*num_noises+300))
    ax1.scatter(df[0], df[3], edgecolors='none', marker='s', label=r"wiener filter", alpha=300/(num_samples*num_noises+300))
    ax1.legend(loc=2)
    for lh in ax1.legend().legendHandles:
        lh.set_alpha(1)
    ax1.set_xlabel("snr")
    ax1.set_ylabel("si_sdr")
    ax1.set_yticks(np.arange(-40,41,10))
    plt.grid(linestyle="--")
    plt.savefig('../si_sdr.svg')
    plt.show()

    import pandas as pd
    ranges = ((-40, -30), (-30, -20), (-20, -10), (-10, 0), (0, 10), (10, 20), (20, 30), (30, 40), (40, 50))
    means = np.zeros((len(ranges),3+len(alpha_list)))
    for i in range(len(ranges)):
        means[i] = np.apply_along_axis(np.mean, 1, df[:,np.logical_and(ranges[i][0] <  df[0], df[0] <= ranges[i][1])])

    d_frame = pd.DataFrame(means.T[1:])
    d_frame.set_axis(list(map(lambda a: "$["+str(a[0])+", "+str(a[1])+"]$", ranges)), axis='columns', inplace=True)
    d_frame.set_axis(["Kein Filter","Baseline","Wiener Filter $\\alpha="+str(alpha_list[0])+"$"] \
                    + list(map(lambda i: "$="+str(alpha_list[i])+"$", range(1, len(alpha_list)))), axis='index', inplace=True)
    print(d_frame)

    table = d_frame.style.format(precision=2) \
        .to_latex(hrules=True,column_format="r" + ('c' * len(ranges)))
    with open("../table.tex", "w") as file:
        file.write(table)

if __name__ == "__main__":
    # test_accuracy()
    # test_perf()
    # display()
    asyncio.run(evaluate())
