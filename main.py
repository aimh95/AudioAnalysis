# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import librosa, librosa.display
import os
import numpy as np
import matplotlib.pyplot as plt
import math


import soundfile as sf

def sigFFT(sig, sr=22050):
    sig_fft = np.fft.fft(sig)
    magnitude = np.abs(sig_fft)
    freq_data = np.linspace(0, sr, len(magnitude))
    left_spectrum = magnitude[:int(len(magnitude) / 2)]
    left_freq = freq_data[:int(len(magnitude) / 2)]
    return left_spectrum, left_freq

def getVoice(sig, sr=22050):
    start_freq = 1800
    end_freq = 18000
    fft_sig = np.fft.fft(sig)
    sig_left = fft_sig[start_freq:end_freq]
    sig_right = fft_sig[len(fft_sig)-end_freq:len(fft_sig)-start_freq]
    voice_sig = np.zeros(len(fft_sig))

    voice_sig[start_freq:end_freq] = sig_left
    voice_sig[len(fft_sig)-end_freq:len(fft_sig)-start_freq] = sig_right

    # fft_sig = fft_sig[1800:18000]
    voice_ifft = np.fft.ifft(voice_sig)

    return voice_ifft

def getRMS(sig):
    square = np.square(sig)
    mean = square.mean()
    root = math.pow(mean, 0.5)

    return root

def MagPlot(sig,sr=22050, x_label="Time", y_label="Magnitude"):
    plt.figure()
    librosa.display.waveshow(sig, sr, alpha=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(audio_list[i])
    plt.show()

def FFTPlot(spec, freq, x_label="Freq", y_label="Magnitude"):
    plt.figure()
    plt.plot(freq, spec)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(audio_list[i])
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = "whateveryouwant"
    audio_list = os.listdir(path)

    for i in range(len(audio_list)):
        audio_name = path + audio_list[i]

        audio_sig, sr = librosa.load(audio_name, sr=20000)
        left_spectrum, left_freq = sigFFT(audio_sig, 20000)

        voice_sig = getVoice(audio_sig)

        # MagPlot(audio_sig,20000)
        # FFTPlot(left_spectrum, left_freq)

        plt.figure()
        plt.plot(audio_sig, "b")
        plt.plot(voice_sig, "r")
        plt.xlabel("Time")
        plt.ylabel("Magnitude")
        plt.title(audio_list[i])
        plt.show()

        rms = getRMS(audio_sig)
        rms_voice = getRMS(voice_sig)
        dBFS = 20*math.log10(rms)
        dBFS_voice = 20*math.log10(rms_voice)

        print(audio_list[i],"\t",rms,"\t",rms_voice,"\t", dBFS, "\t",dBFS_voice)
