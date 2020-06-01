import sys

import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile

from fastica import FastIca, plot_signals

# ------------------------------------------------------------------------
PLOT = False  # set for generate plots
PLAY = False  # set for play signals after separation
# ------------------------------------------------------------------------


def read_data():
    fs1, first_signal = wavfile.read('data\\first.wav')
    fs2, second_signal = wavfile.read('data\\second.wav')

    samples = min(len(first_signal), len(second_signal))
    assert fs1 == fs2, "Cannot separate signals with different sample rate!"

    return fs1, first_signal[:samples], second_signal[:samples]


def main():
    try:
        fs, first_signal, second_signal = read_data()
        if PLOT:
            plot_signals(first_signal, second_signal, "Source signals")

        fast_ica = FastIca(first_signal, second_signal, PLOT)
        new_first_signal, new_second_signal = fast_ica.run_algorithm()
        if PLOT:
            plot_signals(new_first_signal, new_second_signal, "Signals after separation")

    except Exception as e:
        print("Exception! " + str(e))
        sys.exit(-1)

    if PLOT:
        plt.show()

    if PLAY:
        # play signals before mixing
        sd.play(first_signal, fs, blocking=True)
        sd.play(second_signal, fs, blocking=True)

        # play signals after separation
        sd.play(new_first_signal, fs, blocking=True)
        sd.play(new_second_signal, fs, blocking=True)


if __name__ == '__main__':
    main()
