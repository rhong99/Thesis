# sound analysis on waveform (dB percentile)


from pydub import AudioSegment
import numpy as np
import soundfile
import matplotlib.pyplot as plt


def convert(filepath):
    audio = AudioSegment.from_mp3(filepath)
    # signal, sr = soundfile.read(filepath)
    signal = audio.get_array_of_samples()
    samples_sf = 0

    signal_list = signal.tolist()
    maximum = max(signal)
    minimum = abs(min(signal))
    if maximum < minimum:
        maximum = minimum

    # try dual channel, if not mono
    try:
        samples_sf = signal[:, 0]
    except:
        samples_sf = signal

    # normalize to be bounded by 1
    samples_norm = [i / maximum for i in samples_sf]

    # convert to decibels
    data = [convert_to_decibel(i) for i in samples_norm]

    # get data
    percentile = np.percentile(data, [25, 50, 75, 90, 95, 99])
    print(f"1st Quartile : {percentile[0]}")
    print(f"2nd Quartile : {percentile[1]}")
    print(f"3rd Quartile : {percentile[2]}")
    print(f"Mean : {np.mean(data)}")
    print(f"Median : {np.median(data)}")
    print(f"Standard Deviation : {np.std(data)}")
    print(f"Variance : {np.var(data)}")

    data_prune75 = []
    for i in data:
        if i > percentile[2]:
            data_prune75.append(i)
        else:
            data_prune75.append(-60)

    data_prune90 = []
    for i in data:
        if i > percentile[3]:
            data_prune90.append(i)
        else:
            data_prune90.append(-60)

    data_prune95 = []
    for i in data:
        if i > percentile[4]:
            data_prune95.append(i)
        else:
            data_prune95.append(-60)

    data_prune99 = []
    for i in data:
        if i > percentile[5]:
            data_prune99.append(i)
        else:
            data_prune99.append(-60)

    plt.subplot(3, 1, 1)
    plt.plot(samples_norm)
    plt.xlabel('Samples')
    plt.ylabel('Data: Soundfile')
    plt.subplot(3, 1, 2)
    plt.plot(data)
    plt.xlabel('Samples')
    plt.ylabel('dB Full Scale (dB)')
    plt.subplot(3, 1, 3)
    plt.plot(data_prune75)
    plt.xlabel('Samples 75th Percentile')
    plt.ylabel('dB Full Scale (dB)')
    plt.tight_layout()
    plt.show()

    plt.subplot(3, 1, 1)
    plt.plot(data_prune90)
    plt.xlabel('Samples 90th Percentile')
    plt.ylabel('dB Full Scale (dB)')
    plt.subplot(3, 1, 2)
    plt.plot(data_prune95)
    plt.xlabel('Samples 95th Percentile')
    plt.ylabel('dB Full Scale (dB)')
    plt.subplot(3, 1, 3)
    plt.plot(data_prune99)
    plt.xlabel('Samples 99th Percentile')
    plt.ylabel('dB Full Scale (dB)')
    plt.tight_layout()
    plt.show()


def convert_to_decibel(x):
    ref = 1
    if x != 0:
        return 20 * np.log10(abs(x) / ref)
    else:
        return -60


convert('sample.mp3')
