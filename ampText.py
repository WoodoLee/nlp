import numpy as np
import librosa, librosa.display 
import speech_recognition as sReco
from BaselineRemoval import BaselineRemoval

import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler

from scipy.signal import find_peaks, peak_prominences

# font format
plt.rcParams["figure.figsize"] = (1920 / 100 , 902 / 100)
plt.rcParams['font.family'] = 'Times New Roman'
rc('text', usetex=True)

# input
# harvard sentece : https://www.voiptroubleshooter.com/open_speech/american.html
# CMU http://festvox.org/cmu_arctic/dbs_rms.html
# http://homepage.ntu.edu.tw/~karchung/miniconversations/MC.htm
inputWav = "./data/iloveu2.wav"

#sampling rate
sR = 1000

#amplitude
a, sr = librosa.load(inputWav, sr=sR)

#x for plotting
aL = len(a)/sR
tA = np.arange(0, aL , step = aL/len(a))

# min max scaler
scaler = MinMaxScaler()
aS = scaler.fit_transform(a.reshape(-1,1))
aS = aS.flatten()

#baseline Removal
polynomial_degree=2 #only needed for Modpoly and IModPoly algorithm
baseObj=BaselineRemoval(aS)
aSBR=baseObj.ModPoly(polynomial_degree)
# Imodpoly_output=baseObj.IModPoly(polynomial_degree)
# Zhangfit_output=baseObj.ZhangFit()


aSBRA = np.abs(aSBR)

#input for STFT, plotting
# a : raw
# aS : Scaled
# aSBR : Scaled, Baseline Removal
# aSBRA : Scaled, Baseline Removal, Absolute
aIn = aSBRA
peaks, properties = find_peaks(aIn, height=None, distance = 200)

aInAvg = np.average(aIn)
aInAvg = 0.1
peaksOver = peaks[ np.where(aIn[peaks]>aInAvg)]

peaksIn = peaksOver
print(len(peaks))
print(len(peaksIn))

print(tA[peaksIn])
# text data
sText = sReco.Recognizer()
with sReco.AudioFile(inputWav) as source:
  # listen for the data (load audio to memory)
  audio_data = sText.record(source)
  # recognize (convert from speech to text)
  outText = sText.recognize_google(audio_data)

print(len(outText))
print(outText)


# STFT -> spectrogram
# total sampling 
hop_length = int(512 / 2) 
# Sampling number per frame
n_fft = 2048

# calculate duration hop length and window in seconds
hop_length_duration = float(hop_length)/sr
n_fft_duration = float(n_fft)/sr

# STFT
stft = librosa.stft(aIn, n_fft=n_fft, hop_length=hop_length)
# amp
magnitude = np.abs(stft)
# magnitude > Decibels 
log_spectrogram = librosa.amplitude_to_db(magnitude)


## Draw
fSize = 35
sSize = 1
sColor = 'blue'

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
# plt.scatter(tA,aScaled, s = sSize, c = sColor)
plt.plot(tA,aIn, c = sColor)

plt.scatter(tA[peaks],aIn[peaks], c='black')
plt.scatter(tA[peaksIn],aIn[peaksIn], c='red', marker='x', s = 100)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("Time (s)", fontsize=fSize, loc = 'right')
plt.ylabel("Amplitude", fontsize=fSize, loc = 'top')
plt.grid(b=True, which='both', axis='both')
plt.tight_layout()

ax2 = fig.add_subplot(2, 1, 2)
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("Time", fontsize=fSize, loc = 'right')
plt.ylabel("Frequency", fontsize=fSize, loc = 'top')
cbar = plt.colorbar(format="%+2.0f dB")
cbar.set_label(label = "%+2.0f dB",  fontsize=fSize)
# cbar..tick_parms(labelsize=20)
plt.title("Spectrogram (dB)", fontsize=fSize)
plt.tight_layout()


plt.show()
