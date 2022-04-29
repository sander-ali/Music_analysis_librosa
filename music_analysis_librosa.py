import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import scipy

x, sampling_rate = librosa.load('AtifAslam.mp3', sr=None)
#Default sampling rate for librosa is 22050
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x[:5*sampling_rate], sr=sampling_rate)
plt.show()

#plotting STFT-based spectrogram
S = librosa.stft(x)
fig, ax = plt.subplots(figsize=(15,9))
img = librosa.display.specshow(S, x_axis='time',
                         y_axis='linear', sr=sampling_rate,
                         fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='STFT linear scale spectrogram')
plt.savefig('stft-librosa-linear.png')

#Extracting Spectral Features
plt.plot(librosa.feature.spectral_centroid(x, sr=sampling_rate)[0])
plt.xlabel('Frame number')
plt.ylabel('frequency (Hz)')
plt.title('Spectral centroids')
plt.show()

spec_bw = librosa.feature.spectral_bandwidth(x, sr=sampling_rate)
plt.xlabel('Frame number')
plt.ylabel('frequency (Hz)')
plt.title('Spectral bandwidth')
plt.show()

#visualizing deviations from the centroid
times = librosa.times_like(spec_bw)
centroid = librosa.feature.spectral_centroid(S=np.abs(S))

fig, ax = plt.subplots(figsize=(15,9))
img = librosa.display.specshow(S, x_axis='time',
                         y_axis='log', sr=sampling_rate,
                         fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Spectral centroid plus/minus spectral bandwidth')
ax.fill_between(times, centroid[0] - spec_bw[0], centroid[0] + spec_bw[0],
                alpha=0.5, label='Centroid +- bandwidth')
ax.plot(times, centroid[0], label='Spectral centroid', color='w')
ax.legend(loc='lower right')
plt.savefig('centroid-vs-bw-librosa.png')

#spectral contrast
contrast = librosa.feature.spectral_contrast(S=np.abs(S), sr=sampling_rate)
fig, ax = plt.subplots(figsize=(15,9))
img2 = librosa.display.specshow(contrast, x_axis='time', ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(ylabel='Frequency bands', title='Spectral contrast')
plt.savefig('spectral-contrast-librosa.png')

#Spectrogram on a logarithmic scale
S = librosa.stft(x)
fig, ax = plt.subplots(figsize=(15,9))
img = librosa.display.specshow(S, x_axis='time',
                         y_axis='log', sr=sampling_rate,
                         fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='STFT log scale spectrogram')
plt.savefig('stft-librosa-log.png')

#linear scale
S_dB = librosa.amplitude_to_db(S, ref=np.max)
fig, ax = plt.subplots(figsize=(15,9))
img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='linear', sr=sampling_rate,
                         fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='STFT (amplitude to DB scaled) linear scale spectrogram')
plt.savefig('stft-librosa-linear-db.png')

#log scale
S_dB = librosa.amplitude_to_db(S, ref=np.max)
fig, ax = plt.subplots(figsize=(15,9))
img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='log', sr=sampling_rate,
                         fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='STFT (amplitude to DB scaled) log scale spectrogram')
plt.savefig('stft-librosa-log-db.png')

#Mel Spectrogram
fig, ax = plt.subplots(figsize=(15,9))
img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=sampling_rate,
                         fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel scaled STFT spectrogram')
plt.savefig('stft-librosa-mel.png')

#Generating MEL Filters
n_fft = 2048 # number of FFT components
mel_basis = librosa.filters.mel(sampling_rate, n_fft)
mel_spectrogram = librosa.core.power_to_db(mel_basis.dot(S**2))
fig, ax = plt.subplots(figsize=(15,9))
img = librosa.display.specshow(mel_spectrogram, x_axis='time',
                         y_axis='mel', sr=sampling_rate,
                         fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency (power to DB scaled) spectrogram')
plt.savefig('mel-spec-librosa-db.png')

#Calculate MFCC
mfcc = scipy.fftpack.dct(mel_spectrogram, axis=0)
mfcc = librosa.core.power_to_db(librosa.feature.mfcc(x, sr=sampling_rate))
fig, ax = plt.subplots(figsize=(15,9))
img = librosa.display.specshow(mfcc, x_axis='time',
                         y_axis='mel', sr=sampling_rate,
                         fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='MFCCs')
plt.savefig('mfcc-librosa-db.png')

#Chroma Representation
S = librosa.stft(x)**2
chroma = librosa.feature.chroma_stft(S=S, sr=sampling_rate)
fig, ax = plt.subplots(figsize=(15,9))
img = librosa.display.specshow(chroma, y_axis='chroma', 
                               x_axis='time', ax=ax)
fig.colorbar(img, ax=ax)
ax.set(title='Chromagram')

#Onset Detection Features
o_env = librosa.onset.onset_strength(x, sr=sampling_rate)
times = librosa.times_like(o_env, sr=sampling_rate)
onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sampling_rate)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(15,9))
librosa.display.specshow(S_dB, x_axis='time', 
                         y_axis='log', ax=ax[0])
#points out the estimated locations of places where a new note is played through the audio
ax[0].set(title='Power spectrogram')
ax[0].label_outer()
ax[1].plot(times, o_env, label='Onset strength')
ax[1].vlines(times[onset_frames], 0, o_env.max(), 
             color='r', alpha=0.9,
             linestyle='--', label='Onsets')
ax[1].legend()

plt.savefig('onsets.png')

#visualization of first five seconds
o_env = librosa.onset.onset_strength(x, sr=sampling_rate)
total_time = librosa.get_duration(x)
five_sec_mark = int(o_env.shape[0]/total_time*5)
o_env = o_env[:five_sec_mark]
times = librosa.times_like(o_env, sr=sampling_rate)
onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sampling_rate)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(15,9))
librosa.display.specshow(S_dB[:, :five_sec_mark], x_axis='time', 
                         y_axis='log', ax=ax[0])

ax[0].set(title='Power spectrogram')
ax[0].label_outer()
ax[1].plot(times, o_env, label='Onset strength')
ax[1].vlines(times[onset_frames], 0, o_env.max(), 
             color='r', alpha=0.9,
             linestyle='--', label='Onsets')
ax[1].legend()

plt.savefig('onsets-5secs.png')

#Calculate static and dynamic tempo along with beats per minute
onset_env = librosa.onset.onset_strength(x, sr=sampling_rate)
tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sampling_rate)
print(tempo)
dtempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sampling_rate,
                            aggregate=None)
print(dtempo)
#visualize dynamic tempo with tempogram
fig, ax = plt.subplots(figsize=(15,9))
tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sampling_rate)
librosa.display.specshow(tg, x_axis='time', y_axis='tempo', cmap='magma', ax=ax)
ax.plot(librosa.times_like(dtempo), dtempo,
         color='c', linewidth=1.5, label='Tempo estimate (default prior)')
ax.set(title='Dynamic tempo estimation')
ax.legend()
plt.savefig('tempogram.png')

#Extracting and visualizing Harmonic Percussive Source Separation Features
S = librosa.stft(x)
H, P = librosa.decompose.hpss(S)

fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(S), ref=np.max),
                         y_axis='log', x_axis='time', ax=ax[0])
ax[0].set(title='Full power spectrogram')
ax[0].label_outer()
librosa.display.specshow(librosa.amplitude_to_db(np.abs(H), ref=np.max(np.abs(S))),
                         y_axis='log', x_axis='time', ax=ax[1])
ax[1].set(title='Harmonic power spectrogram')
ax[1].label_outer()
librosa.display.specshow(librosa.amplitude_to_db(np.abs(P), ref=np.max(np.abs(S))),
                         y_axis='log', x_axis='time', ax=ax[2])
ax[2].set(title='Percussive power spectrogram')
fig.colorbar(img, ax=ax, format='%+2.0f dB')

#Get segregated signals
y_harm = librosa.istft(H)
y_perc = librosa.istft(P)