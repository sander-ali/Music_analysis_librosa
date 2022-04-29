# Music_analysis_librosa
The code in this repository represents a tutorial to perform music analysis with librosa package python.

In order to use the code you need to install the following
librosa (pip install librosa)
FFMPEG (you can follow the guidelines from [1] to install FFMPEG)

The code performs analysis on Atif Aslam's song (file is included with the repository) but you can choose any of your custom file.

Furthermore, you can also choose among various examples available in librosa package.  
you can run the following code to see examples available:  
librosa.util.list_examples()

output:  
AVAILABLE EXAMPLES
--------------------------------------------------------------------
brahms    	Brahms - Hungarian Dance #5
choice    	Admiral Bob - Choice (drum+bass)
fishin    	Karissa Hobbs - Let's Go Fishin'
nutcracker	Tchaikovsky - Dance of the Sugar Plum Fairy
trumpet   	Mihai Sorohan - Trumpet loop
vibeace   	Kevin MacLeod - Vibe Ace

you can also play the audio files in jupyter notebook using the following set of commands:  
from IPython import display

example_name = 'fishin' 
audio_path = librosa.ex(example_name)  
display.Audio(audio_path, rate=sampling_rate)  

From this code,  
- you can learn how to plot the waves of an audio file.  
- Visualizing linear scale spectrogram  ![stft-librosa-linear](https://user-images.githubusercontent.com/26203136/165922361-89ab23b2-e99d-4a6e-8ba0-cc3fc10d5bde.png)  
- Visualizing spectral centroids and spectral bandwidth  
- Visualizing deviations from the spectal centroids  ![centroid-vs-bw-librosa](https://user-images.githubusercontent.com/26203136/165922558-29ccf594-adcb-4b0f-8481-7be20c0ee54b.png)  
- Visualizing spectral contrast  ![spectral-contrast-librosa](https://user-images.githubusercontent.com/26203136/165922686-ddcb064b-2012-475e-b6af-b9409d8c7eb0.png)  
- Visualize STFT on a linear and logarithmic scale  ![stft-librosa-linear-db](https://user-images.githubusercontent.com/26203136/165922795-b4b946d8-cc25-49c8-9f08-4044e982ea22.png)  
![stft-librosa-log-db](https://user-images.githubusercontent.com/26203136/165922849-52da5e32-183b-4bb7-b1d4-4bec0d37cd0c.png)  
- Visualize MEL spectrogram  ![stft-librosa-mel](https://user-images.githubusercontent.com/26203136/165922916-a3f32527-0e04-4b31-a094-32adbfe71d6a.png)  
- Generating MEL filters  ![mel-spec-librosa-db](https://user-images.githubusercontent.com/26203136/165922975-d1ded490-d298-40c3-bd94-3c15d35e3000.png)  
- Calculating MFCC  ![mfcc-librosa-db](https://user-images.githubusercontent.com/26203136/165923030-794133ba-3368-4d9f-a10f-19d92f13da7a.png)  
- Generating Chroma Representation  
- Generating general and variable Onset Detection Features  ![onsets](https://user-images.githubusercontent.com/26203136/165923148-b4854976-cb61-41c3-b5b6-358af6b2c62c.png)  
![onsets-5secs](https://user-images.githubusercontent.com/26203136/165923181-ff476709-e95e-4767-8f27-1cdc9bdde950.png)  
- Visualize Static and Dynamic Tempos with Tempogram  ![tempogram](https://user-images.githubusercontent.com/26203136/165923280-77fec030-07ef-4e16-b94f-98417fc3c5b4.png)  
- Extracting Harmonic Percussive Source Separation Features that provides Full power Spectrogram, Harmonic Power Spectrogram, and Percussive Power Spectrogram.















References
[1] https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/
