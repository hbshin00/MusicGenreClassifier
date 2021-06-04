import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

#example code for generating a mel spectrogram from a single file
#y, sr = librosa.load('./genres/blues/blues.00000.wav')
#mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
#mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
#fig = plt.Figure()
#canvas = FigureCanvas(fig)
#ax = fig.add_subplot(111)
#spect = librosa.display.specshow(mel_spect, y_axis='mel',ax=ax, fmax=8000, x_axis='time')
#filename = 'blues00.png'
#fig.savefig(filename)
#librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time')
#plt.show()


genre_counter = 0
#number of genre labels
num_genres = 10
genres_list = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
song_counter = 0
#number of songs desired (per genre), max 100
num_songs = 80
#loop through the first 80 songs of each genre, save mel spectrogram as .png
while genre_counter < num_genres and song_counter < num_songs:
    genre = genres_list[genre_counter]
    filepath = './genres/' + genre + '/' + genre + '.000'
    if song_counter < 10:
        song = '0' + str(song_counter)
    else:
        song = str(song_counter)
    filepath = filepath + song + '.wav'
    y, sr = librosa.load(filepath)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    spect = librosa.display.specshow(mel_spect, y_axis='mel', ax=ax, fmax=8000, x_axis='time')
    filename = genre + song + '.png'
    fig.savefig('trainingdata/'+filename)
    if song_counter < (num_songs - 1):
        song_counter = song_counter + 1
    else:
        genre_counter = genre_counter + 1
        song_counter = 0
