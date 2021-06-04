#imports

labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
songs_bank = {'blues': [('song1', 'artist1'), ('song2', 'artist2'), ('song3', 'artist3'), ('song4', 'artist4')],
    'classical': [('song1', 'artist1'), ('song2', 'artist2'), ('song3', 'artist3'), ('song4', 'artist4')],
    'country': [('song1', 'artist1'), ('song2', 'artist2'), ('song3', 'artist3'), ('song4', 'artist4')],
    'disco': [('song1', 'artist1'), ('song2', 'artist2'), ('song3', 'artist3'), ('song4', 'artist4')],
    'hiphop': [('song1', 'artist1'), ('song2', 'artist2'), ('song3', 'artist3'), ('song4', 'artist4')],
    'jazz': [('song1', 'artist1'), ('song2', 'artist2'), ('song3', 'artist3'), ('song4', 'artist4')],
    'metal': [('song1', 'artist1'), ('song2', 'artist2'), ('song3', 'artist3'), ('song4', 'artist4')],
    'pop': [('song1', 'artist1'), ('song2', 'artist2'), ('song3', 'artist3'), ('song4', 'artist4')],
    'reggae': [('song1', 'artist1'), ('song2', 'artist2'), ('song3', 'artist3'), ('song4', 'artist4')],
    'rock': [('song1', 'artist1'), ('song2', 'artist2'), ('song3', 'artist3'), ('song4', 'artist4')]}

def recommend(song_name, label):
    songs = songs_bank[label]
    str = "Based on your input song's genre, we recommend:"
    print(str)
    counter = 0
    while i < len(songs):
        name, artist = songs[i]
        test_name = name.strip()
        test_name = test_name.replace(" ","")
        test_name = "".join(test_name.split())
        test_name = test_name.translate(None, string.punctuation)
        test_song_name = song_name.strip()
        test_song_name = test_song_name.replace(" ","")
        test_song_name = "".join(test_song_name.split())
        test_song_name = test_name.translate(None, string.punctuation)
        if test_name == test_song_name:
            pass 
        elif counter >= 3:
            pass
        else:
            print(str(name) + ", by: " + str(artist))
            counter = counter + 1
        i = i + 1
