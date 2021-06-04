# CNN for Music Genre Classification
Project for CS 4701: Practicum in AI (Spring 2021)

This project includes two different CNN architectures, both of which perform on mel spectrograms generated from 30 second audio clips (the training and testing were done on the GTZAN dataset) and consider 10 genre labels: pop, country, classical, metal, jazz, disco, reggae, blues, rock, and hiphop.

Peak observed performance for the 2-layer architecture was a correct prediction rate of about 55%.

The 5-layer architecture is still being tested, with the peak observed performance so far being a correct prediction rate of about 45%.

For both architectures, there is somewhat of an overfitting issue; if you mean to replicate this experiment, I recommend using longer audio clips or multiple mel spectrograms to generate a label.



References: 

towardsdatascience.com/musical-genre-classification-with-convolutional-neural-networks-ff04f9601a74

github.com/keunwoochoi/music-auto_tagging-keras

github.com/eeveeking/351

github.com/yangdanny97/genre_classification

github.com/jsalbert/Music-Genre-Classification-with-Deep-Learning
