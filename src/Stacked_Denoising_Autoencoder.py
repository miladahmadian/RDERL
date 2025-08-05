import pandas as pd
from datetime import datetime
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from keras import regularizers
from keras.optimizers import Adam, RMSprop


start=datetime.now()

number_latent_feature = 20

#extracting artist_based latent features using stacked denoising autoencoder

artist = pd.read_csv('C:/Users/m.ahmadian/Desktop/RDERL/Prepared_Data/train_user_artist.csv',encoding='ansi')
d = artist.pivot_table(index='userID',columns='artistID',values='value').fillna(0)
data = np.array(d, dtype = 'int')

#import noise to input
train_noisy = np.array(data, dtype = 'int')
noise_factor = 0.5
train_noisy = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape) 
train_noisy = np.clip(train_noisy, 0., 1.)

num_users = int(artist.userID.nunique())
num_artists = int(artist.artistID.nunique())

inputs = Input(shape=(num_artists,))
encoded = Dense(500, activation='relu',activity_regularizer=regularizers.l1(10e-5))(inputs)
encoded = Dense(300, activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)
encoded = Dense(100, activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)
encoded = Dense(number_latent_feature , activation='relu')(encoded)
decoded = Dense(100, activation='relu')(encoded)
decoded = Dense(300, activation='relu')(decoded)
decoded = Dense(500, activation='relu')(decoded)
decoded = Dense(num_artists, activation='sigmoid')(decoded)

autoencoder = Model(inputs, decoded)

encoder = Model(inputs, encoded)

autoencoder.summary()

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

x_train = train_noisy 
x_test = train_noisy


autoencoder.fit(x_train, x_train, epochs=10, batch_size=8, shuffle=True)
#extract encode layer
encoded_vec = encoder.predict(x_test)



f = open("deep_user_artist.txt", "a")        
for i in range (num_users):
    for j in range (number_latent_feature ):
        if j != (number_latent_feature - 1):
            f.write("%f\t" %(encoded_vec[i][j]))
        else:
            f.write("%f\n" %(encoded_vec[i][j]))

f.close()




#extracting trust_based  latent features using stacked denoising autoencoder

trust=pd.read_csv('C:/Users/m.ahmadian/Desktop/RDERL/Prepared_Data/user_friends.csv',encoding='ansi')
d = trust.pivot_table(index='userID',columns='friendID',values='value').fillna(0)
data = np.array(d, dtype = 'int')

#import noise to input
train_noisy = np.array(data, dtype = 'int')
noise_factor = 0.5
train_noisy = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape) 
train_noisy = np.clip(train_noisy, 0., 1.)

num_users=int(trust.userID.nunique())
num_friends=int(trust.friendID.nunique())


inputs = Input(shape=(train_noisy.shape[1],), name='input_layer')
encoded = Dense(500, activation='relu',activity_regularizer=regularizers.l1(10e-5))(inputs)
encoded = Dense(300, activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)
encoded = Dense(100, activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)
encoded = Dense(number_latent_feature , activation='relu')(encoded)
decoded = Dense(100, activation='relu')(encoded)
decoded = Dense(300, activation='relu')(decoded)
decoded = Dense(500, activation='relu')(decoded)
decoded = Dense(num_friends, activation='sigmoid')(decoded)

autoencoder = Model(inputs, decoded)

encoder = Model(inputs, encoded)

autoencoder.summary()

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

x_train = train_noisy 
x_test = train_noisy


autoencoder.fit(x_train, x_train, epochs=10, batch_size=8, shuffle=True)
#extract encode layer
encoded_vec = encoder.predict(x_test)



f = open("deep_user_trust.txt", "a")        
for i in range (num_users):
    for j in range (number_latent_feature ):
        if j != (number_latent_feature - 1) :
            f.write("%f\t" %(encoded_vec[i][j]))
        else:
            f.write("%f\n" %(encoded_vec[i][j]))

f.close()



#extracting tag_based latent features using stacked denoising autoencoder

tag=pd.read_csv('C:/Users/m.ahmadian/Desktop/RDERL/Prepared_Data/user_tag.csv',encoding='ansi')
d = tag.pivot_table(index='userID',columns='tagID',values='value').fillna(0)
data = np.array(d, dtype = 'int')

#import noise to input
train_noisy = np.array(data, dtype = 'int')
noise_factor = 0.5
train_noisy = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape) 
train_noisy = np.clip(train_noisy, 0., 1.)

num_users=int(tag.userID.nunique())
num_tagggs=int(tag.tagID.nunique())



inputs = Input(shape=(train_noisy.shape[1],), name='input_layer')
encoded = Dense(500, activation='relu',activity_regularizer=regularizers.l1(10e-5))(inputs)
encoded = Dense(300, activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)
encoded = Dense(100, activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)
encoded = Dense(number_latent_feature , activation='relu')(encoded)
decoded = Dense(100, activation='relu')(encoded)
decoded = Dense(300, activation='relu')(decoded)
decoded = Dense(500, activation='relu')(decoded)
decoded = Dense(num_tagggs, activation='sigmoid')(decoded)



autoencoder = Model(inputs, decoded)

encoder = Model(inputs, encoded)

autoencoder.summary()

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

x_train = train_noisy 
x_test = train_noisy


autoencoder.fit(x_train, x_train, epochs=10, batch_size=8, shuffle=True)
#extract encode layer
encoded_vec = encoder.predict(x_test)


f = open("deep_user_tag.txt", "a")        
for i in range (num_users):
    for j in range (number_latent_feature ):
        if j != (number_latent_feature - 1 ):
            f.write("%f\t" %(encoded_vec[i][j]))
        else:
            f.write("%f\n" %(encoded_vec[i][j]))

f.close()




print (datetime.now() - start)
