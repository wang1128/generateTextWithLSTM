# Small LSTM Network to Generate Text for Alice in Wonderland
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

# to create the "char to int " dictionary
chars_list = sorted(list(set(raw_text))) #distinct char list
char_to_int = {}
for i , char in enumerate(chars_list):
    char_to_int[char] = i

total_chars = len(raw_text)
total_dist_chars = len(chars_list)

# create data X and data Y. Use a sequence of chars to predict next char
seq_len = 100
dataX = []
dataY = []
for i in range(0, total_chars - seq_len , 1):
    seq_in = raw_text[i : i + seq_len]   #[0:100] 1-99 char
    seq_out = raw_text[i + seq_len]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
    #print(dataX[i], dataY[i])

pattern_num = len(dataX)

# adjust data
X = numpy.reshape(dataX, (pattern_num, seq_len, 1)) # Reshape X that can be used by keras
#X = numpy.reshape(dataX, (pattern_num, seq_len)) # 是不是也一样？ 后面改下# 不一样 100 个time step 每次输入是一个char
# normalize X
X = X / float(total_dist_chars)
y = np_utils.to_categorical(dataY)
print(X.shape)
print(y.shape) #(144243, 43)

#create the LSTM model
model = Sequential()
model.add(LSTM(output_dim= 256,
               input_shape=(X.shape[1], X.shape[2])
               ,return_sequences= True
               )) #  input_length=100, input_dim=1
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation= 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath="my_model_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, nb_epoch=2, batch_size=128, callbacks=callbacks_list)


