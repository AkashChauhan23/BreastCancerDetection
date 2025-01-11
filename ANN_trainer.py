import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequencial
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# data load
data = pd.read_csv('breast-cancer.csv')
data = data.drop('id', axis=1)

# apply encoder
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

# dumping encoder file
with open('diagnosis_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

x = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# data split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)

scaller = StandardScaler()
x_train = scaller.fit_transform(x_train)
x_test = scaller.transform(x_test)
with open('scaller.pkl', 'wb') as file:
    pickle.dump(scaller, file)

model = Sequencial([
    Input(shape=[x_test.shape[1],]),
    Dense(16, activation = 'relu'),
    Dense(8, activation = 'relu'),
    Dense(1, activation = 'Sigmoid')
])

opt = tf.optimizer.Adam(learning_rate = 0.01)
model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
histroy = model.fit(x_train, x_test, validation_data = (x_test, y_test), epochs = 80, callbacks = [early_stopping_callback])
model.save('breastcancerdetection.keras')
