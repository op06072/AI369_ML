import tensorflow as tf
import numpy as np
import os

from tensorflow.python.keras.layers import *

num_digits = 10
numbers = np.arange(1, 101)
l = len(str(2**num_digits))
L = l
l0 = 7
epoch = 300
batch = 300

checkpoint_path = './model/369_{}d_{}e_{}b/cp.ckpt'.format(num_digits, epoch, batch)
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=5)

def tsn_encode(i):
    I0 = str(i)
    y = 0
    for I in I0:
        if I in "369":
            y += 1
    return np.array([1 if j == y else 0 for j in range(l)])

def tsn(i):
    List = [str(i)] + ['x'*i for i in range(1, l)]
    return List[np.argmax(tsn_encode(i))]

def tsn_decode(i, prediction):
    List = [str(i)] + ['x'*i for i in range(1, l)]
    return List[prediction]

def binary_encode(i):
    i = list(map(str, list(i)))
    i = [list("0" * (L - len(j)) * (len(j) < L) + j) for j in i]
    i = np.array([np.array(list(map(ord, j))) for j in i])
    i = np.array([[[k >> d & 1 for d in range(l0)] for k in j] for j in i])
    return i

def ct(data):
    lookup = {}
    for i in range(1, l):
        lookup['x'*i] = i
    grid = [[0 for _ in range(l)] for _ in range(l)]
    for i, output in enumerate(data):
        actual = np.argmax(tsn_encode(i+1))
        predicted = lookup.get(output, 0)
        grid[predicted][actual] += 1
    return grid

trX = np.array(binary_encode(np.array(range(101, 2**num_digits))))
trY = np.array([tsn_encode(i) for i in range(101, 2 ** num_digits)])
teX = binary_encode(numbers)

num_hidden1 = 400
num_hidden2 = 800
num_hidden3 = 1200
num_hidden4 = 1600
num_hidden5 = 2000
num_hidden6 = 2000

model = tf.keras.models.Sequential([
    Flatten(input_shape = (L, l0)),
    Dense(num_hidden1, activation="relu"),
    Dropout(0.5),
    Dense(num_hidden2, activation="relu"),
    Dropout(0.4),
    Dense(num_hidden3, activation="relu"),
    Dropout(0.3),
    Dense(num_hidden4, activation="relu"),
    Dropout(0.2),
    Dense(num_hidden5, activation="relu"),
    Dropout(0.1),
    Dense(num_hidden6, activation="relu"),
    Dense(l, activation="softmax")
])

model2 = tf.keras.models.Sequential([
    Flatten(input_shape = (L, l0)),
    Dense(num_hidden5, activation="relu"),
    Dropout(0.5),
    Dense(num_hidden6, activation="relu"),
    Dense(l, activation="softmax")
])

model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

try:
	model.load_weights(checkpoint_path)
except Exception:
	pass

model.fit(trX, trY, epochs=epoch, batch_size=batch, shuffle=True, callbacks=[cp_callback])

model2.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

#model2.fit(trX, trY, nb_epoch=300, batch_size=400, shuffle=True)

predictions = [tsn_decode(i+1, y) for i, y in enumerate(np.argmax(model.predict(teX), axis=1))]
print(ct(predictions))
#am = model2.predict(teX)
#predictions = [tsn_decode(i+1, y) for i, y in enumerate(np.argmax(model2.predict(teX), axis=1))]
#print(ct(predictions))

errors = 0
correct = 0

for i in range(1, 101):
    x = binary_encode([i])
    y = np.argmax(model.predict(np.array(x)), axis=1)
    print(tsn_decode(i, y[0]))
    if tsn_decode(i, y[0]) == tsn(i):
        correct += 1
    else:
        errors += 1
print("Errors : ", errors, " Correct : ", correct)

errors = 0
correct = 0
"""
for i in range(1, 101):
    x = binary_encode([i])
    y = np.argmax(model2.predict(np.array(x)), axis=1)
    print(tsn_decode(i, y[0]))
    if tsn_decode(i, y[0]) == tsn(i):
        correct += 1
    else:
        errors += 1
print("Errors : ", errors, " Correct : ", correct)"""