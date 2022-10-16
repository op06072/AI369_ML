import numpy as np
import tensorflow as tf
import sys
from keras.layers import *
import keras
import os

num_digits = 10
epoch = 300
batch = 300
l=4
n = 1
if not os.path.isdir('./model'):
    os.mkdir('./model')
checkpt = './model/369_{}d_{}e_{}b.h5'.format(num_digits,epoch,batch)
checkpt2 = './model/369_{}d_{}e_400b-2.h5'.format(num_digits,epoch)
checkpoint_path = './model/369_{}d_{}e_{}b/cp.ckpt'.format(num_digits, epoch, batch)

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
    i = [list("0" * (l - len(j)) * (len(j) < l) + j) for j in i]
    i = np.array([np.array(list(map(ord, j))) for j in i])
    i = np.array([[[k >> d & 1 for d in range(7)] for k in j] for j in i])
    return i

if os.path.isfile(checkpt):
    model = keras.models.load_model(checkpt)

num_hidden1 = 400
num_hidden2 = 800
num_hidden3 = 1200
num_hidden4 = 1600
num_hidden5 = 2000
num_hidden6 = 2000

model2 = keras.models.Sequential([
    Flatten(input_shape = (4, 7)),
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
if os.path.isfile(checkpoint_path):
    model2.load_weights(checkpoint_path)

model3 = keras.models.Sequential([
    Flatten(input_shape = (4, 7)),
    Dense(num_hidden2, activation="relu"),
    Dropout(0.5),
    Dense(num_hidden4, activation="relu"),
    Dropout(0.4),
    Dense(num_hidden6, activation="relu"),
    Dense(l, activation="softmax")
])
if os.path.isfile(checkpt2):
    model3.load_weights(checkpt2)

turn = 0

def play():
	print("=========Welcome to 369 game with AI=========")
	while 1:
		if turn == 0:
			print("     human:", end=' ')
			ans = input()
		elif turn == 1:
			x = binary_encode([n])
			Y = np.argmax(model.predict(np.array(x)), axis=1)
			ans = tsn_decode(n, Y[0])
			print("     AI1: {}".format(ans))
		elif turn == 2:
			x = binary_encode([n])
			Y = np.argmax(model2.predict(np.array(x)), axis=1)
			ans = tsn_decode(n, Y[0])
			print("     AI2: {}".format(ans))
		elif turn == 3:
			x = binary_encode([n])
			Y = np.argmax(model3.predict(np.array(x)), axis=1)
			ans = tsn_decode(n, Y[0])
			print("     AI3: {}".format(ans))
	
		if ans != tsn(n):
			break
		n += 1
		turn += 1
		if turn > 3:
			turn = 0
	if turn == 0:
		print("lose")
	else:
		print("win")

def benchmark():
	start1=1
	while 1:
		x = binary_encode([start1])
		Y = np.argmax(model.predict(np.array(x)), axis=1)
		ans = tsn_decode(start1, Y[0])
		if ans != tsn(start1):
			break
		start1 += 1
	start2=1
	while 1:
		x = binary_encode([start2])
		Y = np.argmax(model2.predict(np.array(x)), axis=1)
		ans = tsn_decode(start2, Y[0])
		if ans != tsn(start2):
			break
		start2 += 1
	start3=1
	while 1:
		x = binary_encode([start3])
		Y = np.argmax(model3.predict(np.array(x)), axis=1)
		ans = tsn_decode(start3, Y[0])
		if ans != tsn(start3):
			break
		start3 += 1
	print("     AI1 : to{}\n     AI2 : to{}\n     AI3 : to{}".format(start1, start2, start3))

if __name__ == '__main__':
	if  '--help' in sys.argv or '-h' in sys.argv:
		print("usage: python3 AIvsHuman_369.py [-h] [-p/--play] [-b/--benchmarking]\noptional arguments:\n-h, --help   show this help message and exit.\n-p, --play   play 369game with AIs.\n-b, --benchmarking   show the performace of each of AIs.")
	elif '--play' in sys.argv or '-p' in sys.argv:
		play()
	elif '--benchmarking' in sys.argv or '-b' in sys.argv:
		benchmark()
