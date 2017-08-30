import pyaudio
import struct
import pygame
import numpy as np
from math import sin, cos, pi, sqrt, exp
import random
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import struct

# helper functions
def create_arr(size):
	lst = [[],[],[],[]]
	for i in range(4):
		lst[i] = (np.zeros(size))
	return lst

# Music related parameters

SEMITONE = 0.031   
l_c4 = 0.604 # length for middle C
notes = {'a': l_c4, 's': (l_c4 - (SEMITONE * 2)), \
'd': l_c4 - SEMITONE * 4, 'f': l_c4 - SEMITONE * 5}

# Equations related parameters

S4 = 1.56          # S is the stiffness of the string
c2 = 1e5           # c is speed of propagation
d1 = 0.13          # frequency independent damping 
d3 = -0.01         # frequency dependent damping

Mmax = 5         # Maximum number of modes.

#l = 0.0          # length of string

#xe = 0.3*l         # Excitation point along string (where it is plucked)
#xa = 0.375*l       # Measurement point along string (where we observe)


# Audio processing related parameters

fs = 32000        # sampling frequency (1/s)
T = 1/float(fs)    # sampling period (s)

BLOCKSIZE = 32      # Number of frames per block
WIDTH = 2           # Bytes per sample
CHANNELS = 1        # Mono
MAXVALUE = 2**15-1
BLOCKSIZE = 32      # Number of frames per block
WIDTH = 2           # Bytes per sample
CHANNELS = 1        # Mono

# Open the audio output stream
p = pyaudio.PyAudio()
PA_FORMAT = p.get_format_from_width(WIDTH)
stream = p.open(format = PA_FORMAT,
                channels = CHANNELS,
                frames_per_buffer = BLOCKSIZE,  # Make small to avoid latency...
                rate = fs,
                input = False,
                output = True)

pygame.init()  # Initializes pygame

stop = False

class Key:
    # Declares variables


	def __init__(self, leng):
		self.b1 = np.zeros(Mmax)
		self.c1 = np.zeros(Mmax)
		self.c0 = np.zeros(Mmax)   
		self.gamma = np.zeros(Mmax)
		self.sigma = np.zeros(Mmax)
		self.X1 = np.zeros(Mmax)
		self.X2 = np.zeros(Mmax)
		self.X3 = -(d1 ** 2)/4
		self.omega = np.zeros(Mmax)
		self.l = leng
		for m in range(1, Mmax+1):
			
			self.gamma[m-1] = m * pi / self.l
			self.sigma[m-1] = (d3 * self.gamma[m-1] ** 2 - d1) / 2

			self.X1[m-1] = ((S4 - (d3 ** 2)) / 2) * (self.gamma[m-1] ** 4)
			self.X2[m-1] = (c2 + ((d1 * d3) / 2) ) * (self.gamma[m-1] ** 2)

			self.omega[m-1] = sqrt(self.X1[m-1] + self.X2[m-1] + self.X3)

		for m in range(Mmax):

			self.b1[m] = T * sin(self.omega[m] * T) / (self.omega[m] * T)
			self.b1[m] = self.b1[m] * exp(self.omega[m] * T)

			self.c1[m] = -2 * exp(self.sigma[m] * T) * cos(self.omega[m] * T)

			self.c0[m] = exp(2 * self.sigma[m] * T)
	def display(self):
		print "string length: " + str(self.l)


key_objects = {}
for key in notes:
    k = Key(notes[key])
    key_objects[key] = k

y = np.zeros(BLOCKSIZE)

x = create_arr(BLOCKSIZE)
yk = create_arr(BLOCKSIZE)
zi = [[], [], [], []]
for i in range(4):
	zi[i] = np.zeros((Mmax,2))
c0 = create_arr(Mmax)
c1 = create_arr(Mmax)
b1 = create_arr(Mmax)

print "Play! a = do, s = re, d = mi, f = fa"
print "press q to quit"

while stop == False:
	y = np.zeros(BLOCKSIZE)

	for event in pygame.event.get():
		if event.type == pygame.KEYDOWN:
			if event.unicode not in notes and event.unicode != 'q':
				print "wrong key"
			else:
				if event.type == pygame.KEYDOWN and event.key == ord('a'):
					key_objects['a']
					x[0][0] = 15000
					b1[0] = key_objects['a'].b1
					c1[0] = key_objects['a'].c1
					c0[0] = key_objects['a'].c0
                if event.type == pygame.KEYDOWN and event.key == ord('s'):
                    x[1][0] = 15000
                    b1[1] = key_objects['s'].b1
                    c1[1] = key_objects['s'].c1
                    c0[1] = key_objects['s'].c0
                if event.type == pygame.KEYDOWN and event.key == ord('d'):
                    x[2][0] = 15000
                    b1[2] = key_objects['d'].b1
                    c1[2] = key_objects['d'].c1
                    c0[2] = key_objects['d'].c0
                if event.type == pygame.KEYDOWN and event.key == ord('f'):
                    x[3][0] = 15000
                    b1[3] = key_objects['f'].b1
                    c1[3] = key_objects['f'].c1
                    c0[3] = key_objects['f'].c0

		if event.key == pygame.K_q:
			stop = True

	for k in range(Mmax):
		b_1 = np.array([0, b1[0][k]])        
		a_1 = np.array([1, c1[0][k], c0[0][k]]) 
		#print x[0]
		yk[0], zf_1 = lfilter(b_1, a_1, x[0], zi = zi[0][k,:])        
		y = np.add(y,yk[0])

		zi[0][k,0] = zf_1[0]
		zi[0][k,1] = zf_1[1]

		b_2 = np.array([0, b1[1][k]])        
		a_2 = np.array([1, c1[1][k], c0[1][k]]) 

		yk[1], zf_2 = lfilter(b_2, a_2, x[1], zi = zi[1][k,:])        
		y = np.add(y,yk[1])

		zi[1][k,0] = zf_2[0]
		zi[1][k,1] = zf_2[1]

		b_3 = np.array([0, b1[2][k]])        
		a_3 = np.array([1, c1[2][k], c0[2][k]]) 

		yk[2], zf_3  = lfilter(b_3, a_3, x[2], zi = zi[2][k,:])        
		y = np.add(y,yk[2])

		zi[2][k,0] = zf_3[0]
		zi[2][k,1] = zf_3[1]

		b_4 = np.array([0, b1[3][k]])        
		a_4 = np.array([1, c1[3][k], c0[3][k]]) 

		yk[3], zf_4 = lfilter(b_4, a_4, x[3], zi = zi[3][k,:])        
		y = np.add(y,yk[3])

		zi[3][k,0] = zf_4[0]
		zi[3][k,1] = zf_4[1]
	for i in range(4):
		x[i][0] = 0

	out = np.clip(80*y, -MAXVALUE, MAXVALUE)     # Clipping

	# Convert numeric list to binary string
	data = struct.pack('h' * BLOCKSIZE, *out)

	# Write binary string to audio output stream
	stream.write(data, BLOCKSIZE)


print 'Done!!!'

# Close audio stream
stream.stop_stream()
stream.close()
p.terminate()


