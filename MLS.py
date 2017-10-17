import os
import math
import skimage.io as io
import skimage.filters as filters
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import linalg as LA

class Deformation(object):
	"""docstring for Deformation"""
	def __init__(self, img):
		super(Deformation, self).__init__()

		self.img = img
		self.p = []
		self.q = []
		h, w = self.img.shape
		self.n_grid = 4  # 4 * 4 * 4
		self.h_grid = int(math.ceil(float(h - 1) / self.n_grid))
		self.w_grid = int(math.ceil(float(w - 1) / self.n_grid))
		self.n_nodes = self.n_grid + 1
		self.interval = 4
		self.h_target = self.n_grid * self.h_grid + 1
		self.w_target = self.n_grid * self.w_grid + 1
		self.padded_image = np.pad(img, ((0, self.h_target - h), (0, self.w_target - w)), mode='edge')

	def MLS(self, v):
		A = np.zeros([2,2])
		B = np.zeros([2,2])
		M = np.zeros([2,2])
		w = np.zeros((self.n_nodes)**2, float)

		#compute Wi
		for i in range(len(self.p)):
			if self.p[i][0] == v[0] and self.p[i][1] == v[1]:
				w[i] = float(100) # infinite
			else:
				w[i] = float(1)/((self.p[i][0]-v[0])**2+(self.p[i][1]-v[1])**2)
		pc = np.zeros(2)
		qc = np.zeros(2)
		sw = 0
		for i in range(len(self.p)):
			
			pc += w[i]*self.p[i]
			qc += w[i]*self.q[i]
			sw += w[i]

		pc = pc/sw
		qc = qc/sw
		# print(pc)
		ph = []
		qh = []


		for i in range(len(self.p)):
			# print(self.p[i].shape)
			# print(pc.shape)
			ph.append(self.p[i]-pc)
			qh.append(self.q[i]-qc)

		for i in range(len(self.p)):
			A += w[i]*np.transpose([ph[i]]).dot([ph[i]])
			B += w[i]*np.transpose([ph[i]]).dot([qh[i]])

		A = LA.inv(A)
		M = A.dot(B)
		t = np.array(v)-pc;

		return t.dot(M)+qc

	def getPandQ(self):
		self.p = [] # before deformation
		self.q = [] # after deformation
		sigma = 512 // (self.n_grid * 2 * 3)
		for i in range(0, self.h_target, self.h_grid):
			for j in range(0, self.w_target, self.w_grid):
				self.p.append([i, j]);
				delta = np.random.normal(0, sigma, 2)
				newx = max(0,min(i+delta[0], self.h_target-1))
				newy = max(0,min(j+delta[1], self.w_target-1))
				self.q.append([newx, newy])
		self.p = np.array(self.p)
		self.q = np.array(self.q)

	def deform(self):
		

		target_image = np.zeros([self.h_target, self.w_target], dtype='uint8')
		
		self.getPandQ()
		# padded_image = np.pad(image, ((0, h_target - h), (0, w_target - w), (0, 0)), mode='edge')

		# h, w = self.padded_image.shape
		isVisited = np.zeros([self.h_target, self.w_target])
		

		for i in range(0, self.h_target, self.interval):
			for j in range(0, self.w_target, self.interval):
				v = [i, j]
				vv = self.MLS(v)
				newx = int(max(0, min(vv[0], self.h_target - 1)))
				newy = int(max(0, min(v[1], self.w_target - 1)))

				target_image[i, j] = self.padded_image[newx][newy]
				isVisited[i, j] = 1

				

		for i in range(self.h_target-1):
			for j in range(self.w_target-1):
				if isVisited[i, j] == 0:
					isVisited[i,j] = 1
					minh = (i/self.interval)*self.interval
					minw = (j/self.interval)*self.interval
					maxh = minh+self.interval
					maxw = minw+self.interval
					
					p1 = float(maxw-j)/self.interval*target_image[minh, minw]+float(j-minw)/self.interval*target_image[minh,maxw]
					p2 = float(maxw-j)/self.interval*target_image[maxh, minw]+float(j-minw)/self.interval*target_image[maxh,maxw]

					p3 = float(i-minh)/self.interval*p2+float(maxh-i)/self.interval*p1
					target_image[i, j] = int(p3)

		h, w = self.img.shape
		return target_image[0:h, 0:w]



if __name__ == '__main__':
	lena = io.imread('lena.jpeg')
	d = Deformation(lena)
	targetimg = d.deform()
	plt.figure()
	plt.subplot(1,2,1)
	plt.imshow(lena)
	plt.subplot(1,2,2)
	plt.imshow(targetimg)
	plt.show()
	# print(np.max(lena))
	# plt.imshow(lena)
	# plt.show()