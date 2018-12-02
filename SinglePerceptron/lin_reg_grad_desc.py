# lin_reg_grad_desc.py

# Towards making a linear regressor for the given data, this code 
# fits the data with a test line and finds the error using cross entropy function

import numpy as np
import matplotlib.pyplot as plt

def draw(x1, x2):
	fig = plt.plot(x1, x2)
	plt.pause(0.0001)
	fig[0].remove()

def lin_regress(w0, X, Y):
	xtx_i = np.linalg.inv(np.matmul(X.T,X))
	W = np.matmul(np.matmul(xtx_i, X.T), Y)
	return W

def sigmoid(z):
	return 1/(1 + np.exp(-z))

def grad_desc(w, data, label):
	alpha = 0.05
	while calc_err(w, data, label)>0.3:
	# for i in range(5000):
		del_E = grad(w, data, label)
		# print(del_E)
		w = w - del_E * alpha
		w1 = w.item(0)
		w2 = w.item(1)
		b = w.item(2)
		x1 = np.array([ data[:,0].min(), data[:,0].max() ])
		x2 = - b / w2 + x1 * (-w1 / w2)
		draw(x1, x2)
	return w

def grad(w, data, label):
	m = data.shape[0]
	p = sigmoid(data * w)
	del_E = (data.T * (p - label)) / m	
	return del_E

def calc_err(w, data, label):	
	m = data.shape[0]
	p = sigmoid(data_X * w)
	return (-label.T * np.log(p) - (1-label).T * np.log(1-p))/m

# Data generation for use
n_pts = 100

# Fixing the seed for same random data generation every time
np.random.seed(0)
bias = np.ones(n_pts)

top_reg = [np.random.normal(10,2,n_pts), np.random.normal(12,2,n_pts), bias]
top_reg = np.array(top_reg).T
Y_top_reg = np.zeros((n_pts,1))

bot_reg = [np.random.normal(5,2,n_pts), np.random.normal(6,2,n_pts), bias]
bot_reg = np.array(bot_reg).T
Y_bot_reg = np.ones((n_pts,1))

data_X = np.vstack((top_reg,bot_reg))
data_Y = np.vstack((Y_top_reg, Y_bot_reg))

# Plotting the data
plt.ion()
plt.scatter(top_reg[:,0],top_reg[:,1],color='b')
plt.scatter(bot_reg[:,0],bot_reg[:,1],color='r')
plt.title('A Sample Randomized Dataset')
plt.legend(['Top Data','Bottom Data'])


# Using the linear regression to model the system

w0 = np.random.normal(-1,0.24,1) 

# Drawing a test line for modeling the data

# Defining the line

w1 = 0
w2 = 0
b = -0

w0 = np.matrix([w1, w2, b]).T


# Checking the model using the test line ––
# positive means class 0, negative means class 1

lin_combs = data_X * w0
# print(lin_combs)


# print(calc_err(w, data_X, data_Y))
w = grad_desc(w0, data_X, data_Y)

################################################

