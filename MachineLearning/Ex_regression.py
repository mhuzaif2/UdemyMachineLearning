# ex_regression.py

# Example code for linear regression in a simple dataset
# The difference from the class is that the data is not one-dimensional


import numpy as np
import matplotlib.pyplot as plt


def lin_regress(w0, X, Y):
	xtx_i = np.linalg.inv(np.matmul(X.T,X))
	W = np.matmul(np.matmul(xtx_i, X.T), Y)
	return W
    # A = np.matmul(xtx_i, X.T)
    # W = np.matmul(A,Y);
# 	# 
# def grad_des(w, x):


# def gradient():

# def error(x, xhat):
# 	return sigmoid(x-xhat)

def sigmoid(z):
	return 1/(1 + np.exp(-z))

# Data generation for use
n_pts = 100

# Fixing the seed for same random data generation every time
np.random.seed(0)
bias = np.ones(n_pts)

top_reg = [np.random.normal(10,3,n_pts), np.random.normal(12,2,n_pts), bias]
top_reg = np.array(top_reg).T
Y_top_reg = np.zeros((n_pts,1))

bot_reg = [np.random.normal(5,2,n_pts), np.random.normal(6,2,n_pts), bias]
bot_reg = np.array(bot_reg).T
Y_bot_reg = np.ones((n_pts,1))

data_X = np.vstack((top_reg,bot_reg))
data_Y = np.vstack((Y_top_reg, Y_bot_reg))



# Using the linear regression to model the system

w0 = np.random.normal(0,1,3) 

# Finds the optimal line that binary classifies the given data
w  = lin_regress(w0, data_X, data_Y)
w_range = np.array([ bot_reg[:,0].min(), top_reg[:,0].max() ])
lin_combs = np.matmul(data_X, w)

line = -w[2]/w[1] + w_range * (-w[0]/w[1])


probabilities = sigmoid(lin_combs)
print(probabilities)

plt.ion()
# Plotting the data
plt.scatter(top_reg[:,0],top_reg[:,1],color='b')
plt.scatter(bot_reg[:,0],bot_reg[:,1],color='r')
plt.plot(w_range, line)
plt.title('A Sample Randomized Dataset')
plt.legend(['Top Data','Bottom Data'])
plt.rc('text', usetex=True)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()
# plt.draw()
plt.pause(5)
# input('Press Enter to Continue')
