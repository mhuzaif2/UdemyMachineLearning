# lin_reg_disp.py

# Towards making a linear regressor for the given data, this code 
# is about displaying a test line across the data


import numpy as np
import matplotlib.pyplot as plt


def lin_regress(w0, X, Y):
	xtx_i = np.linalg.inv(np.matmul(X.T,X))
	W = np.matmul(np.matmul(xtx_i, X.T), Y)
	return W

def sigmoid(z):
	return 1/(1 + np.exp(-z))

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


# Using the linear regression to model the system

w0 = np.random.normal(-1,0.24,1) 

# Drawing a test line for modeling the data

# Defining the line

w1 = 2
w2 = 2
b = -35

w = np.matrix([w1, w2, b]).T
w_range = np.array([ bot_reg[:,0].min(), top_reg[:,0].max() ])

line = - b / w2 + w_range * (-w1 / w2)

# Checking the model using the test line ––
# positive means class 0, negative means class 1

lin_combs = data_X * w
print(lin_combs)

# Finding the probabilities of closeness to the test line
probabilities = sigmoid(lin_combs)
print(100 * probabilities)


# Plotting the data
plt.ion()
plt.scatter(top_reg[:,0],top_reg[:,1],color='b')
plt.scatter(bot_reg[:,0],bot_reg[:,1],color='r')
plt.title('A Sample Randomized Dataset')
plt.legend(['Top Data','Bottom Data'])

# Plotting the modeling line
plt.plot(w_range, line)
plt.rc('text', usetex=True)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()

plt.pause(5)
# input('Press Enter to Continue')
