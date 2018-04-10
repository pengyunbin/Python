# Random Matrix Theory

import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt

n = 100  # size of matrices
t = 5000  # number of samples
v = np.empty((t, n))  # eigenvalue samples
v1 = np.empty(t)  # max eigenvalue samples
delta = 0.2  # histogram bin width

for i in range(t):

    # sample from GOE
    a = np.random.normal(0,1,(n,n))
    s = (a + a.T) / 2

    # compute eigenvalues
    evals = LA.eigvals(s)

    # store eigenvalues
    v[i, :] = evals

    # sample from GUE
    a = np.random.normal(0,1,(n,n)) + 1j * np.random.normal(0,1,(n,n))
    s = (a + np.matrix(a).getH())/2

    # compute eigenvalues
    evals = LA.eigvals(s)

    # store maximum eigenvalue
    v1[i] = np.amax(evals)


# normalize v
v = v / np.sqrt(n/2)

# set histogram bin values to a numpy array containing [-2, -2+delta,
# -2+2*delta, ..., 2]
# Note: both 2 and -2 are to be included
bins = np.arange(-2,2+delta,delta)

# compute histogram
hist, bin_edges = np.histogram(v, bins=bins)

# plot bar chart
plt.bar(bins[:-1], hist/delta/sum(hist), width=delta, facecolor='y')

# plot theoretical prediction, i.e., the semicircle law
plt.plot(bin_edges, np.sqrt(4-bin_edges**2)/(2*np.pi), linewidth=2)

# set axes and save to pdf
plt.ylim([0, .5])
plt.xlim([-2.5, 2.5])
plt.savefig('Semicircle.pdf')
plt.close()

# normalize v1
v1 = (v1 - 2*np.sqrt(n)) * np.power(n, 1.0/6)

# set histogram bin values to a numpy array containing [-5, -5+delta,
# -5+2*delta, ..., 2]
# Note: both -5 and 2 are to be included
bins = np.arange(-5, 2+delta, delta)

# compute histogram
hist, bin_edges = np.histogram(v1, bins=bins)

# plot bar chart
plt.bar(bins[:-1], hist/delta/sum(hist), width=delta, facecolor='y')

# load theoretical prediction, i.e., the Tracy-Widom law, from file
prediction = np.loadtxt('tracy-widom.csv', delimiter=',')

# plot Tracy-Widom law
plt.plot(prediction[:, 0], prediction[:, 1], linewidth=2)

# set axes and save to pdf
plt.ylim([0, .5])
plt.xlim([-5, 2])
plt.savefig('Tracy-Widom.pdf')
plt.close()
