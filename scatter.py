import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from os import walk
import re
import numpy as np
import pandas as pd
sns.set(style="darkgrid")

path = './pendv0/'
N_ITER = 10
N_PARAM = 199
N_EPI = 1000

# Get names of all the files to be imported
f = []
for (dirpath, dirnames, filenames) in walk(path):
        f.extend(filenames)
        break

# Get values of theta and sigma
l = []
for s in f:
        l.append(re.findall("\d+\.\d+", s))

theta = []
sigma = []
for s in l:
        theta.append(float(s[0]))
        sigma.append(float(s[1]))

theta = [theta[i] for i in range(0,1990,10)]
sigma = [sigma[i] for i in range(0,1990,10)]


# Calculate avg reward and std error for avg number of steps to converge for each set of parameters (theta, sigma)
r = np.zeros([N_PARAM, N_EPI])
n_steps_temp = np.empty([N_ITER,])
steps_std_err = np.empty([N_PARAM,])

for i in range(N_PARAM):
        for j in range(N_ITER):
                temp_arr = np.load(path+f[10*i+j])
                r[i] += temp_arr
                for k in range(N_EPI):
                        temp_avg = sum(temp_arr[k:k+10])/10.0
                        if temp_avg > -300:
                                n_steps_temp[j] = k
                                break
        steps_std_err[i] = np.std(n_steps_temp)/np.sqrt(N_ITER)
        r[i] = r[i]/N_ITER


#dev = np.zeros([N_PARAM, N_EPI])
#for i in range(N_PARAM):
#        for j in range(N_ITER):
#                dev[i] += (r[i] - np.load(path+f[10*i+j]))**2
#        dev[i] = np.sqrt(dev[i]/(N_ITER-1))
#        dev[i] = dev[i]/np.sqrt(N_ITER)

# Calculate avg number of steps in which the task is solved 
n_steps = np.zeros([N_PARAM,1])
for i in range(N_PARAM):
        for j in range(N_EPI):
                temp_avg = sum(r[i][j:j+10])/10.0
                if temp_avg > -300:
                        n_steps[i] = j
                        break

n_steps = n_steps.flatten()


# Plotting the results
steps = pd.DataFrame({'N_Steps':n_steps, 'Std_Err':steps_std_err, 'Theta':theta, 'Sigma':sigma})

#plt.pcolor(steps)
#plt.yticks(sigma)
#plt.xticks(theta)
#steps =  steps.pivot_table(index='Theta', columns='Sigma', values='N_Steps')
#fig = plt.figure(1)
#ax = fig.add_subplot(111)
#plt.scatter(theta, sigma, s=n_steps, c=colors, alpha=0.3, cmap='viridis')
#plt.colorbar()
#cmap = sns.cubehelix_palette(dark=0.3, light=0.8, as_cmap=True)
#plt.figure(figsize=(7,7.5))
ax = sns.scatterplot(x='Theta', y='Sigma', hue='N_Steps', size='Std_Err', sizes=(20,200), data=steps)
#ax = sns.heatmap(steps)
#h, l = plt.gca().get_legend_handles_labels()
ax.set_yscale('log')
ax.set_xscale('log')
plt.legend(bbox_to_anchor=(1.11,0.6), loc=4, borderaxespad=0., fontsize='xx-small', frameon=True, framealpha=0.9)
fig = ax.get_figure()
#plt.pcolormesh(theta, sigma, n_steps)
plt.savefig("scatter.png")
