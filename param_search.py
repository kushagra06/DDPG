import subprocess
import numpy as np
import random

submission_text = \
'''
#!/bin/bash
#SBATCH --job-name=pendulum
#SBATCH --time=10:00:00
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=3G
#SBATCH --input=none
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

# module load
module load python/3.5.0
source ~/environment/cpu/tensorflow/bin/activate

python3 ./pendulum.py $1 $2 
'''

f = open('temp.slurm', 'w')
f.write(submission_text[1:])
f.close()

theta = np.exp(np.random.uniform(-1, 2, 100))
sigma = np.exp(np.random.uniform(-1, 2, 100))
for t, s in zip(theta, sigma):
        subprocess.run(['sbatch', 'temp.slurm', str(t), str(s)])
            
subprocess.run(['rm', 'temp.slurm'])
