from linear_prediction import LP
import numpy as np
from matplotlib import pyplot as plt
import pdb

K = np.linspace(1, 29, 29, dtype=int)
alp = np.linspace(2, 6, 5, dtype = int)
p = np.zeros((5, 29))

for i in range(len(alp)):
    for j in range(len(K)):
        p[i,j] = LP(K[j], alp[i])
        
# Graphs

fig, ax = plt.subplots()
ax.set_title("Percentage of Blob Positions Correctly Predicted")
ax.set_xlabel("Number of Previous Frames Taken into Account")
ax.set_ylabel("Percent Correctly Predicted")

a = np.linspace(1, 29, 29)
for i in range(len(alp)):
    ax.plot(a, p[i,:], label="Predicted " +str(i+1) + " Frames into the Future")
plt.legend()

plt.show()
