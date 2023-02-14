import matplotlib.pyplot as plt
import numpy as np

# Generate array
array = np.array([[0,0,1], 
         [0,0,0],
         [1,0,1],
         [1,0,0],
         [0,1,1],
         [0,1,0],
         [1,1,1],
         [1,1,0]
         ])

jittered_x = array[:,0] + np.random.uniform(-0.01, 0.05, size=8)
jittered_y = array[:,1] + np.random.uniform(-0.01, 0.05, size=8)

plt.scatter(jittered_x, jittered_y, c=array[:, 2],
            norm=plt.Normalize(vmin=0, vmax=1),
            cmap='viridis')

# Specify the number of discrete levels in the colormap
colorbar = plt.colorbar()
colorbar.set_label("y")
plt.xlabel("x1")
plt.ylabel("x2")
# Show the plot
plt.show()