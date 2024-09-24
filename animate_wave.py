from matplotlib.animation import FuncAnimation
from Wave2D import Wave2D, Wave2D_Neumann
import matplotlib.pyplot as plt
import numpy as np

mx = 2
my = 2
c = 1
cfl = 1/np.sqrt(2)
N = 10
Nt = 1000

sol = Wave2D_Neumann()
data = sol(N, Nt, cfl=cfl, c=1.0, mx=mx, my=my, store_data=1)

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Surface Plot')

# Set z-axis limits so that they don't change
z_min, z_max = np.min(data[0]), np.max(data[0])


plot = ax.plot_surface(sol.xij, sol.yij, data[0], cmap='viridis')

def frame(i):
    ax.cla()
    ax.set_zlim(z_min, z_max)
    plot = ax.plot_surface(sol.xij, sol.yij, data[i], cmap='viridis')
    return plot,

ani = FuncAnimation(fig=fig, func=frame, frames=Nt)
ani.save(filename="neumannwave.gif.gif", writer="pillow")
