import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Générer des points 3D
np.random.seed(42)

# Points rouges regroupés au centre
num_points_red = 50
radius_red = 1
angles_red = np.linspace(0, 2*np.pi, num_points_red)
x_red = radius_red * np.cos(angles_red) + np.random.normal(0, 0.1, num_points_red)
y_red = radius_red * np.sin(angles_red) + np.random.normal(0, 0.1, num_points_red)
z_red = np.random.uniform(-1, 1, num_points_red)

# Points bleus entourant les points rouges
num_points_blue = 100
radius_blue = 2.5
angles_blue = np.linspace(0, 2*np.pi, num_points_blue)
x_blue = radius_blue * np.cos(angles_blue) + np.random.normal(0, 0.5, num_points_blue)
y_blue = radius_blue * np.sin(angles_blue) + np.random.normal(0, 0.5, num_points_blue)
z_blue = np.random.uniform(2.5, 3, num_points_blue)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Tracer les points rouges
ax.scatter(x_red, y_red, z_red, color='red', label='Points rouges (centre)')

# Tracer les points bleus
ax.scatter(x_blue, y_blue, z_blue, color='blue', label='Points bleus (entourant)')

# Tracer l'hyperplan séparateur à z = 1
x = np.linspace(-3, 3, 10)
y = np.linspace(-3, 3, 10)
X, Y = np.meshgrid(x, y)
Z = np.ones_like(X)  # Positionner l'hyperplan à z = 1
ax.plot_surface(X, Y, Z, color='gray', alpha=0.5)

# Ajouter des étiquettes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Ajouter une légende

# Afficher le graphique
plt.show()