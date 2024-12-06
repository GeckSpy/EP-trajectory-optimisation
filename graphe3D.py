import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Création des données
x = np.linspace(-5, 5, 100)  # Coordonnées en X
y = np.linspace(-5, 5, 100)  # Coordonnées en Y
X, Y = np.meshgrid(x, y)     # Grille 2D pour les coordonnées
Z = np.sin(np.sqrt(X**2 + Y**2))  # Fonction Z = f(X, Y)

# Création de la figure
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')  # Projection 3D

# Tracé de la surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Ajout de la barre de couleur pour une échelle
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Ajout des étiquettes
ax.set_title("3D Surface Plot")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")

# Affichage du graphique
plt.show()
