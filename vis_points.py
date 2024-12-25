import open3d as o3d
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os

def get_pts(pcd):
    points = np.asarray(pcd.points)
    X = []
    Y = []
    Z = []
    for pt in range(points.shape[0]):
        X.append(points[pt][0])
        Y.append(points[pt][1])
        Z.append(points[pt][2])
    return np.asarray(X), np.asarray(Y), np.asarray(Z)


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])




#import numpy as np
#import matplotlib.pyplot as plt

#def plot_xyz(file_path):
    # Read .xyz file
 #   with open(file_path, 'r') as file:
  #      lines = file.readlines()[:]

   # x, y, z = [], [], []
    #for line in lines:
     #   parts = line.split()
      #  if len(parts) >= 3:
       #     x.append(float(parts[0]))
        #    y.append(float(parts[1]))
         #   z.append(float(parts[2]))

   # x = np.array(x)
    #y = np.array(y)
    #z = np.array(z)

    # Plot
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')

    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('Z')
    #plt.show()

# print("Ground Truth")
# file_path = "/content/data/clean/Pyramid.xyz"
# plot_xyz(file_path)



def plot_single_pcd(points, save_path, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('auto')
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    rotation_matrix = np.asarray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    pcd = pcd.transform(rotation_matrix)
    X, Y, Z = get_pts(pcd)
    t = Z
    ax.scatter(X, Y, Z, c=t, cmap='viridis', marker='o') #, s=1.0, linewidths=0)
    ax.grid(False)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    set_axes_equal(ax)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.savefig(save_path, format='png', dpi=500)
    plt.show()

pcd = o3d.io.read_point_cloud("bunny.xyz")  # Replace with your file path
points = np.asarray(pcd.points)
# Ensure the output directory exists
output_dir = "./data/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set the output file path
image_path = os.path.join(output_dir, "bunny.png")
plot_single_pcd(points, image_path, title=None)

print(f"Image saved at: {image_path}")



