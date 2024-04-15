import numpy as np
import open3d as o3d

NAME = "testing/test_files/room.ply"

pcd = o3d.io.read_point_cloud(NAME)
print(pcd)
print(np.asarray(pcd.points))

pts = np.asarray(pcd.points)

alpha = 0.03
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
mesh.compute_vertex_normals()

t1 = o3d.geometry.TriangleMesh.create_coordinate_frame()
t1.translate(np.array([0, 3, 0]))


o3d.visualization.draw_geometries([mesh, t1],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
