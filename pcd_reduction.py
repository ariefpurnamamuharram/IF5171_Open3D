import os
import argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import open3d as o3d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filedir', required=True)
    parser.add_argument('--n_components', required=True)
    args = parser.parse_args()

    files = os.listdir(str(args.filedir))
    files = sorted(files)

    if not os.path.exists('results'):
        os.makedirs('results')

    vertices_load = []
    triangles_load = []

    for file in files:
        filepath = os.path.join(str(args.filedir), file)
        mesh = o3d.io.read_triangle_mesh(filepath)
        vertices_load.append(mesh.vertices)
        triangles_load.append(mesh.triangles)

    vertices_load = np.array(vertices_load)
    _, dim_vertices_2, dim_vertices_3 = vertices_load.shape
    vertices_load = vertices_load.reshape(vertices_load.shape[0],-1)

    pca = PCA(n_components=int(args.n_components), random_state=0)
    vertices_transformed = pca.fit(vertices_load).transform(vertices_load)
    vertices_transformed_inv = pca.inverse_transform(vertices_transformed)

    # Calculate MSE
    mse_train = mean_squared_error(vertices_load, vertices_transformed_inv)
    print('MSE:', mse_train)

    vertices_transformed = vertices_transformed_inv.reshape(vertices_transformed_inv.shape[0], dim_vertices_2, dim_vertices_3)

    for idx, vertices in enumerate(vertices_transformed):
        filename = f'result.{idx}.ply'
        mesh = o3d.geometry.TriangleMesh()
        vertices = vertices.reshape(-1,3)
        triangles = triangles_load[idx]
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        mesh.paint_uniform_color([0.5, 0.5, 0.5])
        o3d.io.write_triangle_mesh(os.path.join('results', filename), mesh)
