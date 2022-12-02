import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import open3d as o3d
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filedir', required=True)
    parser.add_argument('--n_components', required=True)
    args = parser.parse_args()

    if not os.path.exists('results'):
        os.makedirs('results')

    files = os.listdir(str(args.filedir))
    files = sorted(files)

    print('--- Begin Train-Test Splitting --')

    file_train, file_test = train_test_split(
        files, test_size=0.1, random_state=0
    )

    print('-End of the process-\n')

    print('--- Begin Reading the Data ---')

    train_vertices_load = []
    train_triangles_load = []

    for file in file_train:
        filepath = os.path.join(str(args.filedir), file)
        mesh = o3d.io.read_triangle_mesh(filepath)
        train_vertices_load.append(mesh.vertices)
        train_triangles_load.append(mesh.triangles)

    test_vertices_load = []
    test_triangles_load = []

    for file in file_test:
        filepath = os.path.join(str(args.filedir), file)
        mesh = o3d.io.read_triangle_mesh(filepath)
        test_vertices_load.append(mesh.vertices)
        test_triangles_load.append(mesh.triangles)

    train_vertices_load = np.array(train_vertices_load)
    test_vertices_load = np.array(test_vertices_load)

    # Get vertex dimension
    _, dim_vertex_1, dim_vertex_2 = train_vertices_load.shape
    
    # Reshape vertices
    train_vertices_load = train_vertices_load.reshape(train_vertices_load.shape[0], -1)
    test_vertices_load = test_vertices_load.reshape(test_vertices_load.shape[0], -1)

    print('-End of the process-\n')

    print('--- Begin the PCA Analysis ---')

    if int(args.n_components) > len(train_vertices_load):
        print('Warning! n-Components is too many. Using the possible max value.')
        n_components = len(train_vertices_load)
    else:
        n_components = int(args.n_components)

    print('n-Components:', str(n_components))

    # PCA training process
    pca = PCA(n_components=n_components, random_state=0)
    train_vertices_transformed = pca.fit(train_vertices_load).transform(train_vertices_load)
    train_vertices_transformed_inv = pca.inverse_transform(train_vertices_transformed)

    # Calculate train MSE
    mse_train = mean_squared_error(train_vertices_load, train_vertices_transformed_inv)
    print('Train mean square error:', mse_train)

    # PCA testing process
    test_vertices_transformed = pca.transform(test_vertices_load)
    test_vertices_transformed_inv = pca.inverse_transform(test_vertices_transformed)

    # Calculate train MSE
    mse_test = mean_squared_error(test_vertices_load, test_vertices_transformed_inv)
    print('Test mean square error:', mse_test)

    print('-End of the process-\n')

    print('--- Begin Write Out the Data ---')

    def write_out(vertices_load, triangles_load, prefix=''):
        for idx, vertices in tqdm(enumerate(vertices_load)):
            filename = f'{prefix}result.{idx}.ply'
            mesh = o3d.geometry.TriangleMesh()
            vertices = vertices_load.reshape(-1, 3)
            triangles = triangles_load[idx]
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            mesh.paint_uniform_color([0.5, 0.5, 0.5])
            o3d.io.write_triangle_mesh(os.path.join('results', filename), mesh)

    # Write out the train dataset
    train_vertices_transformed = train_vertices_transformed_inv.reshape(train_vertices_transformed_inv.shape[0], dim_vertex_1, dim_vertex_2)
    write_out(train_vertices_transformed, train_triangles_load, prefix='train_')

    # Write out the test dataset
    test_vertices_transformed = test_vertices_transformed_inv.reshape(test_vertices_transformed_inv.shape[0], dim_vertex_1, dim_vertex_2)
    write_out(test_vertices_transformed, test_triangles_load, prefix='test_')

    print('-End of the process-\n')

    print('Finish!\n')
