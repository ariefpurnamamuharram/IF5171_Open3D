import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import open3d as o3d
from tqdm import tqdm
from datetime import datetime
from autoencoder import Autoencoder
from keras import losses


if __name__ == '__main__':
    # Preparing the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--filedir', type=str, required=True)
    parser.add_argument('--n_components', type=int)
    parser.add_argument('--autoencoder_latentdim', type=int, default=64)
    parser.add_argument('--autoencoder_epochs', type=int, default=10)
    parser.add_argument('--translation_comp', type=int)
    parser.add_argument('--translation_factor', type=float)
    parser.add_argument('--write_mesh_train', type=bool, default=False)
    parser.add_argument('--write_mesh_test', type=bool, default=False)
    args = parser.parse_args()

    # Check the arguments
    if not str(args.method) in ('PCA', 'Autoencoder'):
        raise ValueError('Method not supported!')
    if str(args.method) == 'PCA' and (args.n_components == None or int(args.n_components) == 0):
        raise ValueError('n-Components can not be empty!')
    if not args.translation_comp is None:
        if args.translation_factor is None:
            raise ValueError('Translation factor can not leave blank!')

    # Create results folder, if not exists
    if not os.path.exists('results'):
        os.makedirs('results')

    # Create logs folder, if not exists
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Get all files
    files = os.listdir(str(args.filedir))
    files = sorted(files)

    print('--- Begin Train-Test Splitting --')

    # Dataset splitting into train and test dataset
    file_train, file_test = train_test_split(
        files, test_size=0.1, random_state=0
    )

    print('Train size:', len(file_train))
    print('Test size:', len(file_test))

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

    # Method: PCA
    if str(args.method) == 'PCA':
        # Setup PCA n-components
        if int(args.n_components) > len(train_vertices_load):
            print('Warning! n-Components is too many. Using the max possible value.')
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

        # Write the log
        now = datetime.now()
        log_filename = 'log_' + now.strftime('%Y%m%d') + '_' + now.strftime('%H%M%S') + '.txt'
        with open(os.path.join('logs', log_filename), 'w') as fh:
            fh.write('date\ttime\ttrain_mse\ttest_mse\n')
            fh.write(f'{now.strftime("%Y-%m-%d")}\t{now.strftime("%H:%M:%S")}\t{mse_test}\t{mse_train}')

        # Mesh translation
        if not args.translation_comp is None:
            if int(args.translation_comp) > n_components:
                print('PCA component number is out of index! Skipping the process...')
            else:
                print('Mesh translation:', 'Yes')

                idx = int(args.translation_comp)
                factor = float(args.translation_factor)

                print('PCA component no.:', idx)
                print('Translation factor:', factor)

                # Translate the train mesh
                train_vertices_transformed[idx] = train_vertices_transformed[idx] * factor
                train_vertices_transformed_inv = pca.inverse_transform(train_vertices_transformed)

                # Translate the test mesh
                test_vertices_transformed[idx] = test_vertices_transformed[idx] * factor
                test_vertices_transformed_inv = pca.inverse_transform(test_vertices_transformed)
        else:
            print('Mesh translation:', 'No')

    # Setup autoencoder
    if str(args.method) == 'Autoencoder':
        latentdim = int(args.autoencoder_latentdim)
        autoencoder = Autoencoder(latentdim)

        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

        print(train_vertices_load.shape)

        autoencoder.fit(train_vertices_load,
            train_vertices_load, 
            epochs=int(args.autoencoder_epochs), 
            shuffle=True)

    print('-End of the process-\n')

    print('--- Begin Write Out the Data ---')

    # Mesh writing function
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
    if (bool(args.write_mesh_train) == True):
        train_vertices_transformed = train_vertices_transformed_inv.reshape(train_vertices_transformed_inv.shape[0], dim_vertex_1, dim_vertex_2)
        write_out(train_vertices_transformed, train_triangles_load, prefix='train_')

    # Write out the test dataset
    if (bool(args.write_mesh_test) == True):
        test_vertices_transformed = test_vertices_transformed_inv.reshape(test_vertices_transformed_inv.shape[0], dim_vertex_1, dim_vertex_2)
        write_out(test_vertices_transformed, test_triangles_load, prefix='test_')

    print('-End of the process-\n')

    print('Finish!\n')
