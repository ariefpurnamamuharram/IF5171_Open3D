import os
import argparse
import open3d as o3d
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filedir', required=True)
    parser.add_argument('--max_display')
    args = parser.parse_args()

    files = os.listdir(str(args.filedir))
    files = sorted(files)

    if args.max_display is None:
        max_display = int(10)
    else:
        max_display = int(args.max_display)

    if len(files) >= max_display:
        files = files[:max_display]

    meshes = []
    
    for idx, file in tqdm(enumerate(files)):
        filepath = os.path.join(str(args.filedir), file)
        mesh = o3d.io.read_triangle_mesh(filepath)
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        mesh.paint_uniform_color([0.5, 0.5, 0.5])
        mesh = mesh.translate((((idx % 5) * 0.3), ((idx // 5) * 0.35 * (-1)), 0))
        meshes.append(mesh)
    
    o3d.visualization.draw_geometries(meshes, mesh_show_back_face=True)
