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

    pcds = []
    
    for idx, file in tqdm(enumerate(files)):
        filepath = os.path.join(str(args.filedir), file)
        pcd = o3d.io.read_point_cloud(filepath)
        pcd.estimate_normals()
        pcd = pcd.translate((((idx % 5) * 0.3), ((idx // 5) * 0.35 * (-1)), 0))
        pcds.append(pcd)
    
    o3d.visualization.draw_geometries(pcds, mesh_show_back_face=True)
