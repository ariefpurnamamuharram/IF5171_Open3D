# Mesh Feature Editor

The project report can be found [here](https://docs.google.com/document/d/1sj3WmNKaiCxM9auX3gxoSXSBbGuQ81n6jjzdFz1eKhk/edit?usp=sharing).<br>

Example usage:
- Try the feature reduction using PCA<br>
```
python3 mesh_feature_editor.py --method PCA --filedir data/bareteeth --n_components 80 --write_mesh_train True --write_mesh_test True
```
- Try the feature reduction using Autoencoder<br>
```
python3 mesh_feature_editor.py --method Autoencoder --filedir data eyebrow --autoencoder_epochs 10 --autoencoder_latentdim 80 --write_mesh_test True
```
- Visualize the mesh object
```
python3 visualizer_mesh.py --filedir results --max_display 10
```
- Visualize the point cloud object
```
python3 visualizer_pcd.py --filedir results --max_display 10
```
- Modify the mesh object using PCA<br>
```
python3 mesh_feature_editor.py --method PCA --filedir data/bareteeth --n_components 80 --translation_comp 0 --translation_factor -2 --write_mesh_train True --write_mesh_test True
```
- Modify the mesh object using Autoencoder<br>
```
python3 mesh_feature_editor.py --method Autoencoder --filedir data eyebrow --autoencoder_epochs 10 --autoencoder_latentdim 80 --translation_comp 0 -- translation_factor -2 --write_mesh_test True