# Mesh Feature Editor

Example usage:
- Try feature reduction using PCA<br>
```
python3 mesh_feature_editor.py --method PCA --filedir data/bareteeth --n_components 80 --write_mesh_train True --write_mesh_test True
```
- Try feature reduction using Autoencoder<br>
```
python3 mesh_feature_editor.py --method Autoencoder --filedir data eyebrow --autoencoder_epochs 10 --autoencoder_latentdim 80 --write_mesh_test True
```
- Visualize the mesh object
```
python3 visualizer_mesh.py --filedir results --max_display 10
```