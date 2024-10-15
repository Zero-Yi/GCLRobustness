# Introduce

This is the repository of the project: Adversarial Attacks against Graph Neural Networks in the context of Graph Contrastive Learning.

# Usage
## Setup
It is recommended to run the docker container in VScode. The configuration can be adjusted via `.devcontainer/devcontainer.json`.

Install the official extension `Dev Containers`, then press `Ctrl+Shift+P` and type “Dev Container: Rebuild and Reopen in Container” to build image and launch the container.

## Hierarchy
- source
    - node
        - gcl_node.py # GCL in node classification
        - gcn.py # plain counterpart to gcl_node.py
        - pure_dgi.py # DGI in node claasification
    - graph
        - adgcl.py # AD-GCL in graph classification
        - gcl.py # GCL in graph classification
        - gin.py # plain counterpart to gcl.py
        - Infograph.py # Infograph in graph classification
    - utils
- Readme.md
- Dockerfile
- ...

## Running the codes:
To launch a script, first **navigate to the corresponding sub-directory**, e.g., from the root directory, run
```bash
cd source/graph/
python gcl.py
```
if you want to run the evaluation of the model GCL in graph classification.

Please use `-h` to see further instructions. For example:
```bash
python gcl.py -h
```