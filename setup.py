import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="cuda or cpu", default="cuda", type=str)
    args = parser.parse_args()

    # Install torch, matplotlib, numpy
    import os
    os.system('pip install torch torchvision')
    os.system('pip install matplotlib')
    os.system('pip install numpy')
    os.system('pip install pandas')
    os.system('pip install seaborn')

    import torch

    os.environ['TORCH'] = torch.__version__
    os.system("echo Torch version ${TORCH}")

    if args.device == "cuda":
        #NVIDIA GPU version
        os.system('pip uninstall torch-scatter torch-sparse torch-geometric torch-cluster  --y')
        os.system('pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html')
        os.system('pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html')
        os.system('pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}.html')
        os.system('pip install git+https://github.com/pyg-team/pytorch_geometric.git')
    else:
        os.system('pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html')
        os.system('pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html')
        os.system('pip install -q git+https://github.com/pyg-team/pytorch_geometric.git')
    os.system('pip install opacus')
    os.system('pip install wandb')
    os.system('wandb login')