import sys
import os

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../src/")
)  # add the path to the DiffusionNet src
import diffusion_net
import torch

print("diff-net code imported!")


def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("CUDA is not available.")


if __name__ == "__main__":
    check_cuda()
