import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from human_segmentation_original_dataset_vis import HumanSegOrigDataset


# === Options

# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--input_pth", type=str, help="path to the .off data that gets tested", default = 'data/lad_test_data/meshes/test/shrec/1.off')
parser.add_argument("--pretrained_pth", type=str, help="path to the pretrained model", default = 'data/pretrained_models/human_seg_hks_4x128.pth')
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks", default = 'hks')
args = parser.parse_args()

input_pth = args.input_pth
pretrained_pth = args.pretrained_pth

# system things
device = torch.device('cuda:0')
dtype = torch.float32

# problem/dataset things
n_class = 8

# model 
input_features = args.input_features # one of ['xyz', 'hks']
k_eig = 128

# Important paths
base_path = os.path.dirname(__file__)
#op_cache_dir = os.path.join(base_path, "data", "op_cache")
pretrain_path = os.path.join(base_path, pretrained_pth)
dataset_path = os.path.join(base_path, input_pth)
print(f"this is the data path{dataset_path}")
# === Load datasets
test_dataset = HumanSegOrigDataset(dataset_path, k_eig=k_eig) #, use_cache=True) #, op_cache_dir=op_cache_dir)
#test_loader = DataLoader(test_dataset, batch_size=None)

# === Create the model

C_in={'xyz':3, 'hks':16}[input_features] # dimension of input features

model = diffusion_net.layers.DiffusionNet(C_in=C_in,
                                          C_out=n_class,
                                          C_width=128, 
                                          N_block=4, 
                                          last_activation=lambda x : torch.nn.functional.log_softmax(x,dim=-1),
                                          outputs_at='faces', 
                                          dropout=True)

model = model.to(device)

# load the pretrained model
print("Loading pretrained model from: " + str(pretrain_path))
model.load_state_dict(torch.load(pretrain_path))


# Do an evaluation pass on the test dataset 
def test():
    
    model.eval()
    with torch.no_grad():
    

        # Get data
        verts, faces, frames, mass, L, evals, evecs, gradX, gradY = test_dataset()

        # Move to device
        verts = verts.to(device)
        faces = faces.to(device)
        #frames = frames.to(device)
        mass = mass.to(device)
        L = L.to(device)
        evals = evals.to(device)
        evecs = evecs.to(device)
        gradX = gradX.to(device)
        gradY = gradY.to(device)
                
        # Construct features
        if input_features == 'xyz':
            features = verts
        elif input_features == 'hks':
            features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

        # Apply the model
        preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
        
        # track accuracy
        pred_labels = torch.max(preds, dim=1).indices
        pred_labels = pred_labels.cpu().numpy()
    
    return pred_labels 

def colour_off(path_to_object, pred):
    with open(path_to_object, 'r') as f:
        lines = f.readlines()
    
    assert lines[0].strip() == "OFF", "Not a valid OFF file."
    #print(lines[1].strip().split())
    # Read number of vertices, faces, and edges
    n_vertices, n_faces, _ = map(int, lines[1].strip().split())
        
    # Load vertices
    vertices = np.array([list(map(float, line.strip().split())) for line in lines[2:2+n_vertices]])
        
    # Load faces
    faces = []
    colormap = {
        0: [76, 79, 200], #head color
        1: [211, 225, 53], #torso color
        2: [90, 191, 70], #upper arm
        3: [9, 174, 150], # lower arm
        4: [44, 127, 195], #hand
        5: [227, 197, 55], # upper leg
        6: [207, 69, 6], # lower leg        
        7: [214, 27, 4] # foot
        #8: [0, 0, 0]
    }
    for line, ipred in zip(lines[2+n_vertices:2+n_vertices+n_faces], pred):
        parts = list(map(int, line.strip().split()))
        color = colormap.get(ipred)
        #print(color)
        faces.append(parts[0:4] + color)  # Ignore the first number (number of vertices in the face)
    #print(vertices[0:10])
   
    with open("../output/colored.off", 'w') as f:
        # Write the OFF header
        f.write("OFF\n")
        
        # Write the counts: number of vertices, faces, and edges
        f.write(f"{len(vertices)} {len(faces)} 0\n")
        
        # Write vertices
        for vertex in vertices:
            f.write(" ".join(map(str, vertex)) + "\n")
        
        # Write faces
        for face in faces:
            f.write(" ".join(map(str, face[0:4]))+ "  " + " ".join(map(str, face[4:])) + "\n")

    return None
pred = test()
colour_off(input_pth, pred)
print(f"predictions: {0}")
