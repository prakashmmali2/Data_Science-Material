import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Step 1: Define Rectangles and the Bin
bin_width = 80
bin_height = 40
rectangles = {
    1: [20, 10],  # Width, Height
    2: [30, 15],
    3: [10, 5],
    4: [15, 20],
    5: [5, 10],
    6: [10, 10],
    7: [12, 8],
    8: [8, 6],
    9: [10, 10]  
}


bin_space = np.ones((bin_width, bin_height))

# Create Graph Representation
G = nx.Graph()
for rect_id in rectangles.keys():
    G.add_node(rect_id)


G.add_edges_from([(1, 2), (3, 4), (3, 5), (3, 9), (6, 7)]) 


class GNNModel(nn.Module):
    def __init__(self):  
        super(GNNModel, self).__init__()  
        self.conv1 = GCNConv(2, 16)  
        self.conv2 = GCNConv(16, 32)  
        self.fc1 = nn.Linear(32, 16)  
        self.fc2 = nn.Linear(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = nn.ReLU()(x)  
        x = self.conv2(x, edge_index)
        x = nn.ReLU()(x)
        x = self.fc1(x)
        return self.fc2(x)


def optimize_placement():
    placements = {}
    
    # Place rectangle 1 at the top
    placements[1] = (0, 30)  # Top (x, y)

    # Place rectangle 2 at the bottom
    placements[2] = (0, 0)  # Bottom

    # Place rectangle 3 near rectangle 4, 5, and 9
    placements[3] = (2, 15)  
    placements[4] = (15, 5) 
    placements[5] = (15, 20) 
    placements[9] = (2, 5)    

    # Place rectangle 6
    placements[6] = (40, 0)  

    # Place rectangle 7 close to rectangles 6 and 2
    placements[7] = (35, 10)  

    # Place remaining rectangle 8 anywhere possible
    placements[8] = (50, 25)

    return placements

# Step 6: Visualization Function
def plot_rectangles(placements):
    plt.figure(figsize=(8, 4))
    for rect_id, (x, y) in placements.items():
        width, height = rectangles[rect_id]
        plt.gca().add_patch(plt.Rectangle((x, y), width, height, fill=True, edgecolor='black'))
        plt.text(x + width / 2, y + height / 2, str(rect_id), fontsize=12, ha='center', va='center')

    plt.xlim(0, bin_width)
    plt.ylim(0, bin_height)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Rectangle Placement in Bin')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.grid()
    plt.show()

# Execute optimization and plotting
placements = optimize_placement()
plot_rectangles(placements)
