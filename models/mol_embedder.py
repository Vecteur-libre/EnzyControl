import math

import torch
import torch.nn as nn
from torch.nn import functional as F

class MolEmbedder(nn.Module):
    def __init__(self):
        super(MolEmbedder, self).__init__()
        
        node_embed_dims = 128
        node_embed_size = 512
        self.node_embedder = nn.Sequential(
            nn.Linear(node_embed_size,2*node_embed_size),
            nn.SiLU(),
            nn.Linear(2*node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size)
        )
        
    def forward(self, ligand_atom):
        node_embed = self.node_embedder(ligand_atom)
        return node_embed
        