import random
from sympy import false, true
from torch import nn
import torch
import math
from torch_geometric.nn import GATConv
from torch_scatter import scatter_add, scatter_mean

ligand_atom_types = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl']
ligand_atom_add_aromatic_types = ['H', 'C1', 'C2', 'N1', 'N2', 'O1', 'O2', 'F', 'P1', 'P2', 'S1', 'S2', 'Cl']
pocket_atom_types = ['H', 'C', 'N', 'O', 'S', 'Se']
residue_types = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()
    

def get_edges(mask, x=None, edge_cutoff=None): 
    adj = mask[:, None] == mask[None, :]
    if edge_cutoff is not None:
        adj = adj & (torch.cdist(x, x) <= float(edge_cutoff))
    edges = torch.stack(torch.where(adj), dim=0)
    return edges        

def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff

def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))

    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result




class PlicEncoder(nn.Module):
    def __init__(
            self, 
            hidden_nf: int = 128,
            edge_cutoff = None,
            coords_range=15,
            normalization_factor=100, 
            aggregation_method='sum',
            sin_embedding=False,
            act_fn=nn.SiLU(),
            EGNN_layers=1,
            prob_if = 0.6,
            ):
        super(PlicEncoder, self).__init__()

        self.hidden_nf = hidden_nf
        self.ligand_atom_type_embed = nn.Embedding(len(ligand_atom_add_aromatic_types) + 1, hidden_nf)
        self.pocket_atom_type_embed = nn.Embedding(len(pocket_atom_types) + 1, hidden_nf)
        self.pocket_residue_type_embed = nn.Embedding(len(residue_types) + 1, hidden_nf)
        self.residue_type_embed = nn.Embedding(len(residue_types) + 1, hidden_nf)
        self.pocket_feat_fusion = nn.Linear(hidden_nf * 2, hidden_nf)
        self.F_GATlayer = nn.ModuleList([
            GATConv(hidden_nf, hidden_nf*2, heads=2, add_self_loops=True)  # concat=False
        ])
        self.F_mlp = nn.Sequential( 
            nn.Linear(hidden_nf*2*(self.F_GATlayer[0].heads), hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf)
        )
        self.pli_emb = nn.Linear(7, hidden_nf)
        self.I_GATlayer = nn.ModuleList([
            GATConv(hidden_nf+self.pli_emb.out_features, hidden_nf*2, heads=2, add_self_loops=True)  
        ])
        self.pli_mlp = nn.Linear(hidden_nf*2*self.I_GATlayer[0].heads, hidden_nf)
        self.pocket_type_fusion = nn.Linear(hidden_nf * 2, hidden_nf)
        self.EGNN_lig_layers = EGNN_layers
        self.EGNN_pocket_layers = EGNN_layers
        self.id_embed = nn.Embedding(2, 4)
        self.embed_fusion = nn.Linear(hidden_nf + 4, hidden_nf)
        self.id_info_add = nn.Linear(hidden_nf + 4, hidden_nf)
        self.H_lig_fusion = nn.Linear(hidden_nf + 3, hidden_nf)
        self.H_pocket_fusion = nn.Linear(hidden_nf + 3, hidden_nf)


        

        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.prob_if = prob_if
        self.edge_cutoff = edge_cutoff 
        self.coords_range_layer = float(coords_range/1)    
        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2    


        self.EGNN_lig_module = nn.ModuleList([
            EquivariantBlock( 
                hidden_nf, 
                edge_feat_nf=edge_feat_nf,
                act_fn=nn.SiLU(),
                n_layers=1,
                attention=False, 
                norm_diff=True, 
                tanh=False,
                coords_range=coords_range,
                norm_constant=1,
                sin_embedding=self.sin_embedding,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method
            )
        ])

        self.EGNN_pocket_module = nn.ModuleList([
            EquivariantBlock( 
                hidden_nf, 
                edge_feat_nf=edge_feat_nf,
                act_fn=nn.SiLU(),
                n_layers=1,
                attention=False, 
                norm_diff=True, 
                tanh=False,
                coords_range=coords_range,
                norm_constant=1,
                sin_embedding=self.sin_embedding,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method
            )
        ])
    
    def remove_pocket_mean_batch_ligand(self, x_lig, x_pocket, lig_indices, pocket_indices):

        pocket_mean = scatter_mean(x_pocket, pocket_indices, dim=0)
        if x_lig is not None:
            x_lig = x_lig - pocket_mean[lig_indices]
        x_pocket = x_pocket - pocket_mean[pocket_indices]
        x_pocket = x_pocket.float()
        return x_lig, x_pocket
    


    def extract_pli_feat(self, lig_coords, pocket_coords, lig_a_hidx, pocket_a_hidx, pocket_r_hidx, lig_mask, pocket_mask, pli, sample_diffusion=false):
        lig_coords, pocket_coords = self.remove_pocket_mean_batch_ligand(lig_coords, pocket_coords, lig_mask, pocket_mask)  
        device = pocket_coords.device
        if lig_coords is None:
            lig_coords = torch.rand(len(lig_mask), 3).to(pocket_coords.device)
            lig_a_hidx = torch.ones(len(lig_mask)).to(pocket_coords.device)
            lig_a_hidx = lig_a_hidx.int()
            use_raw_ligand_info = false
        else:
            use_raw_ligand_info = true
        num_lig = lig_coords.shape[0]

        complexes_mask = torch.cat([lig_mask, pocket_mask], dim=0)
        # complexes_coords = torch.cat([lig_coords, pocket_coords], dim=0).to(torch.float32)
        complexes_id = torch.LongTensor([0] * lig_coords.shape[0] + [1] * pocket_coords.shape[0]).to(device)
        lig_atom_type = lig_a_hidx
        pocket_atom_type = pocket_a_hidx
        pocket_residue_type = pocket_r_hidx 

        # embedding
        complexes_id_emb = self.id_embed(complexes_id)
        lig_atom_type_emb = self.ligand_atom_type_embed(lig_atom_type)
        pocket_atom_type_emb = self.pocket_atom_type_embed(pocket_atom_type)
        pocket_residue_type_emb = self.pocket_residue_type_embed(pocket_residue_type)
        pocket_type_emb = torch.cat([pocket_atom_type_emb, pocket_residue_type_emb], dim=1)
        pocket_type_emb = self.pocket_type_fusion(pocket_type_emb)

        complexes_type_emb = torch.cat([lig_atom_type_emb, pocket_type_emb], dim=0) 
        complexes_emb = torch.cat([complexes_type_emb, complexes_id_emb], dim=-1)
        complexes_emb = self.embed_fusion(complexes_emb)
        lig_emb, pocket_emb = complexes_emb[: num_lig], complexes_emb[num_lig:]

        ligand_edge_index = get_edges(mask=lig_mask).cpu()
        ligand_edge_index = torch.LongTensor(ligand_edge_index).to(device)
        pocket_edge_index = get_edges(mask=pocket_mask).cpu()
        pocket_edge_index = torch.LongTensor(pocket_edge_index).to(device)
        complexes_edge_index = get_edges(mask=complexes_mask).cpu()
        complexes_edge_index = torch.LongTensor(complexes_edge_index).to(device) 

        pocket_distances, _ = coord2diff(pocket_coords, pocket_edge_index)
        if self.sin_embedding is not None:
            pocket_distances = self.sin_embedding(pocket_distances)

        lig_distances, _ = coord2diff(lig_coords, ligand_edge_index)
        if self.sin_embedding is not None:
            lig_distances = self.sin_embedding(lig_distances)


        h_lig = lig_emb
        h_poc = pocket_emb
        if use_raw_ligand_info:
            for i in range(self.EGNN_lig_layers):
                LigLayer = self.EGNN_lig_module[i]
                h_lig, lig_coords= LigLayer(h_lig, lig_coords, ligand_edge_index, edge_attr=lig_distances, node_mask=None, edge_mask=None)
            H_lig = torch.cat([h_lig, lig_coords], dim=1)
            H_lig = self.H_lig_fusion(H_lig)
        
        for i in range(self.EGNN_pocket_layers):
            PocketLayer = self.EGNN_pocket_module[i]   
            h_poc, pocket_coords = PocketLayer(h_poc, pocket_coords, pocket_edge_index, edge_attr=pocket_distances, node_mask=None, edge_mask=None)
        H_poc = torch.cat([h_poc, pocket_coords], dim=1)
        H_poc = self.H_pocket_fusion(H_poc)

        prob_if = self.prob_if
        random_value = random.random()
        F_GAT_layer = self.F_GATlayer[0]
        I_GAT_layer = self.I_GATlayer[0]
        pli = pli.to(pocket_coords.device)
        pli = self.pli_emb(pli)
        if use_raw_ligand_info:
            if sample_diffusion:
                H_complex = torch.cat([H_lig, H_poc], dim=0)  # # (N_l + N_p, hidden_nf)
                F = F_GAT_layer(H_complex, complexes_edge_index)  # (N_l + N_p, hidden_nf*2)
                F = self.F_mlp(F)    # (N_l + N_p, hidden_nf)
                zero_tensor = torch.zeros(len(lig_coords), self.hidden_nf).to(device)  # (N_l, hidden_nf)
                pli = torch.cat([zero_tensor, pli], dim=0)
                I = torch.cat([F, pli], dim=-1)
                pli = I_GAT_layer(I, complexes_edge_index)
                pli = pli[num_lig:]
            else:
                if random_value < prob_if:
                    H_complex = torch.cat([H_lig, H_poc], dim=0)  # # (N_l + N_p, hidden_nf)
                    F = F_GAT_layer(H_complex, complexes_edge_index)  # (N_l + N_p, hidden_nf*2)
                    F = self.F_mlp(F)    # (N_l + N_p, hidden_nf)

                    zero_tensor = torch.zeros(len(lig_coords), self.hidden_nf).to(device)  # (N_l, hidden_nf)
                    pli = torch.cat([zero_tensor, pli], dim=0)
                    I = torch.cat([F, pli], dim=-1)
                    pli = I_GAT_layer(I, complexes_edge_index)
                    pli = pli[num_lig:]
                else:
                    F = F_GAT_layer(H_poc, pocket_edge_index)  # (N_p, hidden_nf*2)
                    F = self.F_mlp(F)   # (N_p, hidden_nf)
                    I = torch.cat([F, pli], dim=1)
                    pli = I_GAT_layer(I, pocket_edge_index)
        else:
                F = F_GAT_layer(H_poc, pocket_edge_index)  # (N_p, hidden_nf*2)
                F = self.F_mlp(F)   # (N_p, hidden_nf)
                I = torch.cat([F, pli], dim=1)
                pli = I_GAT_layer(I, pocket_edge_index)
        pli = self.pli_mlp(pli)
        return pli




    
class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij

class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask, update_coords_mask=None):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)

        if update_coords_mask is not None:
            agg = update_coords_mask * agg

        coord = coord + agg
        return coord

    def forward(self, h, coord, edge_index, coord_diff, edge_attr=None,
                node_mask=None, edge_mask=None, update_coords_mask=None):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask,
                                 update_coords_mask=update_coords_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord

class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum'):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                              act_fn=act_fn, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, act_fn=nn.SiLU(), tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method))

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None, update_coords_mask=None):
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr,
                                               node_mask=node_mask, edge_mask=edge_mask)
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr,
                                       node_mask, edge_mask, update_coords_mask=update_coords_mask)

        if node_mask is not None:
            h = h * node_mask
        return h, x 
