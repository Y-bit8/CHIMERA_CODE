import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d as BN
from torch_geometric.nn import GCNConv, GAT, GIN, global_mean_pool, SAGPooling
from torch_geometric.utils import to_dense_batch

def group_node_rep(node_rep, batch_size, num_part):
    group = []
    motif_group = []
    super_group = []
                                 
    count = 0
    for i in range(batch_size):
        num_atom = num_part[i][0]
        num_motif = num_part[i][1]
                                                        
                                                          
        num_all = num_atom + num_motif + 1
        group.append(node_rep[count:count + num_atom].float().mean(dim=0))
        motif_group.append(node_rep[count + num_atom : count + num_all-1].float().mean(dim=0))
        super_group.append(node_rep[count + num_atom : count + num_all].float().mean(dim=0))
        count += num_all
    return group, motif_group, super_group
    
                                   

class SubExtractor(nn.Module):
    def __init__(self, hidden_dim, num_clusters, residual=False):
        super().__init__()
        
        self.Q = nn.Parameter(torch.Tensor(1, num_clusters, hidden_dim))
        nn.init.xavier_uniform_(self.Q)
        
        self.W_Q = Linear(hidden_dim, hidden_dim)
        self.W_K = Linear(hidden_dim, hidden_dim)
        self.W_V = Linear(hidden_dim, hidden_dim)
        self.W_O = Linear(hidden_dim, hidden_dim)
 
                             
                                                                      
                  
                                                                         
        self.residual = residual
    
    def forward(self, x, batch, chirality_feas, use_chiral= False):
                                 
        chirality_features = chirality_feas.view(list(chirality_feas.shape)[0]//100,100)                           
                                                                                       
                                                     
                                      
                                                                                                  
        
                                                 
                                                                                   
        
        K = self.W_K(x)
        V = self.W_V(x)
        
        K, mask = to_dense_batch(K, batch)
                                           
        V, _ = to_dense_batch(V, batch)
        
        attn_mask = (~mask).float().unsqueeze(1)
        attn_mask = attn_mask * (-1e9)
        
        Q = self.Q.tile(K.size(0), 1, 1)
        Q = self.W_Q(Q)
        
        A = Q @ K.transpose(-1, -2) / (Q.size(-1) ** 0.5)
        A = A + attn_mask
                                                     
                    
        chirality_weight = chirality_features.unsqueeze(1).repeat(1, 10, 1)[:,:,:A.size(2)]                                                 
         
        if use_chiral ==True:
        
            alpha = float(os.getenv("CHIMERA_CHIRALITY_ALPHA", "2.0"))
            A = A * (chirality_weight * alpha + (1 - chirality_weight))                                              
                                                            
                                                
        A = A.softmax(dim=-2)
                                                   
        
        out = Q + A @ V
        
        if self.residual:
            out = out + self.W_O(out).relu()
        else:
            out = self.W_O(out).relu()
        
        return out, A.detach().argmax(dim=-2), mask




class Transformer(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, num_patterns=10, cfg=None):
        super(Transformer, self).__init__()

        model_cfg = (cfg or {}).get("model", {})
        num_patterns = int(model_cfg.get("num_patterns", num_patterns))
        gin_layers = int(model_cfg.get("gin_layers", 5))
        gat_layers = int(model_cfg.get("gat_layers", 2))
        self.ablation = dict((cfg or {}).get("ablation", {}))
        self.ablation.setdefault("no_fingerprint", False)
        self.ablation.setdefault("no_graph", False)
        self.ablation.setdefault("no_motif", False)
        self.ablation.setdefault("no_interaction", False)
        self.ablation.setdefault("no_loffi", False)
        self.ablation.setdefault("no_condition", False)

        self.conv1 = GIN(dataset[0][0].num_features,hidden,gin_layers,2*hidden,jk='cat')
        self.conv2 = GAT(dataset[0][5].num_features,hidden,gat_layers,2*hidden,jk='cat')
                                                                                                                   
                                                                                                                   
                                                                                
                                            
                                            
                                            
                                            
                                        
 
                                                                                                               
                                                                                                               
                                                                                                               

                                      
        fusion_dim = 2048 + 16 * hidden + 6 * num_patterns * num_patterns
        self.fusion_gate_type = str(model_cfg.get("fusion_gate", "legacy_log_softmax")).lower()
        if self.fusion_gate_type == "legacy_log_softmax":
            self.lin8 = Linear(fusion_dim, fusion_dim)
        elif self.fusion_gate_type == "residual_sigmoid":
            gate_hidden = int(model_cfg.get("gate_hidden", 128))
            if gate_hidden <= 0:
                raise ValueError("model.gate_hidden must be positive")
            self.gate_down = Linear(fusion_dim, gate_hidden)
            self.gate_up = Linear(gate_hidden, fusion_dim)
            nn.init.zeros_(self.gate_up.weight)
            nn.init.zeros_(self.gate_up.bias)
        else:
            raise ValueError(f"Unsupported model.fusion_gate: {self.fusion_gate_type}")
        fusion_norm = str(model_cfg.get("fusion_norm", "batchnorm")).lower()
        if fusion_norm == "batchnorm":
            self.bn1 = BN(fusion_dim)
        elif fusion_norm == "layernorm":
            self.bn1 = nn.LayerNorm(fusion_dim)
        elif fusion_norm in {"none", "identity"}:
            self.bn1 = nn.Identity()
        else:
            raise ValueError(f"Unsupported model.fusion_norm: {fusion_norm}")
                                                                      
                                 
        
        
        default_head_hidden1 = (2048)//2+16*hidden//2+6*num_patterns*num_patterns//2
        default_head_hidden2 = (2048)//4+16*hidden//4+6*num_patterns*num_patterns//4
        head_hidden1 = int(model_cfg.get("head_hidden1", default_head_hidden1))
        head_hidden2 = int(model_cfg.get("head_hidden2", default_head_hidden2))
        if head_hidden1 <= 0 or head_hidden2 <= 0:
            raise ValueError("model head dimensions must be positive")
        self.head_dropout = nn.Dropout(float(model_cfg.get("head_dropout", 0.0)))
        self.lin1 = Linear(fusion_dim, head_hidden1)
        self.lin2 = Linear(head_hidden1, head_hidden2)
        self.lin3 = Linear(head_hidden2, 2)
        
                                          
                                             
                                             
                                          
        
                                                                                                             
                                                
                                               
                                        
        
                                                                 
                                                                 
                                                                 
                                                                 
                                                                 



        self.sagpool1= SAGPooling(2*hidden,0.8,GCNConv )
                                                          
                                                         
                                                          
        
        self.pool1 = SubExtractor(2*hidden, num_patterns, False)
        self.pool2 = SubExtractor(2*hidden, num_patterns, False)
                                                                 
                                                                 
                                                 


    def _apply_ablation(
        self,
        fg_input,
        super_rep_x0, super_rep_x10, super_rep_x20, super_rep_x30,
        sim_cp, sim_c1, sim_c2, sim_x12h, sim_x13h, sim_x14h,
        x, x1, x2, x3,
        loffi_input=None,
        cond_embedding=None,
    ):





        abl = self.ablation
        if abl.get("no_fingerprint", False):
            fg_input = fg_input * 0.0

        if abl.get("no_interaction", False):
            sim_cp = sim_cp * 0.0
            sim_c1 = sim_c1 * 0.0
            sim_c2 = sim_c2 * 0.0
            sim_x12h = sim_x12h * 0.0
            sim_x13h = sim_x13h * 0.0
            sim_x14h = sim_x14h * 0.0

        if abl.get("no_motif", False):
            super_rep_x0 = super_rep_x0 * 0.0
            super_rep_x10 = super_rep_x10 * 0.0
            super_rep_x20 = super_rep_x20 * 0.0
            super_rep_x30 = super_rep_x30 * 0.0
            sim_x12h = sim_x12h * 0.0
            sim_x13h = sim_x13h * 0.0
            sim_x14h = sim_x14h * 0.0

        if abl.get("no_graph", False):
            x = x * 0.0
            x1 = x1 * 0.0
            x2 = x2 * 0.0
            x3 = x3 * 0.0
            sim_cp = sim_cp * 0.0
            sim_c1 = sim_c1 * 0.0
            sim_c2 = sim_c2 * 0.0

        if loffi_input is not None and abl.get("no_loffi", False):
            loffi_input = loffi_input * 0.0

        if cond_embedding is not None and abl.get("no_condition", False):
            cond_embedding = cond_embedding * 0.0

        return (
            fg_input,
            super_rep_x0, super_rep_x10, super_rep_x20, super_rep_x30,
            sim_cp, sim_c1, sim_c2, sim_x12h, sim_x13h, sim_x14h,
            x, x1, x2, x3,
            loffi_input,
            cond_embedding,
        )

    def reset_parameters(self):
        
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
         
                                     
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
                                      
                                      
                                 
                                    
                                 
                                    
                                 
                                    
                                 
                                    
            
        if hasattr(self.bn1, "reset_parameters"):
            self.bn1.reset_parameters()
                                    
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        
                                     
                                     
                                     
                                     
        if self.fusion_gate_type == "legacy_log_softmax":
            self.lin8.reset_parameters()
        else:
            self.gate_down.reset_parameters()
            nn.init.zeros_(self.gate_up.weight)
            nn.init.zeros_(self.gate_up.bias)
        self.sagpool1.reset_parameters()
                                         
                                         
                                         
                                     
        
        
    

        
                                                        
                                                                                                          
    def forward(self, data):
        x, edge_index, edge_attr, batch,temp_x,time_x,metal_x,solvent_x,additive_x,gm_x,elsi_x, add,add1,chirality_fea = data[0].x, data[0].edge_index, data[0].edge_attr, data[0].batch, data[0].temp ,data[0].time ,data[0].metal ,data[0].solvent ,data[0].additive ,data[0].gm ,data[0].elsi ,data[0].add_fea ,data[0].add_fea1, data[0].chirality_fea
        x1, edge_index_x1, edge_attr_x1, batch_x1, chirality_fea1 = data[1].x, data[1].edge_index, data[1].edge_attr, data[1].batch, data[1].chirality_fea
        x2, edge_index_x2, edge_attr_x2, batch_x2, chirality_fea2 = data[2].x, data[2].edge_index, data[2].edge_attr, data[2].batch, data[2].chirality_fea
        x3, edge_index_x3, edge_attr_x3, batch_x3, chirality_fea3 = data[3].x, data[3].edge_index, data[3].edge_attr, data[3].batch, data[3].chirality_fea
        
        xh1, edge_index_xh1, edge_attr_xh1, batch_xh1, num_part_xh1 = data[4].x, data[4].edge_index, data[4].edge_attr, data[4].batch, data[4].num_part
        xh2, edge_index_xh2, edge_attr_xh2, batch_xh2, num_part_xh2 = data[5].x, data[5].edge_index, data[5].edge_attr, data[5].batch, data[5].num_part
        xh3, edge_index_xh3, edge_attr_xh3, batch_xh3, num_part_xh3 = data[6].x, data[6].edge_index, data[6].edge_attr, data[6].batch, data[6].num_part
        xh4, edge_index_xh4, edge_attr_xh4, batch_xh4, num_part_xh4 = data[7].x, data[7].edge_index, data[7].edge_attr, data[7].batch, data[7].num_part
        
        
                                                           
                                                                   
                                                                   
                                                                   
  
        x0 = self.conv1(x, edge_index, edge_attr=edge_attr)
        x10 = self.conv1(x1, edge_index_x1, edge_attr=edge_attr_x1)
        x20 = self.conv1(x2, edge_index_x2, edge_attr=edge_attr_x2)
        x30 = self.conv1(x3, edge_index_x3, edge_attr=edge_attr_x3)
        
        xh1 = self.conv2(xh1, edge_index_xh1, edge_attr=edge_attr_xh1)
        xh2 = self.conv2(xh2, edge_index_xh2, edge_attr=edge_attr_xh2)
        xh3 = self.conv2(xh3, edge_index_xh3, edge_attr=edge_attr_xh3)
        xh4 = self.conv2(xh4, edge_index_xh4, edge_attr=edge_attr_xh4)
                                                     
        bs_xh1 = int(batch_xh1.max().item()) + 1 if batch_xh1.numel() > 0 else 0
        bs_xh2 = int(batch_xh2.max().item()) + 1 if batch_xh2.numel() > 0 else 0
        bs_xh3 = int(batch_xh3.max().item()) + 1 if batch_xh3.numel() > 0 else 0
        bs_xh4 = int(batch_xh4.max().item()) + 1 if batch_xh4.numel() > 0 else 0
        node_rep_x0, motif_node_rep_x0, super_node_rep_x0 = group_node_rep(xh1, bs_xh1, num_part_xh1)
        node_rep_x10, motif_node_rep_x10, super_node_rep_x10 = group_node_rep(xh2, bs_xh2, num_part_xh2)
        node_rep_x20, motif_node_rep_x20, super_node_rep_x20 = group_node_rep(xh3, bs_xh3, num_part_xh3)
        node_rep_x30, motif_node_rep_x30, super_node_rep_x30 = group_node_rep(xh4, bs_xh4, num_part_xh4)
        
        
        motif_rep_x0 = torch.stack(motif_node_rep_x0, dim=0)
        motif_rep_x10 = torch.stack(motif_node_rep_x10, dim=0)
        motif_rep_x20 = torch.stack(motif_node_rep_x20, dim=0)
        motif_rep_x30 = torch.stack(motif_node_rep_x30, dim=0)
        
        super_rep_x0 = torch.stack(super_node_rep_x0, dim=0)
        super_rep_x10 = torch.stack(super_node_rep_x10, dim=0)
        super_rep_x20 = torch.stack(super_node_rep_x20, dim=0)
        super_rep_x30 = torch.stack(super_node_rep_x30, dim=0)
                 
                   
                   
                   
        
                                 
                                                         
                                                                 
                                                                 
                                                                 
                      
                        
                        
                        
        
                                 
                                                                     
                                 
                                                                 
                                                                 
                                 
                                                       
                          
                            
                            
                            

        x, pool_edge_index, pool_edge_weight, pool_batch_x, perm_x ,score_perm_x= self.sagpool1(x0,edge_index,batch=batch)
        x1, pool_edge_index_x1, pool_edge_weight, pool_batch_x1, perm_x1 ,score_perm_x1= self.sagpool1(x10,edge_index_x1,batch=batch_x1)
        x2, pool_edge_index_x2, pool_edge_weight, pool_batch_x2, perm_x2 ,score_perm_x2= self.sagpool1(x20,edge_index_x2,batch=batch_x2)
        x3, pool_edge_index_x3, pool_edge_weight, pool_batch_x3, perm_x3 ,score_perm_x3= self.sagpool1(x30,edge_index_x3,batch=batch_x3)

        
        pool_x, *_ = self.pool1(x0, batch, chirality_fea, use_chiral= True)
        pool_x = F.normalize(pool_x, dim=-1)
        pool_x1, *_ = self.pool1(x10, batch_x1, chirality_fea1, use_chiral= True)
        pool_x1 = F.normalize(pool_x1, dim=-1)
        
        pool_x2, *_ = self.pool1(x20, batch_x2, chirality_fea2, use_chiral= True)
        pool_x2 = F.normalize(pool_x2, dim=-1)
        pool_x3, *_ = self.pool1(x30, batch_x3, chirality_fea3, use_chiral= True)
        pool_x3 = F.normalize(pool_x3, dim=-1)
        
        pool_x1h, *_ = self.pool2(xh1, batch_xh1, chirality_fea, use_chiral= False)
        pool_x1h = F.normalize(pool_x1h, dim=-1)
        pool_x2h, *_ = self.pool2(xh2, batch_xh2, chirality_fea1, use_chiral= False)
        pool_x2h = F.normalize(pool_x2h, dim=-1)
        
        pool_x3h, *_ = self.pool2(xh3, batch_xh3, chirality_fea2, use_chiral= False)
        pool_x3h = F.normalize(pool_x3h, dim=-1)
        pool_x4h, *_ = self.pool2(xh4, batch_xh4, chirality_fea3, use_chiral= False)
        pool_x4h = F.normalize(pool_x4h, dim=-1)
        
                                                                                               
                                                                                                    
                                                                                                    
                                                                                                    
        x = global_mean_pool(x, pool_batch_x)
        x1 = global_mean_pool(x1, pool_batch_x1)
        x2 = global_mean_pool(x2, pool_batch_x2)
        x3 = global_mean_pool(x3, pool_batch_x3)
        
                                         
                                               
                                               
                                              

                                                       
        
        sim_cp = pool_x @ pool_x1.transpose(-1, -2)
        sim_cp = sim_cp.flatten(1)
        
                                                      
                                     
         
        sim_c1 = pool_x @ pool_x2.transpose(-1, -2)
        sim_c1 = sim_c1.flatten(1)
        
        sim_c2 = pool_x @ pool_x3.transpose(-1, -2)
        sim_c2 = sim_c2.flatten(1)
        
        
        
        sim_x12h = pool_x1h @ pool_x2h.transpose(-1, -2)
        sim_x12h = sim_x12h.flatten(1)
        
                                                      
                                     
         
        sim_x13h = pool_x1h @ pool_x3h.transpose(-1, -2)
        sim_x13h = sim_x13h.flatten(1)
        
        sim_x14h = pool_x1h @ pool_x4h.transpose(-1, -2)
        sim_x14h = sim_x14h.flatten(1)
        
                                    
                                                               
                                     
                                     
                                      
                                                                                                                          
       
                                                   
                                                       
                                                         
                                             
                                                 
        
                                                                                                                                                                                              
        temp_x=temp_x.view(len(temp_x),1)
        time_x=time_x.view(len(time_x),1)
        metal_x=metal_x.view(len(metal_x),1)
        solvent_x=solvent_x.view(len(solvent_x),1)
        additive_x=additive_x.view(len(additive_x),1)
        gm_x=gm_x.view(len(gm_x),1)
        elsi_x=elsi_x.view(len(elsi_x),1)
        
        fg_input=add.view(list(add.shape)[0]//2048,2048)
        (
            fg_input,
            super_rep_x0, super_rep_x10, super_rep_x20, super_rep_x30,
            sim_cp, sim_c1, sim_c2, sim_x12h, sim_x13h, sim_x14h,
            x, x1, x2, x3,
            _, _,
        ) = self._apply_ablation(
            fg_input,
            super_rep_x0, super_rep_x10, super_rep_x20, super_rep_x30,
            sim_cp, sim_c1, sim_c2, sim_x12h, sim_x13h, sim_x14h,
            x, x1, x2, x3,
        )
                                                                               
                                                               
                                                                         
                                            
                                            
                                          
                                          
                                                                                                                                                            
        mole_embedding=torch.cat((fg_input, super_rep_x0, super_rep_x10,super_rep_x20, super_rep_x30,sim_cp,sim_c1,sim_c2,sim_x12h,sim_x13h,sim_x14h,x,x1,x2,x3), 1)
                                                                                                                                                           
                                                                                                                        
        z = mole_embedding
                                                           
        if self.fusion_gate_type == "legacy_log_softmax":
            z = torch.mul(F.log_softmax(self.lin8(z), dim=1), z)
        else:
            gate = 0.5 + torch.sigmoid(self.gate_up(F.relu(self.gate_down(z))))
            z = torch.mul(gate, z)
        z1=self.bn1(z)
                                                             
        z1 = self.head_dropout(F.relu(self.lin1(z1)))
        
                                                           
        z2 = self.head_dropout(F.relu(self.lin2(z1)))
                                                    
        output = self.lin3(z2)
                                                                         
                                                                               



                                                                                    
        return F.log_softmax(output, dim=-1), z,z2 , perm_x, score_perm_x, output

    def __repr__(self):
        return self.__class__.__name__
