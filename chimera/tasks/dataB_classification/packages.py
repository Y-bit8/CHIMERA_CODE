import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .data_utils import MolGraph

def data_info_cal(df_data):
    y_label=[]
    y=df_data["ddG"]
    for i in range(len(y)):
        if y[i]>90:
            y_label.append(1)

        else:
            y_label.append(0)
    result=[len(y_label),sum(y_label)/len(y_label)]
    return result

          
def y_label_cal(df_data):





    if "target_label" in df_data.columns:
        y_label = list(df_data["target_label"].astype(int))
        y = df_data["ee"] if "ee" in df_data.columns else df_data["ddG"]
        return y_label, y
    y_label=[]
    y=df_data["ee"]
    for i in range(len(y)):
        y_label.append(1 if y[i] >= 90 else 0)
    return y_label,y

      
def stratified_sampling(df_data, stratify, proportion =0.5):
    
    vc = df_data[stratify].value_counts()
    sam = pd.DataFrame(columns = df_data.columns.tolist())
    
    for vi in vc.index:
    
        dd = df_data[df_data[stratify] == vi ].sample(n = round(vc[vi] * proportion))
        sam = pd.concat([sam, dd ], ignore_index = True)
    return sam


def compute_chirality_features(molecule):
    chirality_features = []
    for atom in molecule.GetAtoms():
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            chirality_features.append(1)            
        else:
            chirality_features.append(0)             
    return chirality_features


def one_hot_encoding(x, permitted_list):





    if x not in permitted_list:
        x = permitted_list[-1]

    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

    return binary_encoding

def get_atom_features(atom, 
                      use_chirality = True, 
                      hydrogens_implicit = True):




                                    
    
                                                                                                                                                                                                                                                           
                                                                                                                                                                                                                  
    permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','Al','I', 'B', 'Bi', 'Ir','Yb','Pd','Co','Zn', 'Li','Cu','Ni','Cd','In','Mn','Zr','Cr','Pt','Rh','Pb','Ru','Unknown'] 
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
                           
    
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    
    is_in_a_ring_enc = [int(atom.IsInRing())]
    
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    
                                                                     
    
                                                                                                   
    
                                                                                                               

    atom_feature_vector = atom_type_enc  + n_heavy_neighbors_enc +formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc 
                                    
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    
                                    
                                                                                                        
                                                
        
             
                                                
                                                                                                    
           
                                                                                                    

                                               
                                                                       
          
                                                                       
            
    return np.array(atom_feature_vector)

def get_bond_features(bond, 
                      use_stereochemistry = True):




    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)
def create_pyg(x_smiles, y,temp,time,metal,solvent,additive,gm,elsi,label,add_fea,add_fea1,condition_features):











                                      
                             
                                              
                                                                                   
    data_list = []

           
                  
    for (smiles, y_val,temp_val,time_val,metal_val,solvent_val,additive_val,gm_val,elsi_val,label_val,add_fea_val,add_fea1_val,condition_val) in zip(x_smiles, y,temp,time,metal,solvent,additive,gm,elsi,label,add_fea,add_fea1,condition_features):
        
                                            
        mol = Chem.MolFromSmiles(smiles,sanitize=False)
                               
    
                                                             
                                
                                             
                                 
                                            

                                                                              
        
        chirality_features = compute_chirality_features(mol)

        
                                
        n_nodes = mol.GetNumAtoms()
        n_edges = 2*mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
            
                                                                             
        X = np.zeros((n_nodes, n_node_features))
     
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)
                                    
                                                                                               
            
                  
                                                                                                               
                     
                                       
                            
                    
        X = torch.tensor(X, dtype = torch.float)
        
                                                            
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0)
        
                                                                             
        EF = np.zeros((n_edges, n_edge_features))
        
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        
        EF = torch.tensor(EF, dtype = torch.float)
        
        
                                                                 
                                                                                        
        
        
                                
        y_tensor = torch.tensor(np.array([y_val]), dtype = torch.long)
        
        temp_tensor = torch.tensor(np.array([temp_val]), dtype = torch.long)
        time_tensor = torch.tensor(np.array([time_val]), dtype = torch.long)
        metal_tensor= torch.tensor(np.array([metal_val]), dtype = torch.long)
        solvent_tensor = torch.tensor(np.array([solvent_val]), dtype = torch.long)
        additive_tensor = torch.tensor(np.array([additive_val]), dtype = torch.long)
        gm_tensor = torch.tensor(np.array([gm_val]), dtype = torch.long)
        elsi_tensor = torch.tensor(np.array([elsi_val]), dtype = torch.long)
        label_tensor = torch.tensor(np.array([label_val]), dtype = torch.long)
        add_fea_tensor = torch.tensor(add_fea_val, dtype = torch.float)
                                                                               
        
        add_fea1_tensor = torch.tensor(add_fea1_val, dtype = torch.float)
        condition_tensor = torch.as_tensor(condition_val, dtype=torch.float32).view(1, -1)
                                                          
                                                                                          
                        
        chirality_tensor = torch.tensor(chirality_features, dtype=torch.float32)
                 
        padding_size = 100 - chirality_tensor.size(0)
        
                             
        chirality_tensor = F.pad(chirality_tensor, (0, padding_size), value=0)
                                                                                   
                                                                         
        data_list.append(Data(x = X, edge_index = E, edge_attr = EF, y = y_tensor,temp=temp_tensor,time=time_tensor,metal=metal_tensor,solvent=solvent_tensor,additive=additive_tensor,gm=gm_tensor,elsi=elsi_tensor, label=label_tensor, add_fea=add_fea_tensor, add_fea1=add_fea1_tensor,condition=condition_tensor,chirality_fea=chirality_tensor))
    

    return data_list

def create_pyg_himol(x_smiles):











    data_list = []


    for smiles in x_smiles:
        
        mol_graph = MolGraph(smiles)
        
        
                                
                                                                        
         
                                                                             
                                                                             
         
         
                                                                              
                                                                                   
                                                                                     
                                                                         
                                                                             
        
        
                                                          
                                                                                          
        
                                                                         
        data_list.append(Data(x = mol_graph.x, edge_index = mol_graph.edge_index, edge_attr = mol_graph.edge_attr, num_part=mol_graph.num_part))
    
    return data_list
    
def extract_subsmiles_chiral(smiles,radius=3):
    if not isinstance(smiles, str):
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    chiral_center = None
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() != rdchem.ChiralType.CHI_UNSPECIFIED:
            chiral_center = atom.GetIdx()
            break

    if chiral_center is not None:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, chiral_center)
        atoms_to_use = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, chiral_center)
        submol = Chem.PathToSubmol(mol, atoms_to_use)
        submol_smiles = Chem.MolToSmiles(submol)
        return submol_smiles
    else:
        return None
    
    
    
def extract_subsmiles_metal(smiles):
    metal_atoms = ["Cu", "Fe", "Ni", "Co", "Pd", "Mg", "Ca", "Ru", "Ce"
                  , "In", "Ir", "Cr", "La", "Li", "Mn", "Nd", "Os", "Re"
                  , "Rh", "Sc", "Yb", "Zn"]                    
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    metal_atom_index = None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in metal_atoms:
            metal_atom_index = atom.GetIdx()
            break

    if metal_atom_index is None:
        return None

    radius = 3
    substructure_atoms = set()
    substructure_atoms.add(metal_atom_index)
    for _ in range(radius):
        neighbors = set()
        for atom_index in substructure_atoms:
            atom = mol.GetAtomWithIdx(atom_index)
            neighbors.update([neighbor.GetIdx() for neighbor in atom.GetNeighbors()])
        substructure_atoms.update(neighbors)

    submol_smiles = Chem.MolFragmentToSmiles(mol, list(substructure_atoms), canonical=False, allBondsExplicit=True, allHsExplicit=True)
    return submol_smiles


def pyg_data_generation(df_data,temp,time,metal,solvent,additive,gm,elsi,label,add_fea,add_fea1,condition_features):
    data_list1=[]
    data_list2=[]
    data_list3=[]
    data_list4=[]
    data_listh1=[]
    data_listh2=[]
    data_listh3=[]
    data_listh4=[]
    y_label,y = y_label_cal(df_data)

    data_list1=create_pyg(df_data['ligand'],y_label,temp,time,metal,solvent,additive,gm,elsi,label, add_fea, add_fea1, condition_features)
    data_list2=create_pyg(df_data['product'],y_label,temp,time,metal,solvent,additive,gm,elsi,label, add_fea, add_fea1, condition_features)
    data_list3=create_pyg(df_data['R1'],y_label,temp,time,metal,solvent,additive,gm,elsi,label, add_fea, add_fea1, condition_features)
    data_list4=create_pyg(df_data['R2'],y_label,temp,time,metal,solvent,additive,gm,elsi,label, add_fea, add_fea1, condition_features)
    data_listh1=create_pyg_himol(df_data['ligand'])
    data_listh2=create_pyg_himol(df_data['product'])
    data_listh3=create_pyg_himol(df_data['R1'])
    data_listh4=create_pyg_himol(df_data['R2'])
    
      
    return list(zip(data_list1,data_list2,data_list3,data_list4,data_listh1,data_listh2,data_listh3,data_listh4))
    
    
    
                                  
