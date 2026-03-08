# Graph_Data.py
# Handles molecular graph dataset creation and loading for PyTorch Geometric pipelines.

from rdkit import Chem
import torch
from torch_geometric.data import InMemoryDataset
import os
from torch_geometric.utils import from_smiles
import pandas as pd


# ------------------------------------------------------------
# Molecular Graph Dataset
# ------------------------------------------------------------
class Molecule_data(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for molecular graphs.

    Converts SMILES strings into graph objects using from_smiles(),
    attaches target labels, and caches processed data to disk.

    Args:
        root     : Directory where processed .pt files are saved/loaded.
        dataset  : Name of the processed file (without .pt extension).
        y        : List of target labels (required only during first-time processing).
        smiles   : List of SMILES strings (required only during first-time processing).
        transform: Optional transform applied to each graph at load time.
    """

    def __init__(self, root='/tmp', dataset='davis', y=None,
                 transform=None, pre_transform=None, smiles=None):
        super(Molecule_data, self).__init__(root, transform, pre_transform)
        self.dataset = dataset

        if os.path.isfile(self.processed_paths[0]):
            # Load cached processed graphs from disk
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print(f'Processed data not found at {self.processed_paths[0]}, building graphs...')
            self.process(y, smiles)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # No raw files needed — data comes directly from SMILES strings
        pass

    @property
    def processed_file_names(self):
        # Cached processed file name on disk
        return [self.dataset + '.pt']

    def download(self):
        # No download required
        pass

    def _download(self):
        pass

    def _process(self):
        # Ensure the processed directory exists before saving
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, y, smiles):
        """
        Converts raw SMILES + labels into PyG Data objects and saves to disk.

        Skips molecules with no bonds (empty edge_attr) and logs them
        to 'incorrect_smiles.csv' for inspection.

        Args:
            y      : List/array of target values aligned with smiles.
            smiles : List of SMILES strings.
        """
        data_list       = []
        incorrect_smiles = []

        for i in range(len(y)):
            smile = smiles[i]
            label = y[i]

            # Convert SMILES to PyG graph using RDKit under the hood
            data          = from_smiles(smile)
            data.x        = data.x.type(torch.FloatTensor)
            data.edge_attr = data.edge_attr.type(torch.FloatTensor)

            # Skip molecules with no bonds — these break GNN message passing
            if data.edge_attr.shape[0] == 0:
                incorrect_smiles.append(smile)
                continue

            # Attach target label
            data.y      = torch.FloatTensor([label])
            data.smiles = smile

            # Attach adjacency matrix (used for some downstream analyses)
            mol      = Chem.MolFromSmiles(smile)
            data.Adj = Chem.rdmolops.GetAdjacencyMatrix(mol)

            data_list.append(data)

        # Apply optional pre-filters and pre-transforms
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        # Save skipped molecules for review
        pd.DataFrame(incorrect_smiles, columns=['SMILES']).to_csv('incorrect_smiles.csv', index=False)

        # Collate and save all graphs to a single .pt file
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])