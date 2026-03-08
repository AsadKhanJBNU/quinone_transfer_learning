import pandas as pd
from sklearn.model_selection import train_test_split
from Data_Prep.Graph_Data import Molecule_data


def prepare_train_test_dataset(df, smilesColumn, labelColumn, savepath):
    """
    Split the dataframe into train and test datasets and convert them
    into graph datasets using the Molecule_data class.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing SMILES and labels.
    smilesColumn : str
        Column name containing SMILES strings.
    labelColumn : str
        Column name containing labels.
    savepath : str
        Directory used by Molecule_data to store processed graph data.

    Returns
    -------
    train_data : Molecule_data
        Graph dataset for training.
    test_data : Molecule_data
        Graph dataset for testing.
    """

    # Split dataset into 80% train and 20% test
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        shuffle=True,
        random_state=42
    )

    # Extract SMILES and labels for training
    train_smiles = train_df[smilesColumn].tolist()
    train_labels = train_df[labelColumn].tolist()

    # Extract SMILES and labels for testing
    test_smiles = test_df[smilesColumn].tolist()
    test_labels = test_df[labelColumn].tolist()

    # Create training graph dataset
    train_data = Molecule_data(
        root='data/' + savepath,
        dataset='train_data_set',
        y=train_labels,
        smiles=train_smiles
    )

    # Create testing graph dataset
    test_data = Molecule_data(
        root='data/' + savepath,
        dataset='test_data_set',
        y=test_labels,
        smiles=test_smiles
    )

    return train_data, test_data