import numpy as np
import pandas as pd
from tqdm import tqdm
from .seq2tensor import s2t

class DataLoader():
    """

    """
    def __init__(self, embedding_file, sequence_file, interactions_file, max_seq_length: int):
        """
        Parameters
        ----------
        embedding_file : string
            path to the file that contains the embedding for each amino acid
        sequence_file : string
            path to the file that contains protein names and the respective sequences [protein_name, sequence]
        interactions_file : string
            path to the file that contains interactions [protein A, protein B, interaction]
        max_seq_length : string
        """

        self.embedding_file = embedding_file
        self.sequence_file = sequence_file
        self.interactions_file = interactions_file
        self.max_seq_length = max_seq_length

        if not self.embedding_file is None:
            self.seq2t = s2t(self.embedding_file)
        self.dim = self.seq2t.dim

        # load sequences and a dictionary that maps protein names with index
        self.seqs, self.protname2index = self.load_sequences()


    def load_sequences(self):
        """
        Loads the protein sequences from sequence_file

        Returns
        -------
        seqs : list, shape [N]
            The list of the sequences
        protname2index : dict
            The dictionary that contains mapping between protein name and index
        """
        id2seq_df = pd.read_csv(self.sequence_file, sep="\t", header=None)
        # id2seq_df = pd.read_pickle(self.sequence_file)
        # print(id2seq_df.head())
        protname2index = {}
        seqs = []
        index = 0
        for row_num in range(id2seq_df.shape[0]):
            row = id2seq_df.iloc[row_num,:]
            # print(row)
            protname2index[row[1]] = index
            seqs.append(row[2])
            index += 1
        return seqs, protname2index

    def load_interactions(self):
        """
        Loads the interactions between proteins
        Returns
        -------
        interactions : array-like, shape [?, 3]
            The interactions between proteins
        """
        interactions_df = pd.read_csv(self.interactions_file, sep="\t", header=None)
        return interactions_df.values

    def convert_seq_to_tensor(self):
        """
        Convert raw amino acid sequence to tensor to pass it as input to the model

        Returns
        -------
        seq_tensor : array-like, shape [N, max_seq_length, num_amino_acid]
            Description
        lengths :  array-like, shape [N]
            The list of lengths of sequences
        """
        seq_tensor = []
        lengths = []
        for line in tqdm(self.seqs):
            seq, lens = self.seq2t.embed_normalized(line, self.max_seq_length)
            seq_tensor.append(seq)
            lengths.append(lens)
        seq_tensor = np.array(seq_tensor)
        lengths = np.array(lengths)

        return seq_tensor, lengths