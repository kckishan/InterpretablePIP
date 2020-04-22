import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import autograd as ta
from torchsparseattn import *


class Gates(nn.Module):

    def __init__(self, att_size, gate_type="sparsemax"):
        """
        Parameters
        ----------
        att_size : number
            The dimension of input vector
        gate_type : string
            The type of the gate
        """

        super(Gates, self).__init__()
        self.att_w = nn.Conv1d(att_size, 1, 1)
        if gate_type == "sparsemax":
            self.gate = Sparsemax()
        elif gate_type == "fusedmax":
            self.gate = Fusedmax()

    def forward(self, input, len_s):
        """
        Computes the sprase gate vector

        Parameters
        ----------
        input : tensor, shape [?, max_length, att_size]
            The tensor that represents the amino acid sequences
        len_s : tensor, shape [?]
            The tensor that contains lengths of the amino acid sequences
        Returns
        -------
        out : tensor, shape [?, max_length]
            The tensor with gate values for each position of amino acid sequence
        """
        input = input.permute(1, 2, 0)
        att = self.att_w(input).squeeze(1)
        lengths = ta.Variable(len_s).type(torch.LongTensor)
        out = self.gate(att.cpu(), lengths).unsqueeze(2)
        out = out.permute(1, 0, 2)
        return out


class Sequence2Gauss(nn.Module):
    def __init__(self, input_dim, emb_dim, rnn_hidden_dim, latent_dim, max_len, gate_type="sparsemax"):
        """
        Parameters
        ----------
        input_dim : number
            The dimension of input vector
        emb_dim : number
            The dimension of embedding
        rnn_hidden_dim : number
            The dimension of hidden representation of RNN (GRU)
        latent_dim : number
            The dimension of latent Gaussian embeddings
        max_len : number
            The maximeanm length of the input sequence considered
        gate_type : string
            The type of the gate
        """
        super(Sequence2Gauss, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.rnn_hidden_dim = rnn_hidden_dim
        self.latent_dim = latent_dim
        self.max_len = max_len

        # Define embedding layer to embed on-hot encoded representation of amino acids to dense vector
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        # Define Bidirectional GRU to read sequence from left to right and right to left
        self.rnn = nn.GRU(emb_dim, self.rnn_hidden_dim, bidirectional=True)

        self.lin = nn.Linear(self.rnn_hidden_dim * 2, self.rnn_hidden_dim * 2)
        self.gating = Gates(self.rnn_hidden_dim * 2, gate_type=gate_type)

        self.mean_encoder = nn.Linear(self.rnn_hidden_dim * 2 * (max_len), self.latent_dim)
        self.var_encoder = nn.Linear(self.rnn_hidden_dim * 2 * (max_len), self.latent_dim)

        self.init_weight(self.lin)
        self.init_weight(self.mean_encoder)
        self.init_weight(self.var_encoder)

    def forward(self, input_sequence, lengths):
        """
        Generates latent Gaussian embeddings for sequence

        Parameters
        ----------
        input_sequence : tensor, shape [?, max_length, att_size]
            The tensor that represents the input amino acid sequences
        lengths : tensor, shape [?]
            The tensor that contains lengths of the input amino acid sequences
        Returns
        -------
        mean : tensor, shape [?, latent_dim]
            The mean of latent gaussian distributions of the input proteins
        var : tensor, shape [?, latent_dim]
            The variance of latent gaussian distributions of the input proteins
        gate : tensor, shape [?, max_length]
            The sparse gate values for each position of amino acid sequence
        """

        # Embed the sequence from amino acids to dense vector
        input_embedding = self.embedding(input_sequence)
        # Pack the sequences based on lengths to allow the model to adapt to variable-length sequences
        packed_seq = pack_padded_sequence(input_embedding, lengths)
        output, _ = self.rnn(packed_seq)

        # pad the output from RNN layer
        output, len_s = pad_packed_sequence(output)

        # compute the gate vector based on RNN output
        emb_h = torch.tanh(self.lin(output))
        self.gate = self.gating(emb_h, len_s).to(self.device)

        # Get the elementwise multiplication of gates with RNN output
        attended = (self.gate * output).permute(1, 0, 2)
        out = torch.zeros(attended.size(0), self.max_len, self.rnn_hidden_dim * 2).to(self.device)
        out[:, :attended.size(1), :] = attended
        attended = out.contiguous().view(attended.size(0), -1)

        # encode the sparse representation to mean and variance
        self.mean = self.mean_encoder(attended)
        self.var = F.elu(self.var_encoder(attended)) + 1 # elu() + 1 ensures the covariance matrix is positive definite.

        return self.mean, self.var, self.gate

    def init_weight(self, layer):
        """
        Initialize the layer with Xavier initialization

        Parameters
        ----------
        layer : Neural network layer
        """
        nn.init.xavier_uniform_(layer.weight)
