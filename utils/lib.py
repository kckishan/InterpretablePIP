import numpy as np
import torch
import scipy.sparse as sp
import warnings
import itertools
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
from texttable import Texttable
from prettytable import PrettyTable

def table_printer(args):
    """
    Print the parameters of the model in a Tabular format
    Parameters
    ---------
    args: argparser object
        The parameters used for the model
    """
    tab = PrettyTable()
    args = vars(args)
    tab.field_names = ["Parameters", "Default"]
    for k, v in args.items():
        tab.add_row([k, v])

    print(tab)
    return tab


def get_batches(X, batch_size):
    """
    Create the batches of X, each batch with size batch_size
    Parameters
    ----------
    X : array-like, shape [?, 3]
        The interactions array
    batch_size : number
        size of the batch
    Returns
    -------
    x : array-like [?, 2]
        interactions data
    y : array-like [?, 1]
        ground-truth of the interactions
    """
    for n in range(0, X.shape[0], batch_size):
        x = X[n:n + batch_size, :2]
        y = X[n:n + batch_size, 2]
        yield x, y.astype(int)


def get_subset_tensor(seq_tensor, lengths, arr, protname2index):
    """
    Create the subset of sequence tensor for proteins that are in arr
    Parameters
    ----------
    seq_tensor : array-like, shape [N, max_seq_length, dim]
        The tensor that contains vector representation for the amino acid sequence of all the proteins
    lengths : array-like, shape [N]
        The list of lengths of the sequences
    arr : array-like, [?, 2]
        interactions between the proteins
    protname2index : dict
        The dictionary that maps protein name to index
    Returns
    -------
    seq_array : array-like [?, max_seq_length, dim]
        Tensor of the sequence representation for proteins in arr
    lengths_array : array-like [?, 1]
        The lengths of the protein sequences in arr
    protid2idx : dict
        The dictionary that maps the id from protname2index to new sorted index
    """
    proteins = np.unique(arr[:, :2])
    seq_list = []
    lengths_list = []

    for i in list(proteins):
        # print(protname2index[i], seq_tensor[protname2index[i]])
        seq_list.append(seq_tensor[protname2index[i]])
        lengths_list.append(lengths[protname2index[i]])

    seq_array = np.array(seq_list)
    lengths_array = np.array(lengths_list)

    indices = np.argsort(-lengths_array)
    lengths_array = lengths_array[indices]
    seq_array = seq_array[indices]

    # create a dictionary to map protein name with sorted index
    protid2idx = {proteins[indices[i]]: i for i in np.arange(len(indices))}
    return seq_array, lengths_array, protid2idx


def gather(mat, dim, index):
    """
    Gather the rows/columns from tensor(based on dim) indexed with index
    Parameters
    ----------
    mat : array-like, shape [?, latent_dim]
        The tensor that contains representations for the sequences
    dim : number
        The value that corresponds to the dimension of mat
    index : array-like, [?, 1]
        The array of indices
    Returns
    -------
    val : array-like [?, latent_dim]
        The tensor that contains the indexed values from mat
    """
    val = torch.index_select(mat, dim, index)
    return val


def _wasserstein(mu_i, sigma_i, mu_j, sigma_j):
    """
    Computes the energy between a set of node pairs as the 2^th Wasserstein distance between their respective Gaussian embeddings.
    Parameters
    ----------
    mu_i : array-like, shape [?, latent_dim]
        The mean of gaussian embeddings of i^th proteins
    mu_j : array-like, shape [?, latent_dim]
        The mean of gaussian embeddings of j^th proteins
    sigma_i : array-like, shape [?, latent_dim]
        The variance of gaussian embeddings of i^th proteins
    sigma_j : array-like, shape [?, latent_dim]
        The variance of gaussian embeddings of j^th proteins
    Returns
    -------
    wd : array-like [?]
        The energy between each pair given the currently trained model
    """
    delta = mu_i - mu_j
    d1 = torch.sum(delta * delta, dim=1)

    x0 = sigma_i - sigma_j
    d2 = torch.sum(x0 * x0, dim=1)

    wd = d1 + d2
    return wd


def get_energy(mu, sigma, indx_i, indx_j):
    """
    Computes the energy between a set of node pairs as the 2^th Wasserstein distance between their respective Gaussian embeddings.
    Parameters
    ----------
    mu : array-like, shape [?, latent_dim]
        The mean of gaussian embeddings of proteins
    sigma : array-like, shape [?, latent_dim]
        The variance of gaussian embeddings of proteins
    indx_i : array-like, [?, 1]
        The array of indices
    indx_j : array-like, [?, 1]
        The array of indices
    Returns
    -------
    wd : array-like [?]
        The energy of each pair (indx_i, indx_j) given the currently trained model
    """
    mu_i = gather(mu, 0, indx_i)
    mu_j = gather(mu, 0, indx_j)

    sigma_i = gather(sigma, 0, indx_i)
    sigma_j = gather(sigma, 0, indx_j)

    energy = _wasserstein(mu_i, sigma_i, mu_j, sigma_j)
    return energy


def get_energy_pairs(mu_batch, sigma_batch, idx2sortedidx, batch, device):
    """
    Wrapper function for get_energy
    Parameters
    ----------
    mu_batch : array-like, shape [?, latent_dim]
       The mean of gaussian embeddings of proteins
    sigma_batch : array-like, shape [?, latent_dim]
       The variance of gaussian embeddings of proteins
    idx2sortedidx : array-like, [?, 1]
       The dictionary that maps the batch proteins with index
    batch : array-like, [?, 2]
       The pair of interactions in a batch
    device : string
        The device type: CPU or GPU as returned by torch.device
    Returns
    -------
    energy : array-like [?]
       The energy between all pairs given the currently trained model
    """
    indx_i = torch.from_numpy(np.array([idx2sortedidx[i] for i in batch[:, 0]])).to(device)
    indx_j = torch.from_numpy(np.array([idx2sortedidx[j] for j in batch[:, 1]])).to(device)
    return get_energy(mu_batch, sigma_batch, indx_i, indx_j)


def compute_loss(mu, sigma, pairs, label, idx2sortedidx, device):
    """
    Computes the loss to train the model
    Parameters
    ----------
    mu : array-like, shape [?, latent_dim]
       The mean of gaussian embeddings of proteins
    sigma : array-like, shape [?, latent_dim]
       The variance of gaussian embeddings of proteins
    pairs : array-like, [?, 2]
       The pair of interactions in a batch
    label: array-like, [?, 1]
        The ground truth label for the interaction pairs
    idx2sortedidx : array-like, [?, 1]
       The dictionary that maps the batch proteins with index
    device : string
        The device type: CPU or GPU as returned by torch.device
    Returns
    -------
    loss : float
       The loss of the model for given pairs of interactions
    """
    pos_batch = pairs[label == 1]
    neg_batch = pairs[label == 0]

    pos_energy = get_energy_pairs(mu, sigma, idx2sortedidx, pos_batch, device)
    neg_energy = get_energy_pairs(mu, sigma, idx2sortedidx, neg_batch, device)

    return torch.mean((pos_energy) ** 2) + torch.mean(torch.exp(-neg_energy))


def score_link_prediction(labels, scores):
    """
    Calculates the area under the ROC curve and the average precision score.
    Parameters
    ----------
    labels : array-like, shape [N]
        The ground truth labels
    scores : array-like, shape [N]
        The (unnormalized) scores of how likely are the instances
    Returns
    -------
    roc_auc : float
        Area under the ROC curve score
    ap : float
        Average precision score
    """

    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def show_plot(iteration, data):
    """
    Draw a line chart
    Parameters
    ----------
    iteration : array-like
        The values for x-axis
    data : array-like
        values for y-axis
    """
    plt.plot(iteration, data)
    plt.show()


def convert_ind_to_protein_pairs(ind_pairs, index2protein):
    """
    Convert the pairs of protein id to pairs of protein name
    Parameters
    ----------
    ind_pairs : array-like, shape [?, 2]
        The pairs of protein indices
    index2protein : dict
        The dictionary that maps protein index to protein name
    Returns
    -------
    prot_pairs : array-like, shape [?, 2]
        The pairs of protein names
    """
    prot_pairs = []
    for i in ind_pairs:
        prot_pairs.append([index2protein[i[0]], index2protein[i[1]]])
    return np.vstack(prot_pairs)

def get_seq_batches(X, lens, batch_size):
    """
    Create the batches of X, each batch with size batch_size
    Parameters
    ----------
    X : array-like, shape [?, 3]
        The interactions array
    batch_size : number
        size of the batch
    Returns
    -------
    x : array-like [?, 2]
        interactions data
    y : array-like [?, 1]
        ground-truth of the interactions
    """
    for n in range(0, X.shape[1], batch_size):
        x = X[:, n:n + batch_size]
        l = lens[n:n + batch_size]
        yield x, l
