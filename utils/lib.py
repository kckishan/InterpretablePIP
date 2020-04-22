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


def edges_to_sparse(edges, N, values=None):
    """
    Create a sparse adjacency matrix from an array of edge indices and (optionally) values.

    Parameters
    ----------
    edges : array-like, shape [n_edges, 2]
        Edge indices
    N : int
        Number of nodes
    values : array_like, shape [n_edges]
        The values to put at the specified edge indices. Optional, default: np.ones(.)

    Returns
    -------
    A : scipy.sparse.csr.csr_matrix
        Sparse adjacency matrix

    """
    if values is None:
        values = np.ones(edges.shape[0])

    return sp.coo_matrix((values, (edges[:, 0], edges[:, 1])), shape=(N, N)).tocsr()


def train_val_test_split_adjacency(A, p_val=0.10, p_test=0.05, seed=0, neg_mul=1,
                                   every_node=True, connected=False, undirected=False,
                                   use_edge_cover=True, set_ops=True, asserts=False):
    """Split the edges of the adjacency matrix into train, validation and test edges
    and randomly samples equal amount of validation and test non-edges.
    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse unweighted adjacency matrix
    p_val : float
        Percentage of validation edges. Default p_val=0.10
    p_test : float
        Percentage of test edges. Default p_test=0.05
    seed : int
        Seed for numpy.random. Default seed=0
    neg_mul : int
        What multiplicity of negative samples (non-edges) to have in the test/validation set
        w.r.t the number of edges, i.e. len(non-edges) = L * len(edges). Default neg_mul=1
    every_node : bool
        Make sure each node appears at least once in the train set. Default every_node=True
    connected : bool
        Make sure the training graph is still connected after the split
    undirected : bool
        Whether to make the split undirected, that is if (i, j) is in val/test set then (j, i) is there as well.
        Default undirected=False
    use_edge_cover: bool
        Whether to use (approximate) edge_cover to find the minimum set of edges that cover every node.
        Only active when every_node=True. Default use_edge_cover=True
    set_ops : bool
        Whether to use set operations to construction the test zeros. Default setwise_zeros=True
        Otherwise use a while loop.
    asserts : bool
        Unit test like checks. Default asserts=False
    Returns
    -------
    train_ones : array-like, shape [n_train, 2]
        Indices of the train edges
    val_ones : array-like, shape [n_val, 2]
        Indices of the validation edges
    val_zeros : array-like, shape [n_val, 2]
        Indices of the validation non-edges
    test_ones : array-like, shape [n_test, 2]
        Indices of the test edges
    test_zeros : array-like, shape [n_test, 2]
        Indices of the test non-edges
    """
    assert p_val + p_test > 0
    assert A.max() == 1  # no weights
    assert A.min() == 0  # no negative edges
    assert A.diagonal().sum() == 0  # no self-loops
    assert not np.any(A.sum(0).A1 + A.sum(1).A1 == 0)  # no dangling nodes

    is_undirected = (A != A.T).nnz == 0

    if undirected:
        assert is_undirected  # make sure is directed
        A = sp.tril(A).tocsr()  # consider only upper triangular
        A.eliminate_zeros()
    else:
        if is_undirected:
            warnings.warn('Graph appears to be undirected. Did you forgot to set undirected=True?')

    np.random.seed(seed)

    E = A.nnz
    N = A.shape[0]
    s_train = int(E * (1 - p_val - p_test))
    
    idx = np.arange(N)

    # hold some edges so each node appears at least once
    if every_node:
        if connected:
            # make sure original graph is connected
            assert sp.csgraph.connected_components(A)[0] == 1
            A_hold = sp.csgraph.minimum_spanning_tree(A)
        else:
            A.eliminate_zeros()  # makes sure A.tolil().rows contains only indices of non-zero elements
            d = A.sum(1).A1

            if use_edge_cover:
                hold_edges = edge_cover(A)

                # make sure the training percentage is not smaller than len(edge_cover)/E when every_node is set to True
                min_size = hold_edges.shape[0]
                if min_size > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
                                     .format(min_size / E))
            else:
                # make sure the training percentage is not smaller than N/E when every_node is set to True
                if N > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
                                     .format(N / E))

                hold_edges_d1 = np.column_stack(
                    (idx[d > 0], np.row_stack(map(np.random.choice, A[d > 0].tolil().rows))))

                if np.any(d == 0):
                    hold_edges_d0 = np.column_stack((np.row_stack(map(np.random.choice, A[:, d == 0].T.tolil().rows)),
                                                     idx[d == 0]))
                    hold_edges = np.row_stack((hold_edges_d0, hold_edges_d1))
                else:
                    hold_edges = hold_edges_d1

            if asserts:
                assert np.all(A[hold_edges[:, 0], hold_edges[:, 1]])
                assert len(np.unique(hold_edges.flatten())) == N

            A_hold = edges_to_sparse(hold_edges, N)

        A_hold[A_hold > 1] = 1
        A_hold.eliminate_zeros()
        A_sample = A - A_hold

        s_train = s_train - A_hold.nnz
    else:
        A_sample = A

    idx_ones = np.random.permutation(A_sample.nnz)
    ones = np.column_stack(A_sample.nonzero())
    train_ones = ones[idx_ones[:s_train]]
    test_ones = ones[idx_ones[s_train:]]

    # return back the held edges
    if every_node:
        train_ones = np.row_stack((train_ones, np.column_stack(A_hold.nonzero())))

    n_test = len(ones) * neg_mul
    if set_ops:
        # generate slightly more completely random non-edge indices than needed and discard any that hit an edge
        # much faster compared a while loop
        # in the future: estimate the multiplicity (currently fixed 1.3/2.3) based on A_obs.nnz
        if undirected:
            random_sample = np.random.randint(0, N, [int(2.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] > random_sample[:, 1]]
        else:
            random_sample = np.random.randint(0, N, [int(1.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] != random_sample[:, 1]]

        # discard ones
        random_sample = random_sample[A[random_sample[:, 0], random_sample[:, 1]].A1 == 0]
        # discard duplicates
        random_sample = random_sample[np.unique(
            random_sample[:, 0] * N + random_sample[:, 1], return_index=True)[1]]
        # only take as much as needed
        test_zeros = np.row_stack(random_sample)[:n_test]
        assert test_zeros.shape[0] == n_test
    else:
        test_zeros = []
        while len(test_zeros) < n_test:
            i, j = np.random.randint(0, N, 2)
            if A[i, j] == 0 and (not undirected or i > j) and (i, j) not in test_zeros:
                test_zeros.append((i, j))
        test_zeros = np.array(test_zeros)

    # split the test set into validation and test set
    s_val_ones = int(len(test_ones) * p_val / (p_val + p_test))
    s_val_zeros = int(len(test_zeros) * p_val / (p_val + p_test))
    s_train_zeros = len(train_ones)

    val_ones = test_ones[:s_val_ones]
    test_ones = test_ones[s_val_ones:]

    train_zeros = test_zeros[:s_train_zeros]
    test_zeros = test_zeros[s_train_zeros:]

    val_zeros = test_zeros[:s_val_ones]
    test_zeros = test_zeros[s_val_ones:]
     
    if undirected:
        # put (j, i) edges for every (i, j) edge in the respective sets and form back original A
        def symmetrize(x): return np.row_stack((x, np.column_stack((x[:, 1], x[:, 0]))))
        train_ones = symmetrize(train_ones)
        train_zeros = symmetrize(train_zeros)
        val_ones = symmetrize(val_ones)
        val_zeros = symmetrize(val_zeros)
        test_ones = symmetrize(test_ones)
        test_zeros = symmetrize(test_zeros)
        A = A.maximum(A.T)

    if asserts:
        set_of_train_ones = set(map(tuple, train_ones))
        set_of_train_zeros = set(map(tuple, train_zeros))
        assert train_ones.shape[0] + test_ones.shape[0] + val_ones.shape[0] == A.nnz
        assert (edges_to_sparse(np.row_stack((train_ones, test_ones, val_ones)), N) != A).nnz == 0
        assert set_of_train_ones.intersection(set(map(tuple, test_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, test_zeros))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_zeros))) == set()
        assert set_of_train_zeros.intersection(set(map(tuple, test_ones))) == set()
        assert set_of_train_zeros.intersection(set(map(tuple, val_ones))) == set()
        assert set_of_train_zeros.intersection(set(map(tuple, test_zeros))) == set()
        assert set_of_train_zeros.intersection(set(map(tuple, val_zeros))) == set()
        assert len(set(map(tuple, train_zeros))) == len(train_ones) * neg_mul
        assert len(set(map(tuple, test_zeros))) == len(test_ones) * neg_mul
        assert len(set(map(tuple, val_zeros))) == len(val_ones) * neg_mul
        assert not connected or sp.csgraph.connected_components(A_hold)[0] == 1
        assert not every_node or ((A_hold - A) > 0).sum() == 0

    dataset = {}
    dataset['train_edges'] = train_ones
    dataset['train_edges_false'] = train_zeros
    dataset['val_edges'] = val_ones
    dataset['val_edges_false'] = val_zeros
    dataset['test_edges'] = test_ones
    dataset['test_edges_false'] = test_zeros

    return dataset

def edge_cover(A):
    """
    Approximately compute minimum edge cover.

    Edge cover of a graph is a set of edges such that every vertex of the graph is incident
    to at least one edge of the set. Minimum edge cover is an  edge cover of minimum size.

    Parameters
    ----------
    A : sp.spmatrix
        Sparse adjacency matrix

    Returns
    -------
    edges : array-like, shape [?, 2]
        The edges the form the edge cover
    """

    N = A.shape[0]
    d_in = A.sum(0).A1
    d_out = A.sum(1).A1

    # make sure to include singleton nodes (nodes with one incoming or one outgoing edge)
    one_in = np.where((d_in == 1) & (d_out == 0))[0]
    one_out = np.where((d_in == 0) & (d_out == 1))[0]

    edges = []
    edges.append(np.column_stack((A[:, one_in].argmax(0).A1, one_in)))
    edges.append(np.column_stack((one_out, A[one_out].argmax(1).A1)))
    edges = np.row_stack(edges)

    edge_cover_set = set(map(tuple, edges))
    nodes = set(edges.flatten())

    # greedly add other edges such that both end-point are not yet in the edge_cover_set
    cands = np.column_stack(A.nonzero())
    for u, v in cands[d_in[cands[:, 1]].argsort()]:
        if u not in nodes and v not in nodes and u != v:
            edge_cover_set.add((u, v))
            nodes.add(u)
            nodes.add(v)
        if len(nodes) == N:
            break

    # add a single edge for the rest of the nodes not covered so far
    not_covered = np.setdiff1d(np.arange(N), list(nodes))
    edges = [list(edge_cover_set)]
    not_covered_out = not_covered[d_out[not_covered] > 0]

    if len(not_covered_out) > 0:
        edges.append(np.column_stack((not_covered_out, A[not_covered_out].argmax(1).A1)))

    not_covered_in = not_covered[d_out[not_covered] == 0]
    if len(not_covered_in) > 0:
        edges.append(np.column_stack((A[:, not_covered_in].argmax(0).A1, not_covered_in)))

    edges = np.row_stack(edges)

    # make sure that we've indeed computed an edge_cover
    assert A[edges[:, 0], edges[:, 1]].sum() == len(edges)
    assert len(set(map(tuple, edges))) == len(edges)
    assert len(np.unique(edges)) == N

    return edges


def generate_train_test_pairs(A, ind2protname, p_val=0.10, p_test=0.05, neg_mul=1):
    """
    Split the edges of the adjacency matrix into train, validation and test edges
    and randomly samples equal amount of validation and test non-edges.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse unweighted adjacency matrix

    Returns
    -------
    train_pairs : array-like, shape [n_train, 3]
        Training pairs
    val_pairs : array-like, shape [n_val, 3]
        Validation pairs
    test_pairs : array-like, shape [n_val, 3]
        Test pairs
    """

    train_ones, train_zeros, val_ones, val_zeros, test_ones, test_zeros = train_val_test_split_adjacency(A, p_val=p_val,
                                                                                                         p_test=p_test,
                                                                                                         seed=0,
                                                                                                         neg_mul=neg_mul,
                                                                                                         every_node=True,
                                                                                                         connected=False,
                                                                                                         undirected=True,
                                                                                                         use_edge_cover=True,
                                                                                                         set_ops=True,
                                                                                                         asserts=True)
    print("Interactions Stats")
    print("Train:", train_ones.shape, train_zeros.shape)
    print("Val:", val_ones.shape, val_zeros.shape)
    print("Test:", test_ones.shape, test_zeros.shape)

    train_ones = convert_ind_to_protein_pairs(train_ones, ind2protname)
    train_zeros = convert_ind_to_protein_pairs(train_zeros, ind2protname)
    val_ones = convert_ind_to_protein_pairs(val_ones, ind2protname)
    val_zeros = convert_ind_to_protein_pairs(val_zeros, ind2protname)
    test_ones = convert_ind_to_protein_pairs(test_ones, ind2protname)
    test_zeros = convert_ind_to_protein_pairs(test_zeros, ind2protname)

    train_ones = np.column_stack((train_ones, np.ones((len(train_ones))).astype(int)))
    val_ones = np.column_stack((val_ones, np.ones((len(val_ones))).astype(int)))
    test_ones = np.column_stack((test_ones, np.ones((len(test_ones))).astype(int)))
    train_zeros = np.column_stack((train_zeros, np.zeros((len(train_zeros))).astype(int)))
    val_zeros = np.column_stack((val_zeros, np.zeros((len(val_zeros))).astype(int)))
    test_zeros = np.column_stack((test_zeros, np.zeros((len(test_zeros))).astype(int)))

    train_pairs = np.vstack((train_ones, train_zeros))
    val_pairs = np.vstack((val_ones, val_zeros))
    test_pairs = np.vstack((test_ones, test_zeros))

    return train_pairs, val_pairs, test_pairs


def get_buckets(arr, diff_size=3):
    """
    Split an array into buckets based on the difference in values

    Parameters
    ----------
    arr : array-like, shape [?]
        An array of values
    diff_size : int
        The threshold to split the array
    Returns
    -------
    list : list
        list of indices of the bucket array
    """

    ind = np.argsort(-arr)
    x = arr[ind]
    return np.split(ind, np.where((-1. * np.diff(x)) > diff_size)[0] + 1)



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