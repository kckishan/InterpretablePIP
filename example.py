from sklearn.model_selection import train_test_split
from torch import optim
from scipy.sparse import csr_matrix
from utils.DataLoader import DataLoader
import pandas as pd
from model.s2g import *
from utils.lib import *
from model.trainer import trainer, evaluate
import warnings
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description="Sequence2Gauss")
parser.add_argument("--dataset", type=str, default='human',
					help="yeast or human")
parser.add_argument("--gate-type", type=str, default="fusedmax",
                    help="Type of gating mechanism")
parser.add_argument("--test", action="store_true", 
                    help="Test the saved model")
parser.add_argument("--max-length", type=int, default=1024,
                    help="Maximum length of sequence")

args = parser.parse_args()
table = table_printer(args)

dataset =args.dataset
# file that contains embedding for each amino acid
embedding_file = "data/embeddings/embeddings.pkl"

# file that contains protein name and sequence
id2seq_file = "data/processed/subset_processed"+dataset+".tsv"

# file that contains interaction dataset
interactions_file = "data/processed/"+dataset+"_interactions_all.tsv"


# check the device type: CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on ", device)

# load sequences, and interactions data
dataloader = DataLoader(embedding_file, id2seq_file, interactions_file, max_seq_length=args.max_length)

seq2t = dataloader.seq2t

# number of amino acids in the sequences
# add the padding index
dim = dataloader.dim + 1

# a dictionary that maps protein name to index
protname2index = dataloader.protname2index

# a dictionary that maps index to protein name
ind2protname = {v: k for k, v in protname2index.items()}

# convert the amino acid sequence to numeric tensor
seq_tensor, lengths = dataloader.convert_seq_to_tensor()
print("Shape of the tensor", seq_tensor.shape)

# load the interactions from interactions file
interactions = dataloader.load_interactions()
pairs = interactions[interactions[:, 0] != interactions[:, 1]]  # remove self-interactions
print("Total number of interactions:", interactions.shape)

# Define the parameters for the model
latent_dim = 128
embedding_dim = 20
n_hidden = 16
clip = 0.5
epochs = 50

gate_type = args.gate_type

# create the Sequence2Gauss model
model = None
model = Sequence2Gauss(input_dim=dim, emb_dim=embedding_dim, rnn_hidden_dim=n_hidden,
                       latent_dim=latent_dim, max_len=args.max_length, gate_type=gate_type)
print(model)
model.to(device)

# name to save the modelß
model_name=dataset+"_"+gate_type

# If args.test is True, evaluates the saved model
if not args.test:
    # Split the interactions to training and testing
    train_pairs, test_pairs = train_test_split(pairs, test_size=0.2)
    train_pairs, val_pairs = train_test_split(train_pairs, test_size=0.2)
    print(train_pairs.shape, val_pairs.shape, test_pairs.shape)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # define data dictionary to pass
    data_dict = {'train_pairs': train_pairs,
                 'val_pairs': val_pairs,  # replace it with validation pairs
                 'seq_tensor': seq_tensor,
                 'lengths': lengths,
                 'protname2index': protname2index}

    # train the model in training interactions
    trainer(model, optimizer, data_dict, device, model_name=dataset+"_"+gate_type,epochs=epochs, clip=clip)

    # evaluate the model in testing interactions
    test_auc, test_ap = evaluate(model, seq_tensor, lengths, test_pairs, protname2index, device)
    print(" Ranking: Test AUC: {:.06f} Test AP: {:.06f} \n".format(test_auc, test_ap))

print("Loading saved model")
model.load_state_dict(torch.load('trained_model/'+model_name+'.pkl'))
# get the dataset for the proteins in pairs
seq, lens, id2idx = get_subset_tensor(seq_tensor, lengths, pairs, protname2index)
seq = torch.from_numpy(seq).transpose(0, 1).type(torch.LongTensor)
lens = torch.from_numpy(lens)


# Turn evaluation mode
with torch.no_grad():
    model.eval()
    # Get the representations for proteins
    mu_list = []
    sigma_list = []
    attn_list = []
    for s, l in get_seq_batches(seq, lens, 256):
        # load tensor to device
        s, l = s.to(device), l.to(device)
        mu, sigma, attn = model(s, l)
        mu = mu.cpu().detach()
        sigma = sigma.cpu().detach()
        attn = attn.cpu().detach()
        mu_list.append(mu)
        sigma_list.append(sigma)
        attn_list.append(attn)

    mu = torch.cat(mu_list, 0)
    sigma = torch.cat(sigma_list, 0)
    
# save_data = {}
# save_data['attn'] = attn_list
# save_data['lengths'] = lens
# save_data['id2idx'] = id2idx
# torch.save(save_data, "attn/"+dataset+"_"+gate_type+"_attn.pt")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

Y = np.column_stack((list(map(id2idx.__getitem__, train_pairs[:, 0])), list(
    map(id2idx.__getitem__, train_pairs[:, 1])), train_pairs[:, 2]))
Y = Y.astype(int)
train_labels = Y[:, 2]

train_mu_i = mu[Y[:, 0], ]
train_mu_j = mu[Y[:, 1], ]
train_sigma_i = sigma[Y[:, 0], ]
train_sigma_j = sigma[Y[:, 1], ]
feat = torch.cat((torch.abs(train_mu_i - train_mu_j), torch.abs(train_sigma_i - train_sigma_j)), dim=1)
feat = torch.cat((feat, train_mu_i * train_mu_j), dim=1)
feat = feat.numpy()

test_Y = np.column_stack((list(map(id2idx.__getitem__, test_pairs[:, 0])), list(
    map(id2idx.__getitem__, test_pairs[:, 1])), test_pairs[:, 2]))

test_Y = test_Y.astype(int)

test_mu_i = mu[test_Y[:, 0], ]
test_mu_j = mu[test_Y[:, 1], ]
test_sigma_i = sigma[test_Y[:, 0], ]
test_sigma_j = sigma[test_Y[:, 1], ]

test_feat = torch.cat((torch.abs(test_mu_i - test_mu_i),
                       torch.abs(test_sigma_i - test_sigma_j)), dim=1)
test_feat = torch.cat((test_feat, test_mu_i * test_mu_j), dim=1)
test_feat = test_feat.numpy()
test_labels = test_Y[:, 2]

print("### Training classifier ###")
clf = None
clf = RandomForestClassifier(max_depth=10, n_estimators=100, random_state=0, n_jobs=4)
clf.fit(feat, train_labels)
prob = clf.predict_proba(test_feat)
prediction = clf.predict(test_feat)
pred = prob[:, 1]
roc = roc_auc_score(test_labels, pred)
ap = average_precision_score(test_labels, pred)

print("Random Forest", roc, ap)