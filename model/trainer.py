import torch
import time
from utils import lib
import numpy as np
from torch import nn

def trainer(model, optimizer, data_dict, device, model_name, epochs=20,clip=5):
    """
    Train the model with training dataset and evaluate on validation dataset
    Parameters
    ----------
    model : torch.nn.Module
        The pytorch model
    optimizer : torch optimizer
        Optimizer for the model
    data_dict : dictionary
        The dictionary of the dataset
    device: torch.device
        The device type: GPU or CPU
    epoch: number   
        Number of epochs to train
    clip: float
        The value to clip the gradient
    """

    best_ap = 0
    patience = 5
    cant_wait = patience
    best_epoch = 0

    for epoch in range(epochs):
        t_total = time.time()
        losses = []
        train_ind = np.random.permutation(len(data_dict['train_pairs']))
        train_interactions = data_dict['train_pairs'][train_ind]
        for batch_id, (batch, label) in enumerate(lib.get_batches(train_interactions, 5000)):
            model.train()
            seq_array, lens_array, id2idx = lib.get_subset_tensor(data_dict['seq_tensor'], data_dict['lengths'], batch, data_dict['protname2index'])
            # move tensors to GPU if available
            seq = torch.from_numpy(seq_array).transpose(0,1).type(torch.LongTensor)
            lens = torch.from_numpy(lens_array)
            seq, lens = seq.to(device), lens.to(device)
            optimizer.zero_grad()

            mu_batch, sigma_batch, _ = model(seq, lens)
    #             print(mu_batch.shape)
            assert mu_batch.shape[0] == seq.shape[1]

            # calculate the batch loss
            loss = lib.compute_loss(mu_batch, sigma_batch, batch, label, id2idx, device)
            b = 0.1
            flooding = (loss - b).abs() + b
            # backward pass: compute the gradients of the loss with respect to model parameters
            loss.backward()

            train_loss = loss.item()
            losses.append(train_loss)

            # clip the gradient of RNN to prevent exploding gradient
            nn.utils.clip_grad_norm_(model.parameters(), clip)

            # perform a single optimization step (parameter update)
            optimizer.step()

            print("Epoch: {:3d} Batch :{:3d} Loss: {:.06f}".format(epoch+1, batch_id, loss.item()))
        
        avg_loss = np.mean(losses)

        val_auc, val_ap = evaluate(model, data_dict['seq_tensor'], data_dict['lengths'], data_dict['val_pairs'], data_dict['protname2index'], device)
        
        # checkpoint the best average precision
        if best_ap < val_ap:
            best_ap = val_ap
            best_epoch = epoch
            status = "*"
            cant_wait = patience
            # Saving the best model
            torch.save(model.state_dict(), 'trained_model/'+model_name+'.pkl')
        else:
            status = ""
            cant_wait -=1

        print("Epoch {}/{} Train Loss: {:.06f} Val AUC: {:.06f} Val AP: {:.06f} Best AP: {:.06f}{}".format(epoch+1, epochs, avg_loss, val_auc, val_ap, best_ap, status))
        t  = time.time() - t_total
        print("Total time elapsed per epoch: {:.4f}s".format((t)))
        with open("time_log.txt","a+") as f:
            f.write(str(t))
            f.write("\n")

        if cant_wait == 0:
            print("Early Stopping")
            print('Loading {}th epoch'.format(best_epoch))
            model.load_state_dict(torch.load('trained_model/'+model_name+'.pkl'))
            break

def evaluate(model, seq_tensor, lengths, pairs, protname2index, device):
    """
    Evaluate the trained model on evaluation dataset
    Parameters
    ----------
    model : torch.nn.Module
        The trained model
    seq_tensor: torch.Tensor
        The dataset that represents the sequences
    lengths: list
        The list of lengths of the sequences
    pairs: array-like, shape [?, 3]
        The interaction array
    protname2index: dict
        The dictionary that maps protein name to index
    device: torch.device
        The device type: CPU or GPU
    Returns
    -------
    auc : float
        Area under the ROC curve
    ap : float
        Average precision
    """

    # get the dataset for the proteins in pairs 
    seq, lens, id2idx = lib.get_subset_tensor(seq_tensor, lengths, pairs, protname2index)
    seq = torch.from_numpy(seq).transpose(0,1).type(torch.LongTensor)
    lens = torch.from_numpy(lens)

    # load tensor to device
    seq, lens = seq.to(device), lens.to(device)

    # Turn evaluation mode

    with torch.no_grad():
        model.eval()
        mu_list = []
        sigma_list = []
        for s, l in lib.get_seq_batches(seq, lens, 256):
            # load tensor to device
            s, l = s.to(device), l.to(device)
            mu, sigma, _ = model(s, l)
            mu = mu.cpu().detach()
            sigma = sigma.cpu().detach()
            mu_list.append(mu)
            sigma_list.append(sigma)

        mu = torch.cat(mu_list, 0)
        sigma = torch.cat(sigma_list, 0)

        # Compute the energy between the pairs given the trained model
        pred = lib.get_energy_pairs(mu, sigma, id2idx, pairs[:,:2], torch.device("cpu"))
        label_test = pairs[:,2].astype(int)

        # Score the prediction
        auc, ap = lib.score_link_prediction(label_test, -pred.cpu())

    return auc, ap

