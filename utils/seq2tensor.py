import numpy as np
import pickle as pkl

class s2t(object):
    def __init__(self, filename):
        """
        Parameters
        ----------
        filename : string
            path to the file that contains the embedding for each amino acid
        """

        # define a dictionary that maps amino acid to one-hot encoding
        self.t2v = {}
        self.dim = None
        
        file = open(filename, "rb")
        self.t2v = pkl.load(file)
        file.close()
        self.dim = len(self.t2v.keys())
    
    def embed(self, seq):
        """
        Embeds the raw amino acid sequence to vector based on t2v

        Returns
        -------
        rst : list, shape [?]
            The list of the amino acid representation for a sequence
        length : number
            The length of the input sequence (seq)
        """
        if seq.find(' ') > 0:
            s = seq.strip().split()
        else:
            s = list(seq.strip())
        rst = []
        for x in s:
            v = self.t2v.get(x)
            if v is None:
                continue
            rst.append(v)
        return rst, len(rst)
    
    def embed_normalized(self, seq, length=50):
        """
        Embeds the raw amino acid sequence to matrix based on t2v considering the sequence length <= length

        Returns
        -------
        rst : list, shape [?]
           The one-hot encoded representation of the amino acid in a sequence
        lens : number
           The length of the input sequence (seq)
        """
        rst, lens = self.embed(seq)
        if len(rst) > length:
            lens = length
            return rst[:length], lens
        elif len(rst) < length:
            lens = len(rst)
            return np.concatenate((rst, np.zeros((length - len(rst))))), lens
        return rst, lens