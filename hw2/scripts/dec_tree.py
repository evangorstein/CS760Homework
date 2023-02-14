import numpy as np

class DecisionTree:
    """
    Implementation of a decision tree
    """
    def __init__(self, D):
        
        self.n = D.shape[0]
        self.counts = np.bincount(D[:,-1].astype(int))
        self.ent = entr(self.counts[0]/self.n)
        self.pred = None 
    
    def make_leaf(self):
        if len(self.counts) == 1:
            self.pred = 0
        elif self.counts[0] == self.counts[1]:
            self.pred = 1
        else:
            self.pred = self.counts.argmax()

    def train(self, D):

        if self.n < 2: #Only one training instance in D -> create leaf
            self.make_leaf()
            return

        if self.ent == 0: #D is pure -> create leaf
            self.make_leaf()
            return

        splits = candidate_splits(D, self.ent)

        if not splits: #No split with positive information gain -> create leaf
            self.make_leaf()
            return

        (j, c) = max(splits, key=splits.get)
        self.test = (j, c)
        test = D[:,j] >= c
        Dleft = D[test,:]
        Dright = D[~test,:]
        self.left = DecisionTree(Dleft)
        self.left.train(Dleft)
        self.right = DecisionTree(Dright)
        self.right.train(Dright)

    def count_nodes(self):
        
        if self.pred is not None:
            return 1
        else: 
            return self.left.count_nodes() + self.right.count_nodes() + 1 

    def predict(self, new_data):
    
        yhat = np.empty(len(new_data), dtype=int)
        for (j, row) in enumerate(new_data):
        
            tree = self
            while tree.pred is None:

                if (row[tree.test[0]] >= tree.test[1]):
                    tree = tree.left
                else: 
                    tree = tree.right

            yhat[j] = tree.pred
    
        return yhat 
    
    def print(self, indent=0):
        print(" " * indent, f"There are {self.n} sample instances at this node.")
        if self.pred is not None:  
            print(" " * indent, f"Reached a leaf")
            print(" " * indent, f"Prediction for instances at this node is {self.pred}")
        else:
            print(" " * indent, f"Reached internal node.")
            print(" " * indent, f"If x_{self.test[0]} is greater than or equal to {self.test[1]}, move to the left branch.")
            print(" " * indent, f"Otherwise, move to the right branch.")
            print(" " * indent, "LEFT:")
            self.left.print(indent + 4)
            print(" " * indent, "RIGHT:")
            self.right.print(indent + 4)


def entr(p):
    """
    Calculates the entropy of a coin flip with probability p
    """
    if p == 0 or p == 1:
        ent = 0
    else:
        ent = -(p*np.log2(p)+(1-p)*np.log2(1-p))
    return ent

def get_gain_ratio(D, j, c, ent):
    
    """
    ent is the entropy of the node, needed for calculating the gain ratio
    """
    
    test = D[:,j] >= c
    Dleft = D[test,:]
    Dright = D[~test,:]
    split_p = np.sum(test)/len(test)
    
    #Calculate entropy of split
    split_info = entr(split_p)
    
    #Calculate condition entropy of label given split
    cond_ent = split_p*entr(sum(Dleft[:,-1])/len(Dleft)) + (1-split_p)*entr(sum(Dright[:,-1])/len(Dright))

    #Calculate gain ratio
    gain_ratio = (ent - cond_ent)/split_info
    return gain_ratio



def candidate_splits(D, ent):
    """
    ent is the entropy of the node, needed for calculating the gain ratio of the candidate splits
    
    Returns dictionary of candidate splits indexed with tuples (j, c) with values given by 
    the gain ratio. Candidate splits with gain ratio 0 are not included
    """

    #Dictionary to store candidate splits
    C = {}

    #Iterate through predictor variables
    for (j, var) in enumerate(D[:,:-1].T):
        
        #Get sorted unique values 
        vals = np.sort(np.unique(var))[1:]
        for val in vals:
            #Calculate gain ratio
            gain_ratio = get_gain_ratio(D, j, val, ent)
            #If gain ratio positive, add the split to the set of candidate splits
            if gain_ratio > 0:
                C[(j, val)] = gain_ratio
    
    return C

    
    

