import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    final_cross=0
    for y,p in zip(Y,P):
        cross=-(y*np.log(p)+(1-y)*np.log(1-p))
        final_cross=final_cross+cross
    return final_cross
    
 #Fancy solution
import numpy as np

def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
