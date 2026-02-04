#import numpy as np
from scipy.signal import hilbert
	
def hilb_transf(data):
    return hilbert(data)