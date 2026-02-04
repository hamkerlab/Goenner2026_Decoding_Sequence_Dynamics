# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 16:53:06 2021

@author: goenner
"""

import scipy.stats as st
import numpy as np

def corr_circ_lin(angles_rad, x_lin):
    # Following the CircStat toolobox by Philip Berens
    # Matlab: corr_cl.m
    n = len(angles_rad)

    rxc, pxc = st.pearsonr(x_lin, np.cos(angles_rad))    
    rxs, pxs = st.pearsonr(x_lin, np.sin(angles_rad))        
    rcs, pcs = st.pearsonr(np.cos(angles_rad), np.sin(angles_rad))    
    
    r = np.sqrt( (rxs**2 + rxc**2 - 2*rxc*rxs*rcs) / (1-rcs**2) )    
    p = 1 - st.chi2.cdf(n * r**2, 2)
    
    return r, p