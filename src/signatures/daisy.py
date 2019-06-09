'''
Computes Daisy features
@author: romue
'''
import numpy as np
from skimage.feature import daisy

def daisy_features(image, name, label, step_ = 4, rings_=3, histograms_=2, orientations_=8):
    
    a = daisy(image,step = step_, rings = rings_, histograms = histograms_, orientations=orientations_)
    result = a.reshape(-1)
    result = np.asarray(result)
    return result,name,label

