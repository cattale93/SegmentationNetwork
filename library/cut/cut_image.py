import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def cut_image(RGB_NIR, tare, tare_splitted,patch_size,percentage_of_acceptable_bad_pixels,overlapping = 1,border = True):
    N_of_band = RGB_NIR.shape[-1]
    N_of_classes = tare_splitted.shape[-1]
    
    #inizialize two empty list
    list_of_RGB_NIR = np.zeros([0,patch_size,patch_size, N_of_band], dtype = np.float32)
    list_of_tare = np.zeros([0,patch_size,patch_size, N_of_classes], dtype = np.float32)
    #memory allocation
    patch_RGB_NIR = np.zeros([1,patch_size,patch_size, N_of_band], dtype = np.float32)
    patch_tare = np.zeros([1,patch_size,patch_size,N_of_classes], dtype = np.float32)

    x_dim = RGB_NIR.shape[0]
    y_dim = RGB_NIR.shape[1]

    for i in range(0, (x_dim - patch_size), int(patch_size/overlapping)):
        for j in range(0, (y_dim - patch_size), int(patch_size/overlapping)):               
            validity_patch = tare[i:i+patch_size,j:j+patch_size]
            u, count = np.unique(validity_patch, return_counts = True)
            percentage_of_bad_pixels = float(count[0])/(patch_size*patch_size)

            if  percentage_of_bad_pixels < percentage_of_acceptable_bad_pixels:
                patch_RGB_NIR[0,:,:,:] = RGB_NIR[i:i+patch_size,j:j+patch_size,:]
                patch_tare[0,:,:,:] = tare_splitted[i:i+patch_size,j:j+patch_size,:]

                list_of_RGB_NIR = np.concatenate((list_of_RGB_NIR, patch_RGB_NIR))
                list_of_tare = np.concatenate((list_of_tare, patch_tare))

    if border:
        for k in range(0, (x_dim - patch_size), patch_size):               
            validity_patch = tare[k:k+patch_size,y_dim - patch_size:y_dim]
            u, count = np.unique(validity_patch, return_counts = True)

            percentage_of_bad_pixels = float(count[0])/(patch_size*patch_size)

            if  percentage_of_bad_pixels < percentage_of_acceptable_bad_pixels:
                patch_RGB_NIR[0,:,:,:] = RGB_NIR[k:k+patch_size,y_dim - patch_size:y_dim,:]
                patch_tare[0,:,:,:] = tare_splitted[k:k+patch_size,y_dim - patch_size:y_dim,:]

                list_of_RGB_NIR = np.concatenate((list_of_RGB_NIR, patch_RGB_NIR))
                list_of_tare = np.concatenate((list_of_tare, patch_tare))

        for k in range(0, (y_dim - patch_size), patch_size):               
            validity_patch = tare[x_dim - patch_size:x_dim,k:k+patch_size]
            u, count = np.unique(validity_patch, return_counts = True)

            percentage_of_bad_pixels = float(count[0])/(patch_size*patch_size)

            if  percentage_of_bad_pixels < percentage_of_acceptable_bad_pixels:
                patch_RGB_NIR[0,:,:,:] = RGB_NIR[x_dim - patch_size:x_dim,k:k+patch_size,:]
                patch_tare[0,:,:,:] = tare_splitted[x_dim - patch_size:x_dim,k:k+patch_size,:]

                list_of_RGB_NIR = np.concatenate((list_of_RGB_NIR, patch_RGB_NIR))
                list_of_tare = np.concatenate((list_of_tare, patch_tare))

    return list_of_RGB_NIR, list_of_tare