import numpy as np

def extract_fault_id_and_assemble(
        n_i: int,
        n_j: int,
        n_k: int,
        fault_id,
        coor_npy,
        save_path
    ):
    
    # initialize a nan array for fault id
    nan_slice = np.full((n_i,n_j,n_k,1), np.nan)
    coor_fault_npy = np.concatenate((coor_npy, nan_slice), axis=3)

    # fill the fault id
    for ii in range(n_i):
        for jj in range(n_j):
            for kk in range(n_k):
                coor_fault_npy[ii,jj,kk,3] = fault_id[kk*n_j*n_i+jj*n_i+ii]

    # coor_fault_npy[:,:,:,3] = coor_fault_npy[:,:,:,3].astype(int)
    coor_fault_npy[:,:,:,3].dtype

    # flip the j direction, because Petrel count j from top to bottom, and CMG count j from bottom to top.
    coor_fault_npy = coor_fault_npy[:,::-1,:,:]

    # fill the non-fault id with nan
    coor_fault_npy[coor_fault_npy == -999] = np.nan

    # save the coor_fault_npy
    np.save(save_path, coor_fault_npy)

    print('Processed file shape (i,j,k, 4):', coor_fault_npy.shape)