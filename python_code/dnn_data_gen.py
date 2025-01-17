import numpy as np
import scipy.io as sio

# Load pre-defined DNN Testing Indices
indices_data = sio.loadmat('python_code/samples_indices_100.mat')
configuration = 'testing'  # training or testing

# Define Simulation parameters
nUSC = 52
nSym = 50
mobility = 'High'
modu = 'QPSK'
ChType = 'VTV_SDWW'
scheme = 'STA'

if configuration == 'training':
    indices = indices_data['training_samples']
    EbN0dB = [40]

elif configuration == 'testing':
    indices = indices_data['testing_samples']
    EbN0dB = np.arange(0, 45, 5)

Dataset_size = indices.shape[0]
Dataset_X = np.zeros((nUSC * 2, Dataset_size * nSym))
Dataset_Y = np.zeros((nUSC * 2, Dataset_size * nSym))

SNR = EbN0dB
N_SNR = len(SNR)

# Simulation loop over SNR values
for n_snr in range(N_SNR):
    # Load pre-saved data for each SNR level
    file_name = f'./{mobility}_{ChType}_{modu}_{configuration}_simulation_{EbN0dB[n_snr]}.mat'
    data = sio.loadmat(file_name, variable_names=['True_Channels_Structure', f'{scheme}_Structure'])
    
    scheme_Channels_Structure = data[f'{scheme}_Structure']
    True_Channels_Structure = data['True_Channels_Structure']
    
    # Reshape data
    DatasetX_expended = scheme_Channels_Structure.reshape(nUSC, nSym * Dataset_size)
    DatasetY_expended = True_Channels_Structure.reshape(nUSC, nSym * Dataset_size)
    
    # Complex to Real domain conversion
    Dataset_X[:nUSC, :] = np.real(DatasetX_expended)
    Dataset_X[nUSC:2*nUSC, :] = np.imag(DatasetX_expended)
    Dataset_Y[:nUSC, :] = np.real(DatasetY_expended)
    Dataset_Y[nUSC:2*nUSC, :] = np.imag(DatasetY_expended)
    
    # Store datasets for training or testing
    DNN_Datasets = {}
    if configuration == 'training':
        DNN_Datasets['Train_X'] = Dataset_X.T
        DNN_Datasets['Train_Y'] = Dataset_Y.T
    elif configuration == 'testing':
        DNN_Datasets['Test_X'] = Dataset_X.T
        DNN_Datasets['Test_Y'] = Dataset_Y.T
    
    # Save the datasets for the current SNR
    save_file_name = f'./{mobility}_{ChType}_{modu}_{scheme}_DNN_{configuration}_dataset_{EbN0dB[n_snr]}.mat'
    sio.savemat(save_file_name, DNN_Datasets)