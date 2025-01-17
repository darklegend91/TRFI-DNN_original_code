import numpy as np
import scipy.io as sio
from commpy.modulation import QAM
from scipy import signal

# Parameters
mobility = 'High'
ChType = 'VTV_SDWW'
modu = 'QPSK'
scheme = 'STA'
testing_samples = 20

# Modulation Order and Power Calculation
if modu == 'QPSK':
    nBitPerSym = 2
elif modu == '16QAM':
    nBitPerSym = 4
elif modu == '64QAM':
    nBitPerSym = 6

M = 2 ** nBitPerSym  # QAM Modulation Order
Pow = np.mean(np.abs(QAM(M).modulate(np.arange(M))) ** 2)  # Normalization factor for QAM

# Load Simulation Parameters
sim_params = sio.loadmat(f'./{mobility}_{ChType}_{modu}_simulation_parameters.mat')
EbN0dB = np.arange(0, 45, 5)  # Eb/N0 in dB
nSym = 50
constlen = 7
trellis = [171, 133]
tbl = 34
scramInit = 93
nDSC = 48
nUSC = 52
dpositions = np.array([1, 6, 8, 20, 22, 31, 33, 45, 47, 52])
Interleaver_Rows = 16
Interleaver_Columns = (nBitPerSym * nDSC * nSym) // Interleaver_Rows
N_SNR = len(EbN0dB)

# Arrays for results
Phf = np.zeros(N_SNR)
Err_scheme_DNN = np.zeros(N_SNR)
Ber_scheme_DNN = np.zeros(N_SNR)

# Loop over SNR values
for n_snr in range(N_SNR):
    # Load Simulation Results
    sim_file = f'./{mobility}_{ChType}_{modu}_testing_simulation_{EbN0dB[n_snr]}.mat'
    sim_data = sio.loadmat(sim_file)
    True_Channels_Structure = sim_data['True_Channels_Structure']
    
    # Load DNN Results
    dnn_file = f'./{mobility}_{ChType}_{modu}_{scheme}_DNN_Results_{EbN0dB[n_snr]}.mat'
    dnn_data = sio.loadmat(dnn_file)
    
    TestY = dnn_data[f'{scheme}_DNN_test_y_{EbN0dB[n_snr]}']
    TestY = TestY.T
    TestY = TestY[:nUSC, :] + 1j * TestY[nUSC:2*nUSC, :]
    TestY = TestY.reshape(nUSC, nSym, testing_samples)
    
    scheme_DNN = dnn_data[f'{scheme}_DNN_corrected_y_{EbN0dB[n_snr]}']
    scheme_DNN = scheme_DNN.T
    scheme_DNN = scheme_DNN[:nUSC, :] + 1j * scheme_DNN[nUSC:2*nUSC, :]
    scheme_DNN = scheme_DNN.reshape(nUSC, nSym, testing_samples)
    
    for u in range(scheme_DNN.shape[2]):  # Iterate over testing samples
        H_scheme_DNN = scheme_DNN[dpositions, :, u]
        
        # Channel Power
        Phf[n_snr] += np.mean(np.sum(np.abs(True_Channels_Structure[:, :, u]) ** 2))
        
        # Error Calculation
        Err_scheme_DNN[n_snr] += np.mean(np.sum(np.abs(H_scheme_DNN - True_Channels_Structure[dpositions, :, u]) ** 2))
        
        # QAM Modulation
        Bits_scheme_DNN = signal.qamdemod(np.sqrt(Pow) * (Received_Symbols_FFT_Structure[dpositions, :, u] / H_scheme_DNN), M)
        
        # Bit Error Rate (BER) Calculation
        Ber_scheme_DNN[n_snr] += np.sum(np.bitwise_xor(Bits_scheme_DNN, TX_Bits_Stream_Structure[:, u]))  # Simulated bit error calculation
    
    # Normalize by the number of testing samples
    Phf[n_snr] /= testing_samples
    Err_scheme_DNN[n_snr] /= (testing_samples * Phf[n_snr])
    Ber_scheme_DNN[n_snr] /= (testing_samples * nSym * 48 * nBitPerSym)

# Final Result
print("Phf: ", Phf)
print("Err_scheme_DNN: ", Err_scheme_DNN)
print("Ber_scheme_DNN: ", Ber_scheme_DNN)

