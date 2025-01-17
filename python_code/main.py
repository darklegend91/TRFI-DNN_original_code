import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift
from commpy import modulations  # You can use commpy for BPSK/QPSK/QAM

# --------OFDM Parameters - Given in IEEE 802.11p Spec--
ofdmBW = 10 * 10**6  # OFDM bandwidth (Hz)
nFFT = 64  # FFT size 
nDSC = 48  # Number of data subcarriers
nPSC = 4  # Number of pilot subcarriers
nZSC = 12  # Number of zeros subcarriers
nUSC = nDSC + nPSC  # Number of total used subcarriers
K = nUSC + nZSC  # Number of total subcarriers
nSym = 50  # Number of OFDM symbols within one frame
deltaF = ofdmBW / nFFT  # Bandwidth for each subcarrier
Tfft = 1 / deltaF  # IFFT or FFT period = 6.4us
Tgi = Tfft / 4  # Guard interval duration = 1.6us
Tsignal = Tgi + Tfft  # Total duration of BPSK-OFDM symbol = 8us
K_cp = int(nFFT * Tgi / Tfft)  # Number of symbols allocated to cyclic prefix
pilots_locations = np.array([8, 22, 44, 58])  # Pilot subcarriers positions
pilots = np.array([1, 1, 1, -1])
data_locations = np.array([i for i in range(2, 8)] + [i for i in range(9, 22)] + [i for i in range(23, 28)] +
                          [i for i in range(39, 44)] + [i for i in range(45, 58)] + [i for i in range(59, 65)])
null_locations = np.array([1] + list(range(28, 39)))

# Pre-defined preamble in frequency domain
dp = np.array([0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
Ep = 1  # Preamble power per sample
dp = fftshift(dp)  # Shift zero-frequency component to center of spectrum
predefined_preamble = dp
Kset = np.where(dp != 0)[0]  # Set of allocated subcarriers
Kon = len(Kset)  # Number of active subcarriers
dp = np.sqrt(Ep) * dp
xp = np.sqrt(K) * ifft(dp)
xp_cp = np.concatenate((xp[-K_cp:], xp))  # Adding CP to the time domain preamble
preamble_80211p = np.tile(xp_cp, (2, 1))  # IEEE 802.11p preamble symbols (two symbols)

# ------ Bits Modulation Technique ------------------------------------------
modu = 'QPSK'
Mod_Type = 1  # 0 for BPSK and 1 for QAM 
if Mod_Type == 0:
    nBitPerSym = 1
    Pow = 1
    M = 1
elif Mod_Type == 1:
    if modu == 'QPSK':
        nBitPerSym = 2
    elif modu == '16QAM':
        nBitPerSym = 4
    elif modu == '64QAM':
        nBitPerSym = 6
    M = 2 ** nBitPerSym
    Pow = np.mean(np.abs(modulations.qammod(np.arange(M), M)) ** 2)  # Normalization factor for QAM

# --------- Scrambler Parameters ---------------------------------------------
scramInit = 93  # As specified in IEEE 802.11p Standard [1011101] in binary representation

# --------- Convolutional Coder Parameters -----------------------------------
constlen = 7
trellis = np.array([171, 133])  # Poly2trellis equivalent in Python (for 7)
tbl = 34
rate = 1 / 2

# ------- Interleaver Parameters ---------------------------------------------
Interleaver_Rows = 16
Interleaver_Columns = (nBitPerSym * nDSC * nSym) // Interleaver_Rows
Random_permutation_Vector = np.random.permutation(nBitPerSym * nDSC * nSym)  # Permutation vector

# ----------------- Vehicular Channel Model Parameters --------------------------
mobility = 'High'
ChType = 'VTV_SDWW'  # Channel model
fs = K * deltaF  # Sampling frequency
fc = 5.9e9  # Carrier Frequency in Hz.
vel = 48  # Moving speed of user in km/h
c = 3e8  # Speed of Light in m/s
fD = 500  # Doppler freq in Hz
rchan = Channel_functions.GenFadingChannel(ChType, fD, fs)

# Simulation Parameters 
configuration = 'testing'  # Can be 'training' or 'testing'

if configuration == 'training':
    indices = sio.loadmat('./samples_indices_100.mat')['training_samples']
    EbN0dB = 40
elif configuration == 'testing':
    indices = sio.loadmat('./samples_indices_100.mat')['testing_samples']
    EbN0dB = np.arange(0, 45, 5)

# --------- Bit to Noise Ratio ------------------
SNR_p = EbN0dB + 10 * np.log10(K / nDSC) + 10 * np.log10(K / (K + K_cp)) + 10 * np.log10(nBitPerSym) + 10 * np.log10(rate)
SNR_p = SNR_p.reshape(-1, 1)
N0 = Ep * 10 ** (-SNR_p / 10)
N_CH = indices.shape[0]
N_SNR = len(SNR_p)

# Normalized mean square error (NMSE) vectors
Err_LS = np.zeros(N_SNR)
Err_STA = np.zeros(N_SNR)
Err_Initial = np.zeros(N_SNR)
Err_CDP = np.zeros(N_SNR)
Err_TRFI = np.zeros(N_SNR)
Err_MMSE_VP = np.zeros(N_SNR)

# Bit error rate (BER) vectors
Ber_Ideal = np.zeros(N_SNR)
Ber_LS = np.zeros(N_SNR)
Ber_STA = np.zeros(N_SNR)
Ber_Initial = np.zeros(N_SNR)
Ber_CDP = np.zeros(N_SNR)
Ber_TRFI = np.zeros(N_SNR)
Ber_MMSE_VP = np.zeros(N_SNR)

# average channel power E(|hf|^2)
Phf_H_Total = np.zeros(N_SNR)

# to be used in the preamble based MMSE Channel Estimation
preamble_diag = np.diag(predefined_preamble[Kset]) @ np.diag(predefined_preamble[Kset].T)

# STA Channel Estimation Scheme Parameters
alpha = 2
Beta = 2
w = 1 / (2 * Beta + 1)
lambda_ = np.arange(-Beta, Beta + 1)

# Simulation Loop
for n_snr in range(N_SNR):
    print(f'Running Simulation, SNR = {EbN0dB[n_snr]} dB')
    TX_Bits_Stream_Structure = np.zeros((nDSC * nSym * nBitPerSym * rate, N_CH))
    Received_Symbols_FFT_Structure = np.zeros((Kon, nSym, N_CH))
    True_Channels_Structure = np.zeros((Kon, nSym, N_CH))
    DPA_Structure = np.zeros((Kon, nSym, N_CH))
    STA_Structure = np.zeros((Kon, nSym, N_CH))
    TRFI_Structure = np.zeros((Kon, nSym, N_CH))

    for n_ch in range(N_CH):
        # Bits Stream Generation
        Bits_Stream_Coded = np.random.randint(0, 2, nDSC * nSym * nBitPerSym * rate)
        # Data Scrambler 
        scrambledData = wlanScramble(Bits_Stream_Coded, scramInit)
        # Convolutional Encoder
        dataEnc = convenc(scrambledData, trellis)
        # Interleaving
        codedata = dataEnc.T
        Matrix_Interleaved_Data = matintrlv(codedata, Interleaver_Rows, Interleaver_Columns).T
        General_Block_Interleaved_Data = intrlv(Matrix_Interleaved_Data, Random_permutation_Vector)
        
        # Bits Mapping: M-QAM Modulation
        TxBits_Coded = np.reshape(General_Block_Interleaved_Data, (nDSC, nSym, nBitPerSym))
        TxData_Coded = np.zeros((nDSC, nSym))
        for m in range(nBitPerSym):
            TxData_Coded += TxBits_Coded[:, :, m] * 2 ** (m - 1)
        
        # M-QAM Modulation
        Modulated_Bits_Coded = 1 / np.sqrt(Pow) * modulations.qammod(TxData_Coded, M)

        # OFDM Frame Generation
        OFDM_Frame_Coded = np.zeros((K, nSym))
        OFDM_Frame_Coded[data_locations, :] = Modulated_Bits_Coded
        OFDM_Frame_Coded[pilots_locations, :] = np.tile(pilots, (1, nSym))

        # Taking FFT
        IFFT_Data_Coded = np.sqrt(K) * ifft(OFDM_Frame_Coded)
        # checking power of transmit signal
        power_Coded = np.var(IFFT_Data_Coded) + np.abs(np.mean(IFFT_Data_Coded)) ** 2

        # Appending Cyclic Prefix
        CP_Coded = IFFT_Data_Coded[(K - K_cp):K, :]
        IFFT_Data_CP_Coded = np.vstack([CP_Coded, IFFT_Data_Coded])

        # Adding Preamble Symbol
        IFFT_Data_CP_Preamble_Coded = np.vstack([preamble_80211p, IFFT_Data_CP_Coded])
        power_transmitted = np.var(IFFT_Data_CP_Preamble_Coded) + np.abs(np.mean(IFFT_Data_CP_Preamble_Coded)) ** 2

        # Ideal Estimation
        h, y = ch_func.ApplyChannel(rchan, IFFT_Data_CP_Preamble_Coded, K_cp)
        yp = y[(K_cp):, 0:2]
        y = y[(K_cp):, 2:]

        # Channel Processing
        yFD = np.sqrt(1 / K) * fft(y)
        yfp = np.sqrt(1 / K) * fft(yp)
        
        h = h[(K_cp):, :]
        hf = fft(h)
        hfp1 = hf[:, 0]
        hfp2 = hf[:, 1]
        hfp = (hfp1 + hfp2) / 2
        hf = hf[:, 2:]

        # Average channel power
        Phf_H_Total[n_snr] += np.mean(np.sum(np.abs(hf[Kset, :]) ** 2))

        # Add noise
        noise_preamble = np.sqrt(N0[n_snr]) * ch_func.GenRandomNoise([K, 2], 1)
        yfp_r = yfp + noise_preamble
        noise_OFDM_Symbols = np.sqrt(N0[n_snr]) * ch_func.GenRandomNoise([K, yFD.shape[1]], 1)
        y_r = yFD + noise_OFDM_Symbols

        # Channel Estimation
        he_LS_Preamble = (yfp_r[Kset, 0] + yfp_r[Kset, 1]) / (2 * predefined_preamble[Kset])
        H_LS = np.tile(he_LS_Preamble, (1, nSym))
        Err_LS[n_snr] += np.mean(np.sum(np.abs(H_LS - hf[Kset, :]) ** 2))

        # STA Channel Estimation
        H_STA, Equalized_OFDM_Symbols_STA = STA(he_LS_Preamble, y_r, Kset, modu, nUSC, nSym, ppositions, alpha, w, lambda_)
        Err_STA[n_snr] += np.mean(np.sum(np.abs(H_STA - hf[Kset, :]) ** 2))

        # CDP Channel Estimation
        H_CDP, Equalized_OFDM_Symbols_CDP = CDP(he_LS_Preamble([1:6, 8:20, 22:31, 33:45, 47:52], 1), y_r, data_locations,
                                                 yfp_r[data_locations, 2], predefined_preamble[0, data_locations], modu, nDSC, nSym)
        Err_CDP[n_snr] += np.mean(np.sum(np.abs(H_CDP - hf[data_locations, :]) ** 2))

        # TRFI Channel Estimation
        H_TRFI, Equalized_OFDM_Symbols_TRFI = TRFI(he_LS_Preamble, y_r, Kset, yfp_r[Kset, 2], predefined_preamble[0, Kset],
                                                   ppositions, modu, nUSC, nSym)
        Err_TRFI[n_snr] += np.mean(np.sum(np.abs(H_TRFI - hf[Kset, :]) ** 2))

        # MMSE + Virtual Pilots
        noise_power_OFDM_Symbols = np.var(noise_OFDM_Symbols)
        H_MMSE_VP, Equalized_OFDM_Symbols_MMSE_VP = MMSE_Vitual_Pilots(he_LS_Preamble, y_r, Kset, modu, ppositions,
                                                                       dpositions, noise_power_OFDM_Symbols)
        Err_MMSE_VP[n_snr] += np.mean(np.sum(np.abs(H_MMSE_VP - hf[Kset, :]) ** 2))

        # Initial Channel Estimation
        H_Initial, Equalized_OFDM_Symbols_Initial = DPA(he_LS_Preamble, y_r, Kset, ppositions, modu, nUSC, nSym)
        Err_Initial[n_snr] += np.mean(np.sum(np.abs(H_Initial - hf[Kset, :]) ** 2))

        # Demodulate and Compute BER
        Bits_Ideal = qamdemod(np.sqrt(Pow) * (y_r[data_locations, :] / hf[data_locations, :]), M)
        # Repeat the demodulation for other methods similarly...
      
        # Save data depending on configuration
        if configuration == 'training':
            sio.savemat(f'./{mobility}_{ChType}_{modu}_training_simulation_{EbN0dB[n_snr]}', {
                'True_Channels_Structure': True_Channels_Structure,
                'DPA_Structure': DPA_Structure,
                'STA_Structure': STA_Structure,
                'TRFI_Structure': TRFI_Structure})
        elif configuration == 'testing':
            sio.savemat(f'./{mobility}_{ChType}_{modu}_testing_simulation_{EbN0dB[n_snr]}', {
                'TX_Bits_Stream_Structure': TX_Bits_Stream_Structure,
                'Received_Symbols_FFT_Structure': Received_Symbols_FFT_Structure,
                'True_Channels_Structure': True_Channels_Structure,
                'DPA_Structure': DPA_Structure,
                'STA_Structure': STA_Structure,
                'TRFI_Structure': TRFI_Structure})

    # Plot BER and NMSE curves, similar to MATLAB plotting
    plt.figure()
    plt.semilogy(EbN0dB, BER_Ideal, 'k-o', label='Perfect Channel', linewidth=2)
    plt.semilogy(EbN0dB, BER_LS, 'k--o', label='LS', linewidth=2)
    plt.semilogy(EbN0dB, BER_STA, 'g-o', label='STA', linewidth=2)
    plt.semilogy(EbN0dB, BER_Initial, 'r-^', label='Initial Channel', linewidth=2)
    plt.semilogy(EbN0dB, BER_CDP, 'b-+', label='CDP', linewidth=2)
    plt.semilogy(EbN0dB, BER_TRFI, 'c-d', label='TRFI', linewidth=2)
    plt.semilogy(EbN0dB, BER_MMSE_VP, 'm-v', label='MMSE-VP', linewidth=2)
    plt.grid(True)
    plt.xlabel('SNR(dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.legend()

    plt.figure()
    plt.semilogy(EbN0dB, ERR_LS, 'k--o', label='LS', linewidth=2)
    plt.semilogy(EbN0dB, ERR_STA, 'g-o', label='STA', linewidth=2)
    plt.semilogy(EbN0dB, ERR_Initial, 'r-^', label='Initial Channel', linewidth=2)
    plt.semilogy(EbN0dB, ERR_CDP, 'b-+', label='CDP', linewidth=2)
    plt.semilogy(EbN0dB, ERR_TRFI, 'c-d', label='TRFI', linewidth=2)
    plt.semilogy(EbN0dB, ERR_MMSE_VP, 'm-v', label='MMSE-VP', linewidth=2)
    plt.grid(True)
    plt.xlabel('SNR(dB)')
    plt.ylabel('Normalized Mean Square Error (NMSE)')
    plt.legend()
