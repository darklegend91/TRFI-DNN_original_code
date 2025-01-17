#Modules importing
import numpy as np
import scipy.io as sio

# Parameters
IDX = 100  # Total number of samples
TR = 0.80  # Training dataset percentage
TE = 0.20  # Testing dataset percentage

# Generate indices
All_IDX = np.arange(1, IDX + 1)  # MATLAB indices start from 1
training_samples = np.random.choice(All_IDX, size=int(TR * IDX), replace=False)  # Choose 80% indices randomly
testing_samples = np.setdiff1d(All_IDX, training_samples)  # Remaining 20% for testing

# Save to .mat file
sio.savemat(f'./samples_indices_{IDX}.mat', {
    'training_samples': training_samples,
    'testing_samples': testing_samples
})

#Message on succesfull run
print("Training and testing indices saved successfully!")
