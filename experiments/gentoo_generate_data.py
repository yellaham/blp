import numpy as np
import matplotlib.pyplot as plt
from models import gentoo
from algorithms import pf

# Model parameters
juvenile_survival = 0.43
adult_survival = 0.82
reproductive_bias = 0.5
reproductive_slope = 0.5
immigration_bias = 20
immigration_slope = 0.02

# Save the parameters
param = {'juvenile survival': juvenile_survival,
         'adult survival': adult_survival,
         'reproductive bias': reproductive_bias,
         'reproductive slope': reproductive_slope,
         'immigration bias': immigration_bias,
         'immigration slope': immigration_slope}

# Length of time-series
time_steps = 20

# Initialize the observations
num_trajectories = 50
trajectories = np.zeros((num_trajectories, time_steps, 2))
# Generate predictions for all the sampled points
for s in range(num_trajectories):
    # Create the model (assuming the noise variances are known)
    regimes = [gentoo.AgeStructuredModel(psi_juv=juvenile_survival, psi_adu=adult_survival,
                                         alpha_r=reproductive_bias, beta_r=reproductive_slope,
                                         phi_0=immigration_bias, phi_1=immigration_slope, nstage=5)]
    draw_regimes = lambda model_idx, num_samp: np.random.choice(np.arange(start=0, stop=1), num_samp, replace=True, p=np.array([1]))
    regimes_log_pdf = lambda model_idx: (1-model_idx)*np.log(1)
    # Create regime switching system
    model = pf.MultiRegimeSSM(regimes, draw_regimes, regimes_log_pdf)
    # Generate synthetic data
    init_state = 1+np.random.randint(low=0, high=5, size=8)
    y, x, m = model.generate_data(init_state=init_state, T=time_steps)
    trajectories[s] = y
# Compute mean prediction and standard deviation
mean_trajectory = np.mean(trajectories, axis=0)
std_dev_prediction = np.sqrt(np.var(trajectories, axis=0))
# Plot the predictive distribution for the sum of adults
plt.figure()
plt.plot(mean_trajectory[:, 0], c='r')
plt.plot(trajectories[:, :, 0].T, c='y', alpha=0.15)
plt.fill_between(np.arange(0, time_steps), mean_trajectory[:, 0] - 1.96 * std_dev_prediction[:, 0],
                 mean_trajectory[:, 0] + 1.96 * std_dev_prediction[:, 0],
                 color='gray', alpha=0.2)
plt.xlabel('Year', fontsize=18)
plt.ylabel('$\~{S}_t$', fontsize=18)
plt.legend(['Mean Prediction', '95% CI'])
plt.show()

# Print out the counts to see what they are like (for a single trajectory)
data = np.ones((time_steps, 4))*0.1
data[:, :2] = y
print(data)
# Save the data
np.savez('gentoo_simulated_data.npz', data=data, true_parameters=param)