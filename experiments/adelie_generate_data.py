import numpy as np
import matplotlib.pyplot as plt
from models import adelie
from algorithms import pf

# Model parameters
juvenile_survival = 0.45
adult_survival = 0.85
reproductive_bias_bad = -0.5
reproductive_bias_good = 1.5
reproductive_slope = 0.5
prob_bad = 0.2

# Save the parameters
param = {'juvenile survival': juvenile_survival,
         'adult survival': adult_survival,
         'reproductive bias (bad)': reproductive_bias_bad,
         'reproductive bias (good)': reproductive_bias_good,
         'reproductive slope': reproductive_slope,
         'switching probability': prob_bad
         }

# Length of time-series
time_steps = 30

# Initialize the observations
num_trajectories = 50
trajectories = np.zeros((num_trajectories, time_steps, 2))
# Generate predictions for all the sampled points
for s in range(num_trajectories):
    # Create the model (assuming the noise variances are known)
    regimes = [adelie.AgeStructuredModel(psi_juv=juvenile_survival, psi_adu=adult_survival,
                                         alpha_r=reproductive_bias_bad, beta_r=reproductive_slope, nstage=5),
               adelie.AgeStructuredModel(psi_juv=juvenile_survival, psi_adu=adult_survival,
                                         alpha_r=reproductive_bias_good, beta_r=reproductive_slope, nstage=5)]
    draw_regimes = lambda model_idx, num_samp: np.random.choice(np.arange(start=0, stop=2), num_samp, replace=True,
                                                                p=[prob_bad, 1-prob_bad])
    regimes_log_pdf = lambda model_idx: model_idx*np.log(1-prob_bad)+(1-model_idx)*np.log(prob_bad)
    # Create regime switching system
    model = pf.MultiRegimeSSM(regimes, draw_regimes, regimes_log_pdf)
    # Generate synthetic data
    init_state = 1+np.random.randint(low=150, high=250, size=8)
    y, x, m = model.generate_data(init_state=init_state, T=time_steps)
    trajectories[s] = y
# Compute mean prediction and standard deviation
mean_trajectory = np.mean(trajectories, axis=0)
std_dev_prediction = np.sqrt(np.var(trajectories, axis=0))
# Plot the predictive distribution for the sum of adults
plt.figure()
plt.plot(mean_trajectory[:, 0], c='r')
plt.plot(trajectories[:, :, 0].T, c='y', alpha=0.15)
plt.xlabel('Year', fontsize=18)
plt.ylabel('$\~{S}_t$', fontsize=18)
plt.legend(['Mean Prediction'])
plt.show()

# Plot the predictive distribution for the sum of adults
plt.figure()
plt.plot(mean_trajectory[:, 1], c='r')
plt.plot(trajectories[:, :, 1].T, c='y', alpha=0.15)
plt.xlabel('Year', fontsize=18)
plt.ylabel('$\~{C}_t$', fontsize=18)
plt.legend(['Mean Prediction'])
plt.show()

# Print out the counts to see what they are like (for a single trajectory)
data = np.ones((time_steps, 4))*0.1
data[:, :2] = y
print(data)
# Save the data
np.savez('adelie_simulated_data.npz', data=data, true_parameters=param)