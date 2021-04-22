import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
from models import gentoo
from algorithms import pf
import seaborn as sns

id = 'synthetic'

with np.load(id+'.npz', allow_pickle=True) as summary:
    data = summary['data']
    prior = summary['prior_parameters'].item()
    sampled_parameters = summary['sampled_parameters']
    parameter_labels = summary['parameter_labels']
    dim = np.shape(sampled_parameters)[1]

# Obtain the predictive distribution
num_samples = np.shape(sampled_parameters)[0]   # Number of trajectories to sample
time_steps = np.shape(data)[0]

# Initialize the observations
num_trajectories = 200
predictive_distribution = np.zeros((num_trajectories, time_steps, 2))
average_reproductive_success = np.zeros(num_trajectories)

# Generate predictions for all the sampled points
for s in range(num_trajectories):
    # Create the model (assuming the noise variances are known)
    regimes = [gentoo.AgeStructuredModel(psi_juv=sampled_parameters[s, 0], psi_adu=sampled_parameters[s, 1],
                                         alpha_r=sampled_parameters[s, 2], beta_r=sampled_parameters[s, 3],
                                         phi_0=sampled_parameters[s, 4], phi_1=sampled_parameters[s, 5], nstage=5)]
    draw_regimes = lambda model_idx, num_samp: np.random.choice(np.arange(start=0, stop=1), num_samp, replace=True, p=np.array([1]))
    regimes_log_pdf = lambda model_idx: (1-model_idx)*np.log(1)
    # Create regime switching system
    model = pf.MultiRegimeSSM(regimes, draw_regimes, regimes_log_pdf)
    # Generate synthetic data
    init_state = 1+np.random.randint(low=0, high=5, size=(8))
    y, x, m = model.generate_data(init_state=init_state, T=time_steps)
    predictive_distribution[s] = y
    average_reproductive_success[s] = np.sum(x[-5:, -3:])/np.sum(x[-5:, 2:5])
# Compute mean prediction and standard deviation
mean_prediction = np.mean(predictive_distribution, axis=0)
std_dev_prediction = np.sqrt(np.var(predictive_distribution, axis=0))


# Plot the predictive distribution for the sum of adults
plt.figure()
plt.plot(data[:, 0], c='k')
plt.plot(mean_prediction[:, 0], c='r')
plt.plot(predictive_distribution[:, :, 0].T, c='b', alpha=0.02)
plt.xlabel('Year', fontsize=16)
plt.ylabel('$\~{S}_t$', fontsize=16)
plt.legend(['Ground Truth', 'Mean Prediction'])
filename = 'predictive_distribution_breeders_'+id
plt.savefig('../figures/gentoo/' + filename + '.pdf', format='pdf')
plt.show()


# Plot the predictive distribution for the sum of chicks
plt.figure()
plt.plot(data[:, 1], c='k')
plt.plot(mean_prediction[:, 1], c='r')
plt.plot(predictive_distribution[:, :, 1].T, c='b', alpha=0.02)
plt.xlabel('Year', fontsize=16)
plt.ylabel('$\~{C}_t$', fontsize=16)
plt.legend(['Ground Truth', 'Predictive Mean'])
filename = 'predictive_distribution_chicks_'+id
plt.savefig('../figures/gentoo/' + filename + '.pdf', format='pdf')
plt.show()

# Plot the posterior for average reproductive success
plt.figure()
plt.hist(average_reproductive_success, bins='auto', density=True, alpha=0.75)
plt.title(id.capitalize(), fontsize=16)
plt.xlabel('Average Reproductive Success', fontsize=12)
plt.ylabel('Density', fontsize=12)
filename = 'avg_reproductive_success_'+id
plt.savefig('../figures/gentoo/' + filename + '.pdf', format='pdf')
plt.show()

# Matrix plot of the approximated target distribution
# plt.figure()
# for i in range(dim):
#     for j in range(dim):
#         plt.subplot(dim, dim, count)
#         if i != j:
#             sns.kdeplot(x=sampled_parameters[:, j], y=sampled_parameters[:, i], cmap="Blues", shade=True, thresh=0.05)
#         else:
#             plt.hist(sampled_parameters[:, i])
# plt.show()

# Plot all of the one-dimensional histograms
for i in range(dim):
    # 1. First get a KDE
    est_post = sp.gaussian_kde(sampled_parameters[:, i])
    # Plot the posterior
    plt.figure()
    plt.hist(sampled_parameters[:, i], bins='auto', density=True, alpha=0.75)
    plt.xlabel(parameter_labels[i], fontsize=12)
    plt.ylabel('Density', fontsize=12)
    xx = np.linspace(min(sampled_parameters[:, i]), max(sampled_parameters[:, i]), 1000)
    plt.plot(xx, est_post(xx), color='b', lw=2)
    # Compute the overlap
    # 2. Evaluate the samples at the estimated KDE
    f_hat = est_post.logpdf(sampled_parameters[:, i])
    # 3. Evaluate the samples at the marginal prior
    if i == 0:
        g_hat = sp.beta.logpdf(sampled_parameters[:, i], prior['juvenile survival'][0], prior['juvenile survival'][1])
    elif i == 1:
        g_hat = sp.beta.logpdf(sampled_parameters[:, i], prior['adult survival'][0], prior['adult survival'][1])
    elif i == 2:
        g_hat = sp.norm.logpdf(sampled_parameters[:, i], prior['reproductive bias'][0], prior['reproductive bias'][1])
    elif i == 3:
        g_hat = sp.norm.logpdf(sampled_parameters[:, i], prior['reproductive slope'][0], np.sqrt(prior['reproductive slope'][1]))
    elif i == 4:
        g_hat = sp.uniform.logpdf(sampled_parameters[:, i], prior['immigration bias'][0], prior['immigration bias'][1])
    elif i == 5:
        g_hat = sp.uniform.logpdf(sampled_parameters[:, i], prior['immigration slope'][0], prior['immigration slope'][1])
    # 4. Evaluate the min function
    min_eval = np.minimum(np.exp(g_hat-f_hat), 1)
    # 5. Compute the overlap by taking a Monte Carlo average
    overlap = np.mean(min_eval)
    # 6. Put overlap in the title of the plot
    plt.title(id.capitalize()+': Overlap = %.3f' % overlap, fontsize=14)
    filename = parameter_labels[i] + '_' + id
    filename = "".join(x for x in filename if (x.isalnum() or x in "._- "))
    filename = filename.replace(' ', '_').lower()
    plt.savefig('../figures/gentoo/' + filename + '.pdf', format='pdf')
    plt.show()


# # Plot all of the two-dimensional histograms
# for i in range(dim):
#     for j in range(dim):
#         if i != j:
#             plt.figure()
#             sns.kdeplot(x=sampled_parameters[:, i], y=sampled_parameters[:, j], cmap="Blues", shade=True, thresh=0.05)
#             plt.xlabel(parameter_labels[i])
#             plt.ylabel(parameter_labels[j])
#             plt.show()