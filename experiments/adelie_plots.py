import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
from models import adelie
from algorithms import pf
import seaborn as sns

id = 'cormorant'

with np.load(id+'.npz', allow_pickle=True) as summary:
    data = summary['data']
    prior = summary['prior_parameters'].item()
    sampled_parameters = summary['sampled_parameters']
    parameter_labels = summary['parameter_labels']
    prior_states = summary['prior_states']
    dim = np.shape(sampled_parameters)[1]

# Obtain the predictive distribution
num_samples = np.shape(sampled_parameters)[0]   # Number of trajectories to sample
time_steps = np.shape(data)[0]

# Compute percentage missing
missing_breeders = np.sum(np.isnan(data[:, 0]))/time_steps
missing_chicks = np.sum(np.isnan(data[:, 1]))/time_steps
print(time_steps)
print('(%.2f, %.2f)' % (missing_breeders, missing_chicks))

# Initialize the observations
num_trajectories = 200
predictive_distribution = np.zeros((num_trajectories, time_steps, 2))
average_reproductive_success = np.zeros(num_trajectories)

# Generate predictions for all the sampled points
for s in range(num_trajectories):
    # Create the model (assuming the noise variances are known)
    regimes = [adelie.AgeStructuredModel(psi_juv=sampled_parameters[s, 0], psi_adu=sampled_parameters[s, 1],
                                         alpha_r=sampled_parameters[s, 2], beta_r=sampled_parameters[s, 4], nstage=5),
               adelie.AgeStructuredModel(psi_juv=sampled_parameters[s, 0], psi_adu=sampled_parameters[s, 1],
                                         alpha_r=sampled_parameters[s, 2]+sampled_parameters[s, 3],
                                         beta_r=sampled_parameters[s, 4], nstage=5)]
    draw_regimes = lambda model_idx, num_samp: np.random.choice(np.arange(start=0, stop=2), num_samp, replace=True,
                                                                p=[sampled_parameters[s, 5], 1-sampled_parameters[s, 5]])
    regimes_log_pdf = lambda model_idx: model_idx*np.log(1-sampled_parameters[s, 5])+(1-model_idx)*np.log(sampled_parameters[s, 5])
    # Create regime switching system
    model = pf.MultiRegimeSSM(regimes, draw_regimes, regimes_log_pdf)
    # Generate synthetic data
    init_state = 1+np.random.randint(low=prior_states[0], high=prior_states[1], size=(8))
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
plt.savefig('../figures/adelie/' + filename + '.pdf', format='pdf')
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
plt.savefig('../figures/adelie/' + filename + '.pdf', format='pdf')
plt.show()

# Plot the posterior for average reproductive success
plt.figure()
plt.hist(average_reproductive_success, bins='auto', density=True, alpha=0.75)
plt.title(id.capitalize(), fontsize=16)
plt.xlabel('Average Reproductive Success', fontsize=12)
plt.ylabel('Density', fontsize=12)
filename = 'avg_reproductive_success_'+id
plt.savefig('../figures/adelie/' + filename + '.pdf', format='pdf')
plt.show()

# # Matrix plot of the approximated target distribution
# plt.figure()
# count = 0
# for i in range(dim):
#     count += 1
#     for j in range(dim):
#         plt.subplot(dim, dim, count)
#         if i != j:
#             sns.kdeplot(x=sampled_parameters[:, j], y=sampled_parameters[:, i], cmap="Blues", shade=True, thresh=0.05)
#         else:
#             plt.hist(sampled_parameters[:, i])
# plt.show()
#
# Plot all of the one-dimensional histograms
posterior_mean = np.zeros(dim)
credible_intervals = np.zeros((dim, 2))
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
        g_hat = sp.norm.logpdf(sampled_parameters[:, i], loc=prior['reproductive bias (bad)'][0],
                               scale=np.sqrt(prior['reproductive bias (bad)'][1]))
    elif i == 3:
        g_hat = sp.uniform.logpdf(sampled_parameters[:, i], loc=prior['reproductive bias (good)'][0],
                                  scale=prior['reproductive bias (good)'][1])
    elif i == 4:
        g_hat = sp.norm.logpdf(sampled_parameters[:, i], loc=prior['reproductive slope'][0],
                               scale=np.sqrt(prior['reproductive slope'][1]))
    elif i == 5:
        g_hat = sp.beta.logpdf(sampled_parameters[:, i], prior['switching probability'][0],
                               prior['switching probability'][1])
    # 4. Evaluate the min function
    min_eval = np.minimum(np.exp(g_hat-f_hat), 1)
    # 5. Compute the overlap by taking a Monte Carlo average
    overlap = np.mean(min_eval)
    # 6. Put overlap in the title of the plot
    plt.title(id.capitalize()+': Overlap = %.3f' % overlap, fontsize=14)
    filename = parameter_labels[i] + '_' + id
    filename = "".join(x for x in filename if (x.isalnum() or x in "._- "))
    filename = filename.replace(' ', '_').lower()
    plt.savefig('../figures/adelie/' + filename + '.pdf', format='pdf')
    plt.show()
    # Compute the posterior mean along with the CIs of all parameters
    posterior_mean[i] = np.mean(sampled_parameters[:, i])
    print('Posterior Mean for theta['+str(i+1)+'] = ', posterior_mean[i])
    # Compute the 95% confidence intervals
    credible_intervals[i, 0] = np.quantile(sampled_parameters[:, i], 0.025)
    credible_intervals[i, 1] = np.quantile(sampled_parameters[:, i], 0.975)
    print('Credible Interval for theta['+str(i+1)+'] = ', credible_intervals[i])
    print('Prior-Posterior Overlap for theta['+str(i+1)+'] = ', overlap)
    print('')

