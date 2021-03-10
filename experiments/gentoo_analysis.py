import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
import multiprocessing
import seaborn as sns
import pickle
from models import gentoo
from algorithms import pf
from algorithms import ais


## DEFINITIONS
def log_jacobian_sigmoid(x): return -x-2*np.log(1+np.exp(-x))



# Set the random seed
np.random.seed(1)

# Parameters of the true model
#   -param[0] - juvenile survival
#   -param[1] - adult survival
#   -param[2] - reproductive success bias (bad year)
#   -param[3] - reproductive success bias (good year)
#   -param[4] - reproductive success slope
#   -param[5] - probability of bad year
#   -param[6] - variance of observations (breeders)
#   -param[7] - variance of observations (chicks)


# PART 1: LOAD THE DATA
data = pickle.load(open( "../data/gentoo_data.pkl", "rb" ))
y = np.zeros((4, data['Orne Islands Years'].shape[0]))
y[:2] = data['Orne Islands Counts']
y[-2:] = data['Orne Islands Errors']*2
y = y.T

# Number of stages to use for model
num_stages = 5

# PART 2: ASSUMED MODEL
# Prior parameters
alpha_juv0 = 3              # Juvenile survival prior - beta prior
beta_juv0 = 3
alpha_adu0 = 3              # Adult survival prior - beta prior
beta_adu0 = 3
mu_int_pb0 = 0              # Breeding success (intercept in logit) - normal prior
var_int_pb0 = 1
mu_slope_pb0 = 0.15         # Breeding success (slope in logit) - normal prior
var_slope_pb0 = 0.01
mu_im_rate = 25             # Immigration Rate - normal prior
var_im_rate = 100


# Likelihood function
def log_likelihood_per_sample(input_parameters):
    # Set the random seed
    np.random.seed()
    # Define the number of particles
    num_particles = 1000
    # Apply relevant transformations to the sample (sigmoid transformation to probability parameters)
    z = np.copy(input_parameters)
    z[0] = 1/(1+np.exp(-z[0]))
    z[1] = 1/(1+np.exp(-z[1]))
    z[3] = np.exp(z[3])
    z[4] = np.exp(z[4])
    # Evaluate prior distribution at transformed samples (don't forget to factor in Jacobian from transformation)
    log_prior = log_jacobian_sigmoid(input_parameters[0])+sp.beta.logpdf(z[0], alpha_juv0, beta_juv0)
    log_prior += log_jacobian_sigmoid(input_parameters[1])+sp.beta.logpdf(z[1], alpha_adu0, beta_adu0)
    log_prior += sp.norm.logpdf(z[2], mu_int_pb0, np.sqrt(var_int_pb0))
    log_prior += input_parameters[3]+sp.norm.logpdf(z[3], mu_slope_pb0, np.sqrt(var_slope_pb0))
    log_prior += input_parameters[4]+sp.norm.logpdf(z[4], mu_im_rate, np.sqrt(var_im_rate))
    # log_prior += log_jacobian_sigmoid(input_parameters[5])+sp.beta.logpdf(z[5], alpha_gamma0, beta_gamma0)
    # Initialize log joint as log prior
    log_joint = log_prior
    # Create the model (assuming the noise variances are known)
    regimes = [gentoo.AgeStructuredModel(psi_juv=z[0], psi_adu=z[1], alpha_r=z[2], beta_r=z[3], phi=z[4], nstage=num_stages)]
    draw_regimes = lambda model_idx, num_samp: np.random.choice(np.arange(start=0, stop=1), num_samp, replace=True, p=np.array([1]))
    regimes_log_pdf = lambda model_idx: (1-model_idx)*np.log(1)
    # Create regime switching system
    model = pf.MultiRegimeSSM(regimes, draw_regimes, regimes_log_pdf)
    # Initialize the particles
    err = 5
    initial_states = 10*np.ones(2 * num_stages - 2)
    init_particles = np.array([initial_states]).T+np.random.randint(low=-err, high=err, size=(2*num_stages-2, num_particles))
    # Run the particle filter and return the log-likelihood
    output = pf.brspf(y, model, init_particles)
    # Update the log joint
    log_joint += output.log_evidence
    return log_joint


# PART 3: PARAMETER INFERENCE
if __name__ == '__main__':
    # Set a random seed
    np.random.seed()
    # Setup the multiprocessing bit
    multiprocessing.Process()
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cores)
    # Define the target distribution
    log_pi = lambda x: pool.map(log_likelihood_per_sample, x)
    # Define the sampler parameters
    dim = 5  # dimension of the unknown parameter
    N = 200  # number of samples per proposal
    I = 50  # number of iterations
    N_w = 50  # number of samples per proposal (warm-up period)
    I_w = 150   # number of iterations (warm-up period)
    D = 10      # number of proposals
    var_0 = 1e-1   # initial variance
    eta_loc = 2e-1  # learning rate for the mean
    eta_scale = 2e-1    # learning rate for the covariance matrix
    # Select initial proposal parameters
    mu_init = np.zeros((D, dim))
    sig_init = np.zeros((D, dim, dim))
    for j in range(D):
        # Prior proposal parameters for juvenile survival
        mu_init[j, 0] = np.random.uniform(0.2, 0.6)
        mu_init[j, 0] = np.log(mu_init[j, 0]/(1-mu_init[j, 0]))
        sig_init[j, 0, 0] = var_0
        # Prior proposal parameters for adult survival
        mu_init[j, 1] = np.random.uniform(0.65, 0.95)
        mu_init[j, 1] = np.log(mu_init[j, 1]/(1-mu_init[j, 1]))
        sig_init[j, 1, 1] = var_0
        # Prior proposal parameters for logit intercept (bad year)
        mu_init[j, 2] = np.random.uniform(-2, -0.5)
        sig_init[j, 2, 2] = var_0
        # Prior proposal parameters for difference in logit intercepts
        mu_init[j, 3] = np.random.uniform(0, 0.3)
        mu_init[j, 3] = np.log(mu_init[j, 3])
        sig_init[j, 3, 3] = var_0
        # Prior proposal parameters for logit slope
        mu_init[j, 4] = np.random.uniform(10, 40)
        mu_init[j, 4] = np.log(mu_init[j, 4])
        sig_init[j, 4, 4] = var_0
    # Warm up the sampler by running it for some number of iterations
    init_sampler = ais.ais(log_target=log_pi, d=dim, mu=mu_init, sig=sig_init, samp_per_prop=N_w, iter_num=I_w,
                           temporal_weights=False, weight_smoothing=True, eta_mu0=eta_loc, eta_sig0=eta_scale,
                           criterion='Moment Matching', optimizer='Constant')
    # Run sampler with initialized parameters
    output = ais.ais(log_target=log_pi, d=dim, mu=init_sampler.means[-D:], sig=init_sampler.covariances[-D:],
                     samp_per_prop=N, iter_num=I, weight_smoothing=True, temporal_weights=True, eta_mu0=0.1*eta_loc,
                     eta_sig0=0.1*eta_scale, criterion='Minimum Variance', optimizer='RMSprop')
    # Use sampling importance resampling to extract posterior samples
    theta = ais.importance_resampling(output.particles, output.log_weights, num_samp=1000)
    # Apply transformations to the samples
    theta[:, 0] = 1/(1+np.exp(-theta[:, 0]))
    theta[:, 1] = 1/(1+np.exp(-theta[:, 1]))
    # theta[:, 3] = theta[:, 2] + np.exp(theta[:, 3])
    theta[:, 3] = np.exp(theta[:, 3])
    theta[:, 4] = np.exp(theta[:, 4])
    # theta[:, 5] = 1/(1+np.exp(-theta[:, 5]))
    # Create labels for each parameter
    labels = ['Juvenile Survival', 'Adult Survival', 'Logit Intercept', 'Logit Slope', 'Immigration Rate']
    # Matrix plot of the approximated target distribution
    plt.figure()
    count = 1
    for i in range(dim):
        for j in range(dim):
            plt.subplot(dim, dim, count)
            if i != j:
                sns.kdeplot(theta[:, j], theta[:, i], cmap="Blues", shade=True, shade_lowest=False)
            else:
                plt.hist(theta[:, i])
            count += 1
    plt.show()
    # Plot all of the one-dimensional histograms
    for i in range(dim):
        # Plot the posterior
        plt.figure()
        plt.hist(theta[:, i])
        # plt.axvline(param[i])
        plt.xlabel(labels[i])
        # Compute the overlap
        # 1. First get a KDE
        est_post = sp.gaussian_kde(theta[:, i])
        # 2. Evaluate the samples at the estimated KDE
        f_hat = est_post.logpdf(theta[:, i])
        # 3. Evaluate the samples at the marginal prior
        if i == 0:
            g_hat = sp.beta.logpdf(theta[:, i], alpha_juv0, beta_juv0)
        elif i == 1:
            g_hat = sp.beta.logpdf(theta[:, i], alpha_adu0, beta_adu0)
        elif i == 2:
            g_hat = sp.norm.logpdf(theta[:, i], mu_int_pb0, np.sqrt(var_int_pb0))
        # elif i == 3:
        #     g_hat = sp.invgamma.logpdf(theta[:, i]-theta[:, i-1], a=alpha_diff0, scale=beta_diff0)
        elif i == 3:
            g_hat = sp.norm.logpdf(theta[:, i], mu_slope_pb0, np.sqrt(var_slope_pb0))
        elif i == 4:
            g_hat = sp.norm.logpdf(theta[:, i], mu_im_rate, np.sqrt(var_im_rate))
        # elif i == 5:
        #     g_hat = sp.beta.logpdf(theta[:, i], alpha_gamma0, beta_gamma0)
        # 4. Evaluate the min function
        min_eval = np.minimum(np.exp(g_hat-f_hat), 1)
        # 5. Compute the overlap by taking a Monte Carlo average
        overlap = np.mean(min_eval)
        # 6. Put overlap in the title of the plot
        plt.title('Overlap = %.3f' % overlap)
        plt.show()
    # Plot all of the two-dimensional histograms
    for i in range(dim):
        for j in range(dim):
            if i != j:
                plt.figure()
                sns.kdeplot(theta[:, i], theta[:, j], cmap="Blues", shade=True, shade_lowest=False)
                plt.xlabel(labels[i])
                plt.ylabel(labels[j])
                plt.show()