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

# REAL DATA
# PART 1: LOAD THE DATA
data = pickle.load(open( "../data/gentoo_data.pkl", "rb" ))
id = 'orne'
site_name = 'Orne Islands'
time_steps = data[site_name+' Years'].shape[0]
y = np.zeros((4, time_steps))
y[:2] = data[site_name+' Counts']
y[-2:] = np.ones((2, time_steps))*0.2 # data[site_name+' Errors']
y = y.T
# y = y[10:] # for Biscoe point

# # SYNTHETIC DATA
# # PART 1: LOAD THE DATA
# simulation = np.load('simulated_data.npz')
# id = 'synthetic'
# y = simulation['data']
# for t in range(np.shape(y)[0]):
#     if t > 0:
#         if np.random.rand() < 0.3:
#             y[t, 0] = np.nan
#             y[t, 2] = np.nan
#         if np.random.rand() < 0.8:
#             y[t, 1] = np.nan
#             y[t, 3] = np.nan

# Number of stages to use for model
num_stages = 5

# PART 2: ASSUMED MODEL
# Prior parameters
alpha_juv0 = 43              # Juvenile survival prior - beta prior (mean of 0.43)
beta_juv0 = 57
alpha_adu0 = 82              # Adult survival prior - beta prior (mean of 0.82)
beta_adu0 = 18
mu_int_pb0 = 0              # Breeding success (intercept in logit) - Normal(0,1)
var_int_pb0 = 10
mu_slope_pb0 = 0.5            # Breeding success (slope in logit) - Normal(0, 1)
var_slope_pb0 = 0.2**2
im_bias_low = 0             # Immigration Rate Bias - assumed positive - Uniform(0, 50)
im_bias_high = 50 - im_bias_low
im_slope_low = -0.5         # Immigration Rate Slope - Uniform(-0.5, 0.5)
im_slope_high = abs(2*im_slope_low)

# Store prior parameters in summary
prior = {'juvenile survival': [alpha_juv0, beta_juv0],
         'adult survival': [alpha_adu0, beta_adu0],
         'reproductive bias': [mu_int_pb0, var_int_pb0],
         'reproductive slope': [mu_slope_pb0, var_slope_pb0],
         'immigration bias': [im_bias_low, im_bias_high],
         'immigration slope': [im_slope_low, im_bias_high]}


# Likelihood function
def log_likelihood_per_sample(input_parameters):
    # Set the random seed
    np.random.seed()
    # Define the number of particles
    num_particles = 2000
    # Apply relevant transformations to the sample (sigmoid transformation to probability parameters)
    z = np.copy(input_parameters)
    z[0] = 1/(1+np.exp(-z[0]))
    z[1] = 1/(1+np.exp(-z[1]))
    z[3] = np.exp(z[3])
    z[4] = np.exp(z[4])
    # Evaluate prior distribution at transformed samples (don't forget to factor in Jacobian from transformation)
    log_prior = sp.beta.logpdf(z[0], alpha_juv0, beta_juv0) + log_jacobian_sigmoid(input_parameters[0])
    log_prior += sp.beta.logpdf(z[1], alpha_adu0, beta_adu0) + log_jacobian_sigmoid(input_parameters[1])
    log_prior += sp.norm.logpdf(z[2], mu_int_pb0, np.sqrt(var_int_pb0))
    log_prior += sp.norm.logpdf(z[3], mu_slope_pb0, np.sqrt(var_slope_pb0)) + input_parameters[3]
    log_prior += sp.uniform.logpdf(z[4], loc=im_bias_low, scale=im_bias_high) + input_parameters[4]
    log_prior += sp.uniform.logpdf(z[5], loc=im_slope_low, scale=im_slope_high)
    # Initialize log joint as log prior
    log_joint = log_prior
    # Create the model (assuming the noise variances are known)
    regimes = [gentoo.AgeStructuredModel(psi_juv=z[0], psi_adu=z[1], alpha_r=z[2], beta_r=z[3], phi_0=z[4],
                                         phi_1=z[5], nstage=num_stages)]
    draw_regimes = lambda model_idx, num_samp: np.random.choice(np.arange(start=0, stop=1), num_samp, replace=True,
                                                                p=np.array([1]))
    regimes_log_pdf = lambda model_idx: (1-model_idx)*np.log(1)
    # Create regime switching system
    model = pf.MultiRegimeSSM(regimes, draw_regimes, regimes_log_pdf)
    # Initialize the particles
    initial_states = 1*np.ones(2*num_stages-2)
    init_particles = np.array([initial_states]).T+np.random.randint(low=0, high=10, size=(2*num_stages-2, num_particles))
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
    dim = 6  # dimension of the unknown parameter
    N = 300     # number of samples per proposal
    I = 20      # number of iterations
    N_w = 50    # number of samples per proposal (warm-up period)
    I_w = 200   # number of iterations (warm-up period)
    D = 10      # number of proposals
    eta_loc = 2e-1  # learning rate for the mean
    eta_scale = 2e-1    # learning rate for the covariance matrix
    # Select initial proposal parameters
    mu_init = np.zeros((D, dim))
    sig_init = np.zeros((D, dim, dim))
    for j in range(D):
        # Prior proposal parameters for juvenile survival
        mu_init[j, 0] = np.random.uniform(0.42, 0.44)
        mu_init[j, 0] = np.log(mu_init[j, 0]/(1-mu_init[j, 0]))
        sig_init[j, 0, 0] = 0.03**2
        # Prior proposal parameters for adult survival
        mu_init[j, 1] = np.random.uniform(0.81, 0.83)
        mu_init[j, 1] = np.log(mu_init[j, 1]/(1-mu_init[j, 1]))
        sig_init[j, 1, 1] = 0.03**2
        # Prior proposal parameters for logit intercept (bad year)
        mu_init[j, 2] = np.random.uniform(0, 1.5)
        sig_init[j, 2, 2] = 0.3**2
        # Prior proposal parameters for logit slope
        mu_init[j, 3] = np.random.uniform(0, 0.2)
        mu_init[j, 3] = np.log(mu_init[j, 3])
        sig_init[j, 3, 3] = 0.1**2
        # Prior proposal parameters for logit slope
        mu_init[j, 4] = np.random.uniform(10, 40)
        mu_init[j, 4] = np.log(mu_init[j, 4])
        sig_init[j, 4, 4] = 0.1**2
        # Prior proposal parameters for logit slope
        mu_init[j, 5] = np.random.uniform(-0.05, 0.05)
        sig_init[j, 5, 5] = 0.02**2
    # Warm up the sampler by running it for some number of iterations
    init_sampler = ais.ais(log_target=log_pi, d=dim, mu=mu_init, sig=sig_init, samp_per_prop=N_w, iter_num=I_w,
                           temporal_weights=False, weight_smoothing=True, eta_mu0=eta_loc, eta_sig0=eta_scale,
                           criterion='Moment Matching', optimizer='Constant')
    # Run sampler with initialized parameters
    output = ais.ais(log_target=log_pi, d=dim, mu=init_sampler.means[-D:], sig=init_sampler.covariances[-D:],
                     samp_per_prop=N, iter_num=I, weight_smoothing=True, temporal_weights=True, eta_mu0=eta_loc,
                     eta_sig0=eta_scale, criterion='Moment Matching', optimizer='RMSprop')
    # Use sampling importance resampling to extract posterior samples
    theta = ais.importance_resampling(output.particles, output.log_weights, num_samp=5000)
    # Apply transformations to the samples
    theta[:, 0] = 1/(1+np.exp(-theta[:, 0]))
    theta[:, 1] = 1/(1+np.exp(-theta[:, 1]))
    theta[:, 3] = np.exp(theta[:, 3])
    theta[:, 4] = np.exp(theta[:, 4])
    # Create labels for each parameter
    state_labels = ['$S_{1,t}$', '$S_{2, t}$', '$S_{3, t}$', '$S_{4, t}$', '$S_{5, t}$', '$C_{3,t}$', '$C_{4, t}$',
                    '$C_{5, t}$']
    parameter_labels = ['Juvenile Survival', 'Adult Survival', 'Reproductive Rate (Logit Intercept)',
                        'Reproductive Rate (Logit Slope)', 'Immigration Rate (Intercept)', 'Immigration Rate (Slope)']
    # Save posterior samples to a file
    # Save the results
    filename = id+'.npz'
    np.savez(filename, data=y, prior_parameters=prior, sampled_parameters=theta, parameter_labels=parameter_labels)

