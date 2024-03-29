import numpy as np
import scipy.stats as sp


class AgeStructuredModel:
    """
    A class which contains all necessary methods for analyzing an age-structured model for Adelie penguin colonies.
    Objects are initialized by the number of assumed adult stages and the demographic parameters.
    """
    def __init__(self, psi_juv, psi_adu, alpha_r, beta_r, nstage):
        self.juvenile_survival = psi_juv                # Juvenile Survival Rate
        self.adult_survival = psi_adu                   # Adult Survival Rate
        self.reproductive_success_bias = alpha_r        # Reproductive Success (logit bias)
        self.reproductive_success_slope = beta_r        # Reproductive Success (logit slope)
        self.num_stages = nstage                        # Number of stages

        # Compute reproductive rates in logit space
        logit_rates = (self.reproductive_success_bias
                       + self.reproductive_success_slope*np.linspace(0, self.num_stages-1, self.num_stages))

        # Obtain probability of reproductive success for each age group
        self.reproductive_rates = 1./(1.+np.exp(-logit_rates))

    def transition_rand(self, old_state):
        """
        Propagates penguin populations from previous year using stage-structured dynamics.
        :param old_state: Latent penguin populations from the previous year
            - old_state[:, :num_stages] references the stage 1 to stage J adult penguins
            - old_state[:, -num_stages-2:] references the stage 1 to stage J-2 chicks
        :return state
        """
        if len(np.shape(old_state)) == 1:
            state = np.zeros(2*self.num_stages-2)
        else:
            # Determine the number of samples
            num_samples = np.shape(old_state)[1]
            # Set up matrix to output everything
            state = np.zeros((2*self.num_stages-2, num_samples))
        # Obtain reproductive rate in real space by applying sigmoid transformation
        pr = self.reproductive_rates
        # Compute the total number of chicks
        ct_old = np.sum(old_state[-(self.num_stages-2):], axis=0)
        # From total number of chicks to state 1 adults
        state[0] = np.array(np.random.binomial((ct_old/2).astype(int), self.juvenile_survival)).flatten()
        # Remainder of cycle
        for j in range(self.num_stages-1):
            # Propagate adults first
            if j < self.num_stages-2:
                state[j+1] = np.random.binomial(old_state[j].astype(int), self.adult_survival)
            else:
                state[j+1] = np.random.binomial((old_state[j]+old_state[j+1]).astype(int), self.adult_survival)
            # Obtain the chicks for the penguins that can breed
            if j >= 1:
                # Chicks obtained = binomial draw
                state[self.num_stages+j-1] = np.random.binomial(2*state[j+1].astype(int), pr[j-1])
        return state

    def transition_log_pdf(self, state, old_state):
        """
        Evaluate the logarithm of the transition distribution for the age-structured penguin model.
        :param old_state: Latent penguin populations (previous year)
            - state[:, :num_stages] references the stage 1 to stage J adult penguins
            - state[:, -num_stages-2:] references the stage 1 to stage J-2 chicks
        :param old_state: Latent penguin populations (previous year)
            - old_state[:, :num_stages] references the stage 1 to stage J adult penguins
            - old_state[:, -num_stages-2:] references the stage 1 to stage J-2 chicks
        :return logarithm of the transition distribution
        """
        if len(np.shape(old_state)) == 1:
            state = np.zeros(2*self.num_stages-2)
            log_transition = np.zeros(1)
        else:
            # Determine the number of samples
            num_samples = np.shape(old_state)[1]
            log_transition = np.zeros(num_samples)
        # Compute the total number of chicks
        ct_old = np.sum(old_state[-(self.num_stages-2):], axis=0)
        # From total number of chicks to state 1 adults
        log_transition += sp.binom.logpmf(state[0], (ct_old/2).astype(int), p=self.juvenile_survival)
        # Remainder of cycle
        for j in range(self.num_stages-1):
            # Propagate adults first
            if j < self.num_stages-2:
                log_transition += sp.binom.logpmf(state[j+1], old_state[j].astype(int), p=self.adult_survival)
            else:
                log_transition += sp.binom.logpmf(state[j+1], (old_state[j]+old_state[j+1]).astype(int),
                                                  p=self.adult_survival)
            # Obtain the chicks for the penguins that can breed
            if j >= 1:
                log_transition += sp.binom.logpmf(state[self.num_stages+j-1], 2*state[j+1].astype(int),
                                                  p=self.reproductive_rates[j-1])
        return log_transition

    def observation_rand(self, x, err_adults=0.1, err_chicks=0.1):
        """
        Generates noisy observations for latent penguin populations
        :param x: Latent penguin populations of the current year
            - x[:, :num_stages] references the stage 1 to stage J adult penguins
            - x[:, -num_stages-2:] references the stage 1 to stage J-2 chicks
        :return observed number of total breeders and observed number of total chicks
        NOTE: Need to keep in mind that I can have numerical errors due to the dimension of the array being used
        """
        # Determine the number of samples and allocate array for sample generation
        if len(np.shape(x)) == 1:
            y = np.zeros(2)
        else:
            num_samples = np.shape(x)[1]
            y = np.zeros((2, num_samples))
        # Extract the total number of breeders and chicks
        st = np.sum(x[2:self.num_stages], axis=0)         # total number of breeders
        ct = np.sum(x[-(self.num_stages-2):], axis=0)    # total number of chicks
        # Generate observations
        y[0] = np.random.normal(loc=st, scale=err_adults*st)
        y[1] = np.random.normal(loc=ct, scale=err_chicks*ct)
        return y.astype(int)

    def observation_log_pdf(self, obs, state):
        """
        Evaluate the logarithm of the observation distribution
        :param obs - Observed data
            - obs[0] - Total number of breeding adults
            - obs[1] - Total number of chicks
            - obs[2] - Percent error in count of breeding adults
            - obs[3] - Percent error in count of chicks
        :param state: Latent penguin populations of the current year
            - state[:, :num_stages] references the stage 1 to stage J adult penguins
            - state[:, -num_stages-2:] references the stage 1 to stage J-2 chicks
        :return logarithm of the observation distribution
        """
        # Determine the number of samples and allocate array for sample generation
        if len(np.shape(state)) == 1:
            log_observation = np.zeros(1)
        else:
            num_samples = np.shape(state)[1]
            log_observation = np.zeros(num_samples)
        # Extract the total number of breeders and chicks
        st = np.sum(state[2:self.num_stages], axis=0)            # total number of breeders
        ct = np.sum(state[-(self.num_stages-2):], axis=0)     # total number of chicks
        # Evaluate log pdf of the observations
        if ~np.isnan(obs[0]):
            log_observation += sp.norm.logpdf(obs[0], loc=st, scale=obs[2]*st)
        if ~np.isnan(obs[1]):
            log_observation += sp.norm.logpdf(obs[1], loc=ct, scale=obs[3]*ct)
        return log_observation