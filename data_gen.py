import math
import numpy as np
import pandas as pd

class Generator:
    DAYS_TO_AVERAGE = 10000

    def __init__(self, 
                 n = 1000, 
                 days = 2, 
                 dietary_components = 2, 
                 additional_variates = 0,
                 bc_lambda = 0,
                 beta = None,
                 alpha = None,
                 sigma_U = None,
                 sigma_Epsilon = None,
                 g_link_function = "identity",
                 g_link_function_inv = None,
                 h_function = None,
                 Z_distribution = None,
                 D_distribution = None):
        
        self.rng = np.random.default_rng()

        self.n = n
        self.m = days
        self.p = dietary_components
        self.q = additional_variates
        self.bc_lambda = bc_lambda

        ## h_function allows to consider transformations of the different dietary components
        ## it should be a function which takes as input true dietary intakes, and outputs 'r' transformed variates
        ## it should act on the np.array with dimension (self.n, self.p) directly. 
        ## if 'None' then the identity is taken
        ## otherwise it must be callable.
        ## 'r' is updated to depend on the output dimension!
        if h_function is None:
            self.h_function = lambda x : x
            self.r = self.p
        elif callable(h_function):
            self.h_function = h_function
            self.r = self.h_function(np.ones((1, self.p))).shape[1]
            
        else:
            raise ValueError("h_function should either be None or a callable object which takes in an np.array and returns a modified np.array.")

        ## Assign beta and alpha coefficients, with checks for correction dimensions 
        #### beta: (p length, size of q+1)
        #### alpha: (q + r + 1)
        if beta is None:
            self.beta = [[1]*(self.q+1)]*self.p
        else:
            if len(beta) != self.p:
                raise ValueError(f"It is expected that beta has 'p' components in it. Beta had {len(beta)}, while p was {self.p}.")
            elif any([len(x) != (self.q+1) for x in beta]):
                raise ValueError(f"It is expected that each component of beta has 'q+1' components. There were {[len(x) for x in beta]} respectively, while q was {self.q}.")
            else:
                self.beta = beta
        if alpha is None:
            self.alpha = [1]*(self.q+self.r+1)
        else:
            if len(alpha) != (1 + self.q + self.r):
                raise ValueError(f"It is expected that alpha has 'q+r+1' components in it. Alpha had {len(alpha)}, while q was {self.q} and r was {self.r}.")
            else:
                self.alpha = alpha

        ## Assign Variance Matrices [U and Epsilon] with dimensionality checks (p-by-p)
        if sigma_U is None:
            self.sigma_U = 0.5*np.ones((self.p, self.p)) + np.diag([0.5]*self.p)
        else:
            if len(sigma_U) != self.p or any([len(x) != self.p for x in sigma_U]):
                raise ValueError(f"It is expected that sigma_U is a 'p'-by-'p' matrix. We have p={self.p}, while there are {len(sigma_U)} rows with lengths {[len(x) for x in sigma_U]} in sigma_U.")
            else:
                self.sigma_U = sigma_U
        if sigma_Epsilon is None:
            self.sigma_Epsilon = 0.5*np.ones((self.p, self.p)) + np.diag([0.5]*self.p)
        else:
            if len(sigma_Epsilon) != self.p or any([len(x) != self.p for x in sigma_Epsilon]):
                raise ValueError(f"It is expected that sigma_Epsilon is a 'p'-by-'p' matrix. We have p={self.p}, while there are {len(sigma_Epsilon)} rows with lengths {[len(x) for x in sigma_Epsilon]} in sigma_Epsilon.")
            else:
                self.sigma_Epsilon = sigma_Epsilon
        
        ## Assign the link function and inverse link function
        # Can be provided either as a function itself, or as a string
        # As String:
        # Possible string values include 'identity', 'logistic', or 'log'
        # only need to provide enough to uniquely identify them
        # As Function:
        # Inverse Link needs to be provided as well
        ### g_link_function
        if isinstance(g_link_function, str):
            g_link_function = g_link_function.lower()

            if g_link_function[0] == 'i':
                # Identity Link Function
                self.g_link = lambda x : x
                self.g_link_function_inv = lambda x : x
            elif g_link_function[0:4] == 'logi':
                # Logistic Link Function
                self.g_link = lambda x : np.log(x/(1-x))
                self.g_link_function_inv = lambda x : (1 + np.exp(-1*x))**(-1)
            elif g_link_function == 'log':
                # Log Link Function
                self.g_link = lambda x : np.log(x)
                self.g_link_function_inv = lambda x : np.exp(x)
        elif callable(g_link_function) and callable(g_link_function_inv):
            self.g_link = g_link_function
            self.g_link_function_inv = g_link_function_inv
        else:
            raise ValueError("The g_link_function needs to be provided either as a string ['identity', 'logistic', 'log'] or as a callable function. If provided as a function, g_link_function_inv, the inverse of the link, needs to be provided as well.")

        ## Distribution generation for the additional covariates
        ## Distribution function should be based on self.rng as the first argument, and 'n' as the second
        if Z_distribution is None or (isinstance(Z_distribution, str) and Z_distribution.lower()[0] in ['g', 'n']):
            self.Z_distribution = lambda n : self.rng.multivariate_normal(mean=np.zeros(self.q), cov=0.5*np.ones((self.q, self.q))+np.diag([0.5]*self.q), size=n)
        elif callable(Z_distribution):
            self.Z_distribution = lambda n : Z_distribution(self.rng, n)
        else:
            raise ValueError("Z_distribution should either be None, a string (either 'normal' or 'gaussian'), or a callable function.")

        ## Distribution generation for the outcome
        # Should be a function which takes expected_D in as the first parameter and returns
        # generated outcomes 
        if D_distribution is None or (isinstance(D_distribution, str) and D_distribution.lower()[0] in ['g', 'n', 'b', 'p']):
            if D_distribution is None or D_distribution.lower()[0] in ['g', 'n']:
                # Generate normal realizations, the means are taken as means, with unit variance
                self.D_distribution = lambda means : self.rng.normal(means, scale=1)
            elif D_distribution.lower()[0] == 'b':
                # Generate Binomial Realizations, here 'means' represent probabilities
                self.D_distribution = lambda means : self.rng.binomial(1, means)
            elif D_distribution.lower()[0] == 'p':
                # Generate poisson realizations, the 'means' represent the rates
                self.D_distribution = lambda means : self.rng.poisson(means)
        elif callable(D_distribution):
            # D_distribution should be based on self.rng and means
            self.D_distribution = lambda means : D_distribution(self.rng, means)
        else:
            raise ValueError("The D_distribution must either be None (in which case a unit normal is assumed), a string from ['normal', 'gaussian', 'poisson', 'binomial'], or a callable function which takes in an np.array of 'means' for each individual and returns draws from the outcome distribution.")

        ## Intialize Data Components to Reserve Memory
        self._reserve_memory()

    def _reserve_memory(self):
        self.Z = np.zeros((self.n, self.q))
        self.Y = [np.zeros((self.n, self.p)) for _ in range(self.m)]
        self.truth = np.zeros((self.n,self.p))
        self.expected_D = np.zeros(self.n)
        self.D = np.zeros((self.n,1))

    ## Define the inverse box-cox transformation to allow for the variate to be properly transformed 
    ## from the linear scale, back.
    def _inv_box_cox(self, Y):
        if self.bc_lambda == 0:
            return np.exp(Y)
        elif self.bc_lambda == -1:
            return Y
        else:
            return np.power(self.bc_lambda*Y + 1, 1/self.bc_lambda)
        
    ## Define helper function to simulate the distribution of the additional variates
    ## Separated out in case this needs to change in the future.
    ## Should not be directly called.
    def _simulate_variates(self):
        if self.q != 0:
            self.Z = self.Z_distribution(self.n)

    ## The simulate_dietary_components will generate the error-prone observations (stored in self.Y) 
    ## in addition to the 'true' long-term average values (stored in self.truth) by averaging
    ## over many repeated iterations.
    ## This will not need to be called directly, as it will be called by simulate_outcome
    def _simulate_dietary_components(self):
        self._simulate_variates()
        u = self.rng.multivariate_normal(mean = [0]*self.p, cov = self.sigma_U, size = self.n)
        mus = []

        for k in range(self.p):
            mus.append(self.beta[k][0] + u[:,k])
            if self.q != 0:
                mus[k] += np.matmul(self.Z, self.beta[k][1:])

        ## Simulate a large number of days
        wt = 1/(self.DAYS_TO_AVERAGE + self.m)
        for _ in range(self.DAYS_TO_AVERAGE):
            eps = self.rng.multivariate_normal(mean = [0]*self.p, cov = self.sigma_Epsilon, size = self.n)
            for k in range(self.p):
                    self.truth[:, k] += wt*(self._inv_box_cox(mus[k] + eps[:,k]))

        for j in range(self.m):
            eps = self.rng.multivariate_normal(mean = [0]*self.p, cov = self.sigma_Epsilon, size = self.n)
            for k in range(self.p):
                self.Y[j][:, k] = self._inv_box_cox(mus[k] + eps[:,k])
                self.truth[:, k] += wt*(self.Y[j][:, k])

    ## The simulate_mean_outcome will generate the mean of the outcomes, after making a call to simulate_dietary_components
    ## in order to store the means self.expected_D.
    ## This will not need to be called directly!
    def _simulate_mean_outcome(self):
        self._simulate_dietary_components()
        eta = self.alpha[0] + np.matmul(self.h_function(self.truth), self.alpha[1:(self.r+1)]) + np.matmul(self.Z, self.alpha[(self.r+1):])
        self.expected_D = np.expand_dims(np.array([self.g_link_function_inv(e) for e in eta]), axis=1)

    ## Generates the true outcomes 
    ## This should not need to be called directly, because it can be accessed from generate sample
    def _generate_outcomes(self):
        self._simulate_mean_outcome()
        self.D = self.D_distribution(self.expected_D)

    ## Helper Function to set the value of 'n'
    def _update_n(self, n):
        self.n = n
        self._reserve_memory()

    ## Actual Generators are provided here
    ## The function calls will generate the necessary data, and save off the results either
    ### As a CSV (filename required, in addition to other parameters)
    ### As a series of np.arrays 
    ### As a pandas data frame
    # For each of these you can specify the sample size(s), a random seed (if desired), and whether to split (if so, val_size required)
    def generate_np(self, n = None, n_val = None, split = False, seed=None):
        if split and (n is None or n_val is None):
            raise ValueError("If you wish to generate a validation sample alongside the training sample, both 'n' and 'n_val' must be provided.")
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        if split:
            old_n = n
            n = n + n_val

        if n is not None:
            self._update_n(n)
        
        self._generate_outcomes()

        if split:
            return self.D[:old_n], self.truth[:old_n, ], [Y[:old_n, ] for Y in self.Y], self.Z[:old_n, ], self.D[old_n:], self.truth[old_n:, ], [Y[old_n:, ] for Y in self.Y], self.Z[old_n:, ]
        
        return self.D, self.truth, self.Y, self.Z

    def generate_pd(self, n = None, n_val = None, split = False, seed=None):
        if split:
            D_train, Truth_train, Y_train, Z_train, D_test, Truth_test, Y_test, Z_test = self.generate_np(n, n_val, split, seed)
            df_Train = pd.DataFrame(np.hstack((D_train, Truth_train, *Y_train, Z_train)), columns = ['Outcome'] + [f'Truth_{i+1}' for i in range(Truth_train.shape[1])] + [f'Y_{i+1}{j+1}' for i in range(len(Y_train)) for j in range(Y_train[i].shape[1])] + [f'Z_{i+1}' for i in range(Z_train.shape[1])])
            df_Test = pd.DataFrame(np.hstack((D_test, Truth_test, *Y_test, Z_test)), columns = ['Outcome'] + [f'Truth_{i+1}' for i in range(Truth_train.shape[1])] + [f'Y_{i+1}{j+1}' for i in range(len(Y_train)) for j in range(Y_train[i].shape[1])] + [f'Z_{i+1}' for i in range(Z_train.shape[1])])
            return df_Train, df_Test
        
        D_train, Truth_train, Y_train, Z_train = self.generate_np(n, n_val, split, seed)

        return pd.DataFrame(np.hstack((D_train, Truth_train, *Y_train, Z_train)), columns = ['Outcome'] + [f'Truth_{i+1}' for i in range(Truth_train.shape[1])] + [f'Y_{i+1}{j+1}' for i in range(len(Y_train)) for j in range(Y_train[i].shape[1])] + [f'Z_{i+1}' for i in range(Z_train.shape[1])])

    def generate_csv(self, filename, folder = None, n = None, n_val = None, split = False, seed=None):
        ## Check valid
        if filename is None:
            raise ValueError("A filename must be provided in order to write to a csv.")
        
        if folder is None:
            folder = ""
        else:
            folder = f"{folder}/"

        if split:
            df_Train, df_Test = self.generate_pd(n, n_val, split, seed)

            df_Train.to_csv(f"{folder}train_{filename}", index = False)
            df_Test.to_csv(f"{folder}test_{filename}", index = False)
            return None
        
        df_Train = self.generate_pd(n, n_val, split, seed)
        df_Train.to_csv(f"{folder}{filename}", index = False)
