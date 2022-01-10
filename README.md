# Nutrition Data: Simulator
> This repository provides a Python class which implements simulated nutrition (dietary) data in a highly customizable manner, as would be represented by (e.g.,) 24-hour recalls.

## Python Generator Parameters
<table>
<tr>
<td>Parameter</td>
<td>Brief Description</td>
<td>Full Description with Allowable Values</td>
</tr>
<tr>
<td>n default = 1000</td>
<td>Default sample size.</td>
<td>Any positive integer value, at least equal to 0. This value can be overwritten when datasets are actually generated, but it will be used to reserve memory.  

```python
n = 1000 
```
</td>
</tr>
<tr>
<td>days default = 2</td>
<td>Number of days of dietary information to simulate.</td>
<td>A positive integer value that specifies the number of days of dietary data to keep. The data are simulated by generating a (very) large number of days, and this parameter specifies how many to save in the final data file.   

```python 
days = 2 
```
</td>
</tr>
<tr>
<td>dietary_components default = 2</td>
<td>The number of dietary components to simulate.</td>
<td>This can take any positive integer value that corresponds to the number of nutritional components to simulate (the dimension of the dietary factors in the data). The dietary factors will be specified based on a transformation of a multivariate normal distribution, and so these components will be correlated (generally).  

```python 
dietary_components = 5 
```
</td>
</tr>
<tr>
<td>additional_variates default = 0</td>
<td>The number of additional (non-dietary) factors to simulate.</td>
<td>This can take any integer value. These factors will be used both in the model for the dietary components and in the outcome model.  

```python 
additional_variates = 4 
```
</td>
</tr>
<tr>
<td>bc_lambda default = 0</td>
<td>The parameter for the Box-Cox transformation.</td>
<td>This can take any non-negative value. It is assumed that the dietary components are the inverse Box-Cox transformation of a linear combination of the additional variates, plus a random intercept term (per individual) plus a random error term (per individual per day).  

```python
bc_lambda = 0.2 
```
</td>
</tr>
<tr>
<td>beta default = None</td>
<td>The regression coefficients for the dietary components linear combinations.</td>
<td>This should be a 2D array, with dimension (dietary_components, additional_variates + 1). Each of the rows corresponds to the coefficients on the additional variates that form the (non-transformed) mean of the nutritional intake variable. The first value will be the intercept, followed by the additional variates in order.  

```python
# E[Y1|...] = 1 - 2*Z1 + 3*Z2 
# # E[Y2|...] = 4 - 0.5*Z2 
beta = np.array([[1, -2, 3], [4, 0, -0.5]]) 
```
</td>
</tr>
<tr>
<td>alpha default = None</td>
<td>The regression coefficients for the linear predictor of the outcome.</td>
<td>This should be a 1D array (or list) with size equal to (dietary_components + additional_variates + 1). These parameters form the coefficients for the linear combination of the dietary components with the first value being the intercept, then for the dietary components (in order), then for the additional_variates.  

```python
# eta = 1 + Y1 - Y2 + 3*Z1 
alpha = [1, 1, -1, 3] 
```
</td>
</tr>
<tr>
<td>sigma_U default = None</td>
<td>The covariance matrix for the random intercept terms (within individuals).</td>
<td>This should be a 2D array with size (dietary_components, dietary_components). It represents the covariance matrix for the random slopes that are computed at the individual level (generated from a multivariate normal distribution).  

```python 
# A Diagonal Matrix specifies independent components 
sigma_U = np.diag([1, 0.5, 1.8]) 
```
</td>
</tr>
<tr>
<td>sigma_Epsilon default = None</td>
<td>The covariance matrix for the measurement error terms (within individuals, within days).</td>
<td>This should be a 2D array with size (dietary_components, dietary_components). It represents the covariance matrix for the random error terms that occur within individuals, within days (generated from a multivariate normal distribution).  

```python
# If we have 2 components
sigma_Epsilon = np.array([[1,0.8],[0.8,1]]) 
```
</td>
</tr>
<tr>
<td>g_link_function default = "identity"</td>
<td>The link function connecting the linear predictor to the mean, for the outcome.</td>
<td>This should be either 'None', a string from {"identity", "logistic", "log"}, or a callable function that takes in and returns a single numeric value. If specifying a string, it only needs to be specified to the point of unique identification. If it is specified as a callable function, the inverse version (see below) needs to be provided as well!  

```python
g_link_function = 'logi' # Logistic link function 
g_link_function = lambda x : np.log(-1*np.log(1-x)) # Implements the c-log-log link function 
```
</td>
</tr>
<tr>
<td>g_link_function_inv default = None</td>
<td>The inverse of the link function specified for the outcome model.</td>
<td>This should either be 'None' (if 'None' or a string was specified for g_link_function), or else a callable function that provides the inverse of the specified g_link_function. 

```python 
g_link_function_inv = None # For the logistic model above 
g_link_function_inv = lambda x : 1-np.exp(-1*np.exp(x)) # Inverts the c-log-log link 
```
</td>
</tr>
<tr>
<td>h_function default = None</td>
<td>A function which transforms the set of dietary factors into the variables that we want included in the linear model.</td>
<td>This should either be 'None' (in which case no transformation takes place), or else a callable function that takes in the 2D array of generated dietary factors and outputs the relevant function forms for the outcome model.  

```python 
h_function = lambda x : x[:, 0] / x[:, 1] # If the ratio between Y1 and Y2 is of interest 
h_function = lambda x : np.hstack((x[:, 0]**2, x[:,0]*x[:,1])) # If Y1^2 and Y1*Y2 are of interest 
```
</td>
</tr>
<tr>
<td>Z_Distribution default = None</td>
<td>The (joint) distribution of the additional variates.</td>
<td>This should be either 'None', a string ('gaussian' or 'normal'), or else a callable function. If it is a callable function it should take two positional arguments: the random number generator (class of numpy.random.default_rng()) and the sample size to generate; it should return an array of size (n, additional_variates). If either string or None is provided, a multivariate normal is used.  

```python
# This will generate a Poisson(2) and N(0,1) random variate 
# independently 
Z_distribution = lambda rng, n : np.hstack((np.expand_dims(rng.poisson(2, n),axis=1), np.expand_dims(rng.normal(0, 1, n),axis=1))) 
```
</td>
</tr>
<tr>
<td>D_Distribution default = None</td>
<td>The distribution for the outcome.</td>
<td>This should be either 'None', a string from {'gaussian', 'normal', 'binomial', 'poisson'}, or a callable function. If it is None, it is treated as 'gaussian'. If the string is provided, only enough digits are required to uniquely specify the string. If it is callable, it should be just as Z_distribution, except this time it takes in the random number generator and a vector of 'means' (linear predictors transformed via the link function).  

```python
D_distribution = 'b' # Will be 'binomial' 
D_distribution = lambda : rng, means : rng.normal(means, 0.5) # Normal with var = 0.5 
```
</td>
</tr>
</table>

## Examples
```python
from data_gen import Generator

# Default Call, save to CSV
generator = Generator()
generator.generate_csv(filename="training_data.csv")

# The generator can be seeded, which is generally advisable
generator.generate_csv(filename="training_data.csv", seed = 3141592)

# Using the same generator, we can generate arbitrary size by specifying 'n'
generator.generate_csv(filename="large_sample.csv", n = 100000, seed = 3141592)

# We can also specify the data to be split into a training/testing set
# here we need to specify 'n' and 'n_val', and the saved files will be
# test_{filename} and train_{filename}
generator.generate_csv(filename="split_example.csv", n = 1000, n_val= 500, split = True, seed = 3141592)

# We can also generate the data directly to a pandas dataframe by calling
# generate_pd() instead
train_df = generator.generate_pd(n = 1000, seed = 3141592)

# If we split with generate_pd, then we need to unpack both train and test
train_df, test_df = generator.generate_pd(n = 1000, n_val = 500, split = True, seed = 3141592)

# We can also generate numpy arrays directly for the various columns
D_train, Truth_train, Y_train, Z_train = generator.generate_np(n=100, seed = 3141592)

# If we split within the numpy generator, then we unpack in order {train, test}
D_train, Truth_train, Y_train, Z_train, D_test, Truth_test, Y_test, Z_test = generator.generate_np(n=100, n_val = 50, split = True, seed = 3141592)

# All of the relevant parameters are changed when we initialize the generator
# These are fully documented on the README page, along with acceptable parameterizations
# For instance, the following will:
### Generate 5 days of food measurements
### Include 3 dietary components
### Include 4 additional variates
### Change the measurement error variances
### Specify explicit values for the coefficients on the outcome
### Change the outcome distribution to be binomial
### Explicitly set the box-cox parameter to be given by lambda = 1
# Then save off a training datafile with all of these variates generated, for n = 10000
import numpy as np

custom_generator = Generator(days = 5,
                             dietary_components = 3, 
                             additional_variates = 4,
                             sigma_Epsilon = np.diag([2,0.8,1.3]) + np.ones((3,3)),
                             alpha = [0.15, -1.5, 1, -0.5, 0.5, -1, 1, -0.8],
                             bc_lambda = 1,
                             D_distribution='binomial',
                             g_link_function='logit')

custom_generator.generate_csv(filename="custom_generator.csv", n = 10000, seed = 3141592)
```
