from data_gen import Generator
import numpy as np

def generate_sequence(min = 0.1, max = 10, per_condition = 5):
    return [np.array([[x, y], [y, x]]) for x in np.arange(min, max, (max - min)/per_condition) for y in np.arange(0, x*0.95, (x*0.95/per_condition))]

nc = 3 # Will end up running [nc^2]^2 total experiments [@ 3 => [3^2]^2 = 81]
variances_matrices = generate_sequence(0.1, 10, nc)

for vidx_U in range(len(variances_matrices)):
    for vidx_E in range(len(variances_matrices)):
        generator_cond = Generator(days = 2,
                                    dietary_components = 2,
                                    additional_variates = 0,
                                    bc_lambda = 1,
                                    beta = None,
                                    alpha = None,
                                    sigma_U = variances_matrices[vidx_U],
                                    sigma_Epsilon = variances_matrices[vidx_E],
                                    g_link_function = "identity",
                                    g_link_function_inv = None,
                                    h_function = None,
                                    Z_distribution = None,
                                    D_distribution = 'g')

        generator_cond.generate_csv(filename=f"custom_generator.id_{vidx_U}.{vidx_E}.csv", n = 1000, seed = 3141592)
