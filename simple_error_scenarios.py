from data_gen import Generator
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

uncorrelated_errors_scenarios = [
    {
        'sigma_Epsilon': [[0.5, 0], [0, 1]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4, -2],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x : np.column_stack((x[:, 0], x[:, 1])),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[1, 0], [0, 2]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4, -2],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x : np.column_stack((x[:, 0], x[:, 1])),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[2, 0], [0, 3]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4, -2],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x : np.column_stack((x[:, 0], x[:, 1])),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[4, 0], [0, 6]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4, -2],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x : np.column_stack((x[:, 0], x[:, 1])),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[6, 0], [0, 9]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4, -2],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x : np.column_stack((x[:, 0], x[:, 1])),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[9, 0], [0, 13.5]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4, -2],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x : np.column_stack((x[:, 0], x[:, 1])),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[12, 0], [0, 18]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4, -2],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x : np.column_stack((x[:, 0], x[:, 1])),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[20, 0], [0, 26]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4, -2],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x : np.column_stack((x[:, 0], x[:, 1])),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[25, 0], [0, 32.5]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4, -2],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x : np.column_stack((x[:, 0], x[:, 1])),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[40, 0], [0, 52]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4, -2],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x : np.column_stack((x[:, 0], x[:, 1])),
        'Z_distribution': None, 'D_distribution': None
    }, 
]

correlated_errors_scenarios = [
    {
        'sigma_Epsilon': [[0.5, 0.25], [0.25, 1]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4, -2],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x : np.column_stack((x[:, 0], x[:, 1])),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[1, 0.5], [0.5, 2]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4, -2],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x : np.column_stack((x[:, 0], x[:, 1])),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[2, 1], [1, 3]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4, -2],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x : np.column_stack((x[:, 0], x[:, 1])),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[4, 2], [2, 6]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4, -2],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x : np.column_stack((x[:, 0], x[:, 1])),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[6, 3], [3, 9]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4, -2],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x : np.column_stack((x[:, 0], x[:, 1])),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[9, 6], [6, 13.5]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4, -2],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x : np.column_stack((x[:, 0], x[:, 1])),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[12, 9], [9, 18]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4, -2],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x : np.column_stack((x[:, 0], x[:, 1])),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[20, 15], [15, 26]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4, -2],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x : np.column_stack((x[:, 0], x[:, 1])),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[25, 19], [19, 32.5]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4, -2],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x : np.column_stack((x[:, 0], x[:, 1])),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[40, 30], [30, 52]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4, -2],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x : np.column_stack((x[:, 0], x[:, 1])),
        'Z_distribution': None, 'D_distribution': None
    }, 
]

uncorrelated_errors_scenarios_ratio = [
    {
        'sigma_Epsilon': [[0.5, 0], [0, 1]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x: np.array(x[:,0]/x[:,1]).reshape((x.shape[0],1)),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[1, 0], [0, 2]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x: np.array(x[:,0]/x[:,1]).reshape((x.shape[0],1)),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[2, 0], [0, 3]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x: np.array(x[:,0]/x[:,1]).reshape((x.shape[0],1)),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[4, 0], [0, 6]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x: np.array(x[:,0]/x[:,1]).reshape((x.shape[0],1)),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[6, 0], [0, 9]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x: np.array(x[:,0]/x[:,1]).reshape((x.shape[0],1)),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[9, 0], [0, 13.5]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x: np.array(x[:,0]/x[:,1]).reshape((x.shape[0],1)),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[12, 0], [0, 18]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x: np.array(x[:,0]/x[:,1]).reshape((x.shape[0],1)),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[20, 0], [0, 26]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x: np.array(x[:,0]/x[:,1]).reshape((x.shape[0],1)),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[25, 0], [0, 32.5]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x: np.array(x[:,0]/x[:,1]).reshape((x.shape[0],1)),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[40, 0], [0, 52]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x: np.array(x[:,0]/x[:,1]).reshape((x.shape[0],1)),
        'Z_distribution': None, 'D_distribution': None
    }, 
]

correlated_errors_scenarios_ratio = [
    {
        'sigma_Epsilon': [[0.5, 0.25], [0.25, 1]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x: np.array(x[:,0]/x[:,1]).reshape((x.shape[0],1)),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[1, 0.5], [0.5, 2]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x: np.array(x[:,0]/x[:,1]).reshape((x.shape[0],1)),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[2, 1], [1, 3]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x: np.array(x[:,0]/x[:,1]).reshape((x.shape[0],1)),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[4, 2], [2, 6]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x: np.array(x[:,0]/x[:,1]).reshape((x.shape[0],1)),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[6, 3], [3, 9]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x: np.array(x[:,0]/x[:,1]).reshape((x.shape[0],1)),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[9, 6], [6, 13.5]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x: np.array(x[:,0]/x[:,1]).reshape((x.shape[0],1)),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[12, 9], [9, 18]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x: np.array(x[:,0]/x[:,1]).reshape((x.shape[0],1)),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[20, 15], [15, 26]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x: np.array(x[:,0]/x[:,1]).reshape((x.shape[0],1)),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[25, 19], [19, 32.5]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x: np.array(x[:,0]/x[:,1]).reshape((x.shape[0],1)),
        'Z_distribution': None, 'D_distribution': None
    }, 
    {
        'sigma_Epsilon': [[40, 30], [30, 52]],
        'n': 10000, 'days': 1, 'dietary_components': 2,
        'additional_variates': 0, 'bc_lambda': -1, 
        'beta': [[36], [28]], 'alpha': [100, 4],
        'sigma_U': [[20, 15.5], [15.5, 26]],
        'g_link_function': 'identity', 'g_link_function_inv': None,
        'h_function': lambda x: np.array(x[:,0]/x[:,1]).reshape((x.shape[0],1)),
        'Z_distribution': None, 'D_distribution': None
    }, 
]

scenario_sets = {
    'uncorrelated_errors_scenarios': uncorrelated_errors_scenarios,
    'correlated_errors_scenarios': correlated_errors_scenarios,
    'uncorrelated_errors_scenarios_ratio': uncorrelated_errors_scenarios_ratio,
    'correlated_errors_scenarios_ratio': correlated_errors_scenarios_ratio
}

if __name__ == "__main__":
    for scenario_name in scenario_sets:
        print(f"Starting {scenario_name}:")
        for idx, scenario in enumerate(scenario_sets[scenario_name]):
            current_generator = Generator(**scenario)
            train_df, test_df = current_generator.generate_pd(n = 1000, n_val = 1000, split = True, seed = 3141592)

            if scenario_name in ['uncorrelated_errors_scenarios', 'correlated_errors_scenarios']:
                X_train_true = train_df[['Truth_1', 'Truth_2']]
                X_train_error = train_df[['Y_11', 'Y_12']]

                X_test_true = test_df[['Truth_1', 'Truth_2']]
                X_test_error = test_df[['Y_11', 'Y_12']]
            else:
                X_train_true = (train_df['Truth_1']/train_df['Truth_2']).values.reshape(-1,1)
                X_train_error = (train_df['Y_11']/train_df['Y_12']).values.reshape(-1,1)

                X_test_true = (test_df['Truth_1']/test_df['Truth_2']).values.reshape(-1,1)
                X_test_error = (test_df['Y_11']/test_df['Y_12']).values.reshape(-1,1)
            Y_train = train_df['Outcome']
            Y_test = test_df['Outcome']

            model_true = LinearRegression().fit(X_train_true, Y_train)
            model_error = LinearRegression().fit(X_train_error, Y_train)
            
            true_preds = model_true.predict(X_test_true)
            error_preds = model_error.predict(X_test_error)

            print(f"\t For scenario {idx} of '{scenario_name}' the true regression model had a PMSE of {round(mean_squared_error(true_preds, Y_test), 4)} and the error-prone model had a PMSE of {round(mean_squared_error(error_preds, Y_test), 4)}.")