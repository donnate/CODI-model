import argparse
import numpy as np
import pandas as pd
import time

from utils import *
from helper_classification import *
from algs import *
import sys


def main():
    # passs command line arguments
    parser = argparse.ArgumentParser(description='Run Synthetic Experiments')
    parser.add_argument('-n','--n_samples', 
                        help='number of samples', type=int, default=200)
    parser.add_argument('-sens','--sensitivity', 
                        help='sensitivity', type=float, default=0.8)
    parser.add_argument('-spec','--specificity', 
                        help='specificity', type=float, default=0.9)
    parser.add_argument('-s','--sigma', 
                        help='sigma', type=float, default=1.0)
    args = parser.parse_args()
    PARAMETERS = {
      'n_samples': args.n_samples,
      'n_factors': 2,
      'beta': [1, -0.8, 0.3],
      'sigma': args.sigma,
      'sensitivity': args.sensitivity,
      'specificity': args.specificity,
      'proba_sick': [0.8, 0.9, 0.3],
      'proba_healthy': [0.6, 0.3, 0.1],
      'csv_prefix': 'n_factors_2_n_symptoms_3_' + str(np.random.randint(0, 500000))
    }
    SYMPTOMS = ['symptom','symptom2','symptom3']
    SYMPTOMS_QUANT = [ ]
    CONTEXT_INFO = ['risk_factor','risk_factor2']
    TRUST_TEST = ['delta_time_symptoms_onset', ]
    VARIABLES = SYMPTOMS + SYMPTOMS_QUANT + CONTEXT_INFO
    SENSITIVITY = {'asymptomatic': [PARAMETERS['sensitivity'], PARAMETERS['sensitivity']-0.05, np.max([1.0,PARAMETERS['sensitivity'] + 0.05])]}
    SPECIFICITY = {'asymptomatic': [PARAMETERS['specificity'], PARAMETERS['specificity']-0.05, np.max([1.0,PARAMETERS['specificity'] + 0.05])]}
    SENSITIVITY_PRIORS = {k: moment_matching_beta(SENSITIVITY[k][0],
                                      (1.0/(2*1.96) *(SENSITIVITY[k][2] - SENSITIVITY[k][1]))**2) for k in SENSITIVITY.keys()}
    SPECIFICITY_PRIORS = {k: moment_matching_beta(1-SPECIFICITY[k][0],
                                      (1.0/(2*1.96) *(SPECIFICITY[k][2] - SPECIFICITY[k][1]))**2) for k in SENSITIVITY.keys()}
    print(SPECIFICITY_PRIORS)
    print(SENSITIVITY_PRIORS)
    MODEL = LogisticRegression(max_iter=10000)
    PARAMETERS_REG = {'C':np.logspace(-4, 4, num=50),
              #'max_depth': [2,3,4,5,6,7,8,9,10]
              }
    B = 100
    
    results = np.zeros((3*B, 20))  
    results[:,0] =  args.sigma
    results[:,1] =  args.sensitivity
    results[:,2] =  args.specificity
    results[:,3] =  args.n_samples
    for sim in np.arange(B):
        data, diagnostic = generate_fake_dataset(PARAMETERS)
        T = data['test_outcome'].values
        data['label'] = diagnostic
        data['label_t'] = 'asymptomatic'
        for bb in [0, 1, 2]:
            b = 3 * sim + bb
            if bb == 0:
                test = VanillaClassifier(parameters=PARAMETERS_REG, list_variables=VARIABLES,
                                             B=500)
                tic = time.time()
                test.fit(MODEL, data, T)
                imp = test.posterior_given_XandT(data, T,
                                                 sensitivity=SENSITIVITY, specificity=SPECIFICITY)
                imputed = imp[:,0]
                toc = time.time()
            elif bb == 1:
                test = EMClassifier2(EM_steps=100, beta = 0. * np.ones(3),
                                        list_symptoms=SYMPTOMS,
                                        list_symptoms_quant=SYMPTOMS_QUANT,
                                        list_context_variables=CONTEXT_INFO,
                                        specificity_priors=SPECIFICITY_PRIORS,
                                        sensitivity_priors=SENSITIVITY_PRIORS)
                                        #beta = np.array([logreg.intercept_[0]] + [u for u in logreg.coef_[0]]))
                tic = time.time()
                imputed = test.fit(data, T, T)
                toc = time.time()
            else:
                test = EMClassifier3(EM_steps=100, beta = 0. * np.ones(3),
                                        list_symptoms=SYMPTOMS,
                                        list_symptoms_quant=SYMPTOMS_QUANT,
                                        list_context_variables=CONTEXT_INFO,
                                        specificity_priors=SPECIFICITY_PRIORS,
                                        sensitivity_priors=SENSITIVITY_PRIORS)
                                        #beta = np.array([logreg.intercept_[0]] + [u for u in logreg.coef_[0]]))
                tic = time.time()
                imputed = test.fit(data, T, T)
                toc = time.time()
            results[b, 4] = bb
            results[b, 5]= np.sum((imputed>0.5) == data['label'].values) * 1.0 / data.shape[0]
            results[b, 6]= np.sum((imputed>0.5) == data['test_outcome'].values) * 1.0 / data.shape[0]
            results[b, 7]= np.sum(data['test_outcome'].values == data['label'].values) * 1.0 / data.shape[0]
            if bb >0:
                print(test.beta_reg)
                results[b, 8:11] = test.beta_reg - PARAMETERS['beta']
                results[b, 11] = test.params['p_symptom_0'] - PARAMETERS['proba_healthy'][0]
                results[b, 12] = test.params['p_symptom_1'] - PARAMETERS['proba_sick'][0]
                results[b, 13] = test.params['p_symptom2_0'] - PARAMETERS['proba_healthy'][1]
                results[b, 14] = test.params['p_symptom2_1'] - PARAMETERS['proba_sick'][1]
                results[b, 15] = test.params['p_symptom3_0'] - PARAMETERS['proba_healthy'][2]
                results[b, 16] = test.params['p_symptom3_1'] - PARAMETERS['proba_sick'][2]
                results[b, 17] = test.params['p_T_1_asymptomatic'] - PARAMETERS['sensitivity']
                results[b, 18] = test.params['p_T_0_asymptomatic'] - (1-PARAMETERS['specificity'])
            results[b, 19] = toc - tic
            np.savetxt("/scratch/users/cdonnat/COVID-testing/sim/results/" + simulations_" + \
            PARAMETERS['csv_prefix'] + '.csv', results, delimiter=',', fmt='%f')
    
    results = pd.DataFrame(results,
                           columns= ["sigma", "specificity", "sensitivity",
                                 "n_samp", "algorithm",'imputed_vs_label_agreement',
                                 'imputed_vs_outcome_agreement','outcome_vs_label_agreement',
                                 'beta_1', 'beta_2','beta_3',
                                 'p_symptom_0','p_symptom_1',
                                 'p_symptom2_0','p_symptom2_1',
                                 'p_symptom3_0','p_symptom3_1',
                                 'p_T_1_asymptomatic','p_T_0_asymptomatic', "time"])
    
    results.to_csv('/scratch/users/cdonnat/COVID-testing/sim/' + PARAMETERS['csv_prefix'] + '.csv')


if __name__ == "__main__":
    main()
    
    
