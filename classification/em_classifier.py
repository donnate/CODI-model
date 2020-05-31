''' Module class for training the EM algorithm
'''
import json
import numpy as np
import pandas as pd
import pickle
import scipy as sc
import sklearn as sk

from sklearn.linear_model import LogisticRegression

import classification.helper as helper


PARAMETERS = {'C':np.logspace(-4, 4, num=50)
              }
SYMPTOMS = ['fever', 'loss_taste_smell', 'dry_cough', 'cough_with_sputum',
            'fatigue', 'shortness_of_breath','achy_joints_muscles',
            'sore_throat', 'headache', 'chills', 'nausea',
            'congested_nose', 'stomach_upset_diarrhoea',]
SYMPTOMS_QUANT = ['index_duration_symptoms','index_severity',
                  'index_pulmonary_impact', 'index_anxiety' ]
CONTEXT_INFO = ['flu_vaccine','household_illness','nb_household']
TRUST_TEST = ['delta_time_symptoms_onset', ]
VARIABLES = SYMPTOMS + SYMPTOMS_QUANT + CONTEXT_INFO

SENSITIVITY = {'asymptomatic': [0.694, 0.519, 0.837],
               '2-10 days': [0.811, 0.724, 0.881],
               '11-20 days': [0.935, 0.843, 0.982],
               '21+ days': [0.98, 0.9, 1.0]}
SPECIFICITY = {'asymptomatic': [0.005, 0.003, 0.008],
               '2-10 days': [0.005, 0.003, 0.008],
               '11-20 days': [0.005, 0.003, 0.008],
               '21+ days': [00.005, 0.003, 0.008]}
SENSITIVITY_PRIORS = np.zeros((4,2))
SPECIFICITY_PRIORS = np.zeros((4,2))
SENSITIVITY_PRIORS = {k: helper.moment_matching_beta(SENSITIVITY[k][0],
                                      (1.0/(2*1.96) *(SENSITIVITY[k][2] - SENSITIVITY[k][1]))**2) for k in SENSITIVITY.keys()}
SPECIFICITY_PRIORS = {k: helper.moment_matching_beta(SPECIFICITY[k][0],
                                      (1.0/(2*1.96) *(SPECIFICITY[k][2] - SPECIFICITY[k][1]))**2) for k in SENSITIVITY.keys()}

class EMClassifier:
    def __init__(self,parameters=PARAMETERS, list_symptoms=SYMPTOMS,
                 list_symptoms_quant=SYMPTOMS_QUANT,
                 list_context_variables=CONTEXT_INFO, params_dist=None,
                 hyperparams_dist=None, EM_steps=20, B=1000, beta=None,
                 specificity_priors=SPECIFICITY_PRIORS,
                 sensitivity_priors=SENSITIVITY_PRIORS):
        self.model = None
        self.parameters = parameters
        self.EM_steps = EM_steps
        self.B = B  ### number of posterior samples
        self.list_context_variables = list_context_variables
        self.list_symptoms = list_symptoms
        self.list_symptoms_quant = list_symptoms_quant
        self.specificity = specificity_priors
        self.sensitivity = sensitivity_priors
        if beta is None:
            self.beta_reg = np.zeros(len(self.list_context_variables)+1) ##coef for the context log reg
        else:
            self.beta_reg = beta
        ##### Initialize all the parameters in the Bayesian Model
        list_params_names = [a + 'T_' + e + k for a in ['alpha_', 'beta_']
                            for e in ['0_','1_'] for k in self.specificity.keys()]
        self.hyperparams = {k: 1 for k in list_params_names}
        for k in self.specificity.keys():
              self.hyperparams['alpha_T_1_' + k] = self.sensitivity[k][0]
              self.hyperparams['beta_T_1_' + k] = self.sensitivity[k][1]
              self.hyperparams['alpha_T_0_' + k] = self.specificity[k][0]
              self.hyperparams['beta_T_0_' + k] = self.specificity[k][1]
        ##### Option to pass different more informative initial values
        if hyperparams_dist is not None:
        	for k in hyperparams.keys(): self.hyperparams[k] = hyperparams[k]
        self.params = {'p_'+ k + e: 0.5 for k in self.list_symptoms
                       for e in ['_0','_1']}
        self.params.update({ 'p_T_' + e + k: (self.hyperparams['alpha_T_' + e + k]/(self.hyperparams['alpha_T_' + e + k] + self.hyperparams['beta_T_' + e + k]))
                            for e in ['0_','1_'] for k in self.specificity.keys()})
        self.params.update({'sigma_' + n + e: 1.0 for n in list_symptoms_quant
                            for e in ['_0','_1']})
        self.params.update({'mu_' + n + e: 5.0 for n in list_symptoms_quant
                            for e in ['_0','_1']})
        if params_dist is not None:
        	for k in params.keys(): self.params[k] = params[k]

    def expectation(self, data, T, D):
        """ Computes the expectation of the hidden variables given
        the different parameters. The hidden variables in our model are D and self.x, self.y
        INPUTS
        ---------------------------------------------------------------------
        Y                   :       context + questionnaire data
        T                   :       image classification label (binary)
        D                   :       imputed diagnostic
        """
        #### MLE
        N = data.shape[0]
        Y = np.hstack([np.expand_dims(np.ones(data.shape[0]),1),
                       data[test.list_context_variables]])
        log_odds = np.zeros(N)
        for k in self.specificity.keys():
            index = np.where(data['label_T']==k)
            log_odds[index] += T[index] * np.log(self.params['p_T_0_' + k]) \
                      + (1 - T[index]) * np.log(1-self.params['p_T_0_' + k])\
                      - (T[index]) * np.log(self.params['p_T_1_' + k])\
                      - (1 - T[index]) * np.log(1-self.params['p_T_1_' + k])
        print("tests", list(log_odds))
        for n in self.list_symptoms:
            X  = data[n]
            log_odds += (X) * np.log(self.params['p_' + n +'_0']) \
                      + (1 - X) * np.log(1-self.params['p_' + n +'_0'])\
                      - (X) * np.log(self.params['p_' + n +'_1'])\
                      - (1 - X) * np.log(1-self.params['p_' + n +'_1'])
            print("symptoms ", n,  list(log_odds))
        print("symptoms", list(log_odds))
        for n in self.list_symptoms_quant:
            X  = data[n]
            log_odds += -(0.5 * ((X - self.params['mu_' + n +'_0'])**2)/(2 *self.params['sigma_' + n +'_0']**2 )\
                        - np.log(self.params['sigma_' + n +'_0']) \
                        + 0.5 * ((X - self.params['mu_' + n +'_1'])**2)/(2 *self.params['sigma_' + n +'_1']**2 )\
                        + np.log(self.params['sigma_' + n +'_1']))
        print("symptoms quant", list(log_odds))
        ### Now add the prior with context info
        log_odds += - Y.dot(self.beta_reg)
        imputed_labels = np.divide(np.ones(N),
                         np.ones(N) + np.exp(log_odds))
        #print("imputed labels", imputed_labels)
        #imputed_labels
        return imputed_labels

    def maximization(self, data, T, imputed_labels):
        """ Maximization of the parameters for the model, that is the $beta$ and coefficients
        for the symptoms
        """
        print("Begin maximization")
        N,_ = data.shape
        lab_t = data['label_T']
        for k in self.specificity.keys():
            index = np.where(lab_t == k)[0]
            self.params['p_T_1_' + k] = ((np.sum(np.multiply(imputed_labels[index], T[index]))+self.hyperparams['alpha_T_1_' + k])/
                                        (np.sum(np.multiply(imputed_labels[index], T[index]))+self.hyperparams['alpha_T_1_' + k]
                                         +np.sum(np.multiply(imputed_labels[index], 1-T[index]))+self.hyperparams['beta_T_1_' + k])
                                        )
            self.params['p_T_0_' + k] =  ((np.sum(np.multiply(imputed_labels[index], T[index]))+self.hyperparams['alpha_T_0_' + k])/
                                        (np.sum(np.multiply(imputed_labels[index], T[index]))+self.hyperparams['alpha_T_0_' + k]
                                         +np.sum(np.multiply(imputed_labels[index], 1-T[index]))+self.hyperparams['beta_T_0_' + k])
                                        )
        for n in self.list_symptoms:
            X = data[n]
            self.params['p_' + n + '_1'] = (1 + np.sum(np.multiply(imputed_labels + T, X)))/\
                                         (2 + N + np.sum(np.multiply(imputed_labels, X)) + np.sum(np.multiply(imputed_labels, 1-X)))
            self.params['p_' + n + '_0'] = (1 + np.sum(np.multiply(1.-imputed_labels + 1 - T, X))+1)/\
                                         (2 + N + np.sum(np.multiply(1-imputed_labels, X)) + np.sum(np.multiply(1-imputed_labels, 1-X)))
            #print([n,self.params['p_' + n + '_1'],self.params['p_' + n + '_0']])
        for n in self.list_symptoms_quant:
            #print("imputed_label in quand", np.sum(imputed_labels))
            X = data[n]
            self.params['mu_' + n + '_1'] = (5 + np.sum((np.multiply(T, X))) + np.sum(np.multiply((imputed_labels), X)))\
             / (1 + np.sum(T)+ np.sum(imputed_labels))
            self.params['mu_' + n + '_0'] = (5 + np.sum((np.multiply(1-T, X)))  + np.sum(np.multiply((1.-imputed_labels), X)))\
             / (1 + N - np.sum(T) + N - np.sum(imputed_labels))
            self.params['sigma_' + n + '_1'] = np.sqrt(( (5 - self.params['mu_' + n + '_1'])**2+ \
                                                        np.sum((np.multiply(T, X-self.params['mu_' + n + '_1']))**2) \
                                                        + np.sum(np.multiply(imputed_labels,(data[n] -self.params['mu_' + n + '_1'])**2 )))\
                                                       /(1 + np.sum(T) + np.sum(imputed_labels)))
            self.params['sigma_' + n + '_0'] = np.sqrt(((5 - self.params['mu_' + n + '_0'])**2+ \
                                                        np.sum((np.multiply(T, X-self.params['mu_' + n + '_0']))**2) \
                                                        + np.sum(np.multiply(1-imputed_labels, (data[n] -self.params['mu_' + n + '_0'])**2 )))\
                                                       /(1 - np.sum(T) + 2* N - np.sum(imputed_labels)))
            print([n, np.sum(imputed_labels), np.sum(1.-imputed_labels), self.params['mu_' + n + '_1'], self.params['sigma_' + n + '_1'],
                   self.params['mu_' + n + '_0'],self.params['sigma_' + n + '_0']])

        ##### Update on the beta coefficient: use Newton-Raphson
        Y = np.hstack([np.expand_dims(np.ones(data.shape[0]),1),
                       data[test.list_context_variables]])
        for it in np.arange(100):
            sig = np.divide(np.ones(N), np.ones(N) + np.exp(-Y.dot(self.beta_reg)))
            diag_weights = np.diag(np.multiply(sig, 1-sig))
            H = Y.T.dot(diag_weights.dot(Y)) ## Hessian
            #print("Hessian", H)
            grad = (imputed_labels -sig).T.dot(Y) ### gradient
            #print("imputed_labels", imputed_labels)
            #print("update norm beta", np.sum((np.linalg.inv(H + 0.01 * np.eye(len(self.beta_reg))).dot(grad)**2)))
            self.beta_reg += np.linalg.inv(H + 0.001 * np.eye(len(self.beta_reg))).dot(grad)
            #print("beta", self.beta_reg)
        #### This automatically updates all the parameters.

    def fit(self, data, T, priorD):
        """ Run the EM for several steps
        """
        #### initialize values
        imputed_labels = priorD  ### initialize at image?
        self.maximization(data, T, imputed_labels)
        print(self.params)
        for it in np.arange(self.EM_steps):
            print(it)
            imputed_labels = self.expectation(data, T, imputed_labels)
            print(it, "imputed labels")
            print(list(imputed_labels))
            self.maximization(data, T, imputed_labels)
            print("Max done")
            print(self.params)
        return imputed_labels

    def save(self, filename):
        dict_v = {'params': self.params, 'hyperparams': self.hyperparams} ### we just need the prior parameters for our models
        pickle.dump(dict_v, open(filename, 'wb'))

    def load(self, filename):
        dict_v = pickle.load(open(filename, 'rb'))
        self.params = dict_v['params']
        self.hyperparams = dict_v['hyperparams']

    def posterior_given_X(self, data):
        ''' Produces a posterior credible distribution for the immunity score given context
        + questionnaire
        Just have to unfold the model progressively: sample from the different distributions
        the different log ratios
        '''
        Y = np.hstack([np.expand_dims(np.ones(data.shape[0]),1),
                       data[test.list_context_variables]])
        print(log_odds)
        for n in self.list_symptoms:
            X = data[n]
            x = self.params['p_'+ n + '_1']   ### Sample from the distribution?
            y = self.params['p_'+ n + '_0']

            log_odds+= (np.sum(1-X) * np.log(y) + np.sum(1-X) * np.log(1-y)\
                         - np.sum(X) * np.log(x) - np.sum(X) * np.log(1-x))
        for n in self.list_symptoms_quant:
            s0 = self.params['sigma_'+ n + '_0']
            s1 = self.params['sigma_'+ n + '_1']
            log_odds+= (-np.log(s0) - 0.5 * np.sum(( data[n] - self.params['mu_'+ n + '_0'] )**2)/(2 *s0**2) +
                        +np.log(s1) + 0.5 * np.sum(( data[n] - self.params['mu_'+ n + '_1'] )**2)/(2 *s1**2) )
        #### sample the posterior from the logistic regression:
        log_odds += -Y.dot(self.beta_reg)
        preds = np.divide(np.ones(self.B) ,  np.ones(self.B) + np.exp(log_odds))
        return(preds)

    def posterior_given_XandT(self, data, T):
        ''' Produces a posterior credible distribution for the immunity score given context
        + questionnaire + IMAGE LABEL (T: binary) !!
        '''
        Y = np.hstack([np.expand_dims(np.ones(data.shape[0]),1),
                       data[test.list_context_variables]])
        for k in self.specificity.keys():
            index = np.where(data['label_T']==k)
            log_odds[index] += T[index] * np.log(self.params['p_T_0_' + k]) \
                      + (1 - T[index]) * np.log(1-self.params['p_T_0_' + k])\
                      - (T[index]) * np.log(self.params['p_T_1_' + k])\
                      - (1 - T[index]) * np.log(1-self.params['p_T_1_' + k])
        for n in self.list_symptoms:
            X = data[n]
            x = self.params['p_'+ n + '_1']   ### Sample from the distribution?
            y = self.params['p_'+ n + '_0']

            log_odds+= (np.sum(1-X) * np.log(y) + np.sum(1-X) * np.log(1-y)\
                         - np.sum(X) * np.log(x) - np.sum(X) * np.log(1-x))
        for n in self.list_symptoms_quant:
            s0 = self.params['sigma_'+ n + '_0']
            s1 = self.params['sigma_'+ n + '_1']
            log_odds+= (-np.log(s0) - 0.5 * np.sum(( data[n] - self.params['mu_'+ n + '_0'] )**2)/(2 *s0**2) +
                        +np.log(s1) + 0.5 * np.sum(( data[n] - self.params['mu_'+ n + '_1'] )**2)/(2 *s1**2) )
        #### sample the posterior from the logistic regression:
        log_odds += -Y.dot(self.beta_reg)
        preds = np.divide(np.ones(self.B) ,  np.ones(self.B) + np.exp(log_odds))
        return(preds)


class EMClassifier2:
    ''' Adding uncertainty estimates in all the parameters.

    Update symptoms probabilities, normal distributions on quantitative symptoms.
    Does not update sensivity and specificity.
    '''
    def __init__(self, list_symptoms=SYMPTOMS,
                 list_symptoms_quant=SYMPTOMS_QUANT,
                 list_context_variables=CONTEXT_INFO, params_dist=None,
                 hyperparams_dist=None, EM_steps=20, B=1000, beta=None,
                 specificity_priors=SPECIFICITY_PRIORS,
                 sensitivity_priors=SENSITIVITY_PRIORS):
        self.model = None
        self.EM_steps = EM_steps
        self.B = B  ### number of posterior samples
        self.list_context_variables = list_context_variables
        self.list_symptoms = list_symptoms
        self.list_symptoms_quant = list_symptoms_quant
        self.specificity = specificity_priors
        self.sensitivity = sensitivity_priors
        if beta is None:
            self.beta_reg = np.zeros(len(self.list_context_variables)+1) ##coef for the context log reg
        else:
            self.beta_reg = beta
        ##### Initialize all the parameters in the Bayesian Model
        list_params_names = [a + 'T_' + e + k for a in ['alpha_', 'beta_']
                            for e in ['0_','1_'] for k in self.specificity.keys()]
        list_params_names += [a + '_' +  n + '_' + e for a in ['alpha_', 'beta_']
                            for e in ['0_','1_'] for n in self.list_symptoms]
        list_params_names += [a + '_' +  n + '_' + e for a in ['mu_', 'sigma_']
                            for e in ['0_','1_'] for n in self.list_symptoms_quant]
        self.hyperparams = {k: 1.0 for k in list_params_names}
        for k in self.specificity.keys():
            self.hyperparams['alpha_T_1_' + k] = self.sensitivity[k][0]
            self.hyperparams['beta_T_1_' + k] = self.sensitivity[k][1]
            self.hyperparams['alpha_T_0_' + k] = self.specificity[k][0]
            self.hyperparams['beta_T_0_' + k] = self.specificity[k][1]

        ##### Option to pass different more informative initial values
        if hyperparams_dist is not None:
        	for k in hyperparams.keys(): self.hyperparams[k] = hyperparams[k]
        self.params = {'p_'+ k + e: 0.5 for k in self.list_symptoms
                       for e in ['_0','_1']}
        self.params.update({ 'p_T_' + e + k: (self.hyperparams['alpha_T_' + e + k]/(self.hyperparams['alpha_T_' + e + k] + self.hyperparams['beta_T_' + e + k]))
                            for e in ['0_','1_'] for k in self.specificity.keys()})
        self.params.update({'sigma_' + n + e: 1.0 for n in list_symptoms_quant
                            for e in ['_0','_1']})
        self.params.update({'mu_' + n + e: 5.0 for n in list_symptoms_quant
                            for e in ['_0','_1']})
        if params_dist is not None:
        	for k in params.keys(): self.params[k] = params[k]

    def expectation(self, data, T, D):
        """ Computes the expectation of the hidden variables given
        the different parameters. The hidden variables in our model are D and self.x, self.y
        INPUTS
        ---------------------------------------------------------------------
        Y                   :       context + questionnaire data
        T                   :       image classification label (binary)
        D                   :       imputed diagnostic
        """
        #### MLE
        N = data.shape[0]
        Y = np.hstack([np.expand_dims(np.ones(data.shape[0]),1),
                       data[self.list_context_variables]])
        log_odds = np.zeros(N)
        for k in self.specificity.keys():
            index = np.where(data['label_t']==k)
            log_odds[index] += T[index] * np.log(self.params['p_T_0_' + k]) \
                      + (1 - T[index]) * np.log(1-self.params['p_T_0_' + k])\
                      - (T[index]) * np.log(self.params['p_T_1_' + k])\
                      - (1 - T[index]) * np.log(1-self.params['p_T_1_' + k])
        #print("tests", list(log_odds))
        for n in self.list_symptoms:
            X  = data[n]
            log_odds += (X) * np.log(self.params['p_' + n +'_0']) \
                      + (1 - X) * np.log(1-self.params['p_' + n +'_0'])\
                      - (X) * np.log(self.params['p_' + n +'_1'])\
                      - (1 - X) * np.log(1-self.params['p_' + n +'_1'])
            #print("symptoms ", n,  list(log_odds))
        #print("symptoms", list(log_odds))
        for n in self.list_symptoms_quant:
            X  = data[n]
            log_odds += -(0.5 * ((X - self.params['mu_' + n +'_0'])**2)/(2 *self.params['sigma_' + n +'_0']**2 )\
                        - np.log(self.params['sigma_' + n +'_0']) \
                        + 0.5 * ((X - self.params['mu_' + n +'_1'])**2)/(2 *self.params['sigma_' + n +'_1']**2 )\
                        + np.log(self.params['sigma_' + n +'_1']))
        #print("symptoms quant", list(log_odds))
        ### Now add the prior with context info
        log_odds += - Y.dot(self.beta_reg)
        imputed_labels = np.divide(np.ones(N),
                         np.ones(N) + np.exp(log_odds))
        #print("imputed labels", imputed_labels)
        #imputed_labels
        return imputed_labels

    def maximization(self, data, T, imputed_labels):
        ''' Maximization of the parameters for the model, that is the $beta$ and coefficients
        for the symptoms
        '''
        #print("Begin maximization")
        N,_ = data.shape
        lab_t = data['label_t']
        for k in self.specificity.keys():
            index = np.where(lab_t == k)[0]
            self.params['p_T_1_' + k] = ((self.hyperparams['alpha_T_1_' + k]-1)/
                                        (self.hyperparams['alpha_T_1_' + k] +self.hyperparams['beta_T_1_' + k]-2)
                                        )
            #((np.sum(np.multiply(imputed_labels[index], T[index]))+self.hyperparams['alpha_T_1_' + k]-1)/
            #                            (np.sum(np.multiply(imputed_labels[index], T[index]))+  self.hyperparams['alpha_T_1_' + k] +
            #                             np.sum(np.multiply(imputed_labels[index], 1-T[index]))+self.hyperparams['beta_T_1_' + k]-2)
            #                            )
            self.params['p_T_0_' + k] = ((self.hyperparams['alpha_T_0_' + k]-1)/
                                        (self.hyperparams['alpha_T_0_' + k] +self.hyperparams['beta_T_0_' + k]-2)
                                        )
            # ((np.sum(np.multiply(imputed_labels[index], T[index]))+self.hyperparams['alpha_T_0_' + k]-1)/
            #                            (np.sum(np.multiply(imputed_labels[index], T[index]))+self.hyperparams['alpha_T_0_' + k]
            #                             +np.sum(np.multiply(imputed_labels[index], 1-T[index]))+self.hyperparams['beta_T_0_' + k]-2)
            #                            )
        for n in self.list_symptoms:
            X = data[n]
            self.params['p_' + n + '_1'] = ((np.sum(np.multiply(imputed_labels, X))+self.hyperparams['alpha_'+ n + '_1'] -1)/
                                        (np.sum(np.multiply(imputed_labels, X))+self.hyperparams['alpha_'+ n + '_1']
                                         +np.sum(np.multiply(imputed_labels, 1-X))+self.hyperparams['beta_'+ n + '_1']-2)
                                        )
            self.params['p_' + n + '_0'] = ((np.sum(np.multiply(1-imputed_labels, X))+self.hyperparams['alpha_'+ n + '_0']-1)/
                                        (np.sum(np.multiply(1-imputed_labels, X))+self.hyperparams['alpha_'+ n  + '_0']
                                         +np.sum(np.multiply(1-imputed_labels, 1-X))+self.hyperparams['beta_'+ n +  '_0']-2)
                                        )
        for n in self.list_symptoms_quant:
            #print("imputed_label in quand", np.sum(imputed_labels))
            X = data[n].values
            if np.sum(imputed_labels)>1e-2:
              mu1 = np.sum(np.multiply(X, imputed_labels)) / np.sum(imputed_labels)
              sigma1 = (np.sum((np.multiply(imputed_labels, X-mu1))**2) \
                      /(np.sum(imputed_labels)))
            else:
              mu1 = 0
              sigma1 = 1.
            if np.sum(1-imputed_labels)>1e-2:
              mu0 = np.sum(np.multiply(X, 1-imputed_labels)) / np.sum(1-imputed_labels)
              sigma0 = (np.sum((np.multiply(1-imputed_labels, X-mu0))**2) \
                      /(np.sum(1-imputed_labels)))
            else:
              mu0 = 0
              sigma0 = 1.
            self.params['sigma_' + n + '_1'] = 1./np.sqrt(1.0/self.hyperparams['sigma_' + n + '_1']**2 + np.sum(imputed_labels)/sigma1)
            self.params['sigma_' + n + '_0'] = 1.0/np.sqrt(1.0/self.hyperparams['sigma_' + n + '_0']**2 + np.sum(1-imputed_labels)/sigma0)
            self.params['mu_' + n + '_1'] = ( self.hyperparams['mu_' + n + '_1']/self.hyperparams['sigma_' + n + '_1']**2 +\
                                             mu1/sigma1) * self.params['sigma_' + n + '_1']**2
            self.params['mu_' + n + '_0'] = ( self.hyperparams['mu_' + n + '_0']/self.hyperparams['sigma_' + n + '_0']**2 +\
                                             mu0/sigma0) * self.params['sigma_' + n + '_0']**2
        ##### Update on the beta coefficient: use Newton-Raphson
        Y = np.hstack([np.expand_dims(np.ones(data.shape[0]),1),
                       data[self.list_context_variables]])
        #lr = LinearRegression()
        #log_odds = np.log(imputed_labels) - np.log(1.0 - imputed_labels)
        #lr.fit(data[test.list_context_variables], log_odds)
        #self.beta_reg = np.array([lr.intercept_, list(lr.coef_)])

        for it in np.arange(100):
            sig = np.divide(np.ones(N), np.ones(N) + np.exp(-Y.dot(self.beta_reg)))
            diag_weights = np.diag(np.multiply(sig, 1-sig))
            H = Y.T.dot(diag_weights.dot(Y)) ## Hessian
            #print("Hessian", H)
            grad = (imputed_labels -sig).T.dot(Y) ### gradient
            #print("imputed_labels", imputed_labels)
            #print("update norm beta", np.sum((np.linalg.inv(H + 0.01 * np.eye(len(self.beta_reg))).dot(grad)**2)))
            self.beta_reg += np.linalg.inv(H).dot(grad)
            #print("beta", self.beta_reg)
        #### Information matrix can be used to compute confidence intervals for
        #### This automatically updates all the parameters.

    def fit(self, data, T, priorD):
        """ Run the EM for several steps
        """
        #### initialize values
        N = len(T)
        imputed_labels = priorD  ### initialize at image?
        #### initialize hyperparameters
        for n in self.list_symptoms:
            X = data[n].values
            self.hyperparams['alpha_' + n + '_1'] = np.sum(np.multiply(priorD, X))
            self.hyperparams['beta_' + n + '_1'] = np.sum(np.multiply(priorD, 1.- X))
            self.hyperparams['alpha_' + n + '_0'] = np.sum(np.multiply(1-priorD, X))
            self.hyperparams['beta_' + n + '_0'] = np.sum(np.multiply(1-priorD, 1.- X))
        for n in self.list_symptoms_quant:
            X = data[n].values
            self.hyperparams['mu_' + n + '_1'] = (5+np.sum((np.multiply(priorD, X))))/(np.sum(priorD)+1)
            self.hyperparams['sigma_' + n + '_1'] = np.sqrt(((5 - self.params['mu_' + n + '_1'])**2+ \
                                                        np.sum((np.multiply(priorD, X-self.params['mu_' + n + '_1']))**2))/(np.sum(priorD)+1))
            self.hyperparams['mu_' + n + '_0'] = (5+np.sum((np.multiply(1-priorD, X))))/(np.sum(1-priorD)+1)
            self.hyperparams['sigma_' + n + '_0'] = np.sqrt(((5 - self.params['mu_' + n + '_0'])**2+ \
                                                        np.sum((np.multiply(1-priorD, X-self.params['mu_' + n + '_0']))**2))/(np.sum(1-priorD)+1))
        self.maximization(data, T, imputed_labels)
        #print(self.params)
        for it in np.arange(self.EM_steps):
            #print("EM step nb", it)
            imputed_labels = self.expectation(data, T, imputed_labels)
            #print(it, "imputed labels")
            #print(list(imputed_labels))
            self.maximization(data, T, imputed_labels)
            #print("Max done")
            #print(self.params)
        return imputed_labels

    def save(self, filename):
        dict_v = {'params': self.params, 'hyperparams': self.hyperparams} ### we just need the prior parameters for our models
        pickle.dump(dict_v, open(filename, 'wb'))

    def load(self, filename):
        dict_v = pickle.load(open(filename, 'rb'))
        self.params = dict_v['params']
        self.hyperparams = dict_v['hyperparams']

    def posterior_given_X(self, data):
        ''' Produces a posterior credible distribution for the immunity score given context
        + questionnaire
        Just have to unfold the model progressively: sample from the different distributions
        the different log ratios
        '''
        Y = np.hstack([np.expand_dims(np.ones(data.shape[0]),1),
                       data[self.list_context_variables]])
        # print(log_odds)
        for n in self.list_symptoms:
            X = data[n]
            x = self.params['p_'+ n + '_1']   ### Sample from the distribution?
            y = self.params['p_'+ n + '_0']

            log_odds+= (np.sum(1-X) * np.log(y) + np.sum(1-X) * np.log(1-y)\
                         - np.sum(X) * np.log(x) - np.sum(X) * np.log(1-x))
        for n in self.list_symptoms_quant:
            s0 = self.params['sigma_'+ n + '_0']
            s1 = self.params['sigma_'+ n + '_1']
            log_odds+= (-np.log(s0) - 0.5 * np.sum(( data[n] - self.params['mu_'+ n + '_0'] )**2)/(2 *s0**2) +
                        +np.log(s1) + 0.5 * np.sum(( data[n] - self.params['mu_'+ n + '_1'] )**2)/(2 *s1**2) )
        #### sample the posterior from the logistic regression:
        log_odds += -Y.dot(self.beta_reg)
        preds = np.divide(np.ones(self.B) ,  np.ones(self.B) + np.exp(log_odds))
        return(preds)

    def posterior_given_XandT(self, data, T):
        ''' Produces a posterior credible distribution for the immunity score given context
        + questionnaire + IMAGE LABEL (T: binary) !!
        '''
        Y = np.hstack([np.expand_dims(np.ones(data.shape[0]),1),
                       data[self.list_context_variables]])
        for k in self.specificity.keys():
            index = np.where(data['label_t']==k)
            log_odds[index] += T[index] * np.log(self.params['p_T_0_' + k]) \
                      + (1 - T[index]) * np.log(1-self.params['p_T_0_' + k])\
                      - (T[index]) * np.log(self.params['p_T_1_' + k])\
                      - (1 - T[index]) * np.log(1-self.params['p_T_1_' + k])
        for n in self.list_symptoms:
            X = data[n]
            x = self.params['p_'+ n + '_1']   ### Sample from the distribution?
            y = self.params['p_'+ n + '_0']

            log_odds+= (np.sum(1-X) * np.log(y) + np.sum(1-X) * np.log(1-y)\
                         - np.sum(X) * np.log(x) - np.sum(X) * np.log(1-x))
        for n in self.list_symptoms_quant:
            s0 = self.params['sigma_'+ n + '_0']
            s1 = self.params['sigma_'+ n + '_1']
            log_odds+= (-np.log(s0) - 0.5 * np.sum(( data[n] - self.params['mu_'+ n + '_0'] )**2)/(2 *s0**2) +
                        +np.log(s1) + 0.5 * np.sum(( data[n] - self.params['mu_'+ n + '_1'] )**2)/(2 *s1**2) )
        #### sample the posterior from the logistic regression:
        log_odds += -Y.dot(self.beta_reg)
        preds = np.divide(np.ones(self.B) ,  np.ones(self.B) + np.exp(log_odds))
        return(preds)


class EMClassifier3:
    ''' Adding uncertainty estimates in all the parameters.

    Update symptoms probabilities.
    Update sensitivity and specificity.
    '''
    def __init__(self, list_symptoms=SYMPTOMS,
                 list_symptoms_quant=SYMPTOMS_QUANT,
                 list_context_variables=CONTEXT_INFO, params_dist=None,
                 hyperparams_dist=None, EM_steps=20, B=1000, beta=None,
                 specificity_priors=SPECIFICITY_PRIORS,
                 sensitivity_priors=SENSITIVITY_PRIORS):
        self.model = None
        self.EM_steps = EM_steps
        self.B = B  ### number of posterior samples
        self.list_context_variables = list_context_variables
        self.list_symptoms = list_symptoms
        self.list_symptoms_quant = list_symptoms_quant
        self.specificity = specificity_priors
        self.sensitivity = sensitivity_priors
        if beta is None:
            self.beta_reg = np.zeros(len(self.list_context_variables)+1) ##coef for the context log reg
        else:
            self.beta_reg = beta
        ##### Initialize all the parameters in the Bayesian Model
        list_params_names = [a + 'T_' + e + k for a in ['alpha_', 'beta_']
                            for e in ['0_','1_'] for k in self.specificity.keys()]
        list_params_names += [a + '_' +  n + '_' + e for a in ['alpha_', 'beta_']
                            for e in ['0_','1_'] for n in self.list_symptoms]
        list_params_names += [a + '_' +  n + '_' + e for a in ['mu_', 'sigma_']
                            for e in ['0_','1_'] for n in self.list_symptoms_quant]
        self.hyperparams = {k: 1.0 for k in list_params_names}
        for k in self.specificity.keys():
            self.hyperparams['alpha_T_1_' + k] = self.sensitivity[k][0]
            self.hyperparams['beta_T_1_' + k] = self.sensitivity[k][1]
            self.hyperparams['alpha_T_0_' + k] = self.specificity[k][0]
            self.hyperparams['beta_T_0_' + k] = self.specificity[k][1]

        ##### Option to pass different more informative initial values
        if hyperparams_dist is not None:
        	for k in hyperparams.keys(): self.hyperparams[k] = hyperparams[k]
        self.params = {'p_'+ k + e: 0.5 for k in self.list_symptoms
                       for e in ['_0','_1']}
        self.params.update({ 'p_T_' + e + k: (self.hyperparams['alpha_T_' + e + k]/(self.hyperparams['alpha_T_' + e + k] + self.hyperparams['beta_T_' + e + k]))
                            for e in ['0_','1_'] for k in self.specificity.keys()})
        self.params.update({'sigma_' + n + e: 1.0 for n in list_symptoms_quant
                            for e in ['_0','_1']})
        self.params.update({'mu_' + n + e: 5.0 for n in list_symptoms_quant
                            for e in ['_0','_1']})
        if params_dist is not None:
        	for k in params.keys(): self.params[k] = params[k]

    def expectation(self, data, T, D):
        """ Computes the expectation of the hidden variables given
        the different parameters. The hidden variables in our model are D and self.x, self.y
        INPUTS
        ---------------------------------------------------------------------
        Y                   :       context + questionnaire data
        T                   :       image classification label (binary)
        D                   :       imputed diagnostic
        """
        #### MLE
        N = data.shape[0]
        Y = np.hstack([np.expand_dims(np.ones(data.shape[0]),1),
                       data[self.list_context_variables]])
        log_odds = np.zeros(N)
        for k in self.specificity.keys():
            index = np.where(data['label_t']==k)
            log_odds[index] += T[index] * np.log(self.params['p_T_0_' + k]) \
                      + (1 - T[index]) * np.log(1-self.params['p_T_0_' + k])\
                      - (T[index]) * np.log(self.params['p_T_1_' + k])\
                      - (1 - T[index]) * np.log(1-self.params['p_T_1_' + k])
        #print("tests", list(log_odds))
        for n in self.list_symptoms:
            X  = data[n]
            log_odds += (X) * np.log(self.params['p_' + n +'_0']) \
                      + (1 - X) * np.log(1-self.params['p_' + n +'_0'])\
                      - (X) * np.log(self.params['p_' + n +'_1'])\
                      - (1 - X) * np.log(1-self.params['p_' + n +'_1'])
            #print("symptoms ", n,  list(log_odds))
        #print("symptoms", list(log_odds))
        for n in self.list_symptoms_quant:
            X  = data[n]
            log_odds += -(0.5 * ((X - self.params['mu_' + n +'_0'])**2)/(2 *self.params['sigma_' + n +'_0']**2 )\
                        - np.log(self.params['sigma_' + n +'_0']) \
                        + 0.5 * ((X - self.params['mu_' + n +'_1'])**2)/(2 *self.params['sigma_' + n +'_1']**2 )\
                        + np.log(self.params['sigma_' + n +'_1']))
        #print("symptoms quant", list(log_odds))
        ### Now add the prior with context info
        log_odds += - Y.dot(self.beta_reg)
        imputed_labels = np.divide(np.ones(N),
                         np.ones(N) + np.exp(log_odds))
        #print("imputed labels", imputed_labels)
        #imputed_labels
        return imputed_labels

    def maximization(self, data, T, imputed_labels):
        ''' Maximization of the parameters for the model, that is the $beta$ and coefficients
        for the symptoms
        '''
        #print("Begin maximization")
        N,_ = data.shape
        lab_t = data['label_t']
        for k in self.specificity.keys():
            index = np.where(lab_t == k)[0]
            self.params['p_T_1_' + k] = ((np.sum(np.multiply(imputed_labels[index], T[index]))+self.hyperparams['alpha_T_1_' + k]-1)/
                                        (np.sum(np.multiply(imputed_labels[index], T[index]))+  self.hyperparams['alpha_T_1_' + k] +
                                         np.sum(np.multiply(imputed_labels[index], 1-T[index]))+self.hyperparams['beta_T_1_' + k]-2)
                                        )
            self.params['p_T_0_' + k] =  ((np.sum(np.multiply(imputed_labels[index], T[index]))+self.hyperparams['alpha_T_0_' + k]-1)/
                                        (np.sum(np.multiply(imputed_labels[index], T[index]))+self.hyperparams['alpha_T_0_' + k]
                                         +np.sum(np.multiply(imputed_labels[index], 1-T[index]))+self.hyperparams['beta_T_0_' + k]-2)
                                        )
        for n in self.list_symptoms:
            X = data[n]
            self.params['p_' + n + '_1'] = ((np.sum(np.multiply(imputed_labels, X))+self.hyperparams['alpha_'+ n + '_1'] -1)/
                                        (np.sum(np.multiply(imputed_labels, X))+self.hyperparams['alpha_'+ n + '_1']
                                         +np.sum(np.multiply(imputed_labels, 1-X))+self.hyperparams['beta_'+ n + '_1']-2)
                                        )
            self.params['p_' + n + '_0'] = ((np.sum(np.multiply(1-imputed_labels, X))+self.hyperparams['alpha_'+ n + '_0']-1)/
                                        (np.sum(np.multiply(1-imputed_labels, X))+self.hyperparams['alpha_'+ n  + '_0']
                                         +np.sum(np.multiply(1-imputed_labels, 1-X))+self.hyperparams['beta_'+ n +  '_0']-2)
                                        )
        for n in self.list_symptoms_quant:
            #print("imputed_label in quand", np.sum(imputed_labels))
            X = data[n].values
            if np.sum(imputed_labels)>1e-2:
              mu1 = np.sum(np.multiply(X, imputed_labels)) / np.sum(imputed_labels)
              sigma1 = (np.sum((np.multiply(imputed_labels, X-mu1))**2) \
                      /(np.sum(imputed_labels)))
            else:
              mu1 = 0
              sigma1 = 1.
            if np.sum(1-imputed_labels)>1e-2:
              mu0 = np.sum(np.multiply(X, 1-imputed_labels)) / np.sum(1-imputed_labels)
              sigma0 = (np.sum((np.multiply(1-imputed_labels, X-mu0))**2) \
                      /(np.sum(1-imputed_labels)))
            else:
              mu0 = 0
              sigma0 = 1.
            self.params['sigma_' + n + '_1'] = 1./np.sqrt(1.0/self.hyperparams['sigma_' + n + '_1']**2 + np.sum(imputed_labels)/sigma1)
            self.params['sigma_' + n + '_0'] = 1.0/np.sqrt(1.0/self.hyperparams['sigma_' + n + '_0']**2 + np.sum(1-imputed_labels)/sigma0)
            self.params['mu_' + n + '_1'] = ( self.hyperparams['mu_' + n + '_1']/self.hyperparams['sigma_' + n + '_1']**2 +\
                                             mu1/sigma1) * self.params['sigma_' + n + '_1']**2
            self.params['mu_' + n + '_0'] = ( self.hyperparams['mu_' + n + '_0']/self.hyperparams['sigma_' + n + '_0']**2 +\
                                             mu0/sigma0) * self.params['sigma_' + n + '_0']**2
        ##### Update on the beta coefficient: use Newton-Raphson
        Y = np.hstack([np.expand_dims(np.ones(data.shape[0]),1),
                       data[self.list_context_variables]])
        #lr = LinearRegression()
        #log_odds = np.log(imputed_labels) - np.log(1.0 - imputed_labels)
        #lr.fit(data[test.list_context_variables], log_odds)
        #self.beta_reg = np.array([lr.intercept_, list(lr.coef_)])

        for it in np.arange(100):
            sig = np.divide(np.ones(N), np.ones(N) + np.exp(-Y.dot(self.beta_reg)))
            diag_weights = np.diag(np.multiply(sig, 1-sig))
            H = Y.T.dot(diag_weights.dot(Y)) ## Hessian
            #print("Hessian", H)
            grad = (imputed_labels -sig).T.dot(Y) ### gradient
            #print("imputed_labels", imputed_labels)
            #print("update norm beta", np.sum((np.linalg.inv(H + 0.01 * np.eye(len(self.beta_reg))).dot(grad)**2)))
            self.beta_reg += np.linalg.inv(H).dot(grad)
            #print("beta", self.beta_reg)
        #### Information matrix can be used to compute confidence intervals for
        #### This automatically updates all the parameters.

    def fit(self, data, T, priorD):
        """ Run the EM for several steps
        """
        #### initialize values
        N = len(T)
        imputed_labels = priorD  ### initialize at image?
        #### initialize hyperparameters
        for n in self.list_symptoms:
            X = data[n].values
            self.hyperparams['alpha_' + n + '_1'] = np.sum(np.multiply(priorD, X))
            self.hyperparams['beta_' + n + '_1'] = np.sum(np.multiply(priorD, 1.- X))
            self.hyperparams['alpha_' + n + '_0'] = np.sum(np.multiply(1-priorD, X))
            self.hyperparams['beta_' + n + '_0'] = np.sum(np.multiply(1-priorD, 1.- X))
        for n in self.list_symptoms_quant:
            X = data[n].values
            self.hyperparams['mu_' + n + '_1'] = (5+np.sum((np.multiply(priorD, X))))/(np.sum(priorD)+1)
            self.hyperparams['sigma_' + n + '_1'] = np.sqrt(((5 - self.params['mu_' + n + '_1'])**2+ \
                                                        np.sum((np.multiply(priorD, X-self.params['mu_' + n + '_1']))**2))/(np.sum(priorD)+1))
            self.hyperparams['mu_' + n + '_0'] = (5+np.sum((np.multiply(1-priorD, X))))/(np.sum(1-priorD)+1)
            self.hyperparams['sigma_' + n + '_0'] = np.sqrt(((5 - self.params['mu_' + n + '_0'])**2+ \
                                                        np.sum((np.multiply(1-priorD, X-self.params['mu_' + n + '_0']))**2))/(np.sum(1-priorD)+1))
        self.maximization(data, T, imputed_labels)
        #print(self.params)
        for it in np.arange(self.EM_steps):
            #print("EM step nb", it)
            imputed_labels = self.expectation(data, T, imputed_labels)
            #print(it, "imputed labels")
            #print(list(imputed_labels))
            self.maximization(data, T, imputed_labels)
            #print("Max done")
            #print(self.params)
        return imputed_labels

    def save(self, filename):
        dict_v = {'params': self.params, 'hyperparams': self.hyperparams} ### we just need the prior parameters for our models
        pickle.dump(dict_v, open(filename, 'wb'))

    def load(self, filename):
        dict_v = pickle.load(open(filename, 'rb'))
        self.params = dict_v['params']
        self.hyperparams = dict_v['hyperparams']

    def posterior_given_X(self, data):
        ''' Produces a posterior credible distribution for the immunity score given context
        + questionnaire
        Just have to unfold the model progressively: sample from the different distributions
        the different log ratios
        '''
        Y = np.hstack([np.expand_dims(np.ones(data.shape[0]),1),
                       data[self.list_context_variables]])
        print(log_odds)
        for n in self.list_symptoms:
            X = data[n]
            x = self.params['p_'+ n + '_1']   ### Sample from the distribution?
            y = self.params['p_'+ n + '_0']

            log_odds+= (np.sum(1-X) * np.log(y) + np.sum(1-X) * np.log(1-y)\
                         - np.sum(X) * np.log(x) - np.sum(X) * np.log(1-x))
        for n in self.list_symptoms_quant:
            s0 = self.params['sigma_'+ n + '_0']
            s1 = self.params['sigma_'+ n + '_1']
            log_odds+= (-np.log(s0) - 0.5 * np.sum(( data[n] - self.params['mu_'+ n + '_0'] )**2)/(2 *s0**2) +
                        +np.log(s1) + 0.5 * np.sum(( data[n] - self.params['mu_'+ n + '_1'] )**2)/(2 *s1**2) )
        #### sample the posterior from the logistic regression:
        log_odds += -Y.dot(self.beta_reg)
        preds = np.divide(np.ones(self.B) ,  np.ones(self.B) + np.exp(log_odds))
        return(preds)

    def posterior_given_XandT(self, data, T):
        ''' Produces a posterior credible distribution for the immunity score given context
        + questionnaire + IMAGE LABEL (T: binary) !!
        '''
        Y = np.hstack([np.expand_dims(np.ones(data.shape[0]),1),
                       data[self.list_context_variables]])
        for k in self.specificity.keys():
            index = np.where(data['label_t']==k)
            log_odds[index] += T[index] * np.log(self.params['p_T_0_' + k]) \
                      + (1 - T[index]) * np.log(1-self.params['p_T_0_' + k])\
                      - (T[index]) * np.log(self.params['p_T_1_' + k])\
                      - (1 - T[index]) * np.log(1-self.params['p_T_1_' + k])
        for n in self.list_symptoms:
            X = data[n]
            x = self.params['p_'+ n + '_1']   ### Sample from the distribution?
            y = self.params['p_'+ n + '_0']

            log_odds+= (np.sum(1-X) * np.log(y) + np.sum(1-X) * np.log(1-y)\
                         - np.sum(X) * np.log(x) - np.sum(X) * np.log(1-x))
        for n in self.list_symptoms_quant:
            s0 = self.params['sigma_'+ n + '_0']
            s1 = self.params['sigma_'+ n + '_1']
            log_odds+= (-np.log(s0) - 0.5 * np.sum(( data[n] - self.params['mu_'+ n + '_0'] )**2)/(2 *s0**2) +
                        +np.log(s1) + 0.5 * np.sum(( data[n] - self.params['mu_'+ n + '_1'] )**2)/(2 *s1**2) )
        #### sample the posterior from the logistic regression:
        log_odds += -Y.dot(self.beta_reg)
        preds = np.divide(np.ones(self.B) ,  np.ones(self.B) + np.exp(log_odds))
        return(preds)



class EMClassifier4:
    ''' Adding uncertainty estimates in all the parameters.

    Update symptoms probabilities, normal distributions on quantitative symptoms.
    Does not update sensivity and specificity.

    Previously SEMClassifier2.
    '''
    def __init__(self, list_symptoms=SYMPTOMS,
                 list_symptoms_quant=SYMPTOMS_QUANT,
                 list_context_variables=CONTEXT_INFO, params_dist=None,
                 hyperparams_dist=None, EM_steps=20, B=1000, beta=None,
                 specificity_priors=SPECIFICITY_PRIORS,
                 sensitivity_priors=SENSITIVITY_PRIORS):
        self.model = None
        self.EM_steps = EM_steps
        self.B = B  ### number of posterior samples
        self.list_context_variables = list_context_variables
        self.list_symptoms = list_symptoms
        self.list_symptoms_quant = list_symptoms_quant
        self.specificity = specificity_priors
        self.sensitivity = sensitivity_priors
        self.H = np.diagflat(np.zeros((len(list_context_variables)+1)))
        if beta is None:
            self.beta_reg = np.zeros(len(self.list_context_variables)+1) ##coef for the context log reg
        else:
            self.beta_reg = beta
        ##### Initialize all the parameters in the Bayesian Model
        list_params_names = [a + 'T_' + e + k for a in ['alpha_', 'beta_']
                            for e in ['0_','1_'] for k in self.specificity.keys()]
        list_params_names += [a  +  n + '_' + e for a in ['alpha_', 'beta_']
                            for e in ['0','1'] for n in self.list_symptoms + ['symptomatic']]
        list_params_names += [a  +  n + '_' + e for a in ['a_', 'b_']
                            for e in ['0','1'] for n in self.list_symptoms_quant]
        self.hyperparams = {k: 1.0 for k in list_params_names}
        for k in self.specificity.keys():
            self.hyperparams['alpha_T_1_' + k] = self.sensitivity[k][0]
            self.hyperparams['beta_T_1_' + k] = self.sensitivity[k][1]
            self.hyperparams['alpha_T_0_' + k] = self.specificity[k][0]
            self.hyperparams['beta_T_0_' + k] = self.specificity[k][1]

        ##### Option to pass different more informative initial values
        if hyperparams_dist is not None:
        	for k in hyperparams.keys(): self.hyperparams[k] = hyperparams[k]
        self.params = {'p_'+ k + e: 0.5 for k in self.list_symptoms + ['symptomatic']
                       for e in ['_0','_1']}
        self.params.update({ 'p_T_' + e + k: (self.hyperparams['alpha_T_' + e + k]/(self.hyperparams['alpha_T_' + e + k] + self.hyperparams['beta_T_' + e + k]))
                            for e in ['0_','1_'] for k in self.specificity.keys()})
        self.params.update({'a_' + n + e: 1.0 for n in list_symptoms_quant
                            for e in ['_0','_1']})
        self.params.update({'b_' + n + e: 1.0 for n in list_symptoms_quant
                            for e in ['_0','_1']})
        if params_dist is not None:
        	for k in params.keys(): self.params[k] = params[k]

    def expectation(self, data, T, D):
        """ Computes the expectation of the hidden variables given
        the different parameters. The hidden variables in our model are D and self.x, self.y
        INPUTS
        ---------------------------------------------------------------------
        Y                   :       context + questionnaire data
        T                   :       image classification label (binary)
        D                   :       imputed diagnostic
        """
        #### MLE
        N = data.shape[0]
        Y = np.hstack([np.expand_dims(np.ones(data.shape[0]),1),
                       data[self.list_context_variables]])
        index_sympt = np.where(data['symptomatic']==1)[0]
        log_odds = np.zeros(N)
        for k in self.specificity.keys():
            index = np.where(data['label_t']==k)
            log_odds[index] += T[index] * np.log(self.params['p_T_0_' + k]) \
                      + (1 - T[index]) * np.log(1-self.params['p_T_0_' + k])\
                      - (T[index]) * np.log(self.params['p_T_1_' + k])\
                      - (1 - T[index]) * np.log(1-self.params['p_T_1_' + k])
        #print("tests", list(log_odds))
        ### Start by updating the symptomatic columns
        n = 'symptomatic'
        X  = data['symptomatic'].values
        log_odds += (X) * np.log(self.params['p_' + n +'_0']) \
                      + (1 - X) * np.log(1-self.params['p_' + n +'_0'])\
                      - (X) * np.log(self.params['p_' + n +'_1'])\
                      - (1 - X) * np.log(1-self.params['p_' + n +'_1'])
        print("symptoms ", n,  list(log_odds))
        print(T)
        for n in self.list_symptoms:
            ### Update the symptomatic for the ones that got sick
            X  = data[n].iloc[index_sympt].values
            log_odds[index_sympt] += (X) * np.log(self.params['p_' + n +'_0']) \
                      + (1 - X) * np.log(1-self.params['p_' + n +'_0'])\
                      - (X) * np.log(self.params['p_' + n +'_1'])\
                      - (1 - X) * np.log(1-self.params['p_' + n +'_1'])
            print("symptoms ", n,  list(log_odds))
            print(T)
        #print("symptoms", list(log_odds))
        for n in self.list_symptoms_quant:
            #### We sample a bunch as we parametrize it by beta dist
            X  = data[n].iloc[index_sympt].values
            #pX_1 = np.reshape(np.random.beta(self.params['a_' + n +'_1'],
            #                                 self.params['b_' + n +'_1'],
            #                                 50 * len(index_sympt)),
            #                  (len(index_sympt), 50))
            #pX_0 = np.reshape(np.random.beta(self.params['a_' + n +'_0'],
            #                                 self.params['b_' + n +'_0'],
            #                                 50 * len(index_sympt)),
            #                  (len(index_sympt),50))
            B1 = sc.special.beta(self.params['a_' + n +'_1'],
                               self.params['b_' + n +'_1'])
            B0 = sc.special.beta(self.params['a_' + n +'_0'],
                               self.params['b_' + n +'_0'])
            log_odds[index_sympt] +=  (self.params['a_' + n +'_0']-1) * np.log(X)+\
                                    (self.params['b_' + n +'_0']-1) * np.log(1-X)+\
                                    -(self.params['a_' + n +'_1']-1) * np.log(X)+\
                                    - (self.params['b_' + n +'_1']-1) * np.log(1-X)+\
                                    np.ones((len(index_sympt))) * (np.log(B1) - np.log(B0))
            print("symptoms quant", list(log_odds))
            print(T)
        ### Now add the prior with context info
        log_odds += - Y.dot(self.beta_reg)
        print("beta ", n,  list(log_odds))
        imputed_labels = np.divide(np.ones(N),
                         np.ones(N) + np.exp(log_odds))
        print("imputed labels", np.sum(imputed_labels))
        #imputed_labels
        return imputed_labels

    def maximization(self, data, T, imputed_labels):
        ''' Maximization of the parameters for the model, that is the $beta$ and coefficients
        for the symptoms
        '''
        #print("Begin maximization")
        N,_ = data.shape
        lab_t = data['label_t']
        index_sympt = np.where(data['symptomatic']==1)[0]
        for k in self.specificity.keys():
            index = np.where(lab_t == k)[0]
            self.params['p_T_1_' + k] = ((self.hyperparams['alpha_T_1_' + k]-1)/
                                        (self.hyperparams['alpha_T_1_' + k] +self.hyperparams['beta_T_1_' + k]-2)
                                        )
            #((np.sum(np.multiply(imputed_labels[index], T[index]))+self.hyperparams['alpha_T_1_' + k]-1)/
            #                            (np.sum(np.multiply(imputed_labels[index], T[index]))+  self.hyperparams['alpha_T_1_' + k] +
            #                             np.sum(np.multiply(imputed_labels[index], 1-T[index]))+self.hyperparams['beta_T_1_' + k]-2)
            #                            )
            self.params['p_T_0_' + k] = ((self.hyperparams['alpha_T_0_' + k]-1)/
                                        (self.hyperparams['alpha_T_0_' + k] +self.hyperparams['beta_T_0_' + k]-2)
                                        )
            # ((np.sum(np.multiply(imputed_labels[index], T[index]))+self.hyperparams['alpha_T_0_' + k]-1)/
            #                            (np.sum(np.multiply(imputed_labels[index], T[index]))+self.hyperparams['alpha_T_0_' + k]
            #                             +np.sum(np.multiply(imputed_labels[index], 1-T[index]))+self.hyperparams['beta_T_0_' + k]-2))
            #
        n = 'symptomatic'
        X = data[n].values
        self.params['p_' + n + '_1'] = ((np.sum(np.multiply(imputed_labels, X)) + self.hyperparams['alpha_'+ n + '_1'] -1)/
                                        (np.sum(np.multiply(imputed_labels, X)) + self.hyperparams['alpha_'+ n + '_1']
                                         +np.sum(np.multiply(imputed_labels, 1-X))+self.hyperparams['beta_'+ n + '_1']-2)
                                        )
        self.params['p_' + n + '_0'] = ((np.sum(np.multiply(1-imputed_labels, X)) + self.hyperparams['alpha_'+ n + '_0'] -1)/
                                        (np.sum(np.multiply(1-imputed_labels, X)) + self.hyperparams['alpha_'+ n + '_0']
                                         +np.sum(np.multiply(1-imputed_labels, 1-X))+self.hyperparams['beta_'+ n + '_0']-2)
                                        )
        for n in self.list_symptoms:
            X = data[n].iloc[index_sympt].values
            temp_lab =  imputed_labels[index_sympt]
            self.params['p_' + n + '_1'] = ((np.sum(np.multiply(temp_lab , X)) +self.hyperparams['alpha_'+ n + '_1'] -1)/
                                        (np.sum(np.multiply(temp_lab , X)) +self.hyperparams['alpha_'+ n + '_1']
                                         +np.sum(np.multiply(temp_lab, 1-X))+self.hyperparams['beta_'+ n + '_1']-2)
                                        )
            self.params['p_' + n + '_0'] = ((np.sum(np.multiply(1-temp_lab, X)) + self.hyperparams['alpha_'+ n + '_0']-1)/
                                        (np.sum(np.multiply(1-temp_lab , X))+self.hyperparams['alpha_'+ n  + '_0']
                                         +np.sum(np.multiply(1-temp_lab, 1-X))+self.hyperparams['beta_'+ n +  '_0']-2)
                                        )
        for n in self.list_symptoms_quant:
            #print("imputed_label in quand", np.sum(imputed_labels))
            X = data[n].iloc[index_sympt].values
            temp_lab =  imputed_labels[index_sympt]
            #self.params['a_' + n + '_1'] = (np.sum(np.multiply(temp_lab , X)) +self.hyperparams['a_'+ n + '_1'] -1)
            #self.params['b_' + n + '_1'] = np.sum(np.multiply(temp_lab, 1-X))+self.hyperparams['b_'+ n + '_1']
            #self.params['a_' + n + '_0'] = (np.sum(np.multiply(1-temp_lab , X)) +self.hyperparams['a_'+ n + '_1'] -1)
            #self.params['b_' + n + '_0'] = np.sum(np.multiply(1-temp_lab, 1-X))+self.hyperparams['b_'+ n + '_1']
            xbar = np.sum(np.multiply(temp_lab , X)) / np.sum(temp_lab)
            s2 = np.sum(np.multiply(temp_lab , (X -xbar)**2)) / np.sum(temp_lab)
            self.params['a_' + n + '_1'] = xbar * (xbar * (1. - xbar) / s2 - 1.0)
            a = self.params['a_' + n + '_1']
            self.params['b_' + n + '_1'] =  a * (1. - xbar) / xbar
            xbar = np.sum(np.multiply(1-temp_lab , X)) / np.sum(1-temp_lab)
            s2 = np.sum(np.multiply(1-temp_lab , (X -xbar)**2)) / np.sum(1-temp_lab)
            self.params['a_' + n + '_0'] = xbar * (xbar * (1. - xbar) / s2 - 1.0)
            a = self.params['a_' + n + '_0']
            self.params['b_' + n + '_0'] =  a * (1. - xbar) / xbar

        ##### Update on the beta coefficient: use Newton-Raphson
        Y = np.hstack([np.expand_dims(np.ones(data.shape[0]),1),
                       data[self.list_context_variables]])
        #lr = LinearRegression()
        #log_odds = np.log(imputed_labels) - np.log(1.0 - imputed_labels)
        #lr.fit(data[test.list_context_variables], log_odds)
        #self.beta_reg = np.array([lr.intercept_, list(lr.coef_)])

        for it in np.arange(100):
            sig = np.divide(np.ones(N), np.ones(N) + np.exp(-Y.dot(self.beta_reg)))
            diag_weights = np.diag(np.multiply(sig, 1-sig))
            H = Y.T.dot(diag_weights.dot(Y)) ## Hessian
            #print("Hessian", H)
            grad = (imputed_labels -sig).T.dot(Y) ### gradient
            #print("imputed_labels", imputed_labels)
            #print("update norm beta", np.sum((np.linalg.inv(H + 0.01 * np.eye(len(self.beta_reg))).dot(grad)**2)))
            self.beta_reg += np.linalg.inv(H).dot(grad)
            #print("beta", self.beta_reg)
        self.H = H
        #### Information matrix can be used to compute confidence intervals for
        #### This automatically updates all the parameters.

    def fit(self, data, T, priorD):
        """ Run the EM for several steps
        """
        #### initialize values
        N = len(T)
        imputed_labels = priorD  ### initialize at image?
        index_sympt = np.where(data['symptomatic']==1)[0]
        #### initialize hyperparameters
        n = 'symptomatic'
        X = data[n].values
        self.hyperparams['alpha_' + n + '_1'] = np.sum(np.multiply(priorD, X))+ 1.0
        self.hyperparams['beta_' + n + '_1'] = np.sum(np.multiply(priorD, 1.- X))+ 1.0
        self.hyperparams['alpha_' + n + '_0'] = np.sum(np.multiply(1-priorD, X))+ 1.0
        self.hyperparams['beta_' + n + '_0'] = np.sum(np.multiply(1-priorD, 1.- X))+ 1.0
        self.params['p_' + n + '_0'] = (self.hyperparams['alpha_' + n + '_0'] - 1.0)/(self.hyperparams['beta_' + n + '_0'] + self.hyperparams['alpha_' + n + '_0'] -2.)
        self.params['p_' + n + '_0'] = (self.hyperparams['alpha_' + n + '_1'] - 1.0)/(self.hyperparams['beta_' + n + '_1'] + self.hyperparams['alpha_' + n + '_1'] -2.)

        for n in self.list_symptoms:
            X = data[n].iloc[index_sympt].values
            self.hyperparams['alpha_' + n + '_1'] = np.sum(np.multiply(priorD[index_sympt], X)) + 1.0
            self.hyperparams['beta_' + n + '_1'] = np.sum(np.multiply(priorD[index_sympt], 1.- X))+ 1.0
            self.hyperparams['alpha_' + n + '_0'] = np.sum(np.multiply(1-priorD[index_sympt], X))+ 1.0
            self.hyperparams['beta_' + n + '_0'] = np.sum(np.multiply(1-priorD[index_sympt], 1.- X))+ 1.0

            self.params['p_' + n + '_0'] = (self.hyperparams['alpha_' + n + '_0'] - 1.0)/(self.hyperparams['beta_' + n + '_0'] + self.hyperparams['alpha_' + n + '_0'] -2.)
            self.params['p_' + n + '_0'] = (self.hyperparams['alpha_' + n + '_1'] - 1.0)/(self.hyperparams['beta_' + n + '_1'] + self.hyperparams['alpha_' + n + '_1'] -2.)

        for n in self.list_symptoms_quant:
            X = data[n].iloc[index_sympt].values
            temp_lab = priorD[index_sympt]
            #print(np.sum((np.multiply(priorD[index_sympt], X))), (np.sum(priorD[index_sympt])+1.) )
            xbar = np.sum(np.multiply(temp_lab , X)) / np.sum(temp_lab)
            s2 = np.sum(np.multiply(temp_lab , (X -xbar)**2)) / np.sum(temp_lab)
            self.params['a_' + n + '_1'] = xbar * (xbar * (1. - xbar) / s2 - 1.0)
            a = self.params['a_' + n + '_1']
            self.params['b_' + n + '_1'] =  a * (1. - xbar) / xbar
            xbar = np.sum(np.multiply(1-temp_lab , X)) / np.sum(1-temp_lab)
            s2 = np.sum(np.multiply(1-temp_lab , (X -xbar)**2)) / np.sum(1-temp_lab)
            self.params['a_' + n + '_0'] = xbar * (xbar * (1. - xbar) / s2 - 1.0)
            a = self.params['a_' + n + '_0']
            self.params['b_' + n + '_0'] =  a * (1. - xbar) / xbar


        #self.maximization(data, T, imputed_labels)
        print(self.params)
        for it in np.arange(self.EM_steps):
            print("EM step nb", it)
            imputed_labels = self.expectation(data, T, imputed_labels)
            print(it, "imputed labels")
            print(list(imputed_labels))
            print("Starting max")
            self.maximization(data, T, imputed_labels)
            print("Max done")
            print(self.params)
        return imputed_labels

    def save(self, filename):
        dict_v = {'params': self.params, 'hyperparams': self.hyperparams} ### we just need the prior parameters for our models
        pickle.dump(dict_v, open(filename, 'wb'))

    def load(self, filename):
        dict_v = pickle.load(open(filename, 'rb'))
        self.params = dict_v['params']
        self.hyperparams = dict_v['hyperparams']

    def posterior_given_X(self, data):
        ''' Produces a posterior credible distribution for the immunity score given context
        + questionnaire
        Just have to unfold the model progressively: sample from the different distributions
        the different log ratios
        '''
        Y = np.hstack([np.expand_dims(np.ones(data.shape[0]),1),
                       data[self.list_context_variables]])
        index_sympt = np.where(data['symptomatic']==1)[0]
        # print(log_odds)
        N, _ = Y.shape

        n = 'symptomatic'
        X = data[n].values
        x = np.reshape(np.random.beta(self.hyperparams['alpha_'+ n + '_1'],
                           self.hyperparams['beta_'+ n + '_1'],
                           self.B * N),
                       (N, self.B))### Sample from the distribution?
        y = np.reshape(np.random.beta(self.hyperparams['alpha_'+ n + '_0'],
                           self.hyperparams['beta_'+ n + '_0'],
                           self.B * N),
                       (N, self.B))### Sample from the distribution?
        log_odds = (np.einsum('i, ij -> ij', 1-X, np.log(y))+ np.einsum('i, ij -> ij',1-X, np.log(1-y))\
                    - np.einsum('i, ij -> ij',X, np.log(x)) - np.einsum('i, ij -> ij', X, np.log(1-x)))

        for n in self.list_symptoms:
              X = data[n].iloc[index_sympt]
              NN = len(index_sympt)
              x = np.reshape(np.random.beta(self.hyperparams['alpha_'+ n + '_1'],
                                self.hyperparams['beta_'+ n + '_1'],
                                self.B * NN),
                            (NN, self.B))### Sample from the distribution?
              y = np.reshape(np.random.beta(self.hyperparams['alpha_'+ n + '_0'],
                                self.hyperparams['beta_'+ n + '_0'],
                                self.B * NN),
                            (NN, self.B))### Sample from the distribution?
              log_odds[index_sympt]+= (np.einsum('i, ij -> ij',1-X, np.log(y)) \
                          + np.einsum('i, ij -> ij',1-X, np.log(1-y))\
                          - np.einsum('i, ij -> ij',X, np.log(x)) \
                          - np.einsum('i, ij -> ij',X, np.log(1-x)))
        Sigma = np.linalg.inv(self.H)
        sigma = Y.dot(Sigma).dot(Y.T)
        log_odds +=  np.random.multivariate_normal(-Y.dot(self.beta_reg), sigma, self.B).T
        preds = np.divide(np.ones(self.B) ,  np.ones(self.B) + np.exp(log_odds))
        return(preds)

    def posterior_given_XandT(self, data, T):
        ''' Produces a posterior credible distribution for the immunity score given context
        + questionnaire + IMAGE LABEL (T: binary) !!
        '''
        Y = np.hstack([np.expand_dims(np.ones(data.shape[0]),1),
                       data[self.list_context_variables]])
        index_sympt = np.where(data['symptomatic']==1)[0]
        N, _ = Y.shape

        n = 'symptomatic'
        X = data[n].values
        x = np.reshape(np.random.beta(self.hyperparams['alpha_'+ n + '_1'],
                           self.hyperparams['beta_'+ n + '_1'],
                           self.B * N),
                       (N, self.B))### Sample from the distribution?
        y = np.reshape(np.random.beta(self.hyperparams['alpha_'+ n + '_0'],
                           self.hyperparams['beta_'+ n + '_0'],
                           self.B * N),
                       (N, self.B))### Sample from the distribution?
        print("here")
        log_odds = (np.einsum('i, ij -> ij', 1-X, np.log(y))+ np.einsum('i, ij -> ij',1-X, np.log(1-y))\
                    - np.einsum('i, ij -> ij',X, np.log(x)) - np.einsum('i, ij -> ij', X, np.log(1-x)))
        n = 'T'
        for k in self.specificity.keys():
            index = np.where(data['label_t']==k)[0]
            NN = len(index)
            X = T[index]
            x = np.reshape(np.random.beta(self.hyperparams['alpha_'+ n + '_1_' + k],
                           self.hyperparams['beta_'+ n + '_1_' + k],
                           self.B * NN),
                       (NN, self.B))### Sample from the distribution?
            y = np.reshape(np.random.beta(self.hyperparams['alpha_'+ n + '_0_' + k],
                           self.hyperparams['beta_'+ n + '_0_' + k],
                           self.B * NN),
                       (NN, self.B))
            log_odds[index] += (np.einsum('i, ij -> ij', 1-X, np.log(y))+ np.einsum('i, ij -> ij',1-X, np.log(1-y))\
                    - np.einsum('i, ij -> ij',X, np.log(x)) - np.einsum('i, ij -> ij', X, np.log(1-x)))

        for n in self.list_symptoms:
              X = data[n].iloc[index_sympt]
              NN = len(index_sympt)
              x = np.reshape(np.random.beta(self.hyperparams['alpha_'+ n + '_1'],
                                self.hyperparams['beta_'+ n + '_1'],
                                self.B * NN),
                            (NN, self.B))### Sample from the distribution?
              y = np.reshape(np.random.beta(self.hyperparams['alpha_'+ n + '_0'],
                                self.hyperparams['beta_'+ n + '_0'],
                                self.B * NN),
                            (NN, self.B))### Sample from the distribution?
              log_odds[index_sympt]+= (np.einsum('i, ij -> ij',1-X, np.log(y)) \
                          + np.einsum('i, ij -> ij',1-X, np.log(1-y))\
                          - np.einsum('i, ij -> ij',X, np.log(x)) \
                          - np.einsum('i, ij -> ij',X, np.log(1-x)))
        Sigma = np.linalg.inv(self.H)
        sigma = Y.dot(Sigma).dot(Y.T)
        log_odds +=  np.random.multivariate_normal(-Y.dot(self.beta_reg), sigma, self.B).T
        preds = np.divide(np.ones(self.B) ,  np.ones(self.B) + np.exp(log_odds))

        return(preds)
