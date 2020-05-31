''' Module class for training the EM algorithm
'''
import copy
import json
import numpy as np
import pandas as pd
import pickle
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from helper_classification import *


MODEL = LogisticRegression(max_iter=10000)
PARAMETERS_REG = {'C':np.logspace(-4, 4, num=50),
              #'max_depth': [2,3,4,5,6,7,8,9,10]
              }
SYMPTOMS = ['symptom','symptom2','symptom3']
SYMPTOMS_QUANT = [ ]
CONTEXT_INFO = ['risk_factor','risk_factor2']
TRUST_TEST = ['delta_time_symptoms_onset', ]
VARIABLES = SYMPTOMS + SYMPTOMS_QUANT + CONTEXT_INFO
SENSITIVITY = {'asymptomatic': [0.8, 0.75, 0.85]}
SPECIFICITY = {'asymptomatic': [0.2, 0.15, 0.25]}              
SENSITIVITY_PRIORS = np.zeros((4,2))
SPECIFICITY_PRIORS = np.zeros((4,2))
SENSITIVITY_PRIORS = {k: moment_matching_beta(SENSITIVITY[k][0],
                                      (1.0/(2*1.96) *(SENSITIVITY[k][2] - SENSITIVITY[k][1]))**2) for k in SENSITIVITY.keys()}
SPECIFICITY_PRIORS = {k: moment_matching_beta(SPECIFICITY[k][0],
                                      (1.0/(2*1.96) *(SPECIFICITY[k][2] - SPECIFICITY[k][1]))**2) for k in SENSITIVITY.keys()}


SYMPTOMS = ['symptom','symptom2','symptom3']
SYMPTOMS_QUANT = [ ]
CONTEXT_INFO = ['risk_factor','risk_factor2']
TRUST_TEST = ['delta_time_symptoms_onset', ]
VARIABLES = SYMPTOMS + SYMPTOMS_QUANT + CONTEXT_INFO

class EMClassifier2():
    ''' Adding uncertainty estimates in all the parameters.
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


class EMClassifier3():
    ''' Adding uncertainty estimates in all the parameters.
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

          

class VanillaClassifier():
    ''' Simple prediction of the immunity score based on a two-step process
    (A) compute prior for the diagnostic (using logistic regression
    (B) add the test information and uncertainty
    '''
    def __init__(self,parameters=PARAMETERS_REG, list_variables=VARIABLES,
                 main_param='C', B = 500):
        self.model = None    #### model to use (e.g. LogisticRegression)
        self.parameters = parameters #### parameters in the model that we'll have to fit (e.g., using gridsearch)
        self.B = B    #### nb of bootstrap samples to create confidence bounds for the prior
        self.boots = None   #### storing the different regression models obtained via bootstrap
        self.pen = 1.0      #### regularization penalty 
        self.list_variables = list_variables   #### names of the variables to fit the model on
        self.main_param_name = main_param

    def fit(self, model, train_data, train_labels):
        ''' Fits the prior using questionnaire + context information
        The model here (logisitc regression) could be subject to change quite easily to 
        other types of models (random forest, etc)
        '''
        GS = sk.model_selection.GridSearchCV(model, self.parameters, cv=10, verbose=0)
        GS.fit(train_data[self.list_variables], train_labels)
        self.pen = GS.best_params_[self.main_param_name]
        if self.main_param_name == 'C':
          self.model = LogisticRegression(penalty='l2',
                                           C=GS.best_params_['C'],
                                          max_iter=1000)
        else:
          self.model = RandomForestClassifier(max_depth=GS.best_params_['max_depth'],
                             random_state=0)
        self.model.fit(train_data[self.list_variables], train_labels)
        #### Now bootstrap to get confidence scores
        self.boots = {}
        for b in np.arange(self.B):
            index_b = np.random.choice(np.arange(train_data.shape[0]), train_data.shape[0],
                                       replace=True)
            logreg_b = copy.deepcopy(self.model)
            logreg_b.fit(train_data[self.list_variables].iloc[index_b, :],
                        train_labels[index_b])
            self.boots[b] = logreg_b
        
    def predict_proba(self, X):
        ''' returns the posterior probability of being immune given symptoms +context info
        '''
        if self.model is None: 
            print("Error. No model has been fit")
            return None
        return self.model.predict_proba(X[self.list_variables])

    def predict_log_proba(self, X):
        ''' returns the log posterior probability of being immune given symptoms +context info
        '''
        if self.model is None: 
            print("Error. No model has been  fit")
            return None
        return self.model.predict_log_proba(X[self.list_variables]) 
    
    def predict(self, X):
        ''' returns the most likely label of being immune given symptoms +context info
        '''
        if self.model is None: 
            print("Error. No model has been  fit")
            return None
        return self.model.predict(X[self.list_variables])
        
    def save(self, filename):
        dict_v = {'model': self.model,
        'bootstraps': self.boots
        }
        pickle.dump(dict_v, open(filename, 'wb'))
    
    def load(self, filename):
        dict_v = pickle.load(open(filename, 'rb'))
        self.model = dict_v['model']
        self.boots = dict_v['bootstraps']
            
    def posterior_given_X(self, data):
        ''' Produces a posterior credible distribution for the immunity score given context
        + questionnaire
        '''
        y_hat = self.model.predict_log_proba(data[self.list_variables])[:,0]
        - self.model.predict_proba(data[self.list_variables])[:, 1]
        preds = np.hstack([np.expand_dims(self.boots[b].predict_log_proba(data[self.list_variables])[:,0]
                                  - self.boots[b].predict_log_proba(data[self.list_variables])[:, 1],1)
           for b in np.arange(self.B)])
        return(np.hstack([np.expand_dims(y_hat,1),
           np.expand_dims(np.percentile(preds, 2.5, axis=1), 1),
           np.expand_dims(np.percentile(preds, 97.5, axis=1), 1)]), preds)
        
    def posterior_given_XandT(self, X, T, sensitivity=SENSITIVITY, specificity=SPECIFICITY):
        ''' Produces a posterior credible distribution for the immunity score given context
        + questionnaire + IMAGE LABEL (T: binary) !!
        '''
        
        label_t = X['label_t']
        N = X.shape[0]
        pred = np.zeros((N, self.B))
        for lab_t in np.unique(label_t):
            index = np.where(label_t == lab_t)[0]
            #### Sample prior sensivity and specificity from appropriate distribution
            a1, b1 = moment_matching_beta(sensitivity[lab_t][0],
                                          (0.5 *(sensitivity[lab_t][2] - sensitivity[lab_t][1]))**2)
            a0, b0 = moment_matching_beta(specificity[lab_t][0],
                                          (0.5 *(specificity[lab_t][2] - specificity[lab_t][1]))**2)
            a1 = sensitivity[lab_t][0]
            a0 = specificity[lab_t][0]
            b1 = 1.0/(2. * 1.96) *(sensitivity[lab_t][2] - sensitivity[lab_t][1])  ###spread
            b0 = 1.0/(2. * 1.96)  *(specificity[lab_t][2] - specificity[lab_t][1])
            #specificity = np.random.beta(a0, b0, self.B)
            #sensitivity = np.random.beta(a1, b1, self.B)
            specificity = np.reshape(np.random.normal(a0, b0, self.B * len(index)),
                                   (len(index), self.B))
            specificity[np.where(specificity<0)] = 1e-7
            specificity[np.where(specificity>1)] = 1. - 1e-7
            
            sensitivity = np.reshape(np.random.normal(a1, b1, self.B * len(index)),
                                   (len(index), self.B))
            sensitivity[np.where(sensitivity<0)] = 1e-7
            sensitivity[np.where(sensitivity>1)] = 1. - 1e-7
            odds = np.zeros((len(index), self.B))
            odds[T==0, :] = np.log(specificity[T==0, :]) - np.log(1. - sensitivity[T==0, :])
            odds[T==1, :] = np.log(1.0 - specificity[T==1, :]) - np.log(sensitivity[T==1, :])
            
            #### sample odds from bootstrap
            score_q, all_bootstrap_samples = self.posterior_given_X(X.iloc[index,:])
            all_bootstrap_samples[np.where(all_bootstrap_samples < 1e-7)] = 1e-7  ### to avoid division by -                                     
            #### sample from the boostrap
            all_bootstrap_samples[T == 1,:] = -1.0 * all_bootstrap_samples[T == 1,:] 
            pred[index,:] = np.divide(np.ones((len(index), self.B)),
                         np.ones((len(index), self.B)) + np.exp(odds + all_bootstrap_samples))
        return np.hstack([np.expand_dims(np.mean(pred, 1),1),
           np.expand_dims(np.percentile(pred, 2.5, axis=1), 1),
           np.expand_dims(np.percentile(pred, 97.5, axis=1), 1)])

    def load_training_data_from_s3(self, s3_path=None):
        pass
