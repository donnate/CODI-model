''' Module class for training the simple logistic regression model.
'''
import copy
import numpy as np
import pandas as pd
import pickle
import sklearn as sk

from sklearn.linear_model import LogisticRegression
import classification.helper as helper


MODEL = LogisticRegression(max_iter=10000)
PARAMETERS_REG = {'C': np.logspace(-4, 4, num=50)}

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
SPECIFICITY = {'asymptomatic': [0.995, 0.992, 0.997],
               '2-10 days': [0.995, 0.992, 0.997],
               '11-20 days': [0.995, 0.992, 0.997],
               '21+ days': [0.995, 0.992, 0.997]}

class VanillaClassifier:
    ''' Simple prediction of the immunity score based on a two-step process
    (A) compute prior for the diagnostic (using logistic regression
    (B) add the test information and uncertainty
    '''
    def __init__(self, parameters=PARAMETERS_REG, list_variables=VARIABLES,
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
            a1, b1 = helper.moment_matching_beta(sensitivity[lab_t][0],
                                          (0.5 *(sensitivity[lab_t][2] - sensitivity[lab_t][1]))**2)
            a0, b0 = helper.moment_matching_beta(specificity[lab_t][0],
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
