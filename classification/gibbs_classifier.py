''' Module class for training the simple logistic regression
'''
import json
import numpy as np
import pandas as pd
import pickle
import sklearn as sk


from .helper_classification import * 


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
SPECIFICITY = {'asymptomatic': [0.995, 0.992, 0.997],
               '2-10 days': [0.995, 0.992, 0.997],
               '11-20 days': [0.995, 0.992, 0.997],
               '21+ days': [0.995, 0.992, 0.997]}              

class GibbsClassifier():
    def __init__(self,parameters=PARAMETERS, list_variables=VARIABLES):
        self.model = None
        self.parameters = parameters
        self.B = 1000
        self.boots = None
        self.pen = 1.0
        self.list_variables = list_variables
        self.sensitivity_test_prior = []
        
    def fit(self, model, train_data, train_labels):
        GS = sk.model_selection.GridSearchCV(model, self.parameters, cv=10, verbose=0)
        GS.fit(train_data, train_labels)
        self.pen = GS.best_params_['C']
        self.model = LogisticRegression(C=self.pen, max_iter=10000)
        self.model.fit(train_data[self.list_variables], train_labels)
        #### Now bootstrap to get confidence scores
        self.boots = {}
        for b in np.arange(self.B):
            index_b = np.random.choice(np.arange(train_data.shape[0]), train_data.shape[0], replace=True)
            logreg_b = LogisticRegression(C=self.pen, max_iter=10000)
            logreg_b.fit(train_data[self.list_variables].iloc[index_b,:],
                        train_labels.iloc[index_b])
            self.boots[b] = logreg_b
        
    def predict_proba(self, X):
        if self.model is None: 
            print("Error. No model ws fit")
            return None
        return self.model.predict_proba(X[self.list_variables])

    def predict_log_proba(self, X):
        if self.model is None: 
            print("Error. No model ws fit")
            return None
        return self.model.predict_log_proba(X[self.list_variables]) 
    
    def predict(self, X):
        if self.model is None: 
            print("Error. No model ws fit")
            return None
        return self.model.predict(X[self.list_variables])
        
    def save(self, filename):
        dict_v = {'model': self.model,
        'bootstraps': self.boots
        }
        pickle.dump(dict_v, open(filename, 'wb'))
    
    def load(self, filename):
        dict_v = pickle.load(open(DATA_FOLDER + 'lr.pkl', 'rb'))
        self.model = dict_v['model']
        self.boots = dict_v['bootstraps']
            
    def produce_confint(self, X):
        y_hat = self.model.predict_proba(np.expand_dims(X[self.list_variables], 0))[0,1]
        preds = [self.boots[b].predict_proba(np.expand_dims(X[self.list_variables], 0))[0,1] 
           for b in np.arange(self.B)]
        return([y_hat, np.percentile(preds, 2.5), np.percentile(preds, 97.5)], preds)
        
    def combine_image(self, X, T, sensitivity=SENSITIVITY, 
    specificity=SPECIFICITY):
        pred = []
        if X.Ill == "I'm fine, haven't been ill at all since the pandemic began": 
            lab_t = 'asymptomatic'
        else:
            if X.delta_time_symptoms_onset<11:
                lab_t = '2-10 days'
            if X.delta_time_symptoms_onset>10 and X.delta_time_symptoms_onset<21:
                lab_t = '11-20 days'
            else:
                lab_t = '21+ days'
        #### Sample prior sensivity and specificity from appropriate distribution

        a1, b1 = moment_matching_beta(sensitivity[lab_t][0], (0.5 *(sensitivity[lab_t][2] - sensitivity[lab_t][1]))**2)
        a0, b0 = moment_matching_beta(specificity[lab_t][0], (0.5 *(sensitivity[lab_t][2] - sensitivity[lab_t][1]))**2)
        print(lab_t, a1, b1)
        print(lab_t, a0, b0)
        specificity = np.random.beta(a0, b0, self.B)
        sensitivity = np.random.beta(a1, b1, self.B)
        log_odds = np.divide((1-specificity),sensitivity)
        
        #### sample log odds from bootstrap
        odds_q, conf_odds_q = self.produce_confint(X)
        sample_odds_q  =  np.array(conf_odds_q)[np.random.choice(np.arange(self.B), self.B, replace=True)]
        pred = np.divide(np.ones(self.B),
                         np.ones(self.B) + np.multiply(log_odds,
                                                  np.divide(sample_odds_q,
                                                            1-sample_odds_q)))
        return([np.mean(pred), np.percentile(pred, 2.5), np.percentile(pred, 97.5)], pred)

    
