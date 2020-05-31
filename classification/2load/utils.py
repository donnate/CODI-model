"""Generate synthetic dataset of symptoms, risk factors and test outcomes.

Synthetic datasets allow validating experimentally
the multimodal data integration approach.
"""

import os

import numpy as np
import pandas as pd

PARAMS = {
    'n_samples': 1,
    'beta': [1, 2, -3],
    'sigma': 1.2,
    'sensitivity': 0.666,
    'specificity': 0.01,
    'proba_sick': [0.8, 0.9],
    'proba_healthy': [0.6, 0.1],
    'csv_prefix': '00'
}

assert len(PARAMS['proba_sick']) == len(PARAMS['proba_healthy'])

PARAMS['n_factors'] = len(PARAMS['beta']) - 1
PARAMS['n_symptoms'] = len(PARAMS['proba_sick'])

DATA_DIR = 'synthetic_data'
DATA_CSV = os.path.join(
    DATA_DIR, PARAMS['csv_prefix'] + '_data.csv')
LABELS_CSV = os.path.join(
    DATA_DIR, PARAMS['csv_prefix'] + '_labels.csv')


def generate_risk_factors(params=PARAMS):
    """Generate risk factors Y.

    The risk factors are sampled from the uniform distribution.

    :param params: (dict) parameters of generative model
    :return: (array) risk factors, shape n_samples x n_factors
    """
    n_samples = params['n_samples']
    n_factors = params['n_factors']
    risk_factors = np.random.uniform(
        low=0., high=5., size=(n_samples, n_factors))
    return risk_factors


def generate_diagnostic_proba(risk_factors, params=PARAMS):
    """Generate probability of a positive diagnostic D.

    The diagnostic probability is generated from a simple logistic
    regression model.

    :param risk_factors: (array) risk factors, shape n_samples x n_factors
    :param params: (dict) parameters of generative model
    :return: (array) probability of getting the disease, length n_samples
    """
    n_samples, n_factors = risk_factors.shape

    ones = np.ones((n_samples, 1))
    design_matrix = np.concatenate([ones, risk_factors], axis=1)

    beta = np.array(params['beta'])
    sigma = params['sigma']

    epsilon = np.random.normal(loc=0., scale=sigma, size=(n_samples,))

    logodds = np.matmul(design_matrix, beta) + epsilon
    proba = 1 / (1 + np.exp(-logodds))
    return proba


def generate_diagnostic(proba):
    """Generate diagnostic D.

    :param proba: (array) probability of getting the disease, length n_samples
    :return: (array) diagnostics, length n_samples
    """
    diagnostics = np.random.binomial(n=1, p=proba)
    return diagnostics


def generate_test_outcome(diagnostics, params=PARAMS):
    """Generate outcome of immunoassay test T.

    :param diagnostics: (array) diagnostics of patients, length n_samples
    :param params: (dict) parameters of generative model
    :return: (array) test outcomes, length n_samples
    """
    n_samples = len(diagnostics)
    sensitivity = params['sensitivity']  ### P(T=1|D=1)
    specificity = params['specificity']  ### P(T=0|D=0)

    proba = np.zeros((n_samples,))

    proba[diagnostics == 1] = sensitivity 
    proba[diagnostics == 0] = 1 - specificity

    test_outcomes = np.random.binomial(n=1, p=proba)
    return test_outcomes


def generate_symptoms(diagnostics, params=PARAMS):
    """Generate symptoms X.

    :param diagnostics: (array) diagnostics of patients, length n_samples
    :param params: (dict) parameters of generative model
    :return: (array) symptoms, shape n_samples x n_symptoms
    """
    n_samples = len(diagnostics)

    proba_sick = params['proba_sick']
    proba_healthy = params['proba_healthy']
    n_symptoms = len(proba_sick)

    proba = np.zeros((n_samples, n_symptoms))

    proba[diagnostics == 1] = proba_sick
    proba[diagnostics == 0] = proba_healthy

    symptoms = np.random.binomial(n=1, p=proba)
    return symptoms


def write_data_csv(risk_factors, symptoms, test_outcomes, path=DATA_CSV):
    """Generate csv of synthetic data.

    Each row represents one sample, i.e. one patient.
    The length of a row is n_factors + n_symptoms + 1,
    corresponding to the concatenation of:
    - the risk factors Y,
    - the symptoms X,
    - the test outcome,
    on the row.

    :param risk_factors: (array) risk factors, shape n_samples x n_factors
    :param symptoms: (array) symptoms, shape n_samples x n_symptoms
    :param test_outcomes: (array) test outcomes, shape n_samples
    :param path: (str) csv path
    """
    n_factors = risk_factors.shape[-1]
    n_symptoms = symptoms.shape[-1]

    test_outcomes = np.expand_dims(test_outcomes, axis=1)

    data = np.concatenate(
        [risk_factors, symptoms, test_outcomes], axis=1)
    header = \
        'risk factor, ' * n_factors + 'symptom, ' * n_symptoms + 'test outcome'
    fmt = '%1.4f ' * n_factors + '%i ' * n_symptoms + '%i'

    np.savetxt(path, data, header=header, delimiter=',', fmt=fmt)
    print('Saved data at %s.' % path)


def write_labels_csv(diagnostics, path=LABELS_CSV):
    """Generate csv of synthetic diagnostic labels.

    Each row represents one sample, i.e. one patient,
    with a 0/1 value denoting their true diagnostic.

    :param diagnostics: (array) diagnostics, length n_samples
    :param path: (str) csv path
    """
    np.savetxt(path, diagnostics, header='diagnostic', delimiter=',', fmt='%i')
    print('Saved labels at %s.' % path)


def generate_fake_dataset(parameters):
  risk_factors = generate_risk_factors(parameters)  #Y
  diagnostic_proba = generate_diagnostic_proba(risk_factors, parameters) #P(D|Y)
  diagnostic = generate_diagnostic(diagnostic_proba)
  test_outcome = generate_test_outcome(diagnostic, parameters) #P(T|D)
  symptoms = generate_symptoms(diagnostic, params=parameters) #P(X|D)
  n_factors = risk_factors.shape[-1]
  n_symptoms = symptoms.shape[-1]

  test_outcomes = np.expand_dims(test_outcome, axis=1)

  data = np.concatenate(
        [risk_factors, symptoms, test_outcomes], axis=1)
  data = pd.DataFrame(data, columns=["risk_factor", "risk_factor2",
                                     "symptom","symptom2","symptom3",
                                     "test_outcome"])
  return data, diagnostic
