''' Preprocessing
'''
import copy
import numpy as np
import pandas as pd



def standardize_input(data, dict_corr,dict_symptoms,
                      symptoms):
  data2 = copy.deepcopy(data)
  data2 = data2.rename(columns={dict_corr[u][0]:u for u in dict_corr.keys() })
  for n in symptoms:
    data2[n] = 0
  data2['label'] = "NotApplicable"
  data2['test'] = 0
  data2['label_t'] = ""
  data2['date_test_taken'] = pd.to_datetime(data2['date_test_taken'])
  for j in np.where(data2['date_symptoms'] == 'BEFORE 01 January'):
     data2.at[j, 'date_symptoms']= np.nan
  for j in np.where(data2['date_symptoms'] =='I never felt ill'):
     data2.at[j, 'date_symptoms']= np.nan
  data2['date_symptoms' ] = pd.to_datetime(data2['date_symptoms'])
  for j in np.where(data2['patient_prior_on_date_caught'] =="I don't think I caught it"):
     data2.at[j, 'patient_prior_on_date_caught']= np.nan
  for j in np.where(data2['patient_prior_on_date_caught'] =='BEFORE 01 January'):
     data2.at[j, 'patient_prior_on_date_caught']= np.nan
  for j in np.where(data2['date_first_housemate_illness'] =='BEFORE 01 January'):
     data2.at[j, 'date_first_housemate_illness']= np.nan
  data2['patient_prior_on_date_caught'] = pd.to_datetime(data2['patient_prior_on_date_caught'])
  data2['date_first_housemate_illness'] = pd.to_datetime(data2['date_first_housemate_illness'])
  data2['delta_time_symptoms_onset'] = np.nan
  data2['delta_time_symptoms_housemate'] = np.nan
  data2['household_illness'] = 0
  data2['index_duration_symptoms'] = 0
  for i in np.arange(data2.shape[0]):
    if data2['duration_symptoms'].iloc[i] == "I'm still ill!":
      data2.at[i, 'duration_symptoms'] = "I know the exact number of days"
      data2.at[i,'duration_symptoms_days'] = (data2.iloc[i]['date_test_taken'] - data2.iloc[i]['date_symptoms']).days
    if data2['duration_symptoms'].iloc[i] == 'I know the exact number of days': 
      if data2.at[i,'duration_symptoms_days']<4:  data2.at[i,'index_duration_symptoms']= 1
      elif data2.at[i,'duration_symptoms_days']<7:  data2.at[i,'index_duration_symptoms']= 2
      elif data2.at[i,'duration_symptoms_days']<14:  data2.at[i,'index_duration_symptoms']= 3
      elif data2.at[i,'duration_symptoms_days']<14:  data2.at[i,'index_duration_symptoms']= 4
      else: data2.at[i,'index_duration_symptoms']= 5
    elif data2['duration_symptoms'].iloc[i] == 'I never felt ill': data2.at[i,'index_duration_symptoms']= 0
    elif data2['duration_symptoms'].iloc[i] == 'Less than a week': data2.at[i,'index_duration_symptoms']= 2
    elif data2['duration_symptoms'].iloc[i] == 'More than three weeks': data2.at[i,'index_duration_symptoms']= 5
    elif data2['duration_symptoms'].iloc[i] == 'More than two weeks': data2.at[i,'index_duration_symptoms']= 4
    elif data2['duration_symptoms'].iloc[i] == 'On and off for a few days or less': data2.at[i,'index_duration_symptoms']= 1
    elif data2['duration_symptoms'].iloc[i] == 'One to two weeks': data2.at[i,'index_duration_symptoms']= 3
    if data2['results'].iloc[i] is not np.nan:
      if 'C and IgG' == data2['results'].iloc[i] : data2.at[i,'label'] ="positive"
      if 'C, IgM and IgG' == data2['results'].iloc[i] : data2.at[i,'label'] ="positive"
      if 'C and IgM' == data2['results'].iloc[i] : data2.at[i,'label'] ="positive"
      if "C only" == data2['results'].iloc[i] : data2.at[i,'label'] ="negative"

    data2.at[i,'delta_time_symptoms_onset'] = (data2.iloc[i]['date_test_taken'] - data2.iloc[i]['date_symptoms']).days
    data2.at[i,'delta_time_symptoms_housemate'] = (data2.iloc[i]['date_test_taken'] - data2.iloc[i]['date_first_housemate_illness']).days
    if data2['type_test'].iloc[i] is not np.nan:
      if "antibody" in data2['type_test'].iloc[i]:
        data2.at[i, 'test'] = 1
      if "Antibody" in data2['type_test'].iloc[i]:
        data2.at[i,'test'] = 1
      if "PCR" in data2['type_test'].iloc[i]:
        data2.at[i,'test'] = 2 


    x = data2['Did you, or do you, have any of the following symptoms?'].iloc[i]
    if x is not np.nan:
        sympt = x.split(',')
        for n in sympt:
          nn = n.strip()
          if nn != 'NONE': data2.at[i, dict_symptoms[nn]] = 1
          
    x = data2['You said you felt ill but don\'t think you had Coronavirus. Still, please tell us your symptoms.'].iloc[i]    
    if x is not np.nan:
        sympt = x.split(',')
        for n in sympt:
          nn = n.strip()
          if nn != 'NONE': data2.at[i, dict_symptoms[nn]] = 1
    
    if data2.iloc[i].Ill == "I'm fine, haven't been ill at all since the pandemic began": 
      data2.at[i,'label_t'] = 'asymptomatic'
    else:
      if data2.iloc[i].delta_time_symptoms_onset < 11:
          data2.at[i,'label_t'] = '2-10 days'
      if data2.iloc[i].delta_time_symptoms_onset > 10 and data2.iloc[i].delta_time_symptoms_onset < 21:
          data2.at[i,'label_t'] = '11-20 days'
      else:
          data2.at[i,'label_t']= '21+ days'
  data2['index_severity'].loc[np.isnan(data2['index_severity'])] = 0
  data2['index_anxiety'].loc[np.isnan(data2['index_anxiety'])] = 0
  data2['index_pulmonary_impact'].loc[np.isnan(data2['index_pulmonary_impact'])] = 0
  return data2