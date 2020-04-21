# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from sklearn import svm
import seaborn as sns; sns.set(font_scale=1.2)
import pickle


#import data
dataset = pd.read_csv('healthcare_data.csv')

X=dataset.iloc[:,1:11]
y=dataset.iloc[:,11:12]


#format or pre process the data
def convert_symptoms_to_int(word):
    word_dict = {'Yes':1, 'No':2, 'Maybe':3}
    return word_dict[word]

def convert_temp_to_int(word):
    word_dict = {'very high':1, 'high':2, 'normal':3, 'low':4}
    return word_dict[word]

def convert_travel_history_to_int(word):
    word_dict = {'Foreign Travel':1, 'Local Travel':2, 'No':3}
    return word_dict[word]

def convert_result_to_int(word):
    word_dict = {'Cold':1, 'Covid':2, 'Fever':3, 'Flu':4, 'Healthy':5, 'low':6, 'Weak immunity':7}
    return word_dict[word]

X['BodyTemp'] = X['BodyTemp'].apply(lambda x : convert_temp_to_int(x))
X['DryCough'] = X['DryCough'].apply(lambda x : convert_symptoms_to_int(x))
X['Tiredness'] = X['Tiredness'].apply(lambda x : convert_symptoms_to_int(x))
X['runnyNose'] = X['runnyNose'].apply(lambda x : convert_symptoms_to_int(x))
X['SoreThroat'] = X['SoreThroat'].apply(lambda x : convert_symptoms_to_int(x))
X['DifficulyinBreathing'] = X['DifficulyinBreathing'].apply(lambda x : convert_symptoms_to_int(x))
X['achesAndPains'] = X['achesAndPains'].apply(lambda x : convert_symptoms_to_int(x))
X['nasalCongestion'] = X['nasalCongestion'].apply(lambda x : convert_symptoms_to_int(x))
X['diarrhoea'] = X['diarrhoea'].apply(lambda x : convert_symptoms_to_int(x))
X['TravelHistory'] = X['TravelHistory'].apply(lambda x : convert_travel_history_to_int(x))

y['Result'] = y['Result'].apply(lambda y : convert_result_to_int(y))

result_label = np.array(y)
symptoms = X[['BodyTemp', 'DryCough', 'Tiredness', 'runnyNose', 'SoreThroat', 'DifficulyinBreathing', 'achesAndPains', 'nasalCongestion','diarrhoea','TravelHistory']].values


#fit the model
model = svm.SVC(kernel='linear')
model.fit(symptoms, result_label)


#create a function to predict outcome
def health_check(BodyTemp, DryCough, Tiredness, runnyNose, SoreThroat, DifficulyinBreathing, achesAndPains, nasalCongestion, diarrhoea, TravelHistory):
    if(model.predict([[BodyTemp, DryCough, Tiredness, runnyNose, SoreThroat, DifficulyinBreathing, achesAndPains, nasalCongestion, diarrhoea, TravelHistory]]))==1:
        print('Cold')
    elif(model.predict([[BodyTemp, DryCough, Tiredness, runnyNose, SoreThroat, DifficulyinBreathing, achesAndPains, nasalCongestion, diarrhoea, TravelHistory]]))==2:
        print('Covid')
    elif(model.predict([[BodyTemp, DryCough, Tiredness, runnyNose, SoreThroat, DifficulyinBreathing, achesAndPains, nasalCongestion, diarrhoea, TravelHistory]]))==3:
        print('Fever')
    elif(model.predict([[BodyTemp, DryCough, Tiredness, runnyNose, SoreThroat, DifficulyinBreathing, achesAndPains, nasalCongestion, diarrhoea, TravelHistory]]))==4:
        print('Flu')
    elif(model.predict([[BodyTemp, DryCough, Tiredness, runnyNose, SoreThroat, DifficulyinBreathing, achesAndPains, nasalCongestion, diarrhoea, TravelHistory]]))==5:
        print('Healthy')
    elif(model.predict([[BodyTemp, DryCough, Tiredness, runnyNose, SoreThroat, DifficulyinBreathing, achesAndPains, nasalCongestion, diarrhoea, TravelHistory]]))==6:
        print('low')
    elif(model.predict([[BodyTemp, DryCough, Tiredness, runnyNose, SoreThroat, DifficulyinBreathing, achesAndPains, nasalCongestion, diarrhoea, TravelHistory]]))==7:
        print('Weak immunity')
    elif(model.predict([[BodyTemp, DryCough, Tiredness, runnyNose, SoreThroat, DifficulyinBreathing, achesAndPains, nasalCongestion, diarrhoea, TravelHistory]]))==8:
        print('Covid')
   
#test the model
health_check(1,2,2,2,2,2,2,2,2,2)

