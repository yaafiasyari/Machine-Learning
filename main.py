#!python3

import os
import json
import pickle

import pandas as pd
import numpy as np

from sqlalchemy import create_engine

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    
    engine = create_engine('postgresql+psycopg2://postgres:postgres@localhost/digitalskola')

    path = os.getcwd()+"\\"
    path_data = path+"data"+"\\"
    path_model = path+"model"+"\\"

    test = pd.DataFrame({'Age':[21],
                        'Language':[1],
                        'Sex':[2],
                        'Has_Credit':[2],
                        'Region':[1]})
    
    labelAge = pickle.load(open(path_model+'labelAge.pkl','rb'))
    test['AgeScalar'] = labelAge.transform(test[['Age']])

    labelLanguage = pickle.load(open(path_model+'labelLanguage.pkl','rb'))
    test['LanguageEncoder'] = labelLanguage.transform(test[['Language']])

    labelSex = pickle.load(open(path_model+'labelSex.pkl','rb'))
    test['SexEncoder'] = labelSex.transform(test[['Sex']])

    labelHasCredit = pickle.load(open(path_model+'labelHasCredit.pkl','rb'))
    test['HasCreditEncoder'] = labelHasCredit.transform(test[['Has_Credit']])
    
    labelRegion = pickle.load(open(path_model+'labelRegion.pkl','rb'))
    # RegionEncoder = [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]]
    RegionEncoder = labelRegion.transform(test[['Region']].values).toarray()
    dfRegion = pd.DataFrame(RegionEncoder, 
                            columns=["RegionEncoder_"+str(i) for i in range(len(RegionEncoder[0]))])
    test = pd.concat([test, dfRegion], axis=1)

    X = test.drop(['Age','Language','Sex','Has_Credit','Region'], axis=1).values
    model = pickle.load(open(path_model+'modelDecisionTree.pkl','rb'))
    predict = model.predict(X)

    if predict[0] == 0:
        status = "Reject"
        print(test[['Age','Language','Sex','Has_Credit','Region']].to_dict('records'))
        print("Status: Reject")
    else:
        status = "Success"
        print(test[['Age','Language','Sex','Has_Credit','Region']].to_dict('records'))
        print("Status: Success")

    data = test[['Age','Language','Sex','Has_Credit','Region']]
    data['Label_Prediction'] = status
    data['Language'] = data['Language'].map({1:'Native',2:'Billigual'})
    data['Sex'] = data['Sex'].map({1:'Man',2:'Woman'}) 
    data['Has_Credit'] = data['Has_Credit'].map({1:'Yes',2:'No'})
    data.to_sql('user_predict', engine, if_exists='append', index=False)    