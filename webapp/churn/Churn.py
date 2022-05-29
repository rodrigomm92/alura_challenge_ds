import pickle
import inflection
import pandas as pd
import numpy as np

class Churn:
    def __init__(self):
        self.scaler = pickle.load(open('model/scaler.pkl','rb'))
      
    def data_cleaning(self, df_raw):
        # padronizando os headers para snake case
        cols_snake = list( map( lambda x: inflection.underscore( x ), df_raw.columns ) )
        df_raw.columns = cols_snake
        
#         # transformando valores vazios e ausentes em NA
#         df_raw = df_raw.mask(df_raw == ' ').mask(df_raw == '')
        
        # convertendo em float
#         df_raw['charges_total'] = df_raw['charges_total'].astype('float')

        return df_raw
    
    def feature_engineering(self, df_clean):
        # transformando os valores "no xxxx service em no"
        for column in df_clean.columns[7:15]:
            df_clean.loc[:,column] = df_clean.loc[:,column].apply(lambda x: 0 if 'No' in x else 1)
        # -----------------------------------------------------------------------------

#         # transformando a variável churn em int
#         df_clean.loc[:,'churn'] = df_clean['churn'].apply(lambda x: 1 if x == 'Yes' else 0)

        # transformando a variável partner em int
        df_clean.loc[:,'partner'] = df_clean['partner'].apply(lambda x: 1 if x == 'Yes' else 0)

        # transformando a variável dependents em int
        df_clean.loc[:,'dependents'] = df_clean['dependents'].apply(lambda x: 1 if x == 'Yes' else 0)

        # transformando a variável phone_service em int
        df_clean.loc[:,'phone_service'] = df_clean['phone_service'].apply(lambda x: 1 if x == 'Yes' else 0)

        # transformando a variável paperless_billing em int
        df_clean.loc[:,'paperless_billing'] = df_clean['paperless_billing'].apply(lambda x: 1 if x == 'Yes' else 0)

        # criando feature de gasto diário partindo do charges_total
        df_clean.insert(loc=18, column='daily_charge', value=df_clean['charges_total']/(df_clean['tenure']*30))

        # transformando os valores das features para o formato snake_case
        df_clean.loc[:,'contract'] = df_clean['contract'].apply(lambda x: inflection.underscore(str(x)))
        df_clean.loc[:,'contract'] = df_clean['contract'].apply(lambda x: inflection.parameterize(x, separator='_'))
        df_clean.loc[:,'gender'] = df_clean['gender'].apply(lambda x: inflection.parameterize(x, separator='_'))
        df_clean.loc[:,'payment_method'] = df_clean['payment_method'].apply(lambda x: inflection.parameterize(x, separator='_'))
        
        return df_clean

    def data_preparation(self, df_clean):
        df_norm = df_clean.drop(columns=['customer_id'])

        # aplicando one hot encoding nas variáveis categóricas
        df_norm = pd.get_dummies(df_norm, columns=['gender', 'contract', 'payment_method'], drop_first=True)
        
        df_transformed = pd.DataFrame(self.scaler.transform(df_norm))
        df_transformed.columns = df_norm.columns
 
        selected_cols = ['tenure','charges_total','daily_charge', 'charges_monthly', 'contract_two_year',
                'payment_method_electronic_check', 'contract_one_year', 'internet_service', 'gender_male',
                'paperless_billing', 'tech_support', 'online_security', 'partner', 'online_backup',  'dependents',
                'senior_citizen', 'multiple_lines', 'device_protection', 'streaming_movies', 'streaming_tv',
                'payment_method_credit_card_automatic', 'payment_method_mailed_check', 'phone_service']
        
        return df_transformed[selected_cols]
        
    def get_prediction(self, model, original_data, test_data):
        pred = model.predict(test_data)
        original_data['churn'] = pred
        
        return original_data.to_json(orient='records', date_format='iso')
        
                         