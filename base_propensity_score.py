import pandas as pd
import numpy as np
import json

class BasePropensityScore:
    def __init__(self,df,features):
        self.df=df
        self.features=features
        self.calculate_base()
        
    def calculate_base(self):
        base={}
        state_base={}
        city_base={}        
        counts=self.df[self.features].sum().to_dict()
        pan_india_base={k: round((v/self.df.shape[0])*100) for k, v in counts.items()}
        for state in list(self.df.State.unique()):
            tdf_state = self.df[(self.df['State']==state)]
            counts_state=tdf_state[self.features].sum().to_dict()
            perc_counts_state={k: round((v/tdf_state.shape[0])*100) for k, v in counts_state.items()}     
            state_base[state]=perc_counts_state
            city_base[state] = {}
            # print(f"{state}:{tdf_state.City.unique()}")
            for city in list(tdf_state.City.unique()):         
                tdf_city = tdf_state[tdf_state['City']==city]
                counts_city=tdf_city[self.features].sum().to_dict()
                perc_counts_city={k: round((v/tdf_city.shape[0])*100) for k, v in counts_city.items()}         
                city_base[state][city]=perc_counts_city
        base['pan_india']=pan_india_base
        base['state']=state_base
        base['city']=city_base
        file_path = "base_propensity_score_dummy.json"
        with open(file_path, "w") as f:
            json.dump(base, f, indent=4)

if __name__=="__main__":
    # df=pd.read_excel("C:\\Users\\RajmaneD\\OneDrive - Kantar\\DR\\MM_v2_fastapi\\KANTAR_MM_dummy_data.xlsx",sheet_name="OG")
    df=pd.read_csv("C:\\Users\\RajmaneD\\OneDrive - Kantar\\DR\\MM_v2_fastapi\\india_states_cities.csv")
    df = df[~(df.longitude.isna() | df.latitude.isna())]
    features=['Online_Shopping', 'Linear_Television', 'Smart_Television',
       'Internet_Users', 'Smart_Phone_Users', 'Social_Media_Users',
       'Digital_Payment', 'Credit_Card', 'Netbanking', 'Insurance',
       'Stocks_Shares', 'Mutual_Funds', 'Loan', 'Electricity_Connection',
       'Ceiling_Fan', 'LPG_Stove', 'Two_wheeler', 'Colour_TV', 'Refrigerator',
       'Washing_Machine', 'Personal_Computer_Laptop', 'Car_Jeep_Van',
       'Air_Conditioner']
    bps=BasePropensityScore(df=df,features=features)

    
        
    