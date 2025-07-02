import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



class Dashboard:
    def __init__(self,**kwargs):
        d={}
        d.update(kwargs)
        data_file=d['data_fname']
        data_sheet=d['data_sheet']
        self.features=d['selected_features']
        self.feature_map=d['feature_map']
        self.df=pd.read_excel(data_file,sheet_name=data_sheet)
    def creat_donut_chart():
        pass
    def get_crosstabs():
        pass