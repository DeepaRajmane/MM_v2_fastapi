import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, MultiPoint
import folium

class GeoSpatialData:
    def __init__(self,df):
        # self.settings = {}
        # self.settings.update(kwargs)
        # self.file_path = self.settings['file_path']
        # self.sheet = self.settings['sheet']
        self.data = df
        self.gdf = None

    def read_data(self):
        # self.data = pd.read_excel(self.file_path,sheet_name=self.sheet)
        self.data = self.data[~(self.data.longitude.isna() | self.data.latitude.isna())]
        self.data['geometry'] = self.data.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
        self.gdf = gpd.GeoDataFrame(self.data, geometry='geometry')
        df=self.data.head()
        df.to_html("./templates/bfsi.html")
        # print(self.gdf.columns)

    def group_by_pincode(self,features,top_values=0):
        self.features = features
        _d =  {'geometry': lambda x: MultiPoint(list(x)).convex_hull,
                        'unique_serial_number': 'size'
                    }
        for feature in self.features:
            _d[feature] = "sum"
        grouped = self.gdf.groupby('pincode').agg(_d
                       ).rename({'unique_serial_number': 'total'}, axis=1)
        for feature in self.features:
            grouped[feature] = grouped[feature]*100.0/grouped['total']
        self.gdf = gpd.GeoDataFrame(grouped, geometry='geometry').reset_index()
        self.filter_data() # where all the features selected are false in a pincode
        self.gdf = self.gdf[self.gdf['geometry'].notnull()]  # Remove rows with None geometries
        if top_values > 0:
            self.get_topn_pincodes(top_values)
    
    def get_topn_pincodes(self,n=10):      
        features_to_order_on = self.features
        features_to_order_on.insert(0,'total')
        self.gdf = self.gdf.sort_values(features_to_order_on,ascending=False)
        self.gdf = self.gdf[:n]
        features_to_order_on.remove('total')
   
    def filter_data(self):
        selecteds = []
        for i,feature in enumerate(self.features):
            selected = (self.gdf[feature]>0.0)
            selecteds.append(selected)
        selecteds = np.array(selecteds,dtype=int)
        final = np.zeros(self.gdf.shape[0])
        for indx in range(selecteds.shape[0]):
            final += selecteds[indx]
        final = np.array(final,dtype=bool)
        self.gdf = self.gdf[final]

    def plot_data(self,d_features):
        m = folium.Map(location=[self.gdf.geometry.centroid.y.mean(), self.gdf.geometry.centroid.x.mean()], zoom_start=5)       
        for _, row in self.gdf.iterrows():
            tooltip = f"<text style='color:white;background-color:black;'>Pincode: <b>{row['pincode']}</b></text>"
            for feature in self.features:
                tooltip += f"<br>{d_features[feature]}: {row[feature]:.2f}"
            tooltip += f"<br><text style='color:red;background-color:white;'>Sample: {row['total']}</text>"
            folium.GeoJson(row['geometry'], 
                           tooltip=tooltip).add_to(m)        
        return m

    def save_map(self,output_path,d_features):
        m = self.plot_data(d_features)
        m.save(output_path)