import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns
# import io
# import base64

class MarketStrengthAnalyzer:
    def __init__(self, n_components=2, **kwargs):
        """
        Initialize the MarketStrengthAnalyzer class.

        Parameters:
        - n_components: int, number of principal components for PCA (default=2).
        - kwargs: additional keyword arguments for features or feature mapping.
        """
        _d = {}
        _d.update(kwargs)
        # print(f'kwargs: {_d}')
        if len(_d['features']) > 0:
            self.features = list(_d['features'])
        else:
            self.features = list(_d['fmap'].values())
        # self.data_fname=_d['data_file']
        # self.data_sheet=_d['data_sheet']
        self.df=_d['df']
        self.n_components = n_components
        self.market_aggregated = None
        self.data_standardized = None
        self.principal_components = None
        self.loadings = None
        self.explained_variance = None
        self.initialize()

    def initialize(self,):
        """Preprocess the data and apply PCA upon instantiation."""
        self.preprocess_data()
        self.apply_pca()

    def preprocess_data(self):
        """Preprocess the data: aggregate by pincode and standardize."""
        # self.df = pd.read_excel("market_map_data.xlsx")
        # self.df = pd.read_excel(self.data_fname,sheet_name=self.data_sheet)
        self.df.fillna(0, inplace=True)
        feature_list = ['pincode'] + self.features
        print(f"FL: {feature_list}")
        self.market_aggregated = self.df[feature_list].groupby('pincode').mean().reset_index()
        scaler = StandardScaler()
        self.data_standardized = scaler.fit_transform(self.market_aggregated.iloc[:, 1:])

    def apply_pca(self):
        """Apply PCA to the standardized data."""
        pca = PCA(n_components=self.n_components)
        self.principal_components = pca.fit_transform(self.data_standardized)
        self.loadings = pca.components_
        self.explained_variance = pca.explained_variance_ratio_

    def get_cluster_data(self):
        """
        Prepares data for Chart.js scatter plot.

        Returns:
            List[dict]: List of dicts with x, y, and label (pincode).
        """
        data = []
        for i, pincode in enumerate(self.market_aggregated['pincode']):
            point = {
                "x": float(self.principal_components[i, 0]),
                "y": float(self.principal_components[i, 1]),
                "label": str(pincode)
            }
            data.append(point)
        return data,self.explained_variance
    
    def get_biplot_data(self):
        """
        Prepare PCA coordinates, loadings, and labels for Chart.js-based biplot.
        """
        # Normalize the PCA points (as in original)
        xs = self.principal_components[:, 0]
        ys = self.principal_components[:, 1]
        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())

        points = []
        for i, pincode in enumerate(self.market_aggregated['pincode']):
            points.append({
                "x": float(xs[i] * scalex),
                "y": float(ys[i] * scaley),
                "label": str(pincode)
            })

        arrows = []
        for i in range(self.loadings.shape[1]):
            arrows.append({
                "x": float(self.loadings[0, i]),
                "y": float(self.loadings[1, i]),
                "label": self.features[i]
            })
        return {
            "points": points,
            "arrows": arrows
        }
    
    def calculate_market_strength(self):
        """
        Calculate and return the market strength summary.

        Returns:
        - pd.DataFrame: DataFrame with 'Market' and 'Strength' columns.
        """
        market_strength = np.sqrt(np.sum(self.principal_components**2, axis=1))
        market_strength_summary = pd.DataFrame({
            'Market': self.market_aggregated['pincode'],
            'Strength': market_strength
        }).sort_values(by='Strength', ascending=False)
        return market_strength_summary

    def get_market_strength_data(self, markets=None):
        """
        Prepare market strength data for Chart.js.

        Parameters:
        - markets (list): Optional list of market names to include.

        Returns:
        - dict: Dictionary with 'labels' and 'values' for the chart.
        """
        df = self.calculate_market_strength()
        if markets:
            df = df[df['Market'].isin(markets)]

        return {
            "labels": df["Market"].tolist(),
            "values": df["Strength"].tolist()
        }


    

if __name__=="__main__":
    d={'features': ['Mutual_Funds'], 
       'fmap': {'Mutual_Funds': 'Mutual Funds'}, 
       'data_file': 'C:\\Users\\RajmaneD\\OneDrive - Kantar\\DR\\micro_markets_webapp\\KANTAR_ICUBE_2022_TO_2024_coded_Data_for_Mumbai.xlsx', 
       'data_sheet': 'Sheet1'}
    msa=MarketStrengthAnalyzer(**d)
    # mkt_clusters_img64=msa.plot_market_clusters 