import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

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

    def plot_market_clusters(self):
        """
        Generate the market cluster plot and return it as a base64-encoded image.

        Returns:
        - str: Base64-encoded PNG image of the plot.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = sns.scatterplot(
            x=self.principal_components[:, 0],
            y=self.principal_components[:, 1],
            hue=self.market_aggregated['pincode'],
            palette='viridis',
            s=100,
            ax=ax
        )
        ax.set_title('Market Clusters', fontsize=14)
        ax.set_xlabel(f'PC1 (Explained Variance: {self.explained_variance[0]:.2f})', fontsize=10)
        ax.set_ylabel(f'PC2 (Explained Variance: {self.explained_variance[1]:.2f})', fontsize=10)
        ax.legend(title='Markets', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6, title_fontsize=9)
        ax.grid()
        for i, pincode in enumerate(self.market_aggregated['pincode']):
            ax.text(
                self.principal_components[i, 0],
                self.principal_components[i, 1],
                f'{pincode}',
                fontsize=7,
                ha='right'
            )
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64

    def plot_biplot(self):
        """
        Generate the market-image biplot and return it as a base64-encoded image.

        Returns:
        - str: Base64-encoded PNG image of the plot.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        xs = self.principal_components[:, 0]
        ys = self.principal_components[:, 1]
        n = self.loadings.shape[1]
        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())
        ax.scatter(xs * scalex, ys * scaley)
        for i in range(n):
            ax.arrow(0, 0, self.loadings[0, i], self.loadings[1, i], color='r', alpha=0.5)
            ax.text(
                self.loadings[0, i] * 1.15,
                self.loadings[1, i] * 1.15,
                f'{self.features[i]}',
                color='g',
                ha='center',
                va='center',
                fontsize=6
            )
        for i, pincode in enumerate(self.market_aggregated['pincode']):
            ax.text(
                xs[i] * scalex * 1.05,
                ys[i] * scaley * 1.05,
                f'{pincode}',
                fontsize=7,
                ha='center',
                va='bottom'
            )
        ax.set_xlabel('PC1', fontsize=10)
        ax.set_ylabel('PC2', fontsize=10)
        ax.set_title('Market Perceptual Map', fontsize=14)
        ax.grid()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64

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

    def plot_market_strength(self, markets=None):
        """
        Generate the market strength bar plot and return it as a base64-encoded image.

        Parameters:
        - markets: list of str, optional list of market names to include (default=None).

        Returns:
        - str: Base64-encoded PNG image of the plot.
        """
        df = self.calculate_market_strength()
        if markets is not None:
            df = df[df['Market'].isin(markets)]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(df['Market'], df['Strength'], color='skyblue', edgecolor='black')
        ax.set_xlabel('Market Name')
        ax.set_ylabel('Strength')
        ax.set_title('Market Strength')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64

if __name__=="__main__":
    d={'features': ['Mutual_Funds'], 
       'fmap': {'Mutual_Funds': 'Mutual Funds'}, 
       'data_file': 'C:\\Users\\RajmaneD\\OneDrive - Kantar\\DR\\micro_markets_webapp\\KANTAR_ICUBE_2022_TO_2024_coded_Data_for_Mumbai.xlsx', 
       'data_sheet': 'Sheet1'}
    msa=MarketStrengthAnalyzer(**d)
    mkt_clusters_img64=msa.plot_market_clusters