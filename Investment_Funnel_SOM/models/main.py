import numpy as np
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from models.dataAnalyser import mean_an_returns, final_stats
from models.MST import minimum_spanning_tree
from models.Clustering import cluster, pick_cluster
from models.ScenarioGeneration import monte_carlo, bootstrapping
from models.CVaRtargets import get_cvar_targets
from models.CVaRmodel import cvar_model
from data.ETFlist import ETFlist
from pandas_datareader import data

pio.renderers.default = "browser"

# # Get data
# data = pd.read_parquet('data/all_etfs_rets.parquet.gzip')
# tickers = data.columns.values
# data_name = pd.read_parquet('data/all_etfs_rets_name.parquet.gzip')
# names = data_name.columns.values

# initialise data
etf_data = pd.read_csv("../Data/Data_ETFs_Info.csv")
etf_returns = pd.read_parquet("../Data/Data_ETFs_Returns.gzip")

with open("../Data/Data_ETFs_Names.json", "r") as openfile:
    # Reading from json file
    etfs_dict = json.load(openfile)
    

tickers = etfs_dict.get("symbols").get("universe") + etfs_dict.get("symbols").get("nord")
names = etfs_dict.get("names").get("universe") + etfs_dict.get("names").get("nord")
data = etf_returns[tickers]

class TradeBot(object):
    """
    Python class analysing financial products and based on machine learning algorithms and mathematical
    optimization suggesting optimal portfolio of assets.
    """

    # def __init__(self, start, end, assets):
    #     # DOWNLOAD THE ADJUSTED DAILY PRICES FROM YAHOO DATABASE
    #     dailyPrices = data.DataReader(assets, 'yahoo', start, end)["Adj Close"]
    #     ## Extra
    #     # test = dailyPrices
    #     # for k in range(len(test.columns)):
    #     #     for i in range(len(test.index)):
    #     #         if math.isnan(float(test.iloc[i, k])):
    #     #             test.iloc[i, k] = test.iloc[i-1, k]
    #     # dailyPrices=test
    #     # GET WEEKLY RETURNS
    #     # Get prices only for Wednesdays and delete Nan columns
    #     pricesWed = dailyPrices[dailyPrices.index.weekday == 2].dropna(axis=1)
    #     # Get weekly returns
    #     self.weeklyReturns = pricesWed.pct_change().drop(pricesWed.index[0])  # drop first NaN row
    #

    def __init__(self):
        self.weeklyReturns = data
        self.tickers = tickers
        self.names = names

    def __get_stat(self, start, end):
        """
        METHOD COMPUTING ANNUAL RETURNS, ANNUAL STD. DEV. & SHARPE RATIO OF ASSETS
        """

        # ANALYZE THE DATA for a given time period
        weekly_data = self.weeklyReturns[(self.weeklyReturns.index >= start) & (self.weeklyReturns.index <= end)].copy()

        # Create table with summary statistics
        mu_ga = mean_an_returns(weekly_data)  # Annualised geometric mean of returns
        std_dev_a = weekly_data.std(axis=0) * np.sqrt(52)  # Annualised standard deviation of returns
        sharpe = round(mu_ga / std_dev_a, 2)  # Sharpe ratio of each financial product

        # Write all results into a data frame
        stat_df = pd.concat([mu_ga, std_dev_a, sharpe], axis=1)
        stat_df.columns = ["Average Annual Returns", "Standard Deviation of Returns", "Sharpe Ratio"]
        stat_df["ISIN"] = stat_df.index  # Add names into the table
        stat_df["Name"] = self.names

        # IS ETF OR NOT? Set size
        for isin in stat_df.index:
            if isin in ETFlist:
                stat_df.loc[isin, "Type"] = "ETF"
                stat_df.loc[isin, "Size"] = 1
            else:
                stat_df.loc[isin, "Type"] = "ETF"
                stat_df.loc[isin, "Size"] = 1

        return stat_df

    @staticmethod
    def __plot_backtest(performance, performance_benchmark, composition, names, tickers):
        """
        METHOD TO PLOT THE BACKTEST RESULTS
        """

        # FOR Yahoo
        # performance.index = performance.index.date

        # FOR Morningstar
        performance.index = pd.to_datetime(performance.index.values, utc=True)

        # ** PERFORMANCE GRAPH **
        performance.index = [date.date() for date in performance.index]
        df_to_plot = pd.concat([performance, performance_benchmark], axis=1)
        color_discrete_map = {'Portfolio_Value': '#21304f', 'Benchmark_Value': '#f58f02'}
        fig = px.line(df_to_plot, x=df_to_plot.index, y=df_to_plot.columns,
                      title='Comparison of different strategies', color_discrete_map=color_discrete_map)
        fig_performance = fig

        # ** COMPOSITION GRAPH **
        # change ISIN to NAMES in allocation df
        composition_names = []
        for ticker in composition.columns:
            ticker_index = list(tickers).index(ticker)
            composition_names.append(list(names)[ticker_index])
        composition.columns = composition_names

        composition = composition.loc[:, (composition != 0).any(axis=0)]
        data = []
        idx_color = 0
        composition_color = (px.colors.sequential.turbid
                             + px.colors.sequential.Brwnyl
                             + px.colors.sequential.YlOrBr
                             + px.colors.sequential.gray)
        for isin in composition.columns:
            trace = go.Bar(
                x=composition.index,
                y=composition[isin],
                name=str(isin),
                marker_color=composition_color[idx_color]  # custom color
            )
            data.append(trace)
            idx_color += 1

        layout = go.Layout(barmode='stack')
        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(
            title="Portfolio Composition",
            xaxis_title="Number of the Investment Period",
            yaxis_title="Composition",
            legend_title="Name of the Fund")
        fig.layout.yaxis.tickformat = ',.1%'
        fig_composition = fig

        # Show figure if needed
        # fig.show()

        return fig_performance, fig_composition

    def plot_dots(self, start, end, ml=None, ml_subset=None, fund_set=[]):
        """
        METHOD TO PLOT THE OVERVIEW OF THE FINANCIAL PRODUCTS IN TERMS OF RISK AND RETURNS
        """

        # Get statistics for a given time period
        data = self.__get_stat(start, end)

        # IF WE WANT TO HIGHLIGHT THE SUBSET OF ASSETS BASED ON ML
        if ml == "MST":
            data.loc[:, "Type"] = "Funds"
            for fund in ml_subset:
                data.loc[fund, "Type"] = "MST subset"
        if ml == "Clustering":
            data.loc[:, "Type"] = ml_subset.loc[:, "Cluster"]

        # If selected any fund for comparison
        for fund in fund_set:
            isin_idx = list(self.names).index(fund)
            data.loc[self.tickers[isin_idx], "Type"] = str(data.loc[self.tickers[isin_idx], "Name"])
            data.loc[self.tickers[isin_idx], "Size"] = 3

        # PLOTTING Data
        color_discrete_map = {'ETF': '#21304f', 'Mutual Fund': '#f58f02',
                              'Funds': '#21304f', "MST subset": '#f58f02',
                              'Cluster 1': '#21304f', 'Cluster 2': '#f58f02'}
        fig = px.scatter(data,
                         x="Standard Deviation of Returns",
                         y="Average Annual Returns",
                         color="Type",
                         size="Size",
                         size_max=8,
                         hover_name="Name",
                         hover_data={"Sharpe Ratio": True, "ISIN": True, "Size": False},
                         color_discrete_map=color_discrete_map,
                         title="Annual Returns and Standard Deviation of Returns from " + start[:10] + " to " + end[:10]
                         )

        # AXIS IN PERCENTAGES
        fig.layout.yaxis.tickformat = ',.1%'
        fig.layout.xaxis.tickformat = ',.1%'

        # RISK LEVEL MARKER
        min_risk = data['Standard Deviation of Returns'].min()
        max_risk = data['Standard Deviation of Returns'].max()
        risk_level = {"Risk Class 1": 0.005,
                      "Risk Class 2": 0.02,
                      "Risk Class 3": 0.05,
                      "Risk Class 4": 0.10,
                      "Risk Class 5": 0.15,
                      "Risk Class 6": 0.25,
                      "Risk Class 7": max_risk}
        # Initialize dynamic risk levels
        actual_risk_level = set()  
        for i in range(1, 8):
            k = "Risk Class " + str(i)
            if (risk_level[k] >= min_risk) and (risk_level[k] <= max_risk):
                actual_risk_level.add(i)
                
        if max(actual_risk_level) < 7:
            actual_risk_level.add(max(actual_risk_level) + 1)  # Add the final risk level  
            
        for level in actual_risk_level:
            k = "Risk Class " + str(level)
            fig.add_vline(x=risk_level[k], line_width=1, line_dash="dash",
                          line_color="#7c90a0")  # annotation_text=k, annotation_position="top left")
            fig.add_annotation(x=risk_level[k] - 0.01, y=max(data["Average Annual Returns"]), text=k, textangle=-90,
                               showarrow=False)

        # RETURN LEVEL MARKER
        fig.add_hline(y=0, line_width=1.5, line_color="rgba(233, 30, 99, 0.5)")

        # TITLES
        fig.update_annotations(font_color="#000000")
        fig.update_layout(
            xaxis_title="Annualised standard deviation of returns (Risk)",
            yaxis_title="Annualised average returns",
        )
        # Position of legend
        fig.update_layout(legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01
        ))

        return fig

    def setup_data(self, start, end, train_test, train_ratio=0.5, end_train=None, start_test=None):
        """
        METHOD TO PREPARE DATA FOR ML AND BACKTESTING
        """
        
        self.start = start
        self.end = end
        self.train_test = train_test
        self.AIdata = self.__get_stat(start, end)

        # Get data for a given time interval
        data = self.weeklyReturns[(self.weeklyReturns.index >= start) & (self.weeklyReturns.index <= end)].copy()

        # IF WE DIVIDE DATASET
        if train_test:
            # # DIVIDE DATA INTO TRAINING AND TESTING PARTS
            # breakPoint = int(np.floor(len(data.index) * train_ratio))
            #
            # # DEFINITION OF TRAINING AND TESTING DATASETS
            # self.trainDataset = data.iloc[0:breakPoint, :]
            # self.testDataset = data.iloc[breakPoint:, :]

            self.trainDataset = data[data.index <= end_train]
            self.testDataset = data[data.index > start_test]

            # Get dates
            self.endTrainDate = str(self.trainDataset.index.date[-1])
            self.startTestDate = str(self.testDataset.index.date[0])

            self.dataPlot = self.__get_stat(start, self.endTrainDate)
            self.lenTest = len(self.testDataset.index)
        else:
            self.trainDataset = data
            self.endTrainDate = str(self.trainDataset.index.date[-1])
            self.dataPlot = self.__get_stat(start, end)
            self.lenTest = 0

    def mst(self, n_mst_runs, plot):
        """
        METHOD TO RUN MST METHOD AND PRINT RESULTS
        """
        
        # Starting subset of data for MST
        self.subsetMST_df = self.trainDataset
        for i in range(n_mst_runs):
            self.subsetMST, self.subsetMST_df, self.corrMST_avg, self.PDI_MST = minimum_spanning_tree(self.subsetMST_df)

        # PLOTTING RESULTS
        if plot:
            fig = self.plot_dots(start=self.start, end=self.endTrainDate, ml="MST", ml_subset=self.subsetMST)
            return fig

    def clustering(self, n_clusters, n_assets, plot):
        """
        METHOD TO RUN MST METHOD AND PRINT RESULTS
        """
        
        # CLUSTER DATA
        clusters = cluster(self.trainDataset, n_clusters, dendrogram=False)

        # SELECT ASSETS
        self.subsetCLUST, self.subsetCLUST_df = pick_cluster(data=self.trainDataset,
                                                             stat=self.dataPlot,
                                                             ml=clusters,
                                                             n_assets=n_assets)  # Number of assets from each cluster

        # PLOTTING DATA
        if plot:
            fig = self.plot_dots(start=self.start, end=self.endTrainDate, ml="Clustering", ml_subset=clusters)
            return fig

    def backtest(self, assets, benchmark, scenarios, n_simulations):
        """
        METHOD TO COMPUTE THE BACKTEST
        """

        # Find Benchmarks' ISIN codes
        benchmark_isin = [self.tickers[list(self.names).index(name)] for name in benchmark]

        # SELECT THE WORKING SUBSET
        if assets == 'MST':
            subset = self.subsetMST
        elif assets == 'Clustering':
            subset = self.subsetCLUST
        else:
            subset = assets

        # SCENARIO GENERATION
        # ---------------------------------------------------------------------------------------------------
        if scenarios == 'MonteCarlo':
            scenarios = monte_carlo(data=self.trainDataset.loc[:, self.trainDataset.columns.isin(subset)],
                                    # subsetMST_df or subsetCLUST_df
                                    n_simulations=n_simulations,
                                    n_test=self.lenTest)
        else:
            scenarios = bootstrapping(data=self.weeklyReturns[subset],  # subsetMST or subsetCLUST
                                      n_simulations=n_simulations,  # number of scenarios per period
                                      n_test=self.lenTest)

        # TARGETS GENERATION
        # ---------------------------------------------------------------------------------------------------
        targets, benchmark_port_val = get_cvar_targets(start_date=self.start,
                                                       end_date=self.end,
                                                       test_date=self.startTestDate,
                                                       benchmark=benchmark_isin,  # MSCI World benchmark
                                                       test_index=self.testDataset.index.date,
                                                       budget=100,
                                                       cvar_alpha=0.05,
                                                       data=self.weeklyReturns)

        # MATHEMATICAL MODELING
        # ------------------------------------------------------------------
        port_allocation, port_value, port_cvar = cvar_model(test_ret=self.testDataset[subset],
                                                            scenarios=scenarios,  # Scenarios
                                                            targets=targets,  # Target
                                                            budget=100,
                                                            cvar_alpha=0.05,
                                                            trans_cost=0.001,
                                                            max_weight=1)
        # PLOTTING
        # ------------------------------------------------------------------
        fig_performance, fig_composition = self.__plot_backtest(performance=port_value.copy(),
                                                                performance_benchmark=benchmark_port_val.copy(),
                                                                composition=port_allocation,
                                                                names=self.names,
                                                                tickers=self.tickers)

        # RETURN STATISTICS
        # ------------------------------------------------------------------
        optimal_portfolio_stat = final_stats(port_value)
        benchmark_stat = final_stats(benchmark_port_val)

        return optimal_portfolio_stat, benchmark_stat, fig_performance, fig_composition


if __name__ == "__main__":
    # INITIALIZATION OF THE CLASS
    # algo = TradeBot(start="2011-07-01", end="2021-07-01", assets=tickers)
    algo = TradeBot()

    # PLOT INTERACTIVE GRAPH
    algo.plot_dots(start="2018-09-24", end="2019-09-01")

    # SETUP WORKING DATASET, DIVIDE DATASET INTO TRAINING AND TESTING PART?
    algo.setup_data(start="2015-12-23", end="2018-08-22", train_test=True, train_ratio=0.6)

    # RUN THE MINIMUM SPANNING TREE METHOD
    algo.mst(n_mst_runs=3, plot=True)

    # RUN THE CLUSTERING METHOD
    algo.clustering(n_clusters=3, n_assets=10, plot=True)

    # RUN THE BACKTEST
    results = algo.backtest(assets='MST', benchmark=['URTH'], scenarios='Bootstrapping', n_simulations=500)
