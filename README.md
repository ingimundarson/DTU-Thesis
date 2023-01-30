
# Self-Organizing Maps and Strategic Fund Selection

A Study on the Synergy of Machine Learning and Graph Theory Algorithms. 

<p align="center" width="100%">
    <a href = "https://www.dtu.dk/"><img width="60%" src="https://progruppen.dk/wp-content/uploads/2018/04/DTU-logo.png"></a>
</p>

## Description

This GitHub repository contains the code for a Master's thesis project conducted at Denmark's Technical University (DTU), on the synergy between Self-Organizing Maps (SOMs) and traditional machine learning methods for strategic fund selection. The project specifically focuses on the combination of SOMs with Agglomerative Hierarchical Clustering (AHC) and Minimum Spanning Tree (MST) algorithms. The optimization model used to evaluate the out-of-sample performance of the generated ETF subsets was constructed by [Petr Vanek](https://github.com/VanekPetr). The model utilizes a stochastic CVaR optimization to maximize the expected returns of the portfolios while maintaining the level of risk under specific targets. The results showed promise for the synergy between SOMs and MST but not for the combination of SOMs and AHC.

## Authors


| **Names**                     | **Email**      |
|-------------------------------|----------------|
| Arnar Tjörvi Charlesson       | s202024@dtu.dk |
| Thorvaldur Ingi Ingimundarson | s202033@dtu.dk |

We wish to extend great gratitude to  **Kourosh Marjani Rasmussen**  for supervising us  for this thesis and providing us with vital information and helpful guidance during the  process.

## Structure of Files and Folders

| **File Name**                               | **Description**                                                                     | **Type** |
|---------------------------------------------|-------------------------------------------------------------------------------------|----------|
| Code                                        | Folder containing sublamentary code files, such as helper functions and plot setup. | Folder   |
| Data                                        | Folder containing data files and saved results.                                     | Folder   |
| Figures\_Plots                              | Saved figures.                                                                      | Folder   |
| Investment\_Funnel\_SOM                     | Modified Investment Funnel for SOM.                                                 | Folder   |
| Benchmarks - Insights.ipynb                 | Insights into benchmark portfolios.                                                 | Notebook |
| Code - Data Insights.ipynb                  | Data insights.                                                                      | Notebook |
| Code - Optimized portfolios - Bootstr.ipynb | Results and insights into opt. Portfolios using Bootstrapping.                      | Notebook |
| Code - Optimized portfolios - Monte.ipynb   | Results and insights into opt. Portfolios using Monte Carlo.                        | Notebook |
| Code - SOM Cluster MST.ipynb                | Asset subset generation and insights.                                               | Notebook |
| ETF data - Clean Raw Data.ipynb             | Data cleaning.                                                                      | Notebook |
| ETF data - Filter Data.ipynb                | Data filtering.                                                                     | Notebook |
| ETF data - Get ETFs Raw.ipynb               | Get data.                                                                           | Notebook |
| ETF data - Get Regions.ipynb                | Get data.                                                                           | Notebook |
| ETF data - Merge Data.ipynb                 | Mergin data.                                                                        | Notebook |
| TradeBot - Bootstr (Large).ipynb            | Optimized portfolios using the modified Investment Funnel (Bootstrapping).          | Notebook |
| TradeBot - Bootstr (Small).ipynb            | Optimized portfolios using the modified Investment Funnel (Bootstrapping).          | Notebook |
| TradeBot - Monte Carlo (Large).ipynb        | Optimized portfolios using the modified Investment Funnel (Monte Carlo).            | Notebook |
| TradeBot - Monte Carlo (Small).ipynb        | Optimized portfolios using the modified Investment Funnel (Monte Carlo).            | Notebook |



