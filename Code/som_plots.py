import numpy as np
import pandas as pd

import pandas as pd
import numpy as np
import networkx as nx


import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from Code.functions import get_pdi, sharpe_ratio

from matplotlib.colors import LinearSegmentedColormap
from Code.plot_setup import dtu_colors_rgb



colors = {
    'DTU Red': '#990000',
    'DTU Blue': '#2f3eea',
    'DTU Green': '#1fd082',
    'DTU dBlue': '#030f4f',
    'DTU Yellow': '#f6d04d',
    'DTU Orange': '#fc7634',
    'DTU Pink': '#f7bbb1',
    'DTU Gray': '#dadada',
    'DTU dPink': '#e83f48',
    'DTU dGreen': '#008835',
    'DTU Purple': '#79238e'
}

def expandgrid(params):
    param_value_lists = list(params.values())
    n_combinations = np.prod([len(x) for x in param_value_lists])

    param_combinations = np.array(np.meshgrid(*param_value_lists)).reshape(len(param_value_lists),n_combinations)
   
    param_grid = {}
    for param, values in zip(params.keys(), param_combinations):
        param_grid[param] = values.tolist()

    param_grid
    
    return param_grid


def som_params_tune(train_data, params_grid):
    from tqdm import tqdm
    errors = []
    
    N = len(list(params_grid.values())[0])
    for i in tqdm(range(N)):
        params = {}
        for key, item in params_grid.items():
            params[key] = item[i]

        # initialize som
        som = SOM(
            x = int(params["x"]),
            y = int(params["y"]),
            input_len = int(params["input_len"]),
            sigma = params["sigma"],
            learning_rate = params["learning_rate"],
            random_seed = 123

        )

        # initialize weights (randomly)
        som.random_weights_init(train_data)

        som.train(train_data, int(params["iterations"]), random_order=True, verbose=False)
        error = som.quantization_error(train_data)
        errors.append(error)
        
    idx_min = errors.index(min(errors))
    
    best_params = {}
    for key, item in params_grid.items():
        best_params[key] = item[idx_min]
        
    return best_params, errors

def som_cluster(data, n_clusters):
    """
    FUNCTION TO CLUSTER DATA
    """
    from sklearn.metrics.pairwise import euclidean_distances
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import fcluster, complete
    
    dist = np.round(euclidean_distances(data), 6)
    # Person corr distance matrix
    con_dist = squareform(dist)   # the distance matrix to be able to fit the hierarchical clustering
    complete_dist = complete(con_dist)     # apply hierarchical clustering using the single distance measure

    
    cluster_labels = fcluster(complete_dist, n_clusters, criterion="maxclust")

    return cluster_labels


def som_mst(dataset):
    from sklearn.metrics.pairwise import euclidean_distances
    
    nodes = dataset.columns.to_list()
    dist = pd.DataFrame(np.round(euclidean_distances(dataset.values.T), 6), index = nodes, columns = nodes)

    # corr = dataset.corr(method="spearman")              # calculate the correlation
    # distance_corr = (2 * (1 - corr)) ** 0.5             # calculate the distance

    mask = np.triu(np.ones_like(dist, dtype=bool))   # get only the upper half of the matrix
    distance_corr = dist * mask 


    # use the correlation matrix to create links
    links = dist.stack().reset_index(level=1)
    links.columns = ["var2", "value"]
    links = links.reset_index()
    links = links.replace(0, np.nan)                                # drop 0 values from the matrix
    links = links.dropna(how='any', axis=0)
    links.columns = ["var1", "var2", "value"]                       # rename the columns
    links_filtered = links.loc[(links["var1"] != links["var2"])]     # filter out self-correlations

    # Create the graph
    created_graph = nx.Graph()
    for i in range(len(dist)):                                         # add nodes
        created_graph.add_node(dist.index[i])
    tuples = list(links_filtered.itertuples(index=False, name=None))    # add edges with weight
    created_graph.add_weighted_edges_from(tuples)

    # Create a MST from the full graph
    mst = nx.minimum_spanning_tree(created_graph)

    # Save the nodes with degree one
    degrees = [val for (node, val) in mst.degree()]
    df = pd.DataFrame(degrees, dist.index)
    df.columns = ["degree"]
    subset = df[df["degree"] == 1].index.tolist()

    # Create a new dataframe with only the assets from the subset
    subset_df = dataset.loc[:, dataset.columns.isin(subset)]
    
    return subset, subset_df

def get_som_clusters(som, data, clusters):

    som_shape = som.activation_response(data).shape
    node_etf_idxs = {c:{"idx":[]} for c in np.unique(clusters)}

    for node_coordinate, item in som.win_map(data, return_indices=True).items():

        node = np.ravel_multi_index(np.array(node_coordinate), som_shape)
        node_cluster = clusters[node]

        node_etf_idxs.get(node_cluster).get("idx").extend(item)

    cluster_labels  = [None for _ in range(data.shape[0])]
    for c, c_item in node_etf_idxs.items():
        for idx in c_item["idx"]:
            cluster_labels[idx] = c
            
    return np.array(cluster_labels)

def plot_nodemap(som, data, returns, st_returns, symbols, title = "", ax = None, assetclasses = []):
    
    fig, ax = plt.subplots(figsize = (10,5))

    nord_data = st_returns[symbols["nord"]].values.T
    returns = returns[symbols["universe"]]
    
    # coordinates to a monodimensional index
    som_shape = som.activation_response(data).shape

    w_x, w_y = zip(*[som.winner(d) for d in data])
    w_x = np.array(w_x)
    w_y = np.array(w_y)

    # nord coordinates 
    n_x, n_y = zip(*[som.winner(d) for d in nord_data])
    n_x = np.array(n_x)
    n_y = np.array(n_y)


        
    plt.pcolor(som.distance_map().T, cmap="bone", alpha=.15)
    cbar = plt.colorbar(
        label = "Node Distance",
        fraction = 0.15,
        shrink = 0.5,
        anchor = (0.0, 0.0),
        aspect = 10.0,
    )

    cbar.ax.get_yaxis().labelpad = 15

    for ac in np.unique(assetclasses):
        idxs = [ac == value for value in assetclasses]
        # plot everything except most dist
        plt.scatter(w_x[idxs]+.5+(np.random.rand(np.sum(idxs))-.5)*.8,
                    w_y[idxs]+.5+(np.random.rand(np.sum(idxs))-.5)*.8, 
                    s=50, alpha = 0.6, label = ac)

    plt.grid()

    # format x ticks
    plt.xticks(np.arange(som_shape[0]) + 1)
    xtick_loc = ax.get_xticks()
    xtick_labels = [f"{tick}"  for tick in xtick_loc]

    # Hide major tick labels
    ax.xaxis.set_major_formatter(mticker.NullFormatter())

    # Customize minor tick labels
    ax.xaxis.set_minor_locator(mticker.FixedLocator(xtick_loc - 0.5))
    ax.xaxis.set_minor_formatter(mticker.FixedFormatter(xtick_labels))

    # format y ticks
    plt.yticks(np.arange(som_shape[1]) + 1)
    ytick_loc = ax.get_yticks()
    ytick_labels = [f"{tick}"  for tick in ytick_loc]

    # Hide major tick labels
    ax.yaxis.set_major_formatter(mticker.NullFormatter())

    # Customize minor tick labels
    ax.yaxis.set_minor_locator(mticker.FixedLocator(ytick_loc - 0.5))
    ax.yaxis.set_minor_formatter(mticker.FixedFormatter(ytick_labels))

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(title)
    plt.xlabel("Nodes (x coordinate)")
    plt.ylabel("Nodes (y coordinate)")


def plot_cluster_nodemap(som, data, returns, st_returns, symbols, clusters, title = "", ax = None, assetclasses = []):
    
    fig, ax = plt.subplots(figsize = (10,5))

    nord_data = st_returns[symbols["nord"]].values.T
    returns = returns[symbols["universe"]]
    
    # coordinates to a monodimensional index
    som_shape = som.activation_response(data).shape

    w_x, w_y = zip(*[som.winner(d) for d in data])
    w_x = np.array(w_x)
    w_y = np.array(w_y)

    # nord coordinates 
    n_x, n_y = zip(*[som.winner(d) for d in nord_data])
    n_x = np.array(n_x)
    n_y = np.array(n_y)


    if len(assetclasses) == 0:
        
        n_clusters = len(np.unique(clusters))
        clusters_mat = np.reshape(clusters, newshape = som_shape)

        dtu_cmap = LinearSegmentedColormap.from_list("node_clusters", dtu_colors_rgb[:n_clusters], N = n_clusters)
        plt.pcolor(clusters_mat.T, cmap = dtu_cmap, alpha = .3)
        cbar = plt.colorbar(
            label = "Node Clusters",
            fraction = 0.15,
            shrink = 0.8,
            anchor = (0.0, 0.0),
            aspect = 16.0,
        )

        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.get_yaxis().set_ticks([1.4, 2.2, 3, 3.8, 4.6])
        cbar.ax.set_yticklabels([1,2,3,4,5])


        # plot everything except nord
        plt.scatter(w_x+.5+(np.random.rand(len(w_x))-.5)*.8,
                    w_y+.5+(np.random.rand(len(w_y))-.5)*.8, 
                    s=50, alpha = 0.5, label = "ETF Universe")

 
        # plot nord
        plt.scatter(n_x+.5+(np.random.rand(len(n_x))-.5)*.8,
                    n_y+.5+(np.random.rand(len(n_y))-.5)*.8, 
                    s=50, alpha = 1.0, label = "NORD Assets")
        
    else:
        
        n_clusters = len(np.unique(clusters))
        clusters_mat = np.reshape(clusters, newshape = som_shape)

        dtu_cmap = LinearSegmentedColormap.from_list("node_clusters", dtu_colors_rgb[:n_clusters], N = n_clusters)
        plt.pcolor(clusters_mat.T, cmap = dtu_cmap, alpha = .3)
        cbar = plt.colorbar(
            label = "Node Clusters",
            fraction = 0.15,
            shrink = 0.5,
            anchor = (0.0, 0.0),
            aspect = 9.0,
        )

        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.get_yaxis().set_ticks([1.4, 2.2, 3, 3.8, 4.6])
        cbar.ax.set_yticklabels([1,2,3,4,5])
        
        for ac in np.unique(assetclasses):
            idxs = [ac == value for value in assetclasses]
            # plot everything except most dist
            plt.scatter(w_x[idxs]+.5+(np.random.rand(np.sum(idxs))-.5)*.8,
                        w_y[idxs]+.5+(np.random.rand(np.sum(idxs))-.5)*.8, 
                        s=50, alpha = 0.6, label = ac)

    plt.grid()

    # format x ticks
    plt.xticks(np.arange(som_shape[0]) + 1)
    xtick_loc = ax.get_xticks()
    xtick_labels = [f"{tick}"  for tick in xtick_loc]

    # Hide major tick labels
    ax.xaxis.set_major_formatter(mticker.NullFormatter())

    # Customize minor tick labels
    ax.xaxis.set_minor_locator(mticker.FixedLocator(xtick_loc - 0.5))
    ax.xaxis.set_minor_formatter(mticker.FixedFormatter(xtick_labels))

    # format y ticks
    plt.yticks(np.arange(som_shape[1]) + 1)
    ytick_loc = ax.get_yticks()
    ytick_labels = [f"{tick}"  for tick in ytick_loc]

    # Hide major tick labels
    ax.yaxis.set_major_formatter(mticker.NullFormatter())

    # Customize minor tick labels
    ax.yaxis.set_minor_locator(mticker.FixedLocator(ytick_loc - 0.5))
    ax.yaxis.set_minor_formatter(mticker.FixedFormatter(ytick_labels))

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(title)
    plt.xlabel("Nodes (x coordinate)")
    plt.ylabel("Nodes (y coordinate)")



def plot_mst_nodemap(som, data, returns, st_returns, symbols, node_subset, title = "", ax = None, assetclasses = []):
    
    fig, ax = plt.subplots(figsize = (10,5))

    nord_data = st_returns[symbols["nord"]].values.T
    returns = returns[symbols["universe"]]
    
    # coordinates to a monodimensional index
    som_shape = som.activation_response(data).shape

    w_x, w_y = zip(*[som.winner(d) for d in data])
    w_x = np.array(w_x)
    w_y = np.array(w_y)

    # nord coordinates 
    n_x, n_y = zip(*[som.winner(d) for d in nord_data])
    n_x = np.array(n_x)
    n_y = np.array(n_y)

    selected_node_idx = [1 if n in node_subset else 0 for n in range(som_shape[0]*som_shape[1])]

    if len(assetclasses) == 0:
        node_mat = np.reshape(selected_node_idx, newshape = som_shape)
        node_mat = np.ma.masked_array(node_mat, node_mat == 0)
        dtu_cmap = mpl.colors.ListedColormap([colors["DTU Red"]])
        plt.pcolor(node_mat.T, cmap = dtu_cmap, alpha = .1)
        cbar = plt.colorbar(
            label = "Node Subset",
            fraction = 0.15,
            shrink = 0.5,
            anchor = (0.0, 0.0),
            aspect = 9.0,
        )

        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.get_yaxis().set_ticks([])


        # plot everything except nord
        plt.scatter(w_x+.5+(np.random.rand(len(w_x))-.5)*.8,
                    w_y+.5+(np.random.rand(len(w_y))-.5)*.8, 
                    s=50, alpha = 0.5, label = "ETF Universe")


        # plot nord
        plt.scatter(n_x+.5+(np.random.rand(len(n_x))-.5)*.8,
                    n_y+.5+(np.random.rand(len(n_y))-.5)*.8, 
                    s=50, alpha = 1.0, label = "Nord Portfolio")
        
    else:
        
        node_mat = np.reshape(selected_node_idx, newshape = som_shape)
        node_mat = np.ma.masked_array(node_mat, node_mat == 0)
        dtu_cmap = mpl.colors.ListedColormap([colors["DTU Red"]])
        plt.pcolor(node_mat.T, cmap = dtu_cmap, alpha = .1)
        cbar = plt.colorbar(
            label = "Node Subset",
            fraction = 0.15,
            shrink = 0.5,
            anchor = (0.0, 0.0),
            aspect = 9.0,
        )
        
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.get_yaxis().set_ticks([])
        
        # plot etf scatter
        for ac in np.unique(assetclasses):
            idxs = [ac == value for value in assetclasses]
            # plot everything except most dist
            plt.scatter(w_x[idxs]+.5+(np.random.rand(np.sum(idxs))-.5)*.8,
                        w_y[idxs]+.5+(np.random.rand(np.sum(idxs))-.5)*.8, 
                        s=50, alpha = 0.6, label = ac)

    plt.grid()

    # format x ticks
    plt.xticks(np.arange(som_shape[0]) + 1)
    xtick_loc = ax.get_xticks()
    xtick_labels = [f"{tick}"  for tick in xtick_loc]

    # Hide major tick labels
    ax.xaxis.set_major_formatter(mticker.NullFormatter())

    # Customize minor tick labels
    ax.xaxis.set_minor_locator(mticker.FixedLocator(xtick_loc - 0.5))
    ax.xaxis.set_minor_formatter(mticker.FixedFormatter(xtick_labels))

    # format y ticks
    plt.yticks(np.arange(som_shape[1]) + 1)
    ytick_loc = ax.get_yticks()
    ytick_labels = [f"{tick}"  for tick in ytick_loc]

    # Hide major tick labels
    ax.yaxis.set_major_formatter(mticker.NullFormatter())

    # Customize minor tick labels
    ax.yaxis.set_minor_locator(mticker.FixedLocator(ytick_loc - 0.5))
    ax.yaxis.set_minor_formatter(mticker.FixedFormatter(ytick_labels))

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(title)
    plt.xlabel("Nodes (x coordinate)")
    plt.ylabel("Nodes (y coordinate)")



def error_plot(x_error, q_error, t_error, n_epocs, ax = None):
    plt.sca(ax)
    plt.plot(x_error, q_error, label='Quantization Error')
    plt.plot(x_error, t_error, label='Topographic Error')
    plt.title("Error Plot")
    plt.xlim([0,n_epocs])
    plt.ylabel('Error')
    plt.xlabel('Iteration')
    plt.legend()


def plot_subset_heatmap(returns, subset, sub_type = "", cbar = True, ax = None, cbar_kws = {}):

    cmap = sns.diverging_palette(250, 10, s=40, l=50, sep=5, n=256, center='light', as_cmap=True)
    
    subset_returns = returns[subset]
    # subset_pdi = get_pdi(subset_returns)

    if ax != None:
        plt.sca(ax)

    cbar_kws.update({"label":"Correlation"})
    sns.heatmap(
        subset_returns.corr("spearman"), cmap = cmap, vmin = -1, vmax = 1, 
        cbar = cbar, cbar_kws = cbar_kws,# linewidths=0.01
    )
    plt.xticks([])
    plt.yticks([])
    plt.ylabel("Subset")
    plt.xlabel("Subset")
    #plt.title(f"Subset ({sub_type}) - n = {len(subset)}")
    # plt.title(f"PDI = {subset_pdi:0.2f}, n = {len(subset)}")


def plot_subset_heatmaps(returns, subset, sub_type = "", ax = None):
    from Code.som_plots import plot_subset_heatmap

    fig = plt.figure(figsize=(13, 4))
    spec = fig.add_gridspec(
        1,2, 
        # right = 1,
        wspace = 0.15, 
        hspace=0.25, 
        #width_ratios=[1.7,2]
    )

    ax = fig.add_subplot(spec[0])
    plot_subset_heatmap(returns, subset["Small"], sub_type = sub_type, cbar_kws = dict(ticks=[-1., -0.5, 0, 0.5 , 1]))

    ax = fig.add_subplot(spec[1])
    plot_subset_heatmap(returns, subset["Large"], sub_type = sub_type, cbar_kws = dict(ticks=[-1., -0.5, 0, 0.5 , 1]))

def plot_port_heatmap(som, data, returns, clusters, title = "", ax = None):

    # cmap = sns.diverging_palette(359, 359, s=0, l=45, sep=50, n=256, center='light', as_cmap=True)
    # cmap = sns.diverging_palette(10, 10, s=40, l=50, sep=50, n=256, center='light', as_cmap=True)
    cmap = sns.diverging_palette(250, 10, s=40, l=50, sep=50, n=256, center='light', as_cmap=True)

    port = get_som_portfolio(som, data, returns, clusters)
    port_returns = returns[port]
    port_pdi = get_pdi(port_returns)

    plt.sca(ax)
    sns.heatmap(port_returns.corr("spearman"), cmap = cmap, vmin = -1, vmax = 1, cbar_kws = {"label":"Correlation"})
    plt.title(f"Portfolio: PDI = {port_pdi:0.2f}") # Type = {list(port.keys())[0]},

def plot_data_scatter(som, data, returns, symbols, clusters):
    
    nord_returns = returns[symbols["nord"]]
    returns = returns[symbols["universe"]]


    port = get_som_portfolio(som, data, returns, clusters)
    idx_port = returns.columns.isin(port)

    plt.scatter(
        x = returns.iloc[:, ~idx_port].std(),
        y = returns.iloc[:, ~idx_port].mean(), 
        label= "ETF Universe", alpha=.2
    )

    plt.scatter(
        x = returns.iloc[:, idx_port].std(),
        y = returns.iloc[:, idx_port].mean(), 
        label= "SOM Portfolio", alpha=.8
    )

    plt.scatter(
        x = nord_returns.std(),
        y = nord_returns.mean(), 
        label= "Nord Portfolio", alpha=.8
    )

    plt.xlabel('Std')
    plt.ylabel('Mean Return')
    plt.title(f"ETF Universe")
    plt.legend()
    
def som_trainplot(som, data, returns, st_returns, n_clusters, clusters, x_error, q_error, t_error, n_epocs, symbols, assetclasses = None):

    # som_shape = som.activation_response(data).shape
    # node_weights = np.reshape(som.get_weights(), newshape = (som_shape[0]*som_shape[1], data.shape[1]))

    # # Apply the clustering algorithm to the node positions to extract the clusters
    # clusters = np.array(cluster(pd.DataFrame(node_weights).T, n_clusters=n_clusters, dendrogram=False)["Complete_Corr"])

    fig = plt.figure(figsize=(18, 5))
    spec = fig.add_gridspec(
        1,2, 
        # right = 1,
        wspace = 0.15, 
        hspace=0.25
    )

    ax = fig.add_subplot(spec[0,0])
    plot_nodemap(som,data,n_clusters, returns, st_returns, clusters, symbols, title = "", ax = ax, assetclasses = assetclasses)

    # ax = fig.add_subplot(spec[0,1])
    # plot_port_heatmap(som, data, returns[symbols["universe"]], clusters, title = "", ax = ax)

    ax = fig.add_subplot(spec[0,1])
    error_plot(x_error, q_error, t_error, n_epocs, ax = ax)

    # ax = fig.add_subplot(spec[1,1])
    # plot_data_scatter(som, data, returns, symbols, clusters)

    plt.show()