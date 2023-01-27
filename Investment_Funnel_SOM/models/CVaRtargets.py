import pulp
import pandas as pd
import numpy as np
from models.ScenarioGeneration import bootstrapping, monte_carlo
from loguru import logger


# FUNCTION RUNNING THE OPTIMIZATION
# ----------------------------------------------------------------------
def portfolio_risk_target(scenarios, cvar_alpha, weights = None):
    
    # Fixed x
    if weights == None:
        weights = [1/len(scenarios.columns) for _ in scenarios.columns]

    x = pd.DataFrame(columns=scenarios.columns, index=["position"])
    x.loc["position", :] = weights
    
    # define index
    i_idx = scenarios.columns
    j_idx = scenarios.index
    
    # number of scenarios
    scenario_n = scenarios.shape[0]    
    
    # loss deviation
    var_dev = pulp.LpVariable.dicts("VarDev", (t for t in j_idx), lowBound=0, cat='Continuous')
        
    # value at risk
    var = pulp.LpVariable("VaR", lowBound=0, cat='Continuous')
    cvar = pulp.LpVariable("CVaR", lowBound=0, cat='Continuous')
      
    # define model
    model = pulp.LpProblem("Targets_Optimization", pulp.LpMinimize)
     
    # Objective Function
    model += cvar
                      
    # *** CONSTRAINS ***               
    # Var deviation constrain
    for t in j_idx:
        model += -pulp.lpSum([scenarios.loc[t, i] * x[i] for i in i_idx]) - var <= var_dev[t]
    
    # CVaR constrain
    model += var + 1/(scenario_n * cvar_alpha) * pulp.lpSum([var_dev[t] for t in j_idx]) == cvar

    # Budget constrain
    model += pulp.lpSum([x[i] for i in i_idx]) == 1

    # solve model
    solver = pulp.PULP_CBC_CMD(msg = 0)
    optimality = model.solve(solver)
    if not bool(optimality):
        print("Solution Not Optimal")
        
    # Get positions    
    if pulp.LpStatus[model.status] == 'Optimal':
     
        # print variables
        var_model = dict()
        for variable in model.variables():
            var_model[variable.name] = variable.varValue
         
        # solution with variable names   
        var_model = pd.Series(var_model, index=var_model.keys())
        
        return var_model["CVaR"]
    else:
        logger.exception(f"LP does not find optimal solution for CVaR targets with: {pulp.LpStatus[model.status]}")


# ----------------------------------------------------------------------
# Mathematical Optimization: TARGETS GENERATION
# ---------------------------------------------------------------------- 
def get_cvar_targets(start_date, end_date, test_date, benchmark, test_index, budget, cvar_alpha, data=[], weights = None, n_simulations = 250, n_rebalancing = 4, scenario_type = "Monte Carlo"):

    # Define Benchmark
    tickers = benchmark

    # *** For YAHOO version ***
    # # User pandas_reader.data.DataReader to load the desired data.
    # panel_data = data.DataReader(tickers, 'yahoo', start_date, end_date)
    # df_close = panel_data["Adj Close"]
    # # Get weekly data
    # target_weekly_ret = get_weekly_returns(data=df_close)

    # df_test_index = pd.DataFrame(index=test_index)
    # test_weekly_ret = pd.concat([df_test_index, test_weekly_ret], axis=1)
    # test_weekly_ret = test_weekly_ret.fillna(0)

    # *** For Morningstar data ***
    target_weekly_ret = data[tickers].copy()

    # Get weekly data for testing
    test_weekly_ret = target_weekly_ret[(target_weekly_ret.index >= test_date) & (target_weekly_ret.index <= end_date)]

    # Number of weeks for testing
    weeks_n = len(test_weekly_ret.index)

    # Get scenarios
    if scenario_type == "Bootstrapping":
        print(f"Scenario generation: {scenario_type}\n")
        target_scenarios = bootstrapping(data=target_weekly_ret,       
                                         n_simulations=n_simulations,
                                         n_test=weeks_n, n_rebalancing = n_rebalancing)
    elif scenario_type == "Monte Carlo":
        print(f"Scenario generation: {scenario_type}\n")
        target_scenarios = monte_carlo(data=target_weekly_ret,      
                                         n_simulations=n_simulations,
                                         n_test=weeks_n, n_rebalancing = n_rebalancing)
    else:
        print(f"\nScenario generation: Failed\n")

    # Compute the optimal portfolio outperforming zero percentage return
    # ----------------------------------------------------------------------
    p_points = len(target_scenarios[:, 0, 0])       # number of periods
    s_points = len(target_scenarios[0, :, 0])       # number of scenarios

    # DATA FRAME TO STORE CVaR TARGETS
    targets = pd.DataFrame(columns=["CVaR_Target"], index=list(range(p_points)))

    # DATA FRAME TO STORE VALUE OF THE PORTFOLIO
    portfolio_value = pd.DataFrame(columns=["Benchmark_Value"], index=test_weekly_ret.index)

    # COMPUTE CVaR TARGETS
    for p in range(p_points):
        # create data frame with scenarios for a given period p
        scenario_df = pd.DataFrame(target_scenarios[p, :, :],
                                   columns=tickers,
                                   index=list(range(s_points)))

        # run CVaR model to compute CVaR targets
        cvar_target = portfolio_risk_target(scenarios=scenario_df,
                                            cvar_alpha=cvar_alpha,
                                            weights = weights)
        # save the result
        targets.loc[p, "CVaR_Target"] = cvar_target

    # COMPUTE PORTFOLIO VALUE
    if weights == None:
        weights = [1/len(tickers) for _ in tickers]

    for w in test_weekly_ret.index:
        # portfolio_value.loc[w, "Benchmark_Value"] = sum((budget/len(tickers)) * (1 + test_weekly_ret.loc[w, :]))
        # budget = sum(budget/len(tickers) * (1+test_weekly_ret.loc[w, :]))

        portfolio_value.loc[w, "Benchmark_Value"] = (budget * (1 + test_weekly_ret.loc[w, :])) @ np.array(weights)
        budget = (budget * (1 + test_weekly_ret.loc[w, :])) @ np.array(weights)

    return targets, portfolio_value
