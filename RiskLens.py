import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

class MonteCarloPortfolio:
    def __init__(self, stock_list, n_simulations=10000, risk_free_rate=0.04):
        self.stock_list = stock_list
        self.n_simulations = n_simulations
        self.risk_free_rate = risk_free_rate
        self.mean_returns, self.cov_matrix, self.valid_stocks = self.load_stock_data()
        self.results = np.zeros((4, n_simulations))  
        self.weights_record = []

    def load_stock_data(self):
        valid_stocks = []
        all_data = {}
        
        for stock in self.stock_list:
            try:
                data = yf.download(stock, start_date, end_date)
                if data.shape[0] != 0:
                    if 'Adj Close' in data:
                        all_data[stock] = data['Adj Close']
                    else:
                        all_data[stock] = data['Close']
                    valid_stocks.append(stock)
                else:
                    st.write(f"Data for {stock} is unavailable.")
            except Exception as e:
                st.write(f"Error fetching data for {stock}: {e}")
        
        if not valid_stocks:
            st.error("No valid stock data available.")
            st.stop()
        
        data = pd.concat(all_data.values(), axis=1, keys=all_data.keys())
        data.index = data.index.tz_localize(None)
        log_returns = np.log(data / data.shift(1)).dropna()
        
        return log_returns.mean().values, log_returns.cov().values, valid_stocks

    def run_simulation(self):
        for i in range(self.n_simulations):
            weights = np.random.dirichlet(np.ones(len(self.valid_stocks)), size=1)[0]
            self.weights_record.append(weights)
            
            portfolio_return = np.sum(weights * self.mean_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            self.results[0, i] = portfolio_return
            self.results[1, i] = portfolio_volatility
            self.results[2, i] = sharpe_ratio
        
    def get_optimal_portfolio(self):
        max_sharpe_idx = np.argmax(self.results[2])
        return self.weights_record[max_sharpe_idx], self.results[:, max_sharpe_idx]

    def plot_simulation(self):
        fig, ax = plt.subplots()
        sc = ax.scatter(self.results[1, :], self.results[0, :], c=self.results[2, :], cmap='viridis', marker='o')
        ax.set_xlabel('Volatility (Risk)')
        ax.set_ylabel('Expected Return')
        plt.colorbar(sc, label='Sharpe Ratio')
        ax.set_title('Monte Carlo Portfolio Simulation')
        st.pyplot(fig)

st.title("Monte Carlo Portfolio Optimization")

stock_list = st.sidebar.text_input("Enter stock symbols separated by commas", "META, AMZN, GOOGL, PLTR, NVDA").split(",")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))
n_simulations = st.sidebar.slider("Number of Simulations", min_value=1000, max_value=50000, value=10000, step=1000)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=4.0) / 100

if st.sidebar.button("Run Simulation"):
    try:
        sim = MonteCarloPortfolio(stock_list, n_simulations, risk_free_rate)
        sim.run_simulation()
        optimal_weights, optimal_metrics = sim.get_optimal_portfolio()
        
        st.subheader("Optimal Portfolio Allocation")
        allocation_df = pd.DataFrame({'Stock': sim.valid_stocks, 'Weight': optimal_weights})
        st.dataframe(allocation_df)
        
        st.write(f"**Expected Return:** {optimal_metrics[0]:.2%}")
        st.write(f"**Volatility:** {optimal_metrics[1]:.2%}")
        st.write(f"**Sharpe Ratio:** {optimal_metrics[2]:.2f}")
        
        sim.plot_simulation()
    except ValueError as e:
        st.error(e)
