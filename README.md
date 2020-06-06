# StockPrediction
Using HMM (Hidden Markov Models) to predict stocks and ETF values 

# Baseline functions #

def get_AR_predictions(training, n_periods):
    Given, training data, returns n_periods forecast using AR model (StatsModel implementation)

def get_MA_predictions(training, n_periods):
    Given, training data, returns n_periods forecast using MA model (StatsModel implementation)

def get_ARMA_predictions(training, n_periods):
    Given, training data, returns n_periods forecast using ARMA model (StatsModel implementation)

def get_ARIMA_predections(training, n_periods):
    Given, training data, returns n_periods forecast using ARIMA model (StatsModel implementation)

def train_test_split(stock_df, training_days, prediction_days):
    Given dataframe, splits into training and testing data (Original implementation)

def get_complex_future_trend_prediction_baseline(stock_df, predictions):
    Given dataframe of stocks and predictions in the form of labels, returns future trends(Original implementation)

def enumerate_features(data):
    Given data, returns enumerated data (Original implementation)
    
# HMM Functions # taken from git hub
    
def forward(mat_A, mat_B, vec_pi, data, feature_enumerator):
    Forward algorithm implementation 
    
def backward(mat_A, mat_B, data, feature_enumerator, normalization_factor):
    Backword algorithm implementation 
    
def viterbi(mat_A, mat_B, data, vec_pi, feature_enumerator):
    Viterbi algorithm implementation 

def baum_welch (data, n, feature_enumerator, type = "uniform", random = 1, tolerance = 1e-6, max_iter = 10000):
    Baum welch algorithm implementation 

def choose_next(path, data, reverse_enumerator, enumerator, a, b, random = 1):
    Given transitions matrices, chooses next most likely state (Original implementation)
    
def predict(n, k, data, enumerator, reverse_enumerator, prediction = "simple", random = 1, seed = 0): 
    Activates baum_welch algorithm, viterbi and finally predicts path (Original implementation)
    
# Data functions #

def get_ETF_data(data_length):
    Reads ETF data from local folder and returns a dictionary of ETFs (Original implementation)
    
def get_stock_data(relevant_stocks):
    From a list of stocks names, reads stock data from a local folder of stocks (Original implementation)
    
def get_new_fracs_and_bins(stock_df, num_bins):
    Adds bins to ETF and stock dataframes into matching columns in dataframe (Original implementation)
    
def get_symbols(bins_data):
    returns the concatenated symbol of given binned data (Original implementation)
    
def symbolise_df(stock_df):
    adds a column of "symbol" to the given dataframe (Original implementation)
    
def get_training_symbols(stock_df, num_learning, num_symbols):
    return list of symbols to train our model (Original implementation)
    
# Reverse plotting functions #

def get_tricky_fracs(stock_df, labels, frac_type):
    returns relevant fractions from range of numbers representing different labels (Original implementation)
    
def get_gaussian_symbol_dic(stock_df, labels, frac_type):
    build a dictionary from given fraction and labels (Original implementation)
    
def get_series_from_gaussian_symbols(stock_df, predicted):
    given a stock dataframe and a series of predicted labels, the function translates the labels into actual numeric values (Original implementation)
    
def translate(stock_df, predictions, num_tryes):
    Repeats the translation process num_tryes in order to get more accurate results (Original implementation)

def get_MAPE_score(true_vals, predicted_vals):
    calculates the MAPE score between two seriess of values (graphs) (Original implementation)
    
def plot_graphs_and_predictions(k,training_days,predictions_dict, etf_dict):
    plots the original graph with the predicted values (Original implementation)
    
def plot_graphs_and_true_vals(etf, etf_dict, training_days, prediction_days):
    plots the true values of given graph (Original implementation)
    
# Basic Comparison technique functions #

def get_frequant_changes(stock_df):
    the function learns the frequent changes of a given stock in periodic order over time, return a dictionary of time periods and matching series of changes (Original implementation)

def get_complex_future_trend_prediction(stock_df, predictions):
    given predictions in the form of labels, translates the predictions onto numeric values and determines the matching label for each time period (Original implementation)
    
def get_true_trends(etf, etf_dict, training_days, prediction_days):
    given stock data, returns the actual trends of the stock for each time period (Original implementation)
    
def get_f1_acc_scores(suggested_preds, stock_dic):
    given suggested future trends predictions calculates the F1 and accuracy of the predictions (Original implementation)
    
    
