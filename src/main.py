import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

def normalize_and_split_data(X, y, seed=None):
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, 
                                          random_state=seed)
                
    sc_inputs = StandardScaler()
    sc_outputs = StandardScaler()

    sc_inputs.fit(X_tr)
    sc_outputs.fit(y_tr[:,np.newaxis])

    X_tr_sc = sc_inputs.transform(X_tr)
    X_ts_sc = sc_inputs.transform(X_ts)

    y_tr_sc = sc_outputs.transform(y_tr[:,np.newaxis])
    y_ts_sc = sc_outputs.transform(y_ts[:,np.newaxis])

    return X_tr_sc, X_ts_sc, y_tr_sc, y_ts_sc

def baseline_mse(regr, X_tr, X_ts, y_tr, y_ts):
    regr.fit(X_tr, y_tr)
    baseline_tr_mse = mean_squared_error(y_tr, regr.predict(X_tr)) 
    baseline_ts_mse = mean_squared_error(y_ts, regr.predict(X_ts))
    
    return baseline_tr_mse, baseline_ts_mse


class PreppedData(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def sgd_mses(X_tr, X_ts, y_tr, y_ts):
    learning_rate = 3e-4
    bs = 4 # batch_size
    num_epochs = 50
    train_set = PreppedData(X_tr, y_tr)
    test_set = PreppedData(X_ts, y_ts)

    train_loader = DataLoader(train_set, batch_size = bs)

    def eval_model(model):
        model.eval()
        with torch.no_grad():
            preds = model(test_set.X)
            loss = nn.functional.mse_loss(preds, test_set.y)

        return loss.item()


    def eval_model_tr(model):
        model.eval()
        with torch.no_grad():
            preds = model(train_set.X)
            loss = nn.functional.mse_loss(preds, train_set.y)

        return loss.item()   


    def regularized_loss(dict_params):
        prior_weights = torch.from_numpy(regr.coef_).float()
        diff = dict_params['dense.weight'] - prior_weights
        return torch.sum(diff.pow(2))

    def get_losses(opt, penalty = 0):  
        model = nn.Sequential(nn.Linear(X_tr.shape[1], 1))
        optimizer = opt(model.parameters(), lr=learning_rate)
        all_losses = []
        test_losses = []
        train_losses = []
        test_losses.append(eval_model(model))
        train_losses.append(eval_model_tr(model))
        for e in range(num_epochs):
        
            model.train()
            for ix, (Xb, yb) in enumerate(train_loader):
                preds = model(Xb)
                loss = nn.functional.mse_loss(preds, yb)
                if penalty > 0:
                    params = model.named_parameters()
                    dict_params = dict(params)
                    loss += penalty*regularized_loss(dict_params)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                all_losses.append(loss.item())


            test_losses.append(eval_model(model))
            train_losses.append(eval_model_tr(model))  

        return train_losses, test_losses       
    
    tr_runs, ts_runs = [], []
    for _ in range(10):
        train_losses, test_losses = get_losses(torch.optim.SGD, penalty=0)
        tr_runs.append(train_losses)
        ts_runs.append(test_losses)    
    
    return tr_runs, ts_runs

def main():

    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    ls_tr_mses, ls_ts_mses = [], []
    ridge_tr_mses, ridge_ts_mses = [], []
    sgd_tr_mses, sgd_ts_mses = [], []
    for _ in range(10):
        X_tr, X_ts, y_tr, y_ts = normalize_and_split_data(diabetes_X, diabetes_y)
        
        ls_tr_mse, ls_ts_mse = baseline_mse(linear_model.LinearRegression(), X_tr, X_ts, y_tr, y_ts)
        ridge_tr_mse, ridge_ts_mse = baseline_mse(linear_model.RidgeCV(np.logspace(-4,4)), X_tr, X_ts, y_tr, y_ts)
        sgd_tr_mse, sgd_ts_mse = sgd_mses( X_tr, X_ts, y_tr, y_ts)

        ls_tr_mses.append(ls_tr_mse)
        ls_ts_mses.append(ls_ts_mse)
        ridge_tr_mses.append(ridge_tr_mse)
        ridge_ts_mses.append(ridge_ts_mse) 
        sgd_tr_mses.append(sgd_tr_mse)
        sgd_ts_mses.append(sgd_ts_mse)
       

    return ls_tr_mses, ls_ts_mses, ridge_tr_mses, ridge_ts_mses, sgd_tr_mses, sgd_ts_mses