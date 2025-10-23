import os
import sys
sys.path.append("CMSPLOTS")
import ROOT
from CMSPLOTS.myFunction import DrawHistos
from utils.channel_map import buildFERSBoards
from utils.utils import loadRDF, vectorizeFERS,calculateEnergySumFERS,subtractFERSBeamPedestal,correctFERSSaturation,calibrateFERSChannelsBeam,testFERSBeamCalibration,preProcessDRSBoards
from utils.html_generator import generate_html
from utils.fitter import eventFit
from utils.colors import colors
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from scipy.optimize import lsq_linear, minimize
from selections.selections import vetoMuonCounter, applyUpstreamVeto, PSDSelection
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from scipy.optimize import curve_fit
def LinearFitWithUncertainty(X,y):
    reg = LinearRegression(fit_intercept=False,positive=True).fit(X, y)
    #reg = LinearRegression(fit_intercept=False).fit(X, y)
    #reg = Ridge(fit_intercept=False,alpha=0.01).fit(X, y)
    y_pred = reg.predict(X)
    residuals = y - y_pred
    n, p = X.shape
    dof = n - p
    sigma2 = np.sum(residuals**2) / dof
    # Covariance matrix of coefficients
    cov = sigma2 * np.linalg.inv(X.T @ X)
    # Standard errors
    unc = np.sqrt(np.diag(cov))
    return reg, unc, cov


def LinearFitWithUncertainty2Steps(X,y,step1_firstevent,step1_lastevent,step1_fix_channel_indices):
    reg_step1 = LinearRegression(fit_intercept=False,positive=True).fit(X[step1_firstevent:step1_lastevent], y[step1_firstevent:step1_lastevent])
    step2_channels_indices = [ch for ch in list(range(X.shape[-1])) if ch not in step1_fix_channel_indices]
    reg_step2 = LinearRegression(fit_intercept=False,positive=True).fit(X[step1_lastevent:,step2_channels_indices], y[step1_lastevent:]-X[step1_lastevent:,step1_fix_channel_indices].dot(reg_step1.coef_[step1_fix_channel_indices]))
    #reg = LinearRegression(fit_intercept=False).fit(X, y)
    #reg = Ridge(fit_intercept=False,alpha=0.01).fit(X, y)
    coef = np.zeros(X.shape[-1])
    coef[step1_fix_channel_indices] = reg_step1.coef_[step1_fix_channel_indices]
    coef[step2_channels_indices] = reg_step2.coef_
    class Result:
        coef_ = coef
        intercept_ = 0
        def predict(self,X_in):
            return X_in.dot(self.coef_)
    y_pred = X.dot(coef)
    residuals = y - y_pred
    n, p = X.shape
    dof = n - p
    sigma2 = np.sum(residuals**2) / dof
    # Covariance matrix of coefficients
    cov = sigma2 * np.linalg.inv(X.T @ X)
    # Standard errors
    unc = np.sqrt(np.diag(cov))
    return Result(), unc, cov


def LinearFitWithUncertainty3Steps(X,y,step1_firstevent,step1_lastevent,step1_fix_channel_indices,step2_firstevent,step2_lastevent,step2_fix_channel_indices):
    coef = np.zeros(X.shape[-1])
    reg_step1 = LinearRegression(fit_intercept=False,positive=True).fit(X[step1_firstevent:step1_lastevent], y[step1_firstevent:step1_lastevent])
    coef[step1_fix_channel_indices] = reg_step1.coef_[step1_fix_channel_indices]
    step2_channels_indices = [ch for ch in list(range(X.shape[-1])) if ch not in step1_fix_channel_indices]
    step2_fix_indices_in_step2_indices = [step2_channels_indices.index(ch) for ch in step2_fix_channel_indices]
    reg_step2 = LinearRegression(fit_intercept=False,positive=True).fit(X[step2_firstevent:step2_lastevent,step2_channels_indices], y[step2_firstevent:step2_lastevent]-X[step2_firstevent:step2_lastevent,:].dot(coef))
    coef[step2_fix_channel_indices] = reg_step2.coef_[step2_fix_indices_in_step2_indices]
    step3_channels_indices = [ch for ch in list(range(X.shape[-1])) if ch not in step1_fix_channel_indices and ch not in step2_fix_channel_indices]
    reg_step3 = LinearRegression(fit_intercept=False,positive=True).fit(X[step2_lastevent:,step3_channels_indices], y[step2_lastevent:]-X[step2_lastevent:,:].dot(coef))
    coef[step3_channels_indices] = reg_step3.coef_

    class Result:
        coef_ = coef
        intercept_ = 0
        def predict(self,X_in):
            return X_in.dot(self.coef_)
    y_pred = X.dot(coef)
    residuals = y - y_pred
    n, p = X.shape
    dof = n - p
    sigma2 = np.sum(residuals**2) / dof
    # Covariance matrix of coefficients
    cov = sigma2 * np.linalg.inv(X.T @ X)
    # Standard errors
    unc = np.sqrt(np.diag(cov))
    return Result(), unc, cov


def LinearFitWithUncertaintyMultiSteps(X,y,firstevents,lastevents,fix_channel_indices):
    coef = np.zeros(X.shape[-1])
    fit_channels_indices = list(range(0,X.shape[-1]))
    for first,last,fix_index in zip(firstevents,lastevents,fix_channel_indices):
        reg_step = LinearRegression(fit_intercept=False,positive=True).fit(X[first:last,fit_channels_indices], y[first:last]-X[first:last,:].dot(coef))
        fix_indices_in_fit_indices = [fit_channels_indices.index(ch) for ch in fix_index]
        coef[fix_index] = reg_step.coef_[fix_indices_in_fit_indices]
        fit_channels_indices = [ch for ch in fit_channels_indices if ch not in fix_index]
    class Result:
        coef_ = coef
        intercept_ = 0
        def predict(self,X_in):
            return X_in.dot(self.coef_)
    y_pred = X.dot(coef)
    residuals = y - y_pred
    n, p = X.shape
    dof = n - p
    sigma2 = np.sum(residuals**2) / dof
    # Covariance matrix of coefficients
    cov = sigma2 * np.linalg.inv(X.T @ X)
    # Standard errors
    unc = np.sqrt(np.diag(cov))
    return Result(), unc, cov

def LinearFitMultiSteps(X,y,firstevents,lastevents,fix_channel_indices):
    coef = np.zeros(X.shape[-1])
    fit_channels_indices = list(range(0,X.shape[-1]))
    for first,last,fix_index in zip(firstevents,lastevents,fix_channel_indices):
        reg_step = LinearRegression(fit_intercept=False,positive=True).fit(X[first:last,fit_channels_indices], y[first:last]-X[first:last,:].dot(coef))
        fix_indices_in_fit_indices = [fit_channels_indices.index(ch) for ch in fix_index]
        coef[fix_index] = reg_step.coef_[fix_indices_in_fit_indices]
        fit_channels_indices = [ch for ch in fit_channels_indices if ch not in fix_index]
    class Result:
        coef_ = coef
        intercept_ = 0
        def predict(self,X_in):
            return X_in.dot(self.coef_)
    return Result()


def log_loss(theta, X, y):
    beta = np.exp(theta)  # ensures positivity
    y_pred = X @ beta
    return np.sum((y - y_pred) ** 2)



def RidgePositive(X, y, alpha=0.1):
    n_samples, n_features = X.shape

    # Augmented system for ridge regression
    X_aug = np.vstack([X, np.sqrt(alpha) * np.eye(n_features)])
    y_aug = np.concatenate([y, np.zeros(n_features)])

    # Solve with positivity constraint
    res = lsq_linear(X_aug, y_aug, bounds=(0, np.inf))

    # Residuals
    residuals = y - X.dot(res.x)
    dof = max(1, n_samples - n_features)  # avoid division by zero
    sigma2 = np.sum(residuals**2) / dof   # variance estimate

    # Covariance matrix (from ridge normal equation)
    XtX = X.T.dot(X)
    cov = sigma2 * np.linalg.inv(XtX + alpha * np.eye(n_features))

    # Uncertainties are sqrt of diagonal
    unc = np.sqrt(np.diag(cov))

    # Wrap in a simple result object like sklearn
    class Result:
        coef_ = res.x
        intercept_ = 0
        def predict(self,X_in):
            return X_in.dot(self.coef_)

    return Result(), unc, cov

def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

input_suffix = ""

#runNumbers_calibrate = [1234,1231,1235,1232,1236,1237,1238,1239,1240,1241,1243,1245,1246,1248,1249,1250,1252]
##energies_calibrate = [80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80]
#firstEvents_calibrate = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#lastEvents_calibrate = [-1]*17
#runNumber_calibrate_suffix = "_runNumber_1234-1252"
#energy_calibrate_suffix = "_energy_80"
#input_suffix = "_muonveto_holeveto_passPSD_passCC1"

#runNumbers_calibrate = [1355,1358,1359,1360,1361,1362,1363,1364,1365,1366,1367,1368,1369,1370,1372,1373,1389,1390,1398,1408,1404,1405,1406]
#energies_calibrate = [80]*23
#firstEvents_calibrate = [0]*23
#lastEvents_calibrate = [-1]*23


#runNumbers_calibrate = [1355,1357,1358,1359,1360,1361,1362,1363,1364,1365,1366,1367,1368,1369,1370,1372,1373,1389,1390]
#runNumbers_calibrate_inner = [1355,1357,1358,1359,1360,1361,1362,1363,1364]
#energies_calibrate = [80]*19
#firstEvents_calibrate = [0]*19
#lastEvents_calibrate = [-1]*19

#runNumbers_calibrate_inner_indices = [runNumbers_calibrate.index(run) for run in runNumbers_calibrate_inner]
#runNumbers_calibrate = [runNumbers_calibrate[i] for i in runNumbers_calibrate_inner_indices] + [runNumbers_calibrate[i] for i in range(len(runNumbers_calibrate)) if i not in runNumbers_calibrate_inner_indices]
#energies_calibrate = [energies_calibrate[i] for i in runNumbers_calibrate_inner_indices] + [energies_calibrate[i] for i in range(len(runNumbers_calibrate)) if i not in runNumbers_calibrate_inner_indices]
#firstEvents_calibrate = [firstEvents_calibrate[i] for i in runNumbers_calibrate_inner_indices] + [firstEvents_calibrate[i] for i in range(len(runNumbers_calibrate)) if i not in runNumbers_calibrate_inner_indices]
#lastEvents_calibrate = [lastEvents_calibrate[i] for i in runNumbers_calibrate_inner_indices] + [lastEvents_calibrate[i] for i in range(len(runNumbers_calibrate)) if i not in runNumbers_calibrate_inner_indices]

#runNumber_calibrate_suffix = "_runNumber_1355-1390_2step"
#energy_calibrate_suffix = "_energy_80"
#input_suffix = "_muonveto_passPSD_passCC1_passCC2_passCC3"


runNumbers_calibrate = [1355,1357,1358,1359,1360,1361,1362,1363,1364,1365,1366,1367,1368,1369,1370,1372,1373,1389,1390,1398,1408,1404,1405,1406]
runNumbers_calibrate_inner = [1355,1357,1358,1359,1360,1361,1362,1363]
runNumbers_calibrate_outer = [1357,1358,1359,1360,1361,1362,1363,1364,1365,1366,1367,1368,1369,1370,1372,1373,1389,1390,1398,1408,1404,1405,1406]
runNumbers_calibrate_upper = [1364,1365,1366,1367,1389]
runNumbers_calibrate_upperleft = [1359,1366,1367,1398,1408]
runNumbers_calibrate_upperright = [1357,1363,1364,1365,1389,1390,1406]
runNumbers_calibrate_lower = [1368,1369,1370,1372,1373]
runNumbers_calibrate_lowerleft = [1364,1368,1369,1408,1404]
runNumbers_calibrate_lowerright = [1361,1362,1370,1372,1373,1405,1406]
energies_calibrate = [80]*24
firstEvents_calibrate = [0]*24
lastEvents_calibrate = [-1]*24
lastEvents_calibrate[runNumbers_calibrate.index(1366)] = 7500

runNumbers_calibrate_inner_indices = [runNumbers_calibrate.index(run) for run in runNumbers_calibrate_inner]
runNumbers_calibrate_outer_indices = [runNumbers_calibrate.index(run) for run in runNumbers_calibrate_outer]
runNumbers_calibrate_upper_indices = [runNumbers_calibrate.index(run) for run in runNumbers_calibrate_upper]
runNumbers_calibrate_upperleft_indices = [runNumbers_calibrate.index(run) for run in runNumbers_calibrate_upperleft]
runNumbers_calibrate_upperright_indices = [runNumbers_calibrate.index(run) for run in runNumbers_calibrate_upperright]
runNumbers_calibrate_lower_indices = [runNumbers_calibrate.index(run) for run in runNumbers_calibrate_lower]
runNumbers_calibrate_lowerleft_indices = [runNumbers_calibrate.index(run) for run in runNumbers_calibrate_lowerleft]
runNumbers_calibrate_lowerright_indices = [runNumbers_calibrate.index(run) for run in runNumbers_calibrate_lowerright]

runNumbers_calibrate = [runNumbers_calibrate[i] for i in runNumbers_calibrate_inner_indices] + \
                       [runNumbers_calibrate[i] for i in runNumbers_calibrate_lowerleft_indices] + \
                       [runNumbers_calibrate[i] for i in runNumbers_calibrate_lowerright_indices] + \
                       [runNumbers_calibrate[i] for i in runNumbers_calibrate_upperright_indices] + \
                       [runNumbers_calibrate[i] for i in runNumbers_calibrate_upperleft_indices]
energies_calibrate = [energies_calibrate[i] for i in runNumbers_calibrate_inner_indices] + \
                     [energies_calibrate[i] for i in runNumbers_calibrate_lowerleft_indices] + \
                     [energies_calibrate[i] for i in runNumbers_calibrate_lowerright_indices] + \
                     [energies_calibrate[i] for i in runNumbers_calibrate_upperright_indices] + \
                     [energies_calibrate[i] for i in runNumbers_calibrate_upperleft_indices]
firstEvents_calibrate = [firstEvents_calibrate[i] for i in runNumbers_calibrate_inner_indices] + \
                        [firstEvents_calibrate[i] for i in runNumbers_calibrate_lowerleft_indices] + \
                        [firstEvents_calibrate[i] for i in runNumbers_calibrate_lowerright_indices] + \
                        [firstEvents_calibrate[i] for i in runNumbers_calibrate_upperright_indices] + \
                        [firstEvents_calibrate[i] for i in runNumbers_calibrate_upperleft_indices]
lastEvents_calibrate = [lastEvents_calibrate[i] for i in runNumbers_calibrate_inner_indices] + \
                       [lastEvents_calibrate[i] for i in runNumbers_calibrate_lowerleft_indices] + \
                       [lastEvents_calibrate[i] for i in runNumbers_calibrate_lowerright_indices] + \
                       [lastEvents_calibrate[i] for i in runNumbers_calibrate_upperright_indices] + \
                       [lastEvents_calibrate[i] for i in runNumbers_calibrate_upperleft_indices]


#runNumbers_calibrate = [1355,1357,1358,1359,1360,1361,1362,1363,1364,1365,1366,1367,1368,1369,1370,1372,1373,1389,1390,1398,1408,1404,1405,1406]
#energies_calibrate = [80]*24
#firstEvents_calibrate = [0]*24
#lastEvents_calibrate = [-1]*24
#lastEvents_calibrate[runNumbers_calibrate.index(1366)] = 7500


#runNumber_calibrate_suffix = "_runNumber_1355-1406_7500events1366"
#energy_calibrate_suffix = "_energy_80"
#input_suffix = "_muonveto_passPSD_passCC1_passCC2_passCC3"

#runNumbers_calibrate = [1355,1357,1358,1359,1360,1361,1362,1363,1364,1365,1366,1367,1368,1369,1370,1372,1373,1389,1390,1398,1408,1404,1405,1406]
#runNumbers_calibrate_inner = [1355,1357,1358,1359,1360,1361,1362,1363]
#runNumbers_calibrate_outer = [1357,1358,1359,1360,1361,1362,1363,1364,1365,1366,1367,1368,1369,1370,1372,1373,1389,1390,1398,1408,1404,1405,1406]
#runNumbers_calibrate_upper = [1364,1365,1366,1367,1389]
#runNumbers_calibrate_upperleft = [1359,1366,1367,1398,1408]
#runNumbers_calibrate_upperright = [1357,1363,1364,1365,1389,1390,1406]
#runNumbers_calibrate_lower = [1368,1369,1370,1372,1373]
#runNumbers_calibrate_lowerleft = [1364,1368,1369,1408,1404]
#runNumbers_calibrate_lowerright = [1361,1362,1370,1372,1373,1405,1406]
#energies_calibrate = [80]*24
#firstEvents_calibrate = [0]*24
#lastEvents_calibrate = [-1]*24
#lastEvents_calibrate[runNumbers_calibrate.index(1366)] = 7500

#runNumbers_calibrate_inner_indices = [runNumbers_calibrate.index(run) for run in runNumbers_calibrate_inner]
#runNumbers_calibrate_outer_indices = [runNumbers_calibrate.index(run) for run in runNumbers_calibrate_outer]

#runNumbers_calibrate = [runNumbers_calibrate[i] for i in runNumbers_calibrate_inner_indices] + \
#                       [runNumbers_calibrate[i] for i in runNumbers_calibrate_outer_indices]
#energies_calibrate = [energies_calibrate[i] for i in runNumbers_calibrate_inner_indices] + \
#                     [energies_calibrate[i] for i in runNumbers_calibrate_outer_indices]
#firstEvents_calibrate = [firstEvents_calibrate[i] for i in runNumbers_calibrate_inner_indices] + \
#                        [firstEvents_calibrate[i] for i in runNumbers_calibrate_outer_indices]
#lastEvents_calibrate = [lastEvents_calibrate[i] for i in runNumbers_calibrate_inner_indices] + \
#                       [lastEvents_calibrate[i] for i in runNumbers_calibrate_outer_indices]
runNumber_calibrate_suffix = "_runNumber_1355-1406_7500events1366_5steps_addtoys"
energy_calibrate_suffix = "_energy_80"
input_suffix = "_muonveto_passPSD_passCC1_passCC2_passCC3"

#runNumbers_calibrate = [1355,1357,1358,1359,1360,1361,1362,1363,1364,1365,1366,1367,1368,1369,1370,1372,1373,1389,1390]
#runNumbers_calibrate_inner = [1355,1357,1358,1359,1360,1361,1362,1363]
#energies_calibrate = [80]*19
#firstEvents_calibrate = [0]*19
#lastEvents_calibrate = [-1]*19
#runNumber_calibrate_suffix = "_runNumber_1355-1390_linear_logcoeff"
#energy_calibrate_suffix = "_energy_80"
#input_suffix = "_muonveto_passPSD_passCC1_passCC2_passCC3"


#runNumbers_calibrate = [1355,1357,1358,1359,1360,1361,1362,1363,1364,1365,1366]
#energies_calibrate = [80]*11
#firstEvents_calibrate = [0]*11
#lastEvents_calibrate = [-1]*11
#runNumber_calibrate_suffix = "_runNumber_1355-1366"
#energy_calibrate_suffix = "_energy_80"
#input_suffix = "_muonveto_passPSD_passCC1_passCC2_passCC3"

#runNumbers_test = [1267,1268,1270,1271,1272,1273,1274,1275,1261,1260]
#energies_test = [40,40,40,40,40,40,40,40,30,60]
#firstEvents_test = [0,0,0,0,0,0,0,0,0,0]
#lastEvents_test = [-1]*10


#runNumbers_calibrate = [1267,1268,1270,1271,1272,1273,1274,1275]
#energies_calibrate = [40,40,40,40,40,40,40,40]
#firstEvents_calibrate = [0,0,0,0,0,0,0,0]
#lastEvents_calibrate = [-1]*8

#runNumbers_test = [1261,1260,1234,1231,1235,1232,1236,1237,1238,1239]
#energies_test = [30,60,80,80,80,80,80,80,80,80]
#firstEvents_test = [0,0,0,0,0,0,0,0,0,0]
#lastEvents_test = [-1]*10

#runNumbers_calibrate = [1234,1231,1235,1232,1236,1237,1238,1239,1240,1241,1243,1245,1246,1248,1249,1250,1252,1267,1268,1270,1271,1272,1273,1274,1275]
#energies_calibrate = [80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,40,40,40,40,40,40,40,40]
#firstEvents_calibrate = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#lastEvents_calibrate = [-1]*25
#runNumber_calibrate_suffix = "_runNumber_1234-1252_1267-1275"
#energy_calibrate_suffix = "_energy_80_40"
#runNumbers_test = [1261,1260,1267,1268,1270,1271,1272,1273,1274,1275]
#energies_test = [30,60,40,40,40,40,40,40,40,40]
#firstEvents_test = [0,0,0,0,0,0,0,0,0,0]
#lastEvents_test = [-1]*10


runNumbers_test = [1416,1514,1526,1527]
energies_test = [30,40,60,100]
firstEvents_test = [0]*4
lastEvents_test = [-1]*4

runPedestal = 1374

#boardNo_calibrate = [7,8,2,3,4,10,11,12,0,1,5,6,9,13]
#boardNo_calibrate = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#channels_inner = {
#    9: list(range(0,64)),
#    10: list(range(0,64)),
#    4: list(range(24,64)) + [17,19,21,23],
#    5: list(range(32,64)),
#    6: list(range(24,64)) + [16,18,20,22],
#    12: list(range(0,32)),
#    13: list(range(0,32)),
#    14: list(range(0,32))
#}
#channels_upper = {
#    2: list(range(0,32)),
#    3: list(range(0,64)),
#    4: list(range(0,16)) + [16,18,20,22],
#    5: list(range(0,32)),
#    6: list(range(0,16)) + [17,19,21,23],
#    7: list(range(0,64)),
#    8: list(range(0,32))
#}
#channels_upperleft = {
#    2: list(range(0,32)),
#    3: list(range(0,64)),
#    4: list(range(0,16)) + [16,18,20,22]
#}
#channels_upperright = {
#    5: list(range(0,32)),
#    6: list(range(0,16)) + [17,19,21,23],
#    7: list(range(0,64)),
#    8: list(range(0,32))
#}

#channels_lower = {
#    2: list(range(32,64)),
#    11: list(range(0,64)),
#    12: list(range(32,64)),
#    13: list(range(32,64)),
#    14: list(range(32,64)),
#    15: list(range(0,64)),
#    8: list(range(32,64))
#}

#channels_lowerleft = {
#    2: list(range(32,64)),
#    11: list(range(0,64)),
#    12: list(range(32,64))
#}

#channels_lowerright = {
#    13: list(range(32,64)),
#    14: list(range(32,64)),
#    15: list(range(0,64)),
#    8: list(range(32,64))
#}

boardNo_calibrate = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
channels_inner = {
    9: list(range(0,64)),
    10: list(range(0,64)),
    5: list(range(32,64)),
    13: list(range(0,32)),
    4: list(range(32,64)),
    6: list(range(32,64)),
    12: list(range(0,32)),
    14: list(range(0,32))
}
channels_outer = {
    2: list(range(0,64)),
    3: list(range(0,64)),
    4: list(range(0,32)),
    5: list(range(0,32)),
    6: list(range(0,32)),
    7: list(range(0,64)),
    8: list(range(0,64)),
    11: list(range(0,64)),
    12: list(range(32,64)),
    13: list(range(32,64)),
    14: list(range(32,64)),
    15: list(range(0,64))
}
channels_upper = {
    2: list(range(0,32)),
    3: list(range(0,64)),
    4: list(range(0,16)) + [16,18,20,22],
    5: list(range(0,32)),
    6: list(range(0,16)) + [17,19,21,23],
    7: list(range(0,64)),
    8: list(range(0,32))
}
channels_upperleft = {
    2: list(range(0,32)),
    3: list(range(0,64)),
    4: list(range(0,16)) + [16,18,20,22]
}
channels_upperright = {
    5: list(range(0,32)),
    6: list(range(0,16)) + [17,19,21,23],
    7: list(range(0,64)),
    8: list(range(0,32))
}

channels_lower = {
    2: list(range(32,64)),
    11: list(range(0,64)),
    12: list(range(32,64)),
    13: list(range(32,64)),
    14: list(range(32,64)),
    15: list(range(0,64)),
    8: list(range(32,64))
}

channels_lowerleft = {
    2: list(range(32,64)),
    11: list(range(0,64)),
    12: list(range(32,64))
}
channels_lowerright = {
    13: list(range(32,64)),
    14: list(range(32,64)),
    15: list(range(0,64)),
    8: list(range(32,64))
}


runMeanResponse = runNumbers_calibrate[0]


pedestal_cut = {
    30: [10000,40000],
    40: [10000,40000],
    60: [10000,40000],
    80: [40000,100000],
    100: [10000,40000],
    120: [10000,40000]
}

unc_from_random = True

random_seed = 3
random_drop_proportion = 0
n_random = 20

isCer = True
isHG = True
var = "Cer" if isCer else "Sci"
textvar = "Cherenkov" if isCer else "Scintillation"
textgain = "HG" if isHG else "LG"
conversion_factor = 1000 if isCer else 10000
if not isHG: conversion_factor = conversion_factor/10

#dead_channels = {
#    7: [25,29,30,35,41,43,47,51,53],
#    8: [40,44],
#    4: [46],
#    0: [26,27],
#    1: [30],
#    2: [20,36,46,60],
#    3: [56],
#    13: [26],
#    11: [39]
#}
dead_channels = {
    9: [61],
    10: [40,44],
    12: [26],
    2:[26,27],

}

sys.path.append("CMSPLOTS")  # noqa
ROOT.ROOT.EnableImplicitMT(10)
ROOT.gROOT.SetBatch(True)  # Disable interactive mode for batch processing

runNumber_str_calibrate = ', '.join([str(runNumber) for runNumber in runNumbers_calibrate])
if len(runNumbers_calibrate) == 1: runNumber_str_calibrate = str(runNumbers_calibrate[0])
else: runNumber_str_calibrate = str(min(runNumbers_calibrate)) + "-" + str(max(runNumbers_calibrate))
if 'runNumber_calibrate_suffix' not in globals():
    runNumber_calibrate_suffix = "_runNumber"+"_".join([str(runNumber) for runNumber in runNumbers_calibrate])
if 'runNumber_test_suffix' not in globals():
    runNumber_test_suffix = "_runNumber"+"_".join([str(runNumber) for runNumber in runNumbers_test])
if 'boardNo_calibrate_suffix' not in globals():
    boardNo_calibrate_suffix = "_board"+"_".join([str(boardNo) for boardNo in boardNo_calibrate])
if 'energy_calibrate_suffix' not in globals():
    energy_calibrate_suffix = "_energy"+"_".join([str(energy) for energy in energies_calibrate])
if 'energy_test_suffix' not in globals():
    energy_test_suffix = "_energy"+"_".join([str(energy) for energy in energies_test])
var_suffix = "_Cer" if isCer else "_Sci"
gain_suffix = "_HG" if isHG else "_LG"
plot_calibrate_suffix = runNumber_calibrate_suffix + energy_calibrate_suffix + boardNo_calibrate_suffix + var_suffix + gain_suffix + input_suffix
plot_test_suffix = runNumber_test_suffix + energy_test_suffix + boardNo_calibrate_suffix + var_suffix + gain_suffix + input_suffix

xmax = 14
xmin = -14
ymax = 10
ymin = -10
W_ref = 1400
H_ref = 1600
random.seed(random_seed)

FERSBoards = buildFERSBoards(run=runPedestal)
FERSBoards_calibrate = {}
FERSChannels_calibrate = []
FERSChannels_calibrate_inner = []
FERSChannels_calibrate_outer = []
FERSChannels_calibrate_upper = []
FERSChannels_calibrate_lower = []
FERSChannels_calibrate_upperleft = []
FERSChannels_calibrate_upperright = []
FERSChannels_calibrate_lowerleft = []
FERSChannels_calibrate_lowerright = []
hist2d_3mm = None
hist2d_6mm = None
hist2d_unc_3mm = None
hist2d_unc_6mm = None
hist2d_3mm_calibrate_before = []
hist2d_3mm_calibrate_after = []
hist2d_6mm_calibrate_before = []
hist2d_6mm_calibrate_after = []
hist1d_calibrate_before_X = []
hist1d_calibrate_after_X = []
hist1d_calibrate_before_Y = []
hist1d_calibrate_after_Y = []
hist2d_3mm_test_before = []
hist2d_3mm_test_after = []
hist2d_6mm_test_before = []
hist2d_6mm_test_after = []
hist1d_test_before_X = []
hist1d_test_after_X = []
hist1d_test_before_Y = []
hist1d_test_after_Y = []
iTowerX_list = []
iTowerY_list = []
iTower_list_is3mm = []

if len(set([7,8]).intersection(boardNo_calibrate)) > 0:
    hist2d_3mm = ROOT.TH2F(
                f"convertion_{var}_3mm",
                f"FERS response factors;X;Y ({textvar})",
                int(xmax - xmin), xmin, xmax,
                int(ymax - ymin) * 4, ymin, ymax
            )
    hist2d_3mm.SetMarkerSize(0.55)
    hist2d_unc_3mm = ROOT.TH2F(
                f"convertion_{var}_unc_3mm",
                f"FERS response factors uncertainty;X;Y ({textvar})",
                int(xmax - xmin), xmin, xmax,
                int(ymax - ymin) * 4, ymin, ymax
            )
    hist2d_unc_3mm.SetMarkerSize(0.55)
    if unc_from_random:
        hist2d_unc_toy_3mm = ROOT.TH2F(
                f"convertion_{var}_unc_toy_3mm",
                f"FERS response factors uncertainty (toys);X;Y ({textvar})",
                int(xmax - xmin), xmin, xmax,
                int(ymax - ymin) * 4, ymin, ymax 
            )
        hist2d_unc_toy_3mm.SetMarkerSize(0.55)
    for runNumber in runNumbers_calibrate:
        hist2d_3mm_calibrate_before.append(ROOT.TH2F(
                    f"ADC_{var}_3mm_for_fit_run{runNumber}",
                    f"FERS average ADC counts run{runNumber};X;Y ({textvar})",
                    int(xmax - xmin), xmin, xmax,
                    int(ymax - ymin) * 4, ymin, ymax
                ))
        hist2d_3mm_calibrate_before[-1].SetMarkerSize(0.55)
        hist2d_3mm_calibrate_after.append(ROOT.TH2F(
                    f"energy_{var}_3mm_for_fit_run{runNumber}",
                    f"FERS average energy [GeV] run{runNumber};X;Y ({textvar})",
                    int(xmax - xmin), xmin, xmax,
                    int(ymax - ymin) * 4, ymin, ymax
                ))
        hist2d_3mm_calibrate_after[-1].SetMarkerSize(0.55)

    for runNumber in runNumbers_test:
        hist2d_3mm_test_before.append(ROOT.TH2F(
                    f"ADC_{var}_3mm_for_test_run{runNumber}",
                    f"FERS average ADC counts run{runNumber};X;Y ({textvar})",
                    int(xmax - xmin), xmin, xmax,
                    int(ymax - ymin) * 4, ymin, ymax
                ))
        hist2d_3mm_test_before[-1].SetMarkerSize(0.55)
        hist2d_3mm_test_after.append(ROOT.TH2F(
                    f"energy_{var}_3mm_for_test_run{runNumber}",
                    f"FERS average energy [GeV] run{runNumber};X;Y ({textvar})",
                    int(xmax - xmin), xmin, xmax,
                    int(ymax - ymin) * 4, ymin, ymax
                ))
        hist2d_3mm_test_after[-1].SetMarkerSize(0.55)


if len(set([0,1,2,3,4,5,6,9,10,11,12,13]).intersection(boardNo_calibrate)) > 0:
    hist2d_6mm = ROOT.TH2F(
                f"convertion_{var}_6mm",
                f"FERS response factors;X;Y ({textvar})",
                int(xmax - xmin), xmin, xmax,
                int(ymax - ymin), ymin, ymax
            )
    hist2d_6mm.SetMarkerSize(0.55)
    hist2d_unc_6mm = ROOT.TH2F(
                f"convertion_{var}_unc_6mm",
                f"FERS response factors uncertainty;X;Y ({textvar})",
                int(xmax - xmin), xmin, xmax,
                int(ymax - ymin), ymin, ymax
            )
    hist2d_unc_6mm.SetMarkerSize(0.55)
    if unc_from_random:
        hist2d_unc_toy_6mm = ROOT.TH2F(
                f"convertion_{var}_unc_toy_6mm",
                f"FERS response factors uncertainty (toys);X;Y ({textvar})",
                int(xmax - xmin), xmin, xmax,
                int(ymax - ymin), ymin, ymax
            )
        hist2d_unc_toy_6mm.SetMarkerSize(0.55)
    for runNumber in runNumbers_calibrate:
        hist2d_6mm_calibrate_before.append(ROOT.TH2F(
                    f"ADC_{var}_6mm_for_fit_run{runNumber}",
                    f"FERS average ADC counts run{runNumber};X;Y ({textvar})",
                    int(xmax - xmin), xmin, xmax,
                    int(ymax - ymin), ymin, ymax
                ))
        hist2d_6mm_calibrate_before[-1].SetMarkerSize(0.55)
        hist2d_6mm_calibrate_after.append(ROOT.TH2F(
                    f"energy_{var}_6mm_for_fit_run{runNumber}",
                    f"FERS average energy [GeV] run{runNumber};X;Y ({textvar})",
                    int(xmax - xmin), xmin, xmax,
                    int(ymax - ymin), ymin, ymax
                ))
        hist2d_6mm_calibrate_after[-1].SetMarkerSize(0.55)
    for runNumber in runNumbers_test:
        hist2d_6mm_test_before.append(ROOT.TH2F(
                    f"ADC_{var}_6mm_for_test_run{runNumber}",
                    f"FERS average ADC counts run{runNumber};X;Y ({textvar})",
                    int(xmax - xmin), xmin, xmax,
                    int(ymax - ymin), ymin, ymax
                ))
        hist2d_6mm_test_before[-1].SetMarkerSize(0.55)
        hist2d_6mm_test_after.append(ROOT.TH2F(
                    f"energy_{var}_6mm_for_test_run{runNumber}",
                    f"FERS average energy [GeV] run{runNumber};X;Y ({textvar})",
                    int(xmax - xmin), xmin, xmax,
                    int(ymax - ymin), ymin, ymax
                ))
        hist2d_6mm_test_after[-1].SetMarkerSize(0.55)

for runNumber in runNumbers_calibrate:
    hist1d_calibrate_before_X.append(ROOT.TH1F(
                f"ADC_{var}_for_fit_run{runNumber}_X",
                f"FERS average ADC counts in X run{runNumber} {textvar}",
                int(xmax - xmin), xmin, xmax
            ))
    hist1d_calibrate_before_X[-1].SetMarkerSize(0.55)
    hist1d_calibrate_before_Y.append(ROOT.TH1F(
                f"ADC_{var}_for_fit_run{runNumber}_Y",
                f"FERS average ADC counts in Y run{runNumber} {textvar}",
                int(ymax - ymin), ymin, ymax
            ))
    hist1d_calibrate_before_Y[-1].SetMarkerSize(0.55)
    hist1d_calibrate_after_X.append(ROOT.TH1F(
                f"energy_{var}_for_fit_run{runNumber}_X",
                f"FERS average energy in X [GeV] run{runNumber} {textvar}",
                int(xmax - xmin), xmin, xmax
            ))
    hist1d_calibrate_after_X[-1].SetMarkerSize(0.55)
    hist1d_calibrate_after_Y.append(ROOT.TH1F(
                f"energy_{var}_for_fit_run{runNumber}_Y",
                f"FERS average energy in Y [GeV] run{runNumber} {textvar}",
                int(ymax - ymin), ymin, ymax
            ))
    hist1d_calibrate_after_Y[-1].SetMarkerSize(0.55)


for runNumber in runNumbers_test:
    hist1d_test_before_X.append(ROOT.TH1F(
                f"ADC_{var}_for_test_run{runNumber}_X",
                f"FERS average ADC counts in X run{runNumber} {textvar}",
                int(xmax - xmin), xmin, xmax
            ))
    hist1d_test_before_X[-1].SetMarkerSize(0.55)
    hist1d_test_before_Y.append(ROOT.TH1F(
                f"ADC_{var}_for_test_run{runNumber}_Y",
                f"FERS average ADC counts in Y run{runNumber} {textvar}",
                int(ymax - ymin), ymin, ymax
            ))
    hist1d_test_before_Y[-1].SetMarkerSize(0.55)
    hist1d_test_after_X.append(ROOT.TH1F(
                f"energy_{var}_for_test_run{runNumber}_X",
                f"FERS average energy in X [GeV] run{runNumber} {textvar}",
                int(xmax - xmin), xmin, xmax
            ))
    hist1d_test_after_X[-1].SetMarkerSize(0.55)
    hist1d_test_after_Y.append(ROOT.TH1F(
                f"energy_{var}_for_test_run{runNumber}_Y",
                f"FERS average energy in Y [GeV] run{runNumber} {textvar}",
                int(ymax - ymin), ymin, ymax
            ))
    hist1d_test_after_Y[-1].SetMarkerSize(0.55)
    


for boardNo in boardNo_calibrate:
    FERSBoards_calibrate[f"Board{boardNo}"] = FERSBoards[f"Board{boardNo}"]
    print(f"# channels before adding {boardNo}", len(FERSChannels_calibrate))
    print(f"# inner channels before adding {boardNo}",len(FERSChannels_calibrate_inner))
    for iTowerX, iTowerY in FERSBoards[f"Board{boardNo}"].GetListOfTowers():
        chan = FERSBoards[f"Board{boardNo}"].GetChannelByTower(
            iTowerX, iTowerY, isCer=isCer)
        channelNo_var = chan.channelNo
        if boardNo in dead_channels.keys() and channelNo_var in dead_channels[boardNo]: continue
        if isHG:
            FERSChannels_calibrate.append(f"FERS_Board{boardNo}_energyHG_{channelNo_var}_subtracted_saturationcorrected")
            if boardNo in channels_inner.keys() and channelNo_var in channels_inner[boardNo]:
                FERSChannels_calibrate_inner.append(f"FERS_Board{boardNo}_energyHG_{channelNo_var}_subtracted_saturationcorrected")
            if boardNo in channels_outer.keys() and channelNo_var in channels_outer[boardNo]:
                FERSChannels_calibrate_outer.append(f"FERS_Board{boardNo}_energyHG_{channelNo_var}_subtracted_saturationcorrected")    
            if boardNo in channels_upper.keys() and channelNo_var in channels_upper[boardNo]:
                FERSChannels_calibrate_upper.append(f"FERS_Board{boardNo}_energyHG_{channelNo_var}_subtracted_saturationcorrected")
            if boardNo in channels_upperleft.keys() and channelNo_var in channels_upperleft[boardNo]:
                FERSChannels_calibrate_upperleft.append(f"FERS_Board{boardNo}_energyHG_{channelNo_var}_subtracted_saturationcorrected")
            if boardNo in channels_upperright.keys() and channelNo_var in channels_upperright[boardNo]:
                FERSChannels_calibrate_upperright.append(f"FERS_Board{boardNo}_energyHG_{channelNo_var}_subtracted_saturationcorrected")
            if boardNo in channels_lower.keys() and channelNo_var in channels_lower[boardNo]:
                FERSChannels_calibrate_lower.append(f"FERS_Board{boardNo}_energyHG_{channelNo_var}_subtracted_saturationcorrected")
            if boardNo in channels_lowerleft.keys() and channelNo_var in channels_lowerleft[boardNo]:
                FERSChannels_calibrate_lowerleft.append(f"FERS_Board{boardNo}_energyHG_{channelNo_var}_subtracted_saturationcorrected")
            if boardNo in channels_lowerright.keys() and channelNo_var in channels_lowerright[boardNo]:
                FERSChannels_calibrate_lowerright.append(f"FERS_Board{boardNo}_energyHG_{channelNo_var}_subtracted_saturationcorrected")
            
        else:
            FERSChannels_calibrate.append(f"FERS_Board{boardNo}_energyLG_{channelNo_var}_subtracted")
            if boardNo in channels_inner.keys() and channelNo_var in channels_inner[boardNo]:
                FERSChannels_calibrate_inner.append(f"FERS_Board{boardNo}_energyLG_{channelNo_var}_subtracted")
            if boardNo in channels_outer.keys() and channelNo_var in channels_outer[boardNo]:
                FERSChannels_calibrate_outer.append(f"FERS_Board{boardNo}_energyLG_{channelNo_var}_subtracted")
            if boardNo in channels_upper.keys() and channelNo_var in channels_upper[boardNo]:
                FERSChannels_calibrate_upper.append(f"FERS_Board{boardNo}_energyLG_{channelNo_var}_subtracted")
            if boardNo in channels_upperleft.keys() and channelNo_var in channels_upperleft[boardNo]:
                FERSChannels_calibrate_upperleft.append(f"FERS_Board{boardNo}_energyLG_{channelNo_var}_subtracted")
            if boardNo in channels_upperright.keys() and channelNo_var in channels_upperright[boardNo]:
                FERSChannels_calibrate_upperright.append(f"FERS_Board{boardNo}_energyLG_{channelNo_var}_subtracted")
            if boardNo in channels_lower.keys() and channelNo_var in channels_lower[boardNo]:
                FERSChannels_calibrate_lower.append(f"FERS_Board{boardNo}_energyLG_{channelNo_var}_subtracted")
            if boardNo in channels_lowerleft.keys() and channelNo_var in channels_lowerleft[boardNo]:
                FERSChannels_calibrate_lowerleft.append(f"FERS_Board{boardNo}_energyLG_{channelNo_var}_subtracted")
            if boardNo in channels_lowerright.keys() and channelNo_var in channels_lowerright[boardNo]:
                FERSChannels_calibrate_lowerright.append(f"FERS_Board{boardNo}_energyLG_{channelNo_var}_subtracted")
        iTower_list_is3mm.append(FERSBoards[f"Board{boardNo}"].Is3mm())
        iTowerX_list.append(iTowerX)
        iTowerY_list.append(iTowerY)
    print(f"# channels after adding {boardNo}", len(FERSChannels_calibrate))
    print(f"# inner channels after adding {boardNo}",len(FERSChannels_calibrate_inner))

FERSChannels_calibrate_inner_indices = [FERSChannels_calibrate.index(ch) for ch in FERSChannels_calibrate_inner]
FERSChannels_calibrate_outer_indices = [FERSChannels_calibrate.index(ch) for ch in FERSChannels_calibrate_outer]
FERSChannels_calibrate_upper_indices = [FERSChannels_calibrate.index(ch) for ch in FERSChannels_calibrate_upper]
FERSChannels_calibrate_upperleft_indices = [FERSChannels_calibrate.index(ch) for ch in FERSChannels_calibrate_upperleft]
FERSChannels_calibrate_upperright_indices = [FERSChannels_calibrate.index(ch) for ch in FERSChannels_calibrate_upperright]
FERSChannels_calibrate_lower_indices = [FERSChannels_calibrate.index(ch) for ch in FERSChannels_calibrate_lower]
FERSChannels_calibrate_lowerleft_indices = [FERSChannels_calibrate.index(ch) for ch in FERSChannels_calibrate_lowerleft]
FERSChannels_calibrate_lowerright_indices = [FERSChannels_calibrate.index(ch) for ch in FERSChannels_calibrate_lowerright]
#steps_channel_indices = [FERSChannels_calibrate_inner_indices,FERSChannels_calibrate_outer_indices]
steps_channel_indices = [FERSChannels_calibrate_inner_indices,FERSChannels_calibrate_lowerleft_indices,FERSChannels_calibrate_lowerright_indices,FERSChannels_calibrate_upperright_indices,FERSChannels_calibrate_upperleft_indices]
FERSArrays_calibrate = []
EnergyArrays_calibrate = []
EnergyArrays_mean_response_calibrate = []
nEvents_calibrate = []
channel_mean_calibrate_before = []

nevent_calibrate_inner = 0
nevent_calibrate_outer = 0
nevent_calibrate_upper = 0
nevent_calibrate_lower = 0
nevent_calibrate_upperleft = 0
nevent_calibrate_upperright = 0
nevent_calibrate_lowerleft = 0
nevent_calibrate_lowerright = 0
runNumbers_calibrate_inner_counted = []
runNumbers_calibrate_outer_counted = []
runNumbers_calibrate_upper_counted = []
runNumbers_calibrate_upperleft_counted = []
runNumbers_calibrate_upperright_counted = []
runNumbers_calibrate_lower_counted = []
runNumbers_calibrate_lowerleft_counted = []
runNumbers_calibrate_lowerright_counted = []
for runNumber, energy, firstEvent, lastEvent in zip(runNumbers_calibrate,energies_calibrate,firstEvents_calibrate,lastEvents_calibrate):
    if isHG:
        infile = f"results/root/Run{runNumber}/FERS_filtered{input_suffix}.npz"
    else:
        infile = f"results/root/Run{runNumber}/FERS_filtered_LG{input_suffix}.npz"
    FERSData = np.load(infile,"r")
    if isHG:
        selection = ( FERSData[f"FERS_CerEnergyHG_subtracted_saturationcorrected"][firstEvent:lastEvent]>pedestal_cut[energy][0]) & (FERSData[f"FERS_SciEnergyHG_subtracted_saturationcorrected"][firstEvent:lastEvent]>pedestal_cut[energy][1])
    else:
        selection = FERSData[f"FERS_{var}EnergyLG_subtracted"][firstEvent:lastEvent]>-2000
    FERSArrays = np.column_stack([FERSData[c][firstEvent:lastEvent][selection] for c in FERSChannels_calibrate])
    #FERSArrays[FERSArrays < 0] = 0
    print("after filtering ",len(FERSArrays), " events")
    if runNumber == runMeanResponse:
        if isHG:
            mean_response = np.mean(FERSData[f"FERS_{var}EnergyHG_subtracted_saturationcorrected"][firstEvent:lastEvent][selection])/energy
        else:
            mean_response = np.mean(FERSData[f"FERS_{var}EnergyLG_subtracted"][firstEvent:lastEvent][selection])/energy
    if isHG:
        EnergyArrays_mean_response = FERSData[f"FERS_{var}EnergyHG_subtracted_saturationcorrected"][firstEvent:lastEvent][selection]/mean_response
    else:
        EnergyArrays_mean_response = FERSData[f"FERS_{var}EnergyLG_subtracted"][firstEvent:lastEvent][selection]/mean_response
    nEvents_calibrate.append(len(FERSArrays))
    if runNumber in runNumbers_calibrate_inner and runNumber not in runNumbers_calibrate_inner_counted:
        nevent_calibrate_inner  += len(FERSArrays)
        runNumbers_calibrate_inner_counted.append(runNumber)
    if runNumber in runNumbers_calibrate_outer and runNumber not in runNumbers_calibrate_outer_counted:
        nevent_calibrate_outer  += len(FERSArrays)
        runNumbers_calibrate_outer_counted.append(runNumber)
    if runNumber in runNumbers_calibrate_upper and runNumber not in runNumbers_calibrate_upper_counted:
        nevent_calibrate_upper += len(FERSArrays)
        runNumbers_calibrate_upper_counted.append(runNumber)
    if runNumber in runNumbers_calibrate_upperleft and runNumber not in runNumbers_calibrate_upperleft_counted:
        nevent_calibrate_upperleft += len(FERSArrays)
        runNumbers_calibrate_upperleft_counted.append(runNumber)
    if runNumber in runNumbers_calibrate_upperright and runNumber not in runNumbers_calibrate_upperright_counted:
        nevent_calibrate_upperright += len(FERSArrays)
        runNumbers_calibrate_upperright_counted.append(runNumber)
    if runNumber in runNumbers_calibrate_lower and runNumber not in runNumbers_calibrate_lower_counted:
        nevent_calibrate_lower += len(FERSArrays)
        runNumbers_calibrate_lower_counted.append(runNumber)
    if runNumber in runNumbers_calibrate_lowerleft and runNumber not in runNumbers_calibrate_lowerleft_counted:
        nevent_calibrate_lowerleft += len(FERSArrays)
        runNumbers_calibrate_lowerleft_counted.append(runNumber)
    if runNumber in runNumbers_calibrate_lowerright and runNumber not in runNumbers_calibrate_lowerright_counted:
        nevent_calibrate_lowerright += len(FERSArrays)
        runNumbers_calibrate_lowerright_counted.append(runNumber)
    FERSArrays_calibrate.append(FERSArrays)
    channel_mean_calibrate_before.append(np.mean(FERSArrays,axis=0))
    EnergyArrays_calibrate.append(np.full(FERSArrays.shape[0],energy))
    EnergyArrays_mean_response_calibrate.append(EnergyArrays_mean_response)
    print(f"Loaded run {runNumber}")
FERSArrays_calibrate = np.vstack(FERSArrays_calibrate)
EnergyArrays_calibrate = np.concatenate(EnergyArrays_calibrate)
EnergyArrays_mean_response_calibrate = np.concatenate(EnergyArrays_mean_response_calibrate)
steps_firstevent = [0, nevent_calibrate_inner,
                    nevent_calibrate_inner+nevent_calibrate_lowerleft,
                    nevent_calibrate_inner+nevent_calibrate_lowerleft+nevent_calibrate_lowerright,
                    nevent_calibrate_inner+nevent_calibrate_lowerleft+nevent_calibrate_lowerright+nevent_calibrate_upperright
]
#steps_firstevent = [0, nevent_calibrate_inner]
steps_lastevent = [nevent_calibrate_inner,
                    nevent_calibrate_inner+nevent_calibrate_lowerleft,
                    nevent_calibrate_inner+nevent_calibrate_lowerleft+nevent_calibrate_lowerright,
                    nevent_calibrate_inner+nevent_calibrate_lowerleft+nevent_calibrate_lowerright+nevent_calibrate_upperright,
                    nevent_calibrate_inner+nevent_calibrate_lowerleft+nevent_calibrate_lowerright+nevent_calibrate_upperright+nevent_calibrate_upperleft
]
#steps_lastevent = [nevent_calibrate_inner,nevent_calibrate_inner+nevent_calibrate_outer]
print("Calibrating on ",len(FERSArrays_calibrate), " events")

#reg = LinearRegression(fit_intercept=False,positive=True).fit(FERSArrays_calibrate, EnergyArrays_calibrate)
#reg, unc, cov = LinearFitWithUncertainty(FERSArrays_calibrate, EnergyArrays_calibrate)

#reg, unc, cov = LinearFitWithUncertaintyLogSpace(FERSArrays_calibrate, EnergyArrays_calibrate)
#reg, unc, cov = RidgePositive(FERSArrays_calibrate, EnergyArrays_calibrate)
#reg, unc, cov = LinearFitWithUncertainty2Steps(FERSArrays_calibrate, EnergyArrays_calibrate,step1_firstevent=0,step1_lastevent=lastEvent_calibrate_inner,step1_fix_channel_indices=FERSChannels_calibrate_inner_indices)
#reg, unc, cov = LinearFitWithUncertainty3Steps(FERSArrays_calibrate, EnergyArrays_calibrate,step1_firstevent=0,step1_lastevent=lastEvent_calibrate_inner,step1_fix_channel_indices=FERSChannels_calibrate_inner_indices,step2_firstevent=lastEvent_calibrate_inner,step2_lastevent=lastEvent_calibrate_upper,step2_fix_channel_indices=FERSChannels_calibrate_upper_indices)
print("steps_firstevent",steps_firstevent)
print("steps_lastevent",steps_lastevent)
reg, unc, cov = LinearFitWithUncertaintyMultiSteps(FERSArrays_calibrate, EnergyArrays_calibrate,steps_firstevent,steps_lastevent,steps_channel_indices)
print("Calibration done")
df_cov = pd.DataFrame(cov,index = FERSChannels_calibrate, columns = FERSChannels_calibrate)
if unc_from_random:
    coef_toys = []
    for itoy in range(n_random):
        print("toy ",itoy)
        indices_select = random.choices(range(len(FERSArrays_calibrate)), k=int(len(FERSArrays_calibrate) * (1-random_drop_proportion)))
        indices_select.sort()
        FERSArrays_calibrate_toy = FERSArrays_calibrate[indices_select,:]
        EnergyArrays_calibrate_toy = EnergyArrays_calibrate[indices_select]
        #reg_toy = LinearRegression(fit_intercept=False,positive=True).fit(FERSArrays_calibrate_toy, EnergyArrays_calibrate_toy)
        reg_toy = LinearFitMultiSteps(FERSArrays_calibrate_toy, EnergyArrays_calibrate_toy,[int(first*(1-random_drop_proportion)) for first in steps_firstevent],[int(last*(1-random_drop_proportion)) for last in steps_lastevent],steps_channel_indices)
        coef_toys.append(reg_toy.coef_)
        print("Calibration done")
    coef_toys = np.vstack(coef_toys)
    cov_toys = np.cov(coef_toys.T)
    unc_toys = np.sqrt(np.diag(cov_toys))
    df_cov_toys = pd.DataFrame(cov_toys,index = FERSChannels_calibrate, columns = FERSChannels_calibrate)
response_dic = {}
for ch,coef, coef_unc in zip(FERSChannels_calibrate,reg.coef_,unc):
    response_dic[ch] = {
        "response": coef,
        "uncertainty": coef_unc
    }
if unc_from_random:
    for ch, coef_unc, coef_list in zip(FERSChannels_calibrate,unc_toys,coef_toys.T):
        response_dic[ch]["uncertainty_toys"] = coef_unc
        response_dic[ch]["response_toys"] = coef_list.tolist()
    
print("coef ", reg.coef_)
print("intercept ", reg.intercept_)
print("unc ",unc)
print("EnergyArrays_calibrate ",EnergyArrays_calibrate)
print("Predict ",reg.predict(FERSArrays_calibrate))
print("mean square of prediction error ",np.mean((reg.predict(FERSArrays_calibrate)-EnergyArrays_calibrate)**2))
print("mean square of prediction error from mean response ",np.mean((EnergyArrays_mean_response_calibrate-EnergyArrays_calibrate)**2))
for icoef, (iTowerX, iTowerY, iTower_is3mm, coef, coef_unc) in enumerate(zip(iTowerX_list,iTowerY_list,iTower_list_is3mm,reg.coef_,unc)):
    if iTower_is3mm:
        hist2d_3mm.Fill(iTowerX, iTowerY, coef*conversion_factor)
        hist2d_unc_3mm.Fill(iTowerX, iTowerY, coef_unc*conversion_factor)
        if unc_from_random:
            hist2d_unc_toy_3mm.Fill(iTowerX, iTowerY, unc_toys[icoef]*conversion_factor)
        for i in range(len(runNumbers_calibrate)):
            hist2d_3mm_calibrate_before[i].Fill(iTowerX, iTowerY, channel_mean_calibrate_before[i][icoef])
            hist2d_3mm_calibrate_after[i].Fill(iTowerX, iTowerY, channel_mean_calibrate_before[i][icoef]*coef)
    else:
        hist2d_6mm.Fill(iTowerX, iTowerY, coef*conversion_factor)
        hist2d_unc_6mm.Fill(iTowerX, iTowerY, coef_unc*conversion_factor)
        if unc_from_random:
            hist2d_unc_toy_6mm.Fill(iTowerX, iTowerY, unc_toys[icoef]*conversion_factor)
        for i in range(len(runNumbers_calibrate)):
            hist2d_6mm_calibrate_before[i].Fill(iTowerX, iTowerY, channel_mean_calibrate_before[i][icoef])
            hist2d_6mm_calibrate_after[i].Fill(iTowerX, iTowerY, channel_mean_calibrate_before[i][icoef]*coef)
    for i in range(len(runNumbers_calibrate)):
        hist1d_calibrate_before_X[i].Fill(iTowerX,channel_mean_calibrate_before[i][icoef])
        hist1d_calibrate_after_X[i].Fill(iTowerX,channel_mean_calibrate_before[i][icoef]*coef)
        hist1d_calibrate_before_Y[i].Fill(iTowerY,channel_mean_calibrate_before[i][icoef])
        hist1d_calibrate_after_Y[i].Fill(iTowerY,channel_mean_calibrate_before[i][icoef]*coef)

hist_list = []
if hist2d_3mm is not None:
    hist_list.append(hist2d_3mm)
if hist2d_6mm is not None:
    hist_list.append(hist2d_6mm)

hist_unc_list = []
if hist2d_unc_3mm is not None:
    hist_unc_list.append(hist2d_unc_3mm)
if hist2d_unc_6mm is not None:
    hist_unc_list.append(hist2d_unc_6mm)

if unc_from_random:
    hist_unc_toy_list = []
    if hist2d_unc_toy_3mm is not None:
        hist_unc_toy_list.append(hist2d_unc_toy_3mm)
    if hist2d_unc_toy_6mm is not None:
        hist_unc_toy_list.append(hist2d_unc_toy_6mm)
outdir_response = "FERS_response/"
DrawHistos(hist_list, f"", xmin, xmax, "iX",
                       ymin, ymax, "iY", f"FERS_response{plot_calibrate_suffix}",
                       dology=False, drawoptions=["COL,text"]*len(hist_list),
                       zmin=0.0, zmax=max(reg.coef_*conversion_factor)*1.01, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText=f"{var} conv.*{conversion_factor}", runNumber=runNumber_str_calibrate,
                       outdir=outdir_response,nTextDigits=2)

DrawHistos(hist_unc_list, f"", xmin, xmax, "iX",
                       ymin, ymax, "iY", f"FERS_response_unc{plot_calibrate_suffix}",
                       dology=False, drawoptions=["COL,text"]*len(hist_unc_list),
                       zmin=0.0, zmax=max(unc*conversion_factor)*1.01, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText=f"{var} u(conv)*{conversion_factor}", runNumber=runNumber_str_calibrate,
                       outdir=outdir_response,nTextDigits=2)
if unc_from_random:
    DrawHistos(hist_unc_toy_list, f"", xmin, xmax, "iX",
                       ymin, ymax, "iY", f"FERS_response_unc_toy{plot_calibrate_suffix}",
                       dology=False, drawoptions=["COL,text"]*len(hist_unc_toy_list),
                       zmin=0.0, zmax=max(unc_toys*conversion_factor)*1.01, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText=f"{var} u(conv)*{conversion_factor}", runNumber=runNumber_str_calibrate,
                       outdir=outdir_response,nTextDigits=2)               
with open(f"{outdir_response}/FERS_response{plot_calibrate_suffix}.json", "w") as f:
    json.dump(response_dic, f)
df_cov.to_csv(f"{outdir_response}/FERS_response_cov{plot_calibrate_suffix}.csv")
if unc_from_random:
    df_cov_toys.to_csv(f"{outdir_response}/FERS_response_cov_toys{plot_calibrate_suffix}.csv")

outdir_calibrate = "FERS_calibrate/"
for irun,runNumber in enumerate(runNumbers_calibrate):
    hist_list = []
    if len(hist2d_3mm_calibrate_before)>0:
        hist_list.append(hist2d_3mm_calibrate_before[irun])
    if len(hist2d_6mm_calibrate_before)>0:
        hist_list.append(hist2d_6mm_calibrate_before[irun])
    DrawHistos(hist_list, f"", xmin, xmax, "iX",
                       ymin, ymax, "iY", f"FERS_average_ADC_run{runNumber}_{var}_{textgain}",
                       dology=False, drawoptions=["COL,text"]*len(hist_list),
                       zmin=0.0, zmax=max(channel_mean_calibrate_before[irun])*1.01, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText=f"{var} {textgain}", runNumber=runNumber,
                       outdir=outdir_calibrate,nTextDigits=0)
    DrawHistos([hist1d_calibrate_before_X[irun]], f"", xmin, xmax, "iX",
                       0, hist1d_calibrate_before_X[irun].GetMaximum()*1.1, "#ADC", f"FERS_average_ADC_run{runNumber}_{var}_{textgain}_X",
                       dology=False, drawoptions=["HIST"], extraText=f"{var} {textgain}", runNumber=runNumber,
                       outdir=outdir_calibrate)
    DrawHistos([hist1d_calibrate_before_Y[irun]], f"", ymin, ymax, "iY",
                       0, hist1d_calibrate_before_Y[irun].GetMaximum()*1.1, "#ADC", f"FERS_average_ADC_run{runNumber}_{var}_{textgain}_Y",
                       dology=False, drawoptions=["HIST"], extraText=f"{var} {textgain}", runNumber=runNumber,
                       outdir=outdir_calibrate)
    

    hist_list = []
    if len(hist2d_3mm_calibrate_after)>0:
        hist_list.append(hist2d_3mm_calibrate_after[irun])
    if len(hist2d_6mm_calibrate_after)>0:
        hist_list.append(hist2d_6mm_calibrate_after[irun])
    DrawHistos(hist_list, f"", xmin, xmax, "iX",
                       ymin, ymax, "iY", f"FERS_average_energy_fit{plot_calibrate_suffix}_run{runNumber}",
                       dology=False, drawoptions=["COL,text"]*len(hist_list),
                       zmin=0.0, zmax=max(channel_mean_calibrate_before[irun]*reg.coef_)*1.01, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText=f"{var}", runNumber=runNumber,
                       outdir=outdir_calibrate,nTextDigits=1)
    DrawHistos([hist1d_calibrate_after_X[irun]], f"", xmin, xmax, "iX",
                       0, hist1d_calibrate_after_X[irun].GetMaximum()*1.1, "Energy [GeV]", f"FERS_average_energy_fit{plot_calibrate_suffix}_run{runNumber}_{var}_{textgain}_X",
                       dology=False, drawoptions=["HIST"], extraText=f"{var} {textgain}", runNumber=runNumber,
                       outdir=outdir_calibrate)
    DrawHistos([hist1d_calibrate_after_Y[irun]], f"", ymin, ymax, "iY",
                       0, hist1d_calibrate_after_Y[irun].GetMaximum()*1.1, "Energy [GeV]", f"FERS_average_energy_fit{plot_calibrate_suffix}_run{runNumber}_{var}_{textgain}_Y",
                       dology=False, drawoptions=["HIST"], extraText=f"{var} {textgain}", runNumber=runNumber,
                       outdir=outdir_calibrate)                  

bins = 80
iEvent = 0
for runNumber,nEvents, energy in zip(runNumbers_calibrate,nEvents_calibrate,energies_calibrate):
    range_min, range_max = 0, 5*energy
    hist_fit_cal, bin_edges = np.histogram(reg.predict(FERSArrays_calibrate[iEvent:iEvent+nEvents]), bins=bins, range=(range_min, range_max))
    hist_compare_cal, _ = np.histogram(EnergyArrays_mean_response_calibrate[iEvent:iEvent+nEvents], bins=bins, range=(range_min, range_max))
    iEvent = iEvent + nEvents
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    p0 = [hist_fit_cal.max(), bin_centers[np.argmax(hist_fit_cal)], 10.0]
    try:
        popt, pcov = curve_fit(gaussian, bin_centers, hist_fit_cal, p0=p0)
        A_fit, mu_fit, sigma_fit = popt
        perr = np.sqrt(np.diag(pcov))  # uncertainties
        A_err, mu_err, sigma_err = perr
    except RuntimeError:
        A_fit, mu_fit, sigma_fit = np.nan, np.nan, np.nan
        mu_err, sigma_err = np.nan, np.nan
    plt.figure(figsize=(8,6))
    plt.step(bin_centers, hist_fit_cal, where='mid', label="fit per-channel response")
    plt.step(bin_centers, hist_compare_cal, where='mid', label="from average response")
    if not np.isnan(mu_fit):
        x_fit = np.linspace(range_min, range_max, 1000)
        plt.plot(x_fit, gaussian(x_fit, *popt), 'r-', linewidth=2, label="Gaussian Fit")
        fit_text = (f"μ = {mu_fit:.2f} ± {mu_err:.2f}\n"
                f"σ = {sigma_fit:.2f} ± {sigma_err:.2f}")
        plt.gca().text(
            0.95, 0.8, fit_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray")
        )
    # Labels and legend
    plt.xlabel("Energy [GeV]")
    plt.ylabel("Counts")
    plt.legend()
    plt.title(f"Comparison of measured energy {var} {textgain}, Run{runNumber}, E={energy}")

    # Save to PNG
    plt.tight_layout()
    plt.savefig(f"fit_comparison_calibrate{plot_calibrate_suffix}_plot_run{runNumber}_energy{energy}.png", dpi=300)
    plt.close()

del FERSArrays_calibrate
del EnergyArrays_mean_response_calibrate
del EnergyArrays_calibrate

FERSArrays_test = []
EnergyArrays_test = []
EnergyArrays_mean_response_test = []
channel_mean_test_before = []
nEvents_test = []

for runNumber, energy, firstEvent, lastEvent in zip(runNumbers_test,energies_test,firstEvents_test,lastEvents_test):
    if isHG:
        infile = f"results/root/Run{runNumber}/FERS_filtered{input_suffix}.npz"
    else:
        infile = f"results/root/Run{runNumber}/FERS_filtered_LG{input_suffix}.npz"
    FERSData = np.load(infile,"r")
    if isHG:
        selection = ( FERSData[f"FERS_CerEnergyHG_subtracted_saturationcorrected"][firstEvent:lastEvent]>pedestal_cut[energy][0]) & (FERSData[f"FERS_SciEnergyHG_subtracted_saturationcorrected"][firstEvent:lastEvent]>pedestal_cut[energy][1])
    else:
        selection = FERSData[f"FERS_{var}EnergyLG_subtracted"][firstEvent:lastEvent]>1000
    FERSArrays = np.column_stack([FERSData[c][firstEvent:lastEvent][selection] for c in FERSChannels_calibrate])
    #FERSArrays[FERSArrays<0] = 0
    if isHG:
        EnergyArrays_mean_response = FERSData[f"FERS_{var}EnergyHG_subtracted_saturationcorrected"][firstEvent:lastEvent][selection]/mean_response
    else:
        EnergyArrays_mean_response = FERSData[f"FERS_{var}EnergyLG_subtracted"][firstEvent:lastEvent][selection]/mean_response
    print("after filtering the test set ",len(FERSArrays), " events")
    nEvents_test.append(len(FERSArrays))
    FERSArrays_test.append(FERSArrays)
    channel_mean_test_before.append(np.mean(FERSArrays,axis=0))
    EnergyArrays_test.append(np.full(FERSArrays.shape[0],energy))
    EnergyArrays_mean_response_test.append(EnergyArrays_mean_response)

FERSArrays_test = np.vstack(FERSArrays_test)
EnergyArrays_test = np.concatenate(EnergyArrays_test)
EnergyArrays_mean_response_test = np.concatenate(EnergyArrays_mean_response_test)
print("EnergyArrays_test ",EnergyArrays_test)
print("Predict ",reg.predict(FERSArrays_test))
print("mean square of prediction error ",np.mean((reg.predict(FERSArrays_test)-EnergyArrays_test)**2))
print("std of prediction error from mean response ",np.mean((EnergyArrays_mean_response_test-EnergyArrays_test)**2))
for icoef, (iTowerX, iTowerY, iTower_is3mm, coef) in enumerate(zip(iTowerX_list,iTowerY_list,iTower_list_is3mm,reg.coef_)):
    if iTower_is3mm:
        for i in range(len(runNumbers_test)):
            hist2d_3mm_test_before[i].Fill(iTowerX, iTowerY, channel_mean_test_before[i][icoef])
            hist2d_3mm_test_after[i].Fill(iTowerX, iTowerY, channel_mean_test_before[i][icoef]*coef)
    else:
        for i in range(len(runNumbers_test)):
            hist2d_6mm_test_before[i].Fill(iTowerX, iTowerY, channel_mean_test_before[i][icoef])
            hist2d_6mm_test_after[i].Fill(iTowerX, iTowerY, channel_mean_test_before[i][icoef]*coef)
    for i in range(len(runNumbers_test)):
        hist1d_test_before_X[i].Fill(iTowerX,channel_mean_test_before[i][icoef])
        hist1d_test_after_X[i].Fill(iTowerX,channel_mean_test_before[i][icoef]*coef)
        hist1d_test_before_Y[i].Fill(iTowerY,channel_mean_test_before[i][icoef])
        hist1d_test_after_Y[i].Fill(iTowerY,channel_mean_test_before[i][icoef]*coef)

for irun,runNumber in enumerate(runNumbers_test):
    hist_list = []
    if len(hist2d_3mm_test_before)>0:
        hist_list.append(hist2d_3mm_test_before[irun])
    if len(hist2d_6mm_test_before)>0:
        hist_list.append(hist2d_6mm_test_before[irun])
    DrawHistos(hist_list, f"", xmin, xmax, "iX",
                       ymin, ymax, "iY", f"FERS_average_ADC_run{runNumber}_{var}_{textgain}",
                       dology=False, drawoptions=["COL,text"]*len(hist_list),
                       zmin=0.0, zmax=max(channel_mean_test_before[irun])*1.01, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText=f"{var} {textgain}", runNumber=runNumber,
                       outdir=outdir_calibrate,nTextDigits=0)
    DrawHistos([hist1d_test_before_X[irun]], f"", xmin, xmax, "iX",
                       0, hist1d_test_before_X[irun].GetMaximum()*1.1, "#ADC", f"FERS_average_ADC_run{runNumber}_{var}_{textgain}_X",
                       dology=False, drawoptions=["HIST"], extraText=f"{var} {textgain}", runNumber=runNumber,
                       outdir=outdir_calibrate)
    DrawHistos([hist1d_test_before_Y[irun]], f"", ymin, ymax, "iY",
                       0, hist1d_test_before_Y[irun].GetMaximum()*1.1, "#ADC", f"FERS_average_ADC_run{runNumber}_{var}_{textgain}_Y",
                       dology=False, drawoptions=["HIST"], extraText=f"{var} {textgain}", runNumber=runNumber,
                       outdir=outdir_calibrate)

    hist_list = []
    if len(hist2d_3mm_test_after)>0:
        hist_list.append(hist2d_3mm_test_after[irun])
    if len(hist2d_6mm_test_after)>0:
        hist_list.append(hist2d_6mm_test_after[irun])
    DrawHistos(hist_list, f"", xmin, xmax, "iX",
                       ymin, ymax, "iY", f"FERS_average_energy_fit{plot_calibrate_suffix}_run{runNumber}",
                       dology=False, drawoptions=["COL,text"]*len(hist_list),
                       zmin=0.0, zmax=max(channel_mean_test_before[irun]*reg.coef_)*1.01, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText=f"{var} {textgain}", runNumber=runNumber,
                       outdir=outdir_calibrate,nTextDigits=1)
    DrawHistos([hist1d_test_after_X[irun]], f"", xmin, xmax, "iX",
                       0, hist1d_test_after_X[irun].GetMaximum()*1.1, "Energy [GeV]", f"FERS_average_energy_fit{plot_calibrate_suffix}_run{runNumber}_{var}_{textgain}_X",
                       dology=False, drawoptions=["HIST"], extraText=f"{var} {textgain}", runNumber=runNumber,
                       outdir=outdir_calibrate)
    DrawHistos([hist1d_test_after_Y[irun]], f"", ymin, ymax, "iY",
                       0, hist1d_test_after_Y[irun].GetMaximum()*1.1, "Energy [GeV]", f"FERS_average_energy_fit{plot_calibrate_suffix}_run{runNumber}_{var}_{textgain}_Y",
                       dology=False, drawoptions=["HIST"], extraText=f"{var} {textgain}", runNumber=runNumber,
                       outdir=outdir_calibrate)   




iEvent = 0
for runNumber,nEvents, energy in zip(runNumbers_test,nEvents_test,energies_test):
    range_min, range_max = 0, 5*energy
    hist_fit_test, bin_edges = np.histogram(reg.predict(FERSArrays_test[iEvent:iEvent+nEvents]), bins=bins, range=(range_min, range_max))
    hist_compare_test, _ = np.histogram(EnergyArrays_mean_response_test[iEvent:iEvent+nEvents], bins=bins, range=(range_min, range_max))
    iEvent = iEvent + nEvents
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    p0 = [hist_fit_test.max(), bin_centers[np.argmax(hist_fit_test)], 10.0]
    try:
        popt, pcov = curve_fit(gaussian, bin_centers, hist_fit_test, p0=p0)
        A_fit, mu_fit, sigma_fit = popt
        perr = np.sqrt(np.diag(pcov))  # uncertainties
        A_err, mu_err, sigma_err = perr
    except RuntimeError:
        A_fit, mu_fit, sigma_fit = np.nan, np.nan, np.nan
        mu_err, sigma_err = np.nan, np.nan
    plt.figure(figsize=(8,6))
    plt.step(bin_centers, hist_fit_test, where='mid', label="test per-channel response")
    plt.step(bin_centers, hist_compare_test, where='mid', label="from average response")
    if not np.isnan(mu_fit):
        x_fit = np.linspace(range_min, range_max, 1000)
        plt.plot(x_fit, gaussian(x_fit, *popt), 'r-', linewidth=2, label="Gaussian Fit")
        fit_text = (f"μ = {mu_fit:.2f} ± {mu_err:.2f}\n"
                f"σ = {sigma_fit:.2f} ± {sigma_err:.2f}")
        plt.gca().text(
            0.95, 0.8, fit_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray")
        )
    # Labels and legend
    plt.xlabel("Energy [GeV]")
    plt.ylabel("Counts")
    plt.legend()
    plt.title(f"Comparison of measured energy {var} {textgain}, Run{runNumber}, E={energy}")

    # Save to PNG
    plt.tight_layout()
    plt.savefig(f"test_comparison_calibrate{plot_calibrate_suffix}_plot_run{runNumber}_energy{energy}.png", dpi=300)
    plt.close()
