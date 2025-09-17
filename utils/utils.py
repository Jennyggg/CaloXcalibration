from utils.channel_map import mapDRSChannel2TriggerChannel,findTimeReferenceDelay
import ROOT
import json
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import savgol_filter
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from scipy.optimize import lsq_linear
from scipy.optimize import least_squares
from scipy.special import erf
ROOT.gSystem.Load("utils/functions_cc.so")  # Load the compiled C++ functions
def number2string(n):
    s = str(n)
    return s.replace('-', 'm').replace('.', 'p')


def string2number(s):
    return float(s.replace('m', '-').replace('p', '.'))

def readTSconfig(run = 692):
    with open("TSconfig.json", 'r') as f:
        TSconfig = json.load(f)
    runkey = str(run)
    if runkey not in TSconfig.keys():
        runkey = "default"
    config = {}
    for key in TSconfig[runkey].keys():
        config[key] = np.array(TSconfig[runkey][key])
    return config

def round_up_to_1eN(x):
    import math
    """
    Round a number up to the nearest 10^N.
    """
    if x <= 0:
        return 0
    return 10 ** math.ceil(math.log10(x))


def IsScanRun(runNumber):
    import json
    f_scanruns = "data/scanruns.json"
    with open(f_scanruns, 'r') as f:
        temp = json.load(f)
        scanruns = temp["scanruns"]
    return runNumber in scanruns


def getDataFile(runNumber):
    runNum = str(runNumber)
    import json
    jsonFile = "data/datafiles.json"
    with open(jsonFile, 'r') as f:
        data = json.load(f)
    if runNum in data:
        return data[runNum]
    else:
        raise ValueError(f"Run number {runNum} not found in datafiles.json")


def getBranchStats(rdf, branches):
    stats = {
        br: {
            "mean": rdf.Mean(br),
            "min": rdf.Min(br),
            "max": rdf.Max(br)
        } for br in branches
    }
    return stats


def vectorizeFERS(rdf, FERSBoards):
    # FRES board outputs
    # define variables as RDF does not support reading vectors
    # with indices directly
    for _, FERSBoard in FERSBoards.items():
        boardNo = FERSBoard.boardNo
        for channel in FERSBoard:
            rdf = rdf.Define(
                f"FERS_Board{boardNo}_energyHG_{channel.channelNo}",
                f"FERS_Board{boardNo}_energyHG[{channel.channelNo}]")
            rdf = rdf.Define(
                f"FERS_Board{boardNo}_energyLG_{channel.channelNo}",
                f"FERS_Board{boardNo}_energyLG[{channel.channelNo}]"
            )
    return rdf


def subtractFERSPedestal(rdf, FERSBoards, pedestals):
    for _, FERSBoard in FERSBoards.items():
        boardNo = FERSBoard.boardNo
        for channel in FERSBoard:
            channelNo = channel.channelNo
            channelNameHG = channel.GetHGChannelName()
            pedestal = pedestals[channelNameHG]
            # subtract pedestal from HG and LG energies
            rdf = rdf.Define(
                f"FERS_Board{boardNo}_energyHG_{channelNo}_subtracted",
                f"FERS_Board{boardNo}_energyHG_{channelNo} - {pedestal}"
            )
            # not on LG yet
            # rdf = rdf.Define(
            #    f"FERS_Board{boardNo}_energyLG_{channelNo}_subtracted",
            #    f"FERS_Board{boardNo}_energyLG_{channelNo} - 0."
            # )
    return rdf

def subtractFERSBeamPedestal(rdf, FERSBoards, pedestal):
    for _, FERSBoard in FERSBoards.items():
        boardNo = FERSBoard.boardNo
        for channel in FERSBoard:
            var_HG = f"FERS_Board{boardNo}_energyHG_{channel.channelNo}"
            var_LG = f"FERS_Board{boardNo}_energyLG_{channel.channelNo}"
            var_HG_subtract = f"FERS_Board{boardNo}_energyHG_{channel.channelNo}_subtracted"
            pedestal_HG = pedestal[var_HG]
            pedestal_LG = pedestal[var_LG]
            #print(var_HG, "Numpy", rdf.AsNumpy([f"{var_HG}"])[f"{var_HG}"])
            #print(var_HG, "GetValue", list(rdf.Take["unsigned short"](var_HG).GetValue()))
            
            #print("pedestal HG ",pedestal_HG)
            #rdf = rdf.Define(
            #    f"FERS_Board{boardNo}_energyHG_{channel.channelNo}_subtracted",
            #    f"static_cast<float>(FERS_Board{boardNo}_energyHG_{channel.channelNo}) - {pedestal_HG}"
            #)
            rdf = rdf.Define(
                f"FERS_Board{boardNo}_energyHG_{channel.channelNo}_subtracted",
                f"FERS_Board{boardNo}_energyHG_{channel.channelNo} - {pedestal_HG}"
            )
            #print(f"static_cast<float>(FERS_Board{boardNo}_energyHG_{channel.channelNo}) - 0.0")
            #print("origin - subtract (getvalue)", np.array(rdf.Take["unsigned short"](var_HG).GetValue()) - np.array(rdf.Take["double"](var_HG_subtract).GetValue()))

            rdf = rdf.Define(
                f"FERS_Board{boardNo}_energyLG_{channel.channelNo}_subtracted",
                f"FERS_Board{boardNo}_energyLG_{channel.channelNo} - {pedestal_LG}"
            )
    return rdf

def correctFERSSaturation(rdf,FERSBoards, HG_to_LG_coefficient):
    for _, FERSBoard in FERSBoards.items():
        boardNo = FERSBoard.boardNo
        for channel in FERSBoard:
            var_HG = f"FERS_Board{boardNo}_energyHG_{channel.channelNo}"
            var_LG = f"FERS_Board{boardNo}_energyLG_{channel.channelNo}"
            var_HG_subtract = f"FERS_Board{boardNo}_energyHG_{channel.channelNo}_subtracted"
            var_LG_subtract = f"FERS_Board{boardNo}_energyLG_{channel.channelNo}_subtracted"
            intercept = HG_to_LG_coefficient[f"FERS_Board{boardNo}_{channel.channelNo}"][0]
            slope = HG_to_LG_coefficient[f"FERS_Board{boardNo}_{channel.channelNo}"][1]
            rdf = rdf.Define(
                f"{var_HG_subtract}_saturationcorrected",
                f"{var_HG_subtract} < 7500 ? {var_HG_subtract} : ({var_LG_subtract} - {intercept})/{slope}"
            )
    return rdf

def calibrateFERSChannels(rdf, FERSBoards, file_gains, file_pedestals):
    """
    Calibrate FERS channels using gains and pedestals from the provided files.
    """
    import json
    with open(file_gains, 'r') as f:
        gains = json.load(f)
    with open(file_pedestals, 'r') as f:
        pedestals = json.load(f)

    # Subtract pedestal and apply gain calibration
    rdf = subtractFERSPedestal(rdf, FERSBoards, pedestals)

    for _, FERSBoard in FERSBoards.items():
        boardNo = FERSBoard.boardNo
        for channel in FERSBoard:
            channelNo = channel.channelNo
            channelNameHG = channel.GetHGChannelName()
            gain = gains[channelNameHG]
            rdf = rdf.Define(
                f"FERS_Board{boardNo}_energyHG_{channelNo}_subtracted_calibrated",
                f"FERS_Board{boardNo}_energyHG_{channelNo}_subtracted / {gain}"
            )
            rdf = rdf.Define(
                f"FERS_Board{boardNo}_energyHG_{channelNo}_subtracted_calibrated_clipped",
                f"FERS_Board{boardNo}_energyHG_{channelNo}_subtracted_calibrated > 0.8 ? FERS_Board{boardNo}_energyHG_{channelNo}_subtracted_calibrated : 0"
            )
            # Not calibrating LG yet
            # rdf = rdf.Define(
            #    f"FERS_Board{boardNo}_energyLG_{channelNo}_calibrated",
            #    f"FERS_Board{boardNo}_energyLG_{channelNo}_subtracted"
            # )

    return rdf
def solve_least_squares_qr(A, B, F_ref):
    """
    Solves the least squares problem A @ F ≈ B
    for F using QR decomposition for better numerical stability.
    F has values close to F_ref

    Parameters
    ----------
    A : ndarray of shape (m, n)
        Known coefficient matrix.
    B : ndarray of shape (m,) or (m, 1)
        Known target vector.

    Returns
    -------
    F : ndarray of shape (n,)
        Least squares solution.
    residuals : float
        Sum of squared residuals.
    """
    # Ensure B is 1-D
    B = np.ravel(B)
    m, n = A.shape
    lam = 0.01
    lower_bound = 0.1
    upper_bound = 100
    # Augment for ridge penalty

    sqrt_lam = np.sqrt(lam)
    def fun(F):
        r_data = A @ F - B
        r_reg = sqrt_lam * (F-F_ref)
        return np.hstack([r_data, r_reg])
    x0 = np.full(n, (lower_bound + upper_bound) / 2)

    res = least_squares(
        fun, x0,
        bounds=(np.full(n, lower_bound), np.full(n, upper_bound)),
        method='trf'
    )

    F = res.x
    residuals = np.sum((A @ F - B)**2)
    return F, residuals
    #A_aug = np.vstack([A, np.sqrt(lam) * np.eye(n)])
    #B_aug = np.hstack([B, np.zeros(n)])

    # Solve with bounds
    #result = lsq_linear(A_aug, B_aug, bounds=(lower_bound, upper_bound), method='trf')
    #return result.x, result.cost * 2 

def estimateShowerSize(rdf,FERSBoards,refRangeLow = 0, refRangeHigh = 500):
    FERSNames_Sci = []
    tower_x = []
    tower_y = []
    for _, FERSBoard in FERSBoards.items():
        boardNo = FERSBoard.boardNo
        for iTowerX, iTowerY in FERSBoard.GetListOfTowers():
            chan_Sci = FERSBoard.GetChannelByTower(
                iTowerX, iTowerY, isCer=False)
            tower_x.append(iTowerX * 1.2)
            tower_y.append(iTowerY * 1.6)
            channelNo_Sci = chan_Sci.channelNo
            branchName_Sci = f"FERS_Board{boardNo}_energyHG_{channelNo_Sci}_subtracted_calibrated_clipped"
            FERSNames_Sci.append(branchName_Sci)
    tower_x = np.array(tower_x)
    tower_y = np.array(tower_y)
    l = refRangeLow
    if refRangeHigh > 0 and refRangeHigh < rdf.Count().GetValue():
        h = refRangeHigh
    else:
        h = rdf.Count().GetValue()
    FERSData = rdf.Range(l,h).AsNumpy(columns = FERSNames_Sci)
    FERSArrays = np.column_stack([FERSData[c] for c in FERSNames_Sci])
    std_x_arrays = np.sqrt((FERSArrays @ (tower_x ** 2) / np.sum(FERSArrays,axis=-1) - (FERSArrays @ tower_x / np.sum(FERSArrays,axis=-1))**2)*(len(tower_x)-1)/len(tower_x))
    std_y_arrays = np.sqrt((FERSArrays @ (tower_y ** 2) / np.sum(FERSArrays,axis=-1) - (FERSArrays @ tower_y / np.sum(FERSArrays,axis=-1))**2)*(len(tower_y)-1)/len(tower_y))
    return np.average(std_x_arrays),np.average(std_y_arrays)
    
def estimateFERSFactorsBeam(rdf,FERSBoards,beamEnergy,refRangeLow = 0, refRangeHigh = 500):
    FERSNames_Cer = []
    FERSNames_Sci = []
    for _, FERSBoard in FERSBoards.items():
        boardNo = FERSBoard.boardNo
        for iTowerX, iTowerY in FERSBoard.GetListOfTowers():
            chan_Cer = FERSBoard.GetChannelByTower(
                iTowerX, iTowerY, isCer=True)
            chan_Sci = FERSBoard.GetChannelByTower(
                iTowerX, iTowerY, isCer=False)
            channelNo_Cer = chan_Cer.channelNo
            channelNo_Sci = chan_Sci.channelNo
            branchName_Cer = f"FERS_Board{boardNo}_energyHG_{channelNo_Cer}_subtracted_calibrated_clipped"
            branchName_Sci = f"FERS_Board{boardNo}_energyHG_{channelNo_Sci}_subtracted_calibrated_clipped"
            FERSNames_Cer.append(branchName_Cer)
            FERSNames_Sci.append(branchName_Sci)
    l = refRangeLow
    if refRangeHigh > 0 and refRangeHigh < rdf.Count().GetValue():
        h = refRangeHigh
    else:
        h = rdf.Count().GetValue()
    factors = {}
    for var,FERSNames in zip(["Cer","Sci"],[FERSNames_Cer,FERSNames_Sci]):
        FERSData = rdf.Range(l,h).AsNumpy(columns = FERSNames)
        FERSArrays = np.column_stack([FERSData[c] for c in FERSNames])
        factors[var] = np.average(beamEnergy / np.sum(FERSArrays,axis=-1))
    return factors["Cer"],factors["Sci"]

def erf_part(a, b, mu, w):
        return 0.5 * (erf((b - mu) / (np.sqrt(2) * w)) -
                      erf((a - mu) / (np.sqrt(2) * w)))

def calibrateFERSChannelsBeam(rdf, FERSBoards, beamEnergy,rangeLow = 0, rangeHigh=-1, refRangeLow=0,refRangeHigh = 500, centers_x=None, centers_y=None):
    w_x, w_y = estimateShowerSize(rdf,FERSBoards,refRangeLow = refRangeLow, refRangeHigh = refRangeHigh)
    F_ref_Cer,F_ref_Sci = estimateFERSFactorsBeam(rdf,FERSBoards,beamEnergy,refRangeLow = refRangeLow, refRangeHigh = refRangeHigh)
    print("w_x=",w_x, ", w_y=",w_y)
    print("F_ref_Cer=",F_ref_Cer,", F_ref_Sci=",F_ref_Sci)
    FERSNames_Cer = []
    FERSNames_Sci = []
    tower_x = []
    tower_y = []
    size_x = []
    size_y = []
    for _, FERSBoard in FERSBoards.items():
        boardNo = FERSBoard.boardNo
        for iTowerX, iTowerY in FERSBoard.GetListOfTowers():
            chan_Cer = FERSBoard.GetChannelByTower(
                iTowerX, iTowerY, isCer=True)
            chan_Sci = FERSBoard.GetChannelByTower(
                iTowerX, iTowerY, isCer=False)
            channelNo_Cer = chan_Cer.channelNo
            channelNo_Sci = chan_Sci.channelNo
            branchName_Cer = f"FERS_Board{boardNo}_energyHG_{channelNo_Cer}_subtracted_calibrated_clipped"
            branchName_Sci = f"FERS_Board{boardNo}_energyHG_{channelNo_Sci}_subtracted_calibrated_clipped"
            FERSNames_Cer.append(branchName_Cer)
            FERSNames_Sci.append(branchName_Sci)
            tower_x.append(iTowerX*1.2)
            tower_y.append(iTowerY*1.6)
            if FERSBoard.Is3mm():
                size_x.append(1.2)
                size_y.append(0.4)
            else:
                size_x.append(1.2)
                size_y.append(1.6)
    tower_x = np.array(tower_x)
    tower_y = np.array(tower_y)
    size_x = np.array(size_x)
    size_y = np.array(size_y)

    l = rangeLow
    if rangeHigh > 0 and rangeHigh < rdf.Count().GetValue():
        h = rangeHigh
    else:
        h = rdf.Count().GetValue()
    caliFactors = {}
    for var,FERSNames,F_ref in zip(["Cer","Sci"],[FERSNames_Cer,FERSNames_Sci],[F_ref_Cer,F_ref_Sci]):
        FERSData = rdf.Range(l,h).AsNumpy(columns = FERSNames)
        FERSArrays = np.column_stack([FERSData[c] for c in FERSNames])
        if centers_x is None:
            centers_x = tower_x[np.argmax(FERSArrays,axis = -1)]
        if centers_y is None:
            centers_y = tower_y[np.argmax(FERSArrays,axis = -1)]
        beamEnergyDetected = beamEnergy * np.sum(erf_part(tower_x[None,:]-size_x[None,:]/2, tower_x[None,:]+size_x[None,:]/2, centers_x[:,None], w_x)*
                                                 erf_part(tower_y[None,:]-size_y[None,:]/2, tower_y[None,:]+size_y[None,:]/2, centers_y[:,None], w_y),
                                                 axis = -1)
        #beamEnergyArray = np.full(FERSArrays.shape[0],beamEnergy)
        factors,residuals = solve_least_squares_qr(FERSArrays, beamEnergyDetected,F_ref)
        print("var: ",var)
        print("np.sum(FERSArrays,axis=-1): ",np.sum(FERSArrays,axis=-1))
        print("beamEnergyTruth ", rdf.Range(l,h).AsNumpy(columns = ["energysum"])["energysum"])
        print("beamEnergyDetected", beamEnergyDetected)
        print("residuals truth to esitmated ", np.mean((rdf.Range(l,h).AsNumpy(columns = ["energysum"])["energysum"] - beamEnergyDetected) ** 2))
        print("residuals ",residuals/FERSArrays.shape[0])
        for FERSName,fac in zip(FERSNames,factors):
            caliFactors[FERSName] = fac
    return caliFactors

def testFERSBeamCalibration(rdf,FERSBoards,beamEnergy,beamEnergyTruth,caliFactors,rangeLow = 0, rangeHigh=-1):
    print("number of rows" , rdf.Count().GetValue())
    FERSNames_Cer = []
    FERSNames_Sci = []
    for _, FERSBoard in FERSBoards.items():
        boardNo = FERSBoard.boardNo
        for iTowerX, iTowerY in FERSBoard.GetListOfTowers():
            chan_Cer = FERSBoard.GetChannelByTower(
                iTowerX, iTowerY, isCer=True)
            chan_Sci = FERSBoard.GetChannelByTower(
                iTowerX, iTowerY, isCer=False)
            channelNo_Cer = chan_Cer.channelNo
            channelNo_Sci = chan_Sci.channelNo
            branchName_Cer = f"FERS_Board{boardNo}_energyHG_{channelNo_Cer}_subtracted_calibrated_clipped"
            branchName_Sci = f"FERS_Board{boardNo}_energyHG_{channelNo_Sci}_subtracted_calibrated_clipped"
            FERSNames_Cer.append(branchName_Cer)
            FERSNames_Sci.append(branchName_Sci)
    l = rangeLow
    if rangeHigh > 0 and rangeHigh < rdf.Count().GetValue():
        h = rangeHigh
    else:
        h = rdf.Count().GetValue()
    residuals = {}
    for var,FERSNames in zip(["Cer","Sci"],[FERSNames_Cer,FERSNames_Sci]):
        FERSData = rdf.Range(l,h).AsNumpy(columns = FERSNames)
        FERSArrays = np.column_stack([FERSData[c] for c in FERSNames])
        factors = np.array([caliFactors[col] for col in  FERSNames])
        beamEnergyArrayPredict = FERSArrays @ factors
        print("FERSArrays.shape",FERSArrays.shape)
        print("beamEnergyTruth ",beamEnergyTruth)
        print("beamEnergyArrayPredict ",beamEnergyArrayPredict)
        residuals[var] = np.mean((beamEnergyArrayPredict - beamEnergyTruth) ** 2)
        print("residuals truth to fit ", np.mean((beamEnergyArrayPredict - beamEnergyTruth) ** 2))
    return residuals["Cer"],residuals["Sci"]



def preProcessDRSBoards(rdf, debug=False):
    import re
    # Get the list of all branch names
    branches = [str(b) for b in rdf.GetColumnNames()]
    pattern = re.compile(r"DRS.*Group.*Channel.*")
    drs_branches = [b for b in branches if pattern.search(b)]
    stats = getBranchStats(rdf, drs_branches)
    print("DRS branches statistics:")
    for br, res in stats.items():
        print(f"{br}: mean = {res['mean'].GetValue():.4f}, "
              f"min = {res['min'].GetValue():.4f}, "
              f"max = {res['max'].GetValue():.4f}")
        stats[br] = {
            "mean": res['mean'].GetValue(),
            "min": res['min'].GetValue(),
            "max": res['max'].GetValue()
        }

    # Create an array of indices for DRS outputs
    rdf = rdf.Define("TS", "FillIndices(1024)")

    # find the baseline of each DRS channel
    # and subtract it from the DRS outputs
    for varname in drs_branches:
        if "triggerT" in varname or "deltaT" in varname or "peakA" in varname or "peakT" in varname: continue
        rdf = rdf.Define(
            f"{varname}_median",
            f"compute_median({varname})"
        )
        rdf = rdf.Define(
            f"{varname}_subtractMedian",
            f"{varname} - {varname}_median"
        )

    if debug:
        # define relative TS with respect to the StartIndexCell
        for varrname in drs_branches:
            # replace the string "Channel[0-9]+" with "StartIndexCell"
            var_StartIndexCell = re.sub(
                r"Channel[0-9]+", "StartIndexCell", varrname)
            rdf = rdf.Define(
                f"RTS_pos_{varrname}",
                f"(TS + {var_StartIndexCell}) % 1024"
            )
            rdf = rdf.Define(
                f"RTS_neg_{varrname}",
                f"((TS - {var_StartIndexCell}) % 1024 + 1024) % 1024"
            )

    return rdf


def calculateEnergySumFERS(rdf, FERSBoards, subtractPedestal=False, calibrate=False, clip=False, saturationCorrected = False,lowGain = False):
    """
    Calculate the Sci and Cer energy sum for FERS boards, per board and per event.
    """
    suffix = ""
    if subtractPedestal:
        suffix = "_subtracted"
    if calibrate:
        suffix += "_calibrated"
    if clip:
        suffix += "_clipped"
    if saturationCorrected:
        suffix += "_saturationcorrected"

    boardNos = []
    for _, FERSBoard in FERSBoards.items():
        boardNo = FERSBoard.boardNo
        boardNos.append(boardNo)

        channels_Cer = FERSBoard.GetCerChannels()
        channels_Sci = FERSBoard.GetSciChannels()
        if lowGain:
            string_CerEnergyLG = "+".join(
                chan.GetLGChannelName() + suffix for chan in channels_Cer
            )

            string_SciEnergyLG = "+".join(
               chan.GetLGChannelName() + suffix for chan in channels_Sci
            )

            rdf = rdf.Define(
                f"FERS_Board{boardNo}_CerEnergyLG" + suffix,
            f"(double)({string_CerEnergyLG})")

            rdf = rdf.Define(
                f"FERS_Board{boardNo}_SciEnergyLG" + suffix,
            f"(double)({string_SciEnergyLG})")
        else:
            string_CerEnergyHG = "+".join(
                chan.GetHGChannelName() + suffix for chan in channels_Cer
            )
            string_SciEnergyHG = "+".join(
                chan.GetHGChannelName() + suffix for chan in channels_Sci
            )
            rdf = rdf.Define(
                f"FERS_Board{boardNo}_CerEnergyHG" + suffix,
                f"(double)({string_CerEnergyHG})")
        
            rdf = rdf.Define(
                f"FERS_Board{boardNo}_SciEnergyHG" + suffix,
                f"(double)({string_SciEnergyHG})")
        

    if lowGain:
        string_CerEnergyLG_Total = "+".join(
            f"FERS_Board{boardNo}_CerEnergyLG" + suffix for boardNo in boardNos
        )
        string_SciEnergyLG_Total = "+".join(
            f"FERS_Board{boardNo}_SciEnergyLG" + suffix for boardNo in boardNos
        )
        rdf = rdf.Define("FERS_CerEnergyLG" + suffix,
            f"(double)({string_CerEnergyLG_Total})")
        rdf = rdf.Define("FERS_SciEnergyLG" + suffix,
            f"(double)({string_SciEnergyLG_Total})")
    else:
        string_CerEnergyHG_Total = "+".join(
            f"FERS_Board{boardNo}_CerEnergyHG" + suffix for boardNo in boardNos
        )
    
        string_SciEnergyHG_Total = "+".join(
            f"FERS_Board{boardNo}_SciEnergyHG" + suffix for boardNo in boardNos
        )
     
        rdf = rdf.Define("FERS_CerEnergyHG" + suffix,
                         f"(double)({string_CerEnergyHG_Total})")
     
        rdf = rdf.Define("FERS_SciEnergyHG" + suffix,
                         f"(double)({string_SciEnergyHG_Total})")
     

    return rdf


def getDRSSum(rdf, DRSBoards, TS_start=0, TS_end=400):
    # get the mean of DRS outputs per channel
    TS_start = int(TS_start)
    TS_end = int(TS_end)
    for _, DRSBoard in DRSBoards.items():
        for channel in DRSBoard:
            varname = channel.GetChannelName()
            rdf = rdf.Define(
                f"{varname}_subtractMedian_positive",
                f"clipToZero({varname}_subtractMedian)"
            )
            rdf = rdf.Define(
                f"{varname}_sum",
                f"SumRange({varname}_subtractMedian_positive, {TS_start}, {TS_end})"
            )
    return rdf


def getDRSPeakTS(rdf, DRSBoards, TS_start=0, TS_end=400, threshold=1.0):
    # get the peak TS of DRS outputs per channel
    TS_start = int(TS_start)
    TS_end = int(TS_end)
    for _, DRSBoard in DRSBoards.items():
        for channel in DRSBoard:
            varname = channel.GetChannelName()
            rdf = rdf.Define(
                f"{varname}_peakTS",
                f"ArgMaxRange({varname}_subtractMedian_positive, {TS_start}, {TS_end}, {threshold})"
            )
    return rdf


def getDRSPeak(rdf, DRSBoards, TS_start=0, TS_end=400):
    """
    Get the peak value of DRS outputs per channel.
    """
    TS_start = int(TS_start)
    TS_end = int(TS_end)
    for _, DRSBoard in DRSBoards.items():
        for channel in DRSBoard:
            varname = channel.GetChannelName()
            rdf = rdf.Define(
                f"{varname}_peak",
                f"MaxRange({varname}_subtractMedian_positive, {TS_start}, {TS_end})"
            )
    return rdf


def prepareDRSStats(rdf, DRSBoards, TS_start=0, TS_end=400, threshold=1.0):
    """
    Get the statistics of DRS outputs per channel.
    """
    rdf = getDRSSum(rdf, DRSBoards, TS_start, TS_end)
    rdf = getDRSPeakTS(rdf, DRSBoards, TS_start, TS_end, threshold)
    rdf = getDRSPeak(rdf, DRSBoards, TS_start, TS_end)
    return rdf


def loadRDF(runNumber, firstEvent=0, lastEvent=-1,filter=False):
    # Open the input ROOT file
    if filter:
        ifile = f"results/root/Run{runNumber}/filtered_track_FERS_nofanoutcorr.root"
    else:
        ifile = getDataFile(runNumber)
    infile = ROOT.TFile(ifile, "READ")
    rdf_org = ROOT.RDataFrame("EventTree", infile)
    #nevents = rdf_org.Count().GetValue()
    max_event_n = max(rdf_org.Take["unsigned int"]("event_n").GetValue())
    if lastEvent < 0 or lastEvent > max_event_n:
        lastEvent = max_event_n
    print(
        f"\033[94mfiltering events from {firstEvent} to {lastEvent} in run {runNumber}, total {lastEvent-firstEvent} events in the file.\033[0m")
    # Apply the event range filter
    rdf = rdf_org.Filter(f"event_n >= {firstEvent} && event_n < {lastEvent}")

    return rdf, rdf_org


def filterPrefireEvents(rdf, runNumber, TS=350):
    # use the hodo trigger to filter prefire events
    from utils.channel_map import buildHodoTriggerChannels
    trigger_names = buildHodoTriggerChannels(runNumber)
    if not trigger_names:
        return rdf, rdf  # No hodo trigger channels available for this run

    trigger_name_top, trigger_name_bottom = trigger_names[0], trigger_names[1]
    print(
        f"Filtering prefire events with TS >= {TS} using triggers: {trigger_name_top}, {trigger_name_bottom}")
    # index of the minimum value in the trigger channels
    rdf = rdf.Define(
        "TS_fired_up", f"ROOT::VecOps::ArgMin({trigger_name_top})")
    rdf = rdf.Define(
        "TS_fired_down", f"ROOT::VecOps::ArgMin({trigger_name_bottom})")

    rdf = rdf.Define(
        "NormalFired", f"(TS_fired_up >= {TS}) && (TS_fired_down >= {TS})")

    rdf_prefilter = rdf
    rdf = rdf.Filter("NormalFired == 1")

    return rdf, rdf_prefilter

def processDRSPeaks(rdf,drs_channels,trigger_channels,bl_start = 600, bl_end = 700,window_start = 0, window_end = 400,drs_amplified = []):
    for varname in drs_channels:
        #for i in range(10):
        #    print("event ",i, " branch ",varname)
        #    ROOT.fit_riseT(rdf.AsNumpy([varname])[varname][i],0.8)
        if varname in drs_amplified:
            rdf = rdf.Define(
                f"{varname}_peakT",
                f"fit_riseT({varname},0.5,2,1,{bl_start+200},{bl_end+200},{window_start},{window_end+200}, false,false,\"signal\",true)"
            )
            
        else:
            rdf = rdf.Define(
                f"{varname}_peakT",
                f"fit_riseT({varname},0.5,2,1,{bl_start},{bl_end},{window_start},{window_end}, false,false,\"signal\",false)"
            )
        #print("peak processed")
        #rdf = rdf.Define(
        #    f"{varname}_peakT",
        #    f"compute_peakT({varname})"
        #)
        rdf = rdf.Define(
            f"{varname}_peakA",
            f"compute_peakAmp({varname})"
        )

    #for trigname in trigger_channels:
    #    rdf = rdf.Define(
    #        f"{trigname}_triggerT",
    #        f"compute_triggerT({trigname})"
    #    )
    for trigname in trigger_channels:
        #trigdelay = findTimeReferenceDelay(trigname)
        #for i in range(10):
        #    print("event ",i, " branch ",trigname)
        #    ROOT.fit_riseT(rdf.AsNumpy([trigname])[trigname][i],0.5,2,-1,0, 200,  600, 900)
        rdf = rdf.Define(
            f"{trigname}_triggerT",
            #f"fit_riseT({trigname},0.5,2,-1,0, 200,  600, 900,false,false,\"trigger\")-{trigdelay}"
            f"fit_riseT({trigname},0.5,2,-1,0, 200,  600, 900,false,false,\"trigger\")"
        )
    for varname in drs_channels:
        triggername = mapDRSChannel2TriggerChannel(varname)
        print("DRS channel: ", varname, "mapped trigger: ",triggername)
        rdf = rdf.Define(
            f"{varname}_deltaT",
            f"{varname}_peakT - {triggername}_triggerT + 1024"
        )
    return rdf

def processHodoPeaks(rdf,hodo_pos_channels):
    for group, channels in hodo_pos_channels.items():
        for channel in channels:
            # find the minimum index of the pulse shape
            rdf = rdf.Define(f"{channel}_hodopeak_position",
                             f"ROOT::VecOps::ArgMin({channel}_subtractMedian)")
            rdf = rdf.Define(f"{channel}_hodopeak_value",
                             f"ROOT::VecOps::Min({channel}_subtractMedian)")
        rdf = rdf.Define(f"{group}_delta_hodopeak",
                            f"(int){channels[1]}_hodopeak_position - (int){channels[0]}_hodopeak_position")
    return rdf

def denoiseDRS(sig):
    savgol = savgol_filter(sig, window_length=3, polyorder=1)
    #fft_vals = fft(sig)
    # Filter out high frequencies (noise)
    #fft_vals[len(fft_vals)//2:] = 0
    #denoised_fourier = ifft(fft_vals)
    return savgol