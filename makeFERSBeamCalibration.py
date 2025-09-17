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
from runconfig import runNumber, firstEvent, lastEvent, beamEnergy
from sklearn.linear_model import LinearRegression
from selections.selections import vetoMuonCounter, applyUpstreamVeto, PSDSelection
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
def LinearFitWithUncertainty(X,y):
    reg = LinearRegression(fit_intercept=False,positive=True).fit(X, y)
    # Residuals and variance
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

#runNumbers_test = [1267,1260]
#energies_test = [40,60]
#firstEvents_test = [0,0]
#lastEvents_test = [21000,44248]

#runNumbers_calibrate = [1234,1261]
#energies_calibrate = [80,30]
#firstEvents_calibrate = [0,0]
#lastEvents_calibrate = [21426,24803]



#runNumbers_calibrate = [1267,1260]
#energies_calibrate = [40,60]
#firstEvents_calibrate = [0,0]
#lastEvents_calibrate = [21000,44248]

#runNumbers_test = [1234,1261]
#energies_test = [80,30]
#firstEvents_test = [0,0]
#lastEvents_test = [21426,24803]



#runNumbers_calibrate = [1261,1267,1234]
#energies_calibrate = [30,40,80]
#firstEvents_calibrate = [0,0,0]
#lastEvents_calibrate = [24803,21000,21426]

#runNumbers_test = [1260]
#energies_test = [60]
#firstEvents_test = [0]
#lastEvents_test = [44248]



#runNumbers_calibrate = [1261,1260,1234]
#energies_calibrate = [30,60,80]
#firstEvents_calibrate = [0,0,0]
#lastEvents_calibrate = [24803,44248,21426]

#runNumbers_test = [1267]
#energies_test = [40]
#firstEvents_test = [0]
#lastEvents_test = [21000]



#runNumbers_calibrate = [1261,1267,1260]
#energies_calibrate = [30,40,60]
#firstEvents_calibrate = [0,0,0]
#lastEvents_calibrate = [24803,21000,44248]

#runNumbers_test = [1234]
#energies_test = [80]
#firstEvents_test = [0]
#lastEvents_test = [21426]



#runNumbers_calibrate = [1267]
#energies_calibrate = [40]
#firstEvents_calibrate = [0]
#lastEvents_calibrate = [20000]


#runNumbers_test = [1234]
#energies_test = [80]
#firstEvents_test = [0]
#lastEvents_test = [20000]


#runNumbers_test = [1261,1260]
#energies_test = [30,60]
#firstEvents_test = [0,0]
#lastEvents_test = [24803,44248]

#runNumbers_calibrate = [1234,1231,1235,1232,1233,1236,1237,1238,1267,1268,1270,1271,1272,1273,1274,1275]
#energies_calibrate = [80,80,80,80,80,80,80,80,40,40,40,40,40,40,40,40]
#firstEvents_calibrate = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#lastEvents_calibrate = [21426,20623,19277,18276,4965,27378,20684,20593,21444,21585,20235,23264,20092,20034,20049,14094]


#runNumbers_calibrate = [1234,1231,1235,1232,1233,1236,1237,1238]
#energies_calibrate = [80,80,80,80,80,80,80,80]
#firstEvents_calibrate = [0,0,0,0,0,0,0,0]
#lastEvents_calibrate = [21426,20623,19277,18276,4965,27378,20684,20593]

#runNumbers_test = [1267,1268,1270,1271,1272,1273,1274,1275]
#energies_test = [40,40,40,40,40,40,40,40]
#firstEvents_test = [0,0,0,0,0,0,0,0]
#lastEvents_test = [21444,21585,20235,23264,20092,20034,20049,14094]

runNumbers_calibrate = [1234,1231,1235,1232,1236,1237,1238,1239,1240,1241,1243,1245,1246,1248,1249,1250,1252]
energies_calibrate = [80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80]
firstEvents_calibrate = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#lastEvents_calibrate = [21426,20623,19277,18276,27378,20684,20593,20410,20300,22037,20061,20057,20043,20100,20231,20287,20111]
lastEvents_calibrate = [1000]*17


runNumbers_test = [1267,1268,1270,1271,1272,1273,1274,1275]
energies_test = [40,40,40,40,40,40,40,40]
firstEvents_test = [0,0,0,0,0,0,0,0]
lastEvents_test = [21444,21585,20235,23264,20092,20034,20049,14094,20410]


boardNo_calibrate = [7,8,2,3,4,10,11,12,0,1,5,6,9,13]

#runNumbers_calibrate = [1274,1238,1259]
#energies_calibrate = [40,80,0]
#firstEvents_calibrate = [10000,8000,1000]
#lastEvents_calibrate = [20000,20000,2000]

unc_from_random = True

random_seed = 3
random_drop_proportion = 0.1
n_random = 20

dead_channels = {
    7: [35,43,47,51],
    8: [40,44]
}

sys.path.append("CMSPLOTS")  # noqa
ROOT.ROOT.EnableImplicitMT(10)
ROOT.gROOT.SetBatch(True)  # Disable interactive mode for batch processing
ROOT.gSystem.Load("utils/functions_cc.so")  # Load the compiled C++ functions
runNumber_str_calibrate = ', '.join([str(runNumber) for runNumber in runNumbers_calibrate])
runNumber_calibrate_suffix = "_runNumber"+"_".join([str(runNumber) for runNumber in runNumbers_calibrate])
runNumber_test_suffix = "_runNumber"+"_".join([str(runNumber) for runNumber in runNumbers_test])
boardNo_calibrate_suffix = "_board"+"_".join([str(boardNo) for boardNo in boardNo_calibrate])
energy_calibrate_suffix = "_energy"+"_".join([str(energy) for energy in energies_calibrate])
energy_test_suffix = "_energy"+"_".join([str(energy) for energy in energies_test])
#random_drop_suffix = ""
#if random_drop:
#    random_drop_suffix = f"_random_drop_percentage{str(int(random_drop_proportion*100))}_seed{random_seed}"
plot_calibrate_suffix = runNumber_calibrate_suffix + energy_calibrate_suffix + boardNo_calibrate_suffix
plot_test_suffix = runNumber_test_suffix + energy_test_suffix + boardNo_calibrate_suffix


runPedestal = 1259
with open(f"results/root/Run{runPedestal}/testbeam_pedestal.json", "r") as f:
    pedestal = json.load(f)
HGLG_json_dir = f"results/root/positroncali/"
with open(f"{HGLG_json_dir}/testbeam_FERS_HG_to_LG_factors.json", "r") as f:
    factors_HG_to_LG = json.load(f)

xmax = 14
xmin = -14
ymax = 10
ymin = -10
W_ref = 1400
H_ref = 1600
random.seed(random_seed)

FERSBoards = buildFERSBoards(run=runPedestal)
FERSBoards_calibrate = {}
FERSChannels_calibrate_Cer = []
hist2d_Cer_3mm = None
hist2d_Cer_6mm = None
hist2d_Cer_unc_3mm = None
hist2d_Cer_unc_6mm = None
hist2d_Cer_3mm_calibrate_before = []
hist2d_Cer_3mm_calibrate_after = []
hist2d_Cer_6mm_calibrate_before = []
hist2d_Cer_6mm_calibrate_after = []
hist2d_Cer_3mm_test_before = []
hist2d_Cer_3mm_test_after = []
hist2d_Cer_6mm_test_before = []
hist2d_Cer_6mm_test_after = []
iTowerX_list = []
iTowerY_list = []
iTower_list_is3mm = []
if len(set([7,8]).intersection(boardNo_calibrate)) > 0:
    hist2d_Cer_3mm = ROOT.TH2F(
                f"convertion_Cer_3mm",
                f"FERS response factors;X;Y (Cherenkov)",
                int(xmax - xmin), xmin, xmax,
                int(ymax - ymin) * 4, ymin, ymax
            )
    hist2d_Cer_3mm.SetMarkerSize(0.55)
    hist2d_Cer_unc_3mm = ROOT.TH2F(
                f"convertion_Cer_unc_3mm",
                f"FERS response factors uncertainty;X;Y (Cherenkov)",
                int(xmax - xmin), xmin, xmax,
                int(ymax - ymin) * 4, ymin, ymax
            )
    hist2d_Cer_unc_3mm.SetMarkerSize(0.55)
    if unc_from_random:
        hist2d_Cer_unc_toy_3mm = ROOT.TH2F(
                f"convertion_Cer_unc_toy_3mm",
                f"FERS response factors uncertainty (toys);X;Y (Cherenkov)",
                int(xmax - xmin), xmin, xmax,
                int(ymax - ymin) * 4, ymin, ymax
            )
        hist2d_Cer_unc_toy_3mm.SetMarkerSize(0.55)
    for runNumber in runNumbers_calibrate:
        hist2d_Cer_3mm_calibrate_before.append(ROOT.TH2F(
                    f"ADC_Cer_3mm_for_fit_run{runNumber}",
                    f"FERS average ADC counts run{runNumber};X;Y (Cherenkov)",
                    int(xmax - xmin), xmin, xmax,
                    int(ymax - ymin) * 4, ymin, ymax
                ))
        hist2d_Cer_3mm_calibrate_before[-1].SetMarkerSize(0.55)
        hist2d_Cer_3mm_calibrate_after.append(ROOT.TH2F(
                    f"energy_Cer_3mm_for_fit_run{runNumber}",
                    f"FERS average energy [GeV] run{runNumber};X;Y (Cherenkov)",
                    int(xmax - xmin), xmin, xmax,
                    int(ymax - ymin) * 4, ymin, ymax
                ))
        hist2d_Cer_3mm_calibrate_after[-1].SetMarkerSize(0.55)

    for runNumber in runNumbers_test:
        hist2d_Cer_3mm_test_before.append(ROOT.TH2F(
                    f"ADC_Cer_3mm_for_test_run{runNumber}",
                    f"FERS average ADC counts run{runNumber};X;Y (Cherenkov)",
                    int(xmax - xmin), xmin, xmax,
                    int(ymax - ymin) * 4, ymin, ymax
                ))
        hist2d_Cer_3mm_test_before[-1].SetMarkerSize(0.55)
        hist2d_Cer_3mm_test_after.append(ROOT.TH2F(
                    f"energy_Cer_3mm_for_test_run{runNumber}",
                    f"FERS average energy [GeV] run{runNumber};X;Y (Cherenkov)",
                    int(xmax - xmin), xmin, xmax,
                    int(ymax - ymin) * 4, ymin, ymax
                ))
        hist2d_Cer_3mm_test_after[-1].SetMarkerSize(0.55)


if len(set([0,1,2,3,4,5,6,9,10,11,12,13]).intersection(boardNo_calibrate)) > 0:
    hist2d_Cer_6mm = ROOT.TH2F(
                f"convertion_Cer_6mm",
                f"FERS response factors;X;Y (Cherenkov)",
                int(xmax - xmin), xmin, xmax,
                int(ymax - ymin), ymin, ymax
            )
    hist2d_Cer_6mm.SetMarkerSize(0.55)
    hist2d_Cer_unc_6mm = ROOT.TH2F(
                f"convertion_Cer_unc_6mm",
                f"FERS response factors uncertainty;X;Y (Cherenkov)",
                int(xmax - xmin), xmin, xmax,
                int(ymax - ymin), ymin, ymax
            )
    hist2d_Cer_unc_6mm.SetMarkerSize(0.55)
    if unc_from_random:
        hist2d_Cer_unc_toy_6mm = ROOT.TH2F(
                f"convertion_Cer_unc_toy_6mm",
                f"FERS response factors uncertainty (toys);X;Y (Cherenkov)",
                int(xmax - xmin), xmin, xmax,
                int(ymax - ymin), ymin, ymax
            )
        hist2d_Cer_unc_toy_6mm.SetMarkerSize(0.55)
    for runNumber in runNumbers_calibrate:
        hist2d_Cer_6mm_calibrate_before.append(ROOT.TH2F(
                    f"ADC_Cer_6mm_for_fit_run{runNumber}",
                    f"FERS average ADC counts run{runNumber};X;Y (Cherenkov)",
                    int(xmax - xmin), xmin, xmax,
                    int(ymax - ymin), ymin, ymax
                ))
        hist2d_Cer_6mm_calibrate_before[-1].SetMarkerSize(0.55)
        hist2d_Cer_6mm_calibrate_after.append(ROOT.TH2F(
                    f"energy_Cer_6mm_for_fit_run{runNumber}",
                    f"FERS average energy [GeV] run{runNumber};X;Y (Cherenkov)",
                    int(xmax - xmin), xmin, xmax,
                    int(ymax - ymin), ymin, ymax
                ))
        hist2d_Cer_6mm_calibrate_after[-1].SetMarkerSize(0.55)
    for runNumber in runNumbers_test:
        hist2d_Cer_6mm_test_before.append(ROOT.TH2F(
                    f"ADC_Cer_6mm_for_test_run{runNumber}",
                    f"FERS average ADC counts run{runNumber};X;Y (Cherenkov)",
                    int(xmax - xmin), xmin, xmax,
                    int(ymax - ymin), ymin, ymax
                ))
        hist2d_Cer_6mm_test_before[-1].SetMarkerSize(0.55)
        hist2d_Cer_6mm_test_after.append(ROOT.TH2F(
                    f"energy_Cer_6mm_for_test_run{runNumber}",
                    f"FERS average energy [GeV] run{runNumber};X;Y (Cherenkov)",
                    int(xmax - xmin), xmin, xmax,
                    int(ymax - ymin), ymin, ymax
                ))
        hist2d_Cer_6mm_test_after[-1].SetMarkerSize(0.55)

for boardNo in boardNo_calibrate:
    FERSBoards_calibrate[f"Board{boardNo}"] = FERSBoards[f"Board{boardNo}"]
    for iTowerX, iTowerY in FERSBoards[f"Board{boardNo}"].GetListOfTowers():
        chan_Cer = FERSBoards[f"Board{boardNo}"].GetChannelByTower(
            iTowerX, iTowerY, isCer=True)
        channelNo_Cer = chan_Cer.channelNo
        if boardNo in dead_channels.keys() and channelNo_Cer in dead_channels[boardNo]: continue
        FERSChannels_calibrate_Cer.append(f"FERS_Board{boardNo}_energyHG_{channelNo_Cer}_subtracted_saturationcorrected")
        iTower_list_is3mm.append(FERSBoards[f"Board{boardNo}"].Is3mm())
        iTowerX_list.append(iTowerX)
        iTowerY_list.append(iTowerY)


FERSArrays_calibrate = []
EnergyArrays_calibrate = []
EnergyArrays_mean_response_calibrate = []
nEvents_calibrate = []
channel_mean_calibrate_before = []
for runNumber, energy, firstEvent, lastEvent in zip(runNumbers_calibrate,energies_calibrate,firstEvents_calibrate,lastEvents_calibrate):
    rdf, _ = loadRDF(runNumber, firstEvent, lastEvent)
    rdf = preProcessDRSBoards(rdf)
    rdf, rdf_prefilter = vetoMuonCounter(rdf, TSmin=400, TSmax=700, cut=-30)

    rdf, rdf_filterveto = applyUpstreamVeto(rdf, runNumber)
    rdf, rdf_psd = PSDSelection(rdf, runNumber, isHadron=False)
    rdf = vectorizeFERS(rdf, FERSBoards)
    rdf = subtractFERSBeamPedestal(rdf, FERSBoards, pedestal)
    rdf = correctFERSSaturation(rdf,FERSBoards, factors_HG_to_LG)
    FERSBoards_energysum_Cer_str = ""
    for boardNo in boardNo_calibrate:
        FERSBoards_energysum_Cer_str += f"FERS_Board{boardNo}_CerEnergyHG_subtracted_saturationcorrected + "
    FERSBoards_energysum_Cer_str = FERSBoards_energysum_Cer_str[:-3]
    rdf = calculateEnergySumFERS(rdf, FERSBoards, subtractPedestal=True, calibrate=False, clip=False, saturationCorrected=True)
    #if runNumber != runPedestal:
    #    with open(f"results/root/Run{runNumber}/testbeam_energy_sum_fit.json", "r") as f:
    #        fit_energy_sum = json.load(f)
    #    energy_sum_Cer_low = fit_energy_sum["mean_s_Cer"] - 1.5*fit_energy_sum["width_s_Cer"]
    #    energy_sum_Cer_high = fit_energy_sum["mean_s_Cer"] + 1.5*fit_energy_sum["width_s_Cer"]
    #    rdf =  rdf.Filter(f"(FERS_CerEnergyHG_subtracted_saturationcorrected > {energy_sum_Cer_low}) && (FERS_CerEnergyHG_subtracted_saturationcorrected < {energy_sum_Cer_high}) && ({FERSBoards_energysum_Cer_str} > 0.85 * FERS_CerEnergyHG_subtracted_saturationcorrected)","filter Cer energy")
    rdf =  rdf.Filter(f"(FERS_CerEnergyHG_subtracted_saturationcorrected > 20000)","filter Cer energy")
    FERSData = rdf.AsNumpy(columns = FERSChannels_calibrate_Cer + ["FERS_CerEnergyHG_subtracted_saturationcorrected"])
    FERSArrays = np.column_stack([FERSData[c] for c in FERSChannels_calibrate_Cer])
    print("after filtering ",len(FERSArrays), " events")
    #EnergyArrays_mean_response = FERSData["FERS_CerEnergyHG_subtracted_saturationcorrected"]/fit_energy_sum["mean_response_s_Cer"]
    mean_response = np.mean(FERSData["FERS_CerEnergyHG_subtracted_saturationcorrected"])/energy
    EnergyArrays_mean_response = FERSData["FERS_CerEnergyHG_subtracted_saturationcorrected"]/mean_response
    #if random_drop:
    #    indices_select = random.sample(range(len(FERSArrays)), int(len(FERSArrays) * (1-random_drop_proportion)))
    #    FERSArrays = FERSArrays[indices_select,:]
    #    EnergyArrays_mean_response = EnergyArrays_mean_response[indices_select]
    #    print("after random drop ",len(FERSArrays), " events")
    
    nEvents_calibrate.append(len(FERSArrays))
    FERSArrays_calibrate.append(FERSArrays)
    channel_mean_calibrate_before.append(np.mean(FERSArrays,axis=0))
    EnergyArrays_calibrate.append(np.full(FERSArrays.shape[0],energy))
    EnergyArrays_mean_response_calibrate.append(EnergyArrays_mean_response)
    del rdf
    del FERSData

FERSArrays_calibrate = np.vstack(FERSArrays_calibrate)
EnergyArrays_calibrate = np.concatenate(EnergyArrays_calibrate)
EnergyArrays_mean_response_calibrate = np.concatenate(EnergyArrays_mean_response_calibrate)

print("Calibrating on ",len(FERSArrays_calibrate), " events")

#reg = LinearRegression(fit_intercept=False,positive=True).fit(FERSArrays_calibrate, EnergyArrays_calibrate)
reg, unc, cov = LinearFitWithUncertainty(FERSArrays_calibrate, EnergyArrays_calibrate)
df_cov = pd.DataFrame(cov,index = FERSChannels_calibrate_Cer, columns = FERSChannels_calibrate_Cer)
if unc_from_random:
    coef_toys = []
    for itoy in range(n_random):
        indices_select = random.sample(range(len(FERSArrays_calibrate)), int(len(FERSArrays_calibrate) * (1-random_drop_proportion)))
        FERSArrays_calibrate_toy = FERSArrays_calibrate[indices_select,:]
        EnergyArrays_calibrate_toy = EnergyArrays_calibrate[indices_select,:]
        reg_toy = LinearRegression(fit_intercept=False,positive=True).fit(FERSArrays_calibrate_toy, EnergyArrays_calibrate_toy)
        coef_toys.append(reg_toy.coef_)
    coef_toys = np.vstack(coef_toys)
    cov_toys = np.cov(coef_toys.T)
    unc_toys = np.sqrt(np.diag(cov_toys))
    df_cov_toys = pd.DataFrame(cov_toys,index = FERSChannels_calibrate_Cer, columns = FERSChannels_calibrate_Cer)
response_dic = {}
for ch,coef, coef_unc in zip(FERSChannels_calibrate_Cer,reg.coef_,unc):
    response_dic[ch] = {
        "response": coef,
        "uncertainty": coef_unc
    }
if unc_from_random:
    for ch, coef_unc in zip(FERSChannels_calibrate_Cer,unc_toys):
        response_dic[ch]["uncertainty_toys"] = coef_unc
    
print("coef ", reg.coef_)
print("intercept ", reg.intercept_)
print("unc ",unc)
print("EnergyArrays_calibrate ",EnergyArrays_calibrate)
print("Predict ",reg.predict(FERSArrays_calibrate))
print("mean square of prediction error ",np.mean((reg.predict(FERSArrays_calibrate)-EnergyArrays_calibrate)**2))
print("mean square of prediction error from mean response ",np.mean((EnergyArrays_mean_response_calibrate-EnergyArrays_calibrate)**2))
for icoef, (iTowerX, iTowerY, iTower_is3mm, coef, coef_unc) in enumerate(zip(iTowerX_list,iTowerY_list,iTower_list_is3mm,reg.coef_,unc)):
    if iTower_is3mm:
        hist2d_Cer_3mm.Fill(iTowerX, iTowerY, coef*1000)
        hist2d_Cer_unc_3mm.Fill(iTowerX, iTowerY, coef_unc*1000)
        if unc_from_random:
            hist2d_Cer_unc_toy_3mm.Fill(iTowerX, iTowerY, unc_toys[icoef]*1000)
        for i in range(len(runNumbers_calibrate)):
            hist2d_Cer_3mm_calibrate_before[i].Fill(iTowerX, iTowerY, channel_mean_calibrate_before[i][icoef])
            hist2d_Cer_3mm_calibrate_after[i].Fill(iTowerX, iTowerY, channel_mean_calibrate_before[i][icoef]*coef)
    else:
        hist2d_Cer_6mm.Fill(iTowerX, iTowerY, coef*1000)
        hist2d_Cer_unc_6mm.Fill(iTowerX, iTowerY, coef_unc*1000)
        if unc_from_random:
            hist2d_Cer_unc_toy_6mm.Fill(iTowerX, iTowerY, unc_toys[icoef]*1000)
        for i in range(len(runNumbers_calibrate)):
            hist2d_Cer_6mm_calibrate_before[i].Fill(iTowerX, iTowerY, channel_mean_calibrate_before[i][icoef])
            hist2d_Cer_6mm_calibrate_after[i].Fill(iTowerX, iTowerY, channel_mean_calibrate_before[i][icoef]*coef)

hist_list = []
if hist2d_Cer_3mm is not None:
    hist_list.append(hist2d_Cer_3mm)
if hist2d_Cer_6mm is not None:
    hist_list.append(hist2d_Cer_6mm)

hist_unc_list = []
if hist2d_Cer_unc_3mm is not None:
    hist_unc_list.append(hist2d_Cer_unc_3mm)
if hist2d_Cer_unc_6mm is not None:
    hist_unc_list.append(hist2d_Cer_unc_6mm)

if unc_from_random:
    hist_unc_toy_list = []
    if hist2d_Cer_unc_toy_3mm is not None:
        hist_unc_toy_list.append(hist2d_Cer_unc_toy_3mm)
    if hist2d_Cer_unc_toy_6mm is not None:
        hist_unc_toy_list.append(hist2d_Cer_unc_toy_6mm)
outdir_response = "FERS_response/"
DrawHistos(hist_list, f"", xmin, xmax, "iX",
                       ymin, ymax, "iY", f"FERS_response{plot_calibrate_suffix}",
                       dology=False, drawoptions=["COL,text"]*len(hist_list),
                       zmin=0.0, zmax=max(reg.coef_*1000)*1.01, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText="Cer (response*1000)", runNumber=runNumber_str_calibrate,
                       outdir=outdir_response,nTextDigits=2)

DrawHistos(hist_unc_list, f"", xmin, xmax, "iX",
                       ymin, ymax, "iY", f"FERS_response_unc{plot_calibrate_suffix}",
                       dology=False, drawoptions=["COL,text"]*len(hist_unc_list),
                       zmin=0.0, zmax=max(unc*1000)*1.01, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText="Cer (response*1000)", runNumber=runNumber_str_calibrate,
                       outdir=outdir_response,nTextDigits=2)
if unc_from_random:
    DrawHistos(hist_unc_toy_list, f"", xmin, xmax, "iX",
                       ymin, ymax, "iY", f"FERS_response_unc_toy{plot_calibrate_suffix}",
                       dology=False, drawoptions=["COL,text"]*len(hist_unc_toy_list),
                       zmin=0.0, zmax=max(unc_toys*1000)*1.01, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText="Cer (response*1000)", runNumber=runNumber_str_calibrate,
                       outdir=outdir_response,nTextDigits=2)               
with open(f"{outdir_response}/FERS_response{plot_calibrate_suffix}.json", "w") as f:
    json.dump(response_dic, f)
df_cov.to_csv(f"{outdir_response}/FERS_response_cov{plot_calibrate_suffix}.csv")
if unc_from_random:
    df_cov_toys.to_csv(f"{outdir_response}/FERS_response_cov_toys{plot_calibrate_suffix}.csv")

outdir_calibrate = "FERS_calibrate/"
for irun,runNumber in enumerate(runNumbers_calibrate):
    hist_list = []
    if len(hist2d_Cer_3mm_calibrate_before)>0:
        hist_list.append(hist2d_Cer_3mm_calibrate_before[irun])
    if len(hist2d_Cer_6mm_calibrate_before)>0:
        hist_list.append(hist2d_Cer_6mm_calibrate_before[irun])
    DrawHistos(hist_list, f"", xmin, xmax, "iX",
                       ymin, ymax, "iY", f"FERS_average_ADC_run{runNumber}",
                       dology=False, drawoptions=["COL,text"]*len(hist_list),
                       zmin=0.0, zmax=max(channel_mean_calibrate_before[irun])*1.01, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText="Cer", runNumber=runNumber,
                       outdir=outdir_calibrate,nTextDigits=0)
    hist_list = []
    if len(hist2d_Cer_3mm_calibrate_after)>0:
        hist_list.append(hist2d_Cer_3mm_calibrate_after[irun])
    if len(hist2d_Cer_6mm_calibrate_after)>0:
        hist_list.append(hist2d_Cer_6mm_calibrate_after[irun])
    DrawHistos(hist_list, f"", xmin, xmax, "iX",
                       ymin, ymax, "iY", f"FERS_average_energy_fit{plot_calibrate_suffix}_run{runNumber}",
                       dology=False, drawoptions=["COL,text"]*len(hist_list),
                       zmin=0.0, zmax=max(channel_mean_calibrate_before[irun]*reg.coef_)*1.01, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText="Cer", runNumber=runNumber,
                       outdir=outdir_calibrate,nTextDigits=1)

bins = 40
iEvent = 0
for runNumber,nEvents, energy in zip(runNumbers_calibrate,nEvents_calibrate,energies_calibrate):
    range_min, range_max = energy-40, energy+40
    hist_fit_cal, bin_edges = np.histogram(reg.predict(FERSArrays_calibrate[iEvent:iEvent+nEvents]), bins=bins, range=(range_min, range_max))
    hist_compare_cal, _ = np.histogram(EnergyArrays_mean_response_calibrate[iEvent:iEvent+nEvents], bins=bins, range=(range_min, range_max))
    iEvent = iEvent + nEvents
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    plt.figure(figsize=(8,6))
    plt.step(bin_centers, hist_fit_cal, where='mid', label="fit per-channel response")
    plt.step(bin_centers, hist_compare_cal, where='mid', label="from average response")

    # Labels and legend
    plt.xlabel("Energy [GeV]")
    plt.ylabel("Counts")
    plt.legend()
    plt.title(f"Comparison of Responses, Run{runNumber}, E={energy}")

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
    rdf, _ = loadRDF(runNumber, firstEvent, lastEvent)
    rdf = preProcessDRSBoards(rdf)
    rdf, rdf_prefilter = vetoMuonCounter(rdf, TSmin=400, TSmax=700, cut=-30)

    rdf, rdf_filterveto = applyUpstreamVeto(rdf, runNumber)
    rdf, rdf_psd = PSDSelection(rdf, runNumber, isHadron=False)
    rdf = vectorizeFERS(rdf, FERSBoards)
    rdf = subtractFERSBeamPedestal(rdf, FERSBoards, pedestal)
    rdf = correctFERSSaturation(rdf,FERSBoards, factors_HG_to_LG)
    FERSBoards_energysum_Cer_str = ""
    for boardNo in boardNo_calibrate:
        FERSBoards_energysum_Cer_str += f"FERS_Board{boardNo}_CerEnergyHG_subtracted_saturationcorrected + "
    FERSBoards_energysum_Cer_str = FERSBoards_energysum_Cer_str[:-3]
    rdf = calculateEnergySumFERS(rdf, FERSBoards, subtractPedestal=True, calibrate=False, clip=False, saturationCorrected=True)
    #if runNumber != runPedestal:
    #    with open(f"results/root/Run{runNumber}/testbeam_energy_sum_fit.json", "r") as f:
    #        fit_energy_sum = json.load(f)
    #    energy_sum_Cer_low = fit_energy_sum["mean_s_Cer"] - 1.5*fit_energy_sum["width_s_Cer"]
    #    energy_sum_Cer_high = fit_energy_sum["mean_s_Cer"] + 1.5*fit_energy_sum["width_s_Cer"]
    #    rdf =  rdf.Filter(f"(FERS_CerEnergyHG_subtracted_saturationcorrected > {energy_sum_Cer_low}) && (FERS_CerEnergyHG_subtracted_saturationcorrected < {energy_sum_Cer_high}) && ({FERSBoards_energysum_Cer_str} > 0.85 * FERS_CerEnergyHG_subtracted_saturationcorrected)","filter Cer energy")
    rdf =  rdf.Filter(f"(FERS_CerEnergyHG_subtracted_saturationcorrected > 20000)","filter Cer energy")
    FERSData = rdf.AsNumpy(columns = FERSChannels_calibrate_Cer + ["FERS_CerEnergyHG_subtracted_saturationcorrected"])
    FERSArrays = np.column_stack([FERSData[c] for c in FERSChannels_calibrate_Cer])
    mean_response = np.mean(FERSData["FERS_CerEnergyHG_subtracted_saturationcorrected"])/energy
    EnergyArrays_mean_response = FERSData["FERS_CerEnergyHG_subtracted_saturationcorrected"]/mean_response
    #EnergyArrays_mean_response = FERSData["FERS_CerEnergyHG_subtracted_saturationcorrected"]/fit_energy_sum["mean_response_s_Cer"]
    print("after filtering the test set ",len(FERSArrays), " events")
    nEvents_test.append(len(FERSArrays))
    FERSArrays_test.append(FERSArrays)
    channel_mean_test_before.append(np.mean(FERSArrays,axis=0))
    EnergyArrays_test.append(np.full(FERSArrays.shape[0],energy))
    EnergyArrays_mean_response_test.append(EnergyArrays_mean_response)
    del rdf
    del FERSData

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
            hist2d_Cer_3mm_test_before[i].Fill(iTowerX, iTowerY, channel_mean_test_before[i][icoef])
            hist2d_Cer_3mm_test_after[i].Fill(iTowerX, iTowerY, channel_mean_test_before[i][icoef]*coef)
    else:
        for i in range(len(runNumbers_test)):
            hist2d_Cer_6mm_test_before[i].Fill(iTowerX, iTowerY, channel_mean_test_before[i][icoef])
            hist2d_Cer_6mm_test_after[i].Fill(iTowerX, iTowerY, channel_mean_test_before[i][icoef]*coef)


for irun,runNumber in enumerate(runNumbers_test):
    hist_list = []
    if len(hist2d_Cer_3mm_test_before)>0:
        hist_list.append(hist2d_Cer_3mm_test_before[irun])
    if len(hist2d_Cer_6mm_test_before)>0:
        hist_list.append(hist2d_Cer_6mm_test_before[irun])
    DrawHistos(hist_list, f"", xmin, xmax, "iX",
                       ymin, ymax, "iY", f"FERS_average_ADC_run{runNumber}",
                       dology=False, drawoptions=["COL,text"]*len(hist_list),
                       zmin=0.0, zmax=max(channel_mean_test_before[irun])*1.01, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText="Cer", runNumber=runNumber,
                       outdir=outdir_calibrate,nTextDigits=0)
    hist_list = []
    if len(hist2d_Cer_3mm_test_after)>0:
        hist_list.append(hist2d_Cer_3mm_test_after[irun])
    if len(hist2d_Cer_6mm_test_after)>0:
        hist_list.append(hist2d_Cer_6mm_test_after[irun])
    DrawHistos(hist_list, f"", xmin, xmax, "iX",
                       ymin, ymax, "iY", f"FERS_average_energy_fit{plot_calibrate_suffix}_run{runNumber}",
                       dology=False, drawoptions=["COL,text"]*len(hist_list),
                       zmin=0.0, zmax=max(channel_mean_test_before[irun]*reg.coef_)*1.01, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText="Cer", runNumber=runNumber,
                       outdir=outdir_calibrate,nTextDigits=1)




iEvent = 0
for runNumber,nEvents, energy in zip(runNumbers_test,nEvents_test,energies_test):
    range_min, range_max = energy-40, energy+40
    hist_fit_test, bin_edges = np.histogram(reg.predict(FERSArrays_test[iEvent:iEvent+nEvents]), bins=bins, range=(range_min, range_max))
    hist_compare_test, _ = np.histogram(EnergyArrays_mean_response_test[iEvent:iEvent+nEvents], bins=bins, range=(range_min, range_max))
    iEvent = iEvent + nEvents
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    plt.figure(figsize=(8,6))
    plt.step(bin_centers, hist_fit_test, where='mid', label="test per-channel response")
    plt.step(bin_centers, hist_compare_test, where='mid', label="from average response")

    # Labels and legend
    plt.xlabel("Energy [GeV]")
    plt.ylabel("Counts")
    plt.legend()
    plt.title(f"Comparison of Responses, Run{runNumber}, E={energy}")

    # Save to PNG
    plt.tight_layout()
    plt.savefig(f"test_comparison_calibrate{plot_calibrate_suffix}_plot_run{runNumber}_energy{energy}.png", dpi=300)
    plt.close()