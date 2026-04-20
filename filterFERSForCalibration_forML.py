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
from sklearn.linear_model import LinearRegression
from selections.selections import vetoMuonCounter, applyUpstreamVeto, PSDSelection,applyCC1Selection,applyCC2Selection,applyCC3Selection,applyPSDSelection
from selections.selections import vetoMuonCounter, applyUpstreamVeto, PSDSelection,applyCC1Selection,applyCC2Selection,applyCC3Selection,applyPSDSelection,getCCselection
import json
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--runNumbers", type=int,nargs="+",
                    help="give a list of runNumber")
parser.add_argument("--energies", type=int,nargs="+",
                    help="give a list of energies")
parser.add_argument('--muoncounter',
                    action='store_true')
parser.add_argument('--holeveto',
                    action='store_true')
parser.add_argument('--PSD',
                    action='store_true')
parser.add_argument('--CC1',
                    action='store_true')
parser.add_argument('--CC2',
                    action='store_true')
parser.add_argument('--CC3',
                    action='store_true')
parser.add_argument('--particle',type=str,
                    help = 'particle type: muon, pion, positron, kaon, proton')

args = parser.parse_args()

#runNumbers_filter = [1237,1238,1239,1240,1241]
runNumbers_filter = args.runNumbers
energies_filter = args.energies
#runNumbers_filter = [1266]
ROOT.ROOT.EnableImplicitMT(10)
ROOT.gROOT.SetBatch(True)  # Disable interactive mode for batch processing
ROOT.gSystem.Load("utils/functions_cc.so")  # Load the compiled C++ functions

runPedestal = 1374
with open(f"results/root/Run{runPedestal}/testbeam_pedestal.json", "r") as f:
    pedestal = json.load(f)
HGLG_json_dir = f"results/root/positroncali_round2/"
with open(f"{HGLG_json_dir}/testbeam_FERS_HG_to_LG_factors.json", "r") as f:
    factors_HG_to_LG = json.load(f)

filter_suffix = ""
particle = args.particle
if args.muoncounter:
    if particle == "muon":
        filter_suffix += "_muonaccept"
    else:
        filter_suffix += "_muonveto"
if args.holeveto:
    filter_suffix += "_holeveto"
if args.PSD:
    if particle != "positron":
        filter_suffix += "_failPSD"
    else:
        filter_suffix += "_passPSD"
if args.CC1:
    if particle == "muon" or particle == "positron":
        filter_suffix += "_passCC1"
    else:
        filter_suffix += "_selectCC1"
if args.CC2:
    if particle == "muon" or particle == "positron":
        filter_suffix += "_passCC2"
    else:
        filter_suffix += "_selectCC2"
if args.CC3:
    if particle == "muon" or particle == "positron":
        filter_suffix += "_passCC3"
    else:
        filter_suffix += "_selectCC3"
if filter_suffix == "": filter_suffix = "_wofilter"


for runNumber,energy in zip(runNumbers_filter,energies_filter):
    FERSBoards = buildFERSBoards(run=runNumber)
    rdf, _ = loadRDF(runNumber, 0, -1)
    rdf = preProcessDRSBoards(rdf)
    if args.holeveto:
        rdf, rdf_filterveto = applyUpstreamVeto(rdf, runNumber)
    if args.muoncounter:
        rdf, rdf_prefilter = vetoMuonCounter(rdf,runNumber, isMuon=(particle == "muon"))
    if args.PSD:
        rdf = applyPSDSelection(rdf, runNumber, applyCut=False)
        if particle != "positron":
            str_psd = "pass_PSDEle_selection == 0"
        else:
            str_psd = "pass_PSDEle_selection == 1"
        rdf = rdf.Filter(str_psd)
        print("select events "+str_psd)
        print(
        f"Events after PSD selection: {rdf.Count().GetValue()}")
    if args.CC1:
        rdf = applyCC1Selection(rdf, runNumber, energy,particle,applyCut=False)
        if getCCselection(particle, energy, 1):
            str_cc1 = f"pass_CC1Ele_selection == {getCCselection(particle, energy, 1)}"
            rdf = rdf.Filter(str_cc1)
            print("select events "+str_cc1)
        else: print("No selection on pass_CC1Ele_selection")
        print(
        f"Events after CC1 selection: {rdf.Count().GetValue()}")
    if args.CC2:
        rdf = applyCC2Selection(rdf, runNumber, energy,particle,applyCut=False)
        if getCCselection(particle, energy, 2):
            str_cc2 = f"pass_CC2Ele_selection == {getCCselection(particle, energy, 2)}"
            rdf = rdf.Filter(str_cc2)
            print("select events "+str_cc2)
        else: print("No selection on pass_CC2Ele_selection")
        print(
        f"Events after CC2 selection: {rdf.Count().GetValue()}")
    if args.CC3:
        rdf = applyCC3Selection(rdf, runNumber, energy, particle,applyCut=False)
        if getCCselection(particle, energy, 3):
            str_cc3 = f"pass_CC3Ele_selection == {getCCselection(particle, energy, 3)}"
            rdf = rdf.Filter(str_cc3)
            print("select events "+str_cc3)
        else: print("No selection on pass_CC3Ele_selection")
        print(
        f"Events after CC3 selection: {rdf.Count().GetValue()}")
    rdf = vectorizeFERS(rdf, FERSBoards)
    rdf = subtractFERSBeamPedestal(rdf, FERSBoards, pedestal)
    rdf = correctFERSSaturation(rdf,FERSBoards, factors_HG_to_LG)
    rdf = calculateEnergySumFERS(rdf, FERSBoards, subtractPedestal=True, calibrate=False, clip=False, saturationCorrected=True, lowGain = False)
    rdf = calculateEnergySumFERS(rdf, FERSBoards, subtractPedestal=False, calibrate=False, clip=False, saturationCorrected=False, lowGain = False)
    rdf = calculateEnergySumFERS(rdf, FERSBoards, subtractPedestal=True, calibrate=False, clip=False, saturationCorrected=False, lowGain = True)
    FERSChannels_calibrate = []
    FERSChannels_Cer = []
    FERSChannels_Sci = []
    for _, FERSBoard in FERSBoards.items():
        boardNo = FERSBoard.boardNo
        for iTowerX, iTowerY in FERSBoards[f"Board{boardNo}"].GetListOfTowers():
            for var in ["Cer","Sci"]:
                chan = FERSBoards[f"Board{boardNo}"].GetChannelByTower(
                    iTowerX, iTowerY, isCer=(var == "Cer"))
                channelNo_var = chan.channelNo
                if var == "Cer":
                    FERSChannels_Cer.append(f"FERS_Board{boardNo}_energyHG_{channelNo_var}_subtracted_saturationcorrected")
                    FERSChannels_Cer.append(f"FERS_Board{boardNo}_energyLG_{channelNo_var}_subtracted")
                    FERSChannels_Cer.append(f"FERS_Board{boardNo}_energyHG_{channelNo_var}_subtracted")
                if var == "Sci":
                    FERSChannels_Sci.append(f"FERS_Board{boardNo}_energyHG_{channelNo_var}_subtracted_saturationcorrected")
                    FERSChannels_Sci.append(f"FERS_Board{boardNo}_energyLG_{channelNo_var}_subtracted")
                    FERSChannels_Sci.append(f"FERS_Board{boardNo}_energyHG_{channelNo_var}_subtracted")
                FERSChannels_calibrate.append(f"FERS_Board{boardNo}_energyHG_{channelNo_var}_subtracted_saturationcorrected")
                FERSChannels_calibrate.append(f"FERS_Board{boardNo}_energyLG_{channelNo_var}_subtracted")
                FERSChannels_calibrate.append(f"FERS_Board{boardNo}_energyHG_{channelNo_var}_subtracted")
            
    FERSData = rdf.AsNumpy(columns = FERSChannels_calibrate)
    old_keys = []
    new_keys = []
    for key in FERSData.keys():
        new_key = key
        if "saturationcorrected" in key:
            new_key = key.replace("HG","mixed").replace("_saturationcorrected","")
        if key in FERSChannels_Cer:
            new_key+="_Cer"
        elif key in FERSChannels_Sci:
            new_key+="_Sci"
        old_keys.append(key)
        new_keys.append(new_key)
    for old_key,new_key in zip(old_keys,new_keys):
        FERSData[new_key] = FERSData.pop(old_key)
    #FERSData_test = rdf.AsNumpy(columns = ["FERS_Board8_energyHG_26","FERS_Board8_energyHG_26_subtracted","FERS_Board8_energyHG_26_subtracted_saturationcorrected",f"FERS_{var}EnergyHG",f"FERS_{var}EnergyHG_subtracted",f"FERS_{var}EnergyHG_subtracted_saturationcorrected"])
    #print("FERS_Board8_energyHG_26: ",FERSData_test["FERS_Board8_energyHG_26"],", mean: ",np.mean(FERSData_test["FERS_Board8_energyHG_26"]))
    #print("FERS_Board8_energyHG_26_subtracted: ",FERSData_test["FERS_Board8_energyHG_26_subtracted"],", mean: ",np.mean(FERSData_test["FERS_Board8_energyHG_26_subtracted"]))
    #print("FERS_Board8_energyHG_26_subtracted_saturationcorrected: ",FERSData_test["FERS_Board8_energyHG_26_subtracted_saturationcorrected"],", mean: ",np.mean(FERSData_test["FERS_Board8_energyHG_26_subtracted_saturationcorrected"]))
    #print(f"FERS_{var}EnergyHG: ",FERSData_test[f"FERS_{var}EnergyHG"],", mean: ",np.mean(FERSData_test[f"FERS_{var}EnergyHG"]))
    #print(f"FERS_{var}EnergyHG_subtracted: ",FERSData_test[f"FERS_{var}EnergyHG_subtracted"],", mean: ",np.mean(FERSData_test[f"FERS_{var}EnergyHG_subtracted"]))
    #print(f"FERS_{var}EnergyHG_subtracted_saturationcorrected: ",FERSData_test[f"FERS_{var}EnergyHG_subtracted_saturationcorrected"],", mean: ",np.mean(FERSData_test[f"FERS_{var}EnergyHG_subtracted_saturationcorrected"]))
    #print("length ", len(FERSData_test[f"FERS_{var}EnergyHG_subtracted_saturationcorrected"]))
    #outdir = f"results/root/Run{runNumber}"
    outdir = f"/lustre/research/hep/jweijie/HGDREAM/FERS/Run{runNumber}"
    outfile = outdir+f"/FERS_filtered{filter_suffix}.npz"
    os.system(f"mkdir -p {outdir}")
    np.savez(outfile,**FERSData)
    del rdf
    del FERSData
