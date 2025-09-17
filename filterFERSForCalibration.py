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
from selections.selections import vetoMuonCounter, applyUpstreamVeto, PSDSelection,applyCC1Selection
import json
import numpy as np

runNumbers_filter = [1292,1262,1266,1264,1290,1288]
ROOT.ROOT.EnableImplicitMT(10)
ROOT.gROOT.SetBatch(True)  # Disable interactive mode for batch processing
ROOT.gSystem.Load("utils/functions_cc.so")  # Load the compiled C++ functions

runPedestal = 1259
with open(f"results/root/Run{runPedestal}/testbeam_pedestal.json", "r") as f:
    pedestal = json.load(f)
HGLG_json_dir = f"results/root/positroncali/"
with open(f"{HGLG_json_dir}/testbeam_FERS_HG_to_LG_factors.json", "r") as f:
    factors_HG_to_LG = json.load(f)

isCer = False
isHG = True
isFilter = True
isHadron = True
filter_suffix = "_addCC1"
if not isFilter: filter_suffix = "_wofilter"
var = "Cer" if isCer else "Sci"

for runNumber in runNumbers_filter:
    FERSBoards = buildFERSBoards(run=runNumber)
    rdf, _ = loadRDF(runNumber, 0, -1)
    rdf = preProcessDRSBoards(rdf)
    if isFilter:
        rdf, rdf_prefilter = vetoMuonCounter(rdf, TSmin=400, TSmax=700, cut=-30)
        rdf, rdf_filterveto = applyUpstreamVeto(rdf, runNumber)
        rdf, rdf_psd = PSDSelection(rdf, runNumber, isHadron=isHadron)
        rdf, rdf_CC1 = applyCC1Selection(rdf, runNumber, isHadron=isHadron, applyCut=True)
    rdf = vectorizeFERS(rdf, FERSBoards)
    rdf = subtractFERSBeamPedestal(rdf, FERSBoards, pedestal)
    rdf = correctFERSSaturation(rdf,FERSBoards, factors_HG_to_LG)
    rdf = calculateEnergySumFERS(rdf, FERSBoards, subtractPedestal=True, calibrate=False, clip=False, saturationCorrected=isHG, lowGain = not isHG)
    FERSChannels_calibrate = []
    for _, FERSBoard in FERSBoards.items():
        boardNo = FERSBoard.boardNo
        for iTowerX, iTowerY in FERSBoards[f"Board{boardNo}"].GetListOfTowers():
            chan = FERSBoards[f"Board{boardNo}"].GetChannelByTower(
                iTowerX, iTowerY, isCer=isCer)
            channelNo_var = chan.channelNo
            if isHG:
                FERSChannels_calibrate.append(f"FERS_Board{boardNo}_energyHG_{channelNo_var}_subtracted_saturationcorrected")
            else:
                FERSChannels_calibrate.append(f"FERS_Board{boardNo}_energyLG_{channelNo_var}_subtracted")
    if isHG:
        FERSData = rdf.AsNumpy(columns = FERSChannels_calibrate + [f"FERS_{var}EnergyHG_subtracted_saturationcorrected"])
    else:
        FERSData = rdf.AsNumpy(columns = FERSChannels_calibrate + [f"FERS_{var}EnergyLG_subtracted"])
    outdir = f"results/root/Run{runNumber}"
    if isHG:
        outfile = f"results/root/Run{runNumber}/FERS_filtered_{var}{filter_suffix}.npz"
    else:
        outfile = f"results/root/Run{runNumber}/FERS_filtered_{var}_LG{filter_suffix}.npz"
    os.system(f"mkdir -p {outdir}")
    np.savez(outfile,**FERSData)
    del rdf
    del FERSData
