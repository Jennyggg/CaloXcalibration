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
from selections.selections import vetoMuonCounter, applyUpstreamVeto, PSDSelection,applyCC1Selection,applyPSDSelection
import json
import numpy as np
import matplotlib.pyplot as plt


FERS_energe_sum_range = [0,200000]

sys.path.append("CMSPLOTS")  # noqa
ROOT.ROOT.EnableImplicitMT(10)
ROOT.gROOT.SetBatch(True)  # Disable interactive mode for batch processing
ROOT.gSystem.Load("utils/functions_cc.so")  # Load the compiled C++ functions

runPedestal = 1374
with open(f"results/root/Run{runPedestal}/testbeam_pedestal.json", "r") as f:
    pedestal = json.load(f)
HGLG_json_dir = f"results/root/positroncali_round2/"
with open(f"{HGLG_json_dir}/testbeam_FERS_HG_to_LG_factors.json", "r") as f:
    factors_HG_to_LG = json.load(f)

xmax = 14
xmin = -14
ymax = 10
ymin = -10
W_ref = 1400
H_ref = 1600


FERSBoards = buildFERSBoards(run=runNumber)
rdf, _ = loadRDF(runNumber, firstEvent, lastEvent)
rdf = preProcessDRSBoards(rdf)
rdf, _ = vetoMuonCounter(rdf, TSmin=400, TSmax=700, cut=-30)
rdf,_ = applyPSDSelection(rdf, runNumber, applyCut=True)
rdf,_ = applyCC1Selection(rdf, runNumber, applyCut=True)
#rdf, _ = applyUpstreamVeto(rdf, runNumber)
#rdf, _ = PSDSelection(rdf, runNumber, isHadron=False)
rdf = vectorizeFERS(rdf, FERSBoards)
rdf = subtractFERSBeamPedestal(rdf, FERSBoards, pedestal)
rdf = correctFERSSaturation(rdf,FERSBoards, factors_HG_to_LG)

rdf = calculateEnergySumFERS(rdf, FERSBoards, subtractPedestal=True, calibrate=False, clip=False, saturationCorrected=True)
rdf = calculateEnergySumFERS(rdf, FERSBoards, subtractPedestal=True, calibrate=False, clip=False, saturationCorrected=False, lowGain = True)
rdf =  rdf.Filter(f"(FERS_CerEnergyHG_subtracted_saturationcorrected > {FERS_energe_sum_range[0]}) && (FERS_CerEnergyHG_subtracted_saturationcorrected < {FERS_energe_sum_range[1]})","filter Cer energy")
print("After filtering: ",rdf.Count().GetValue()," events")
hist2d_HG_average_Cer_3mm = ROOT.TH2F(
                f"HG_average_Cer_3mm",
                f"FERS HG average ({FERS_energe_sum_range[0]}<HG sum<{FERS_energe_sum_range[1]});X;Y (Cherenkov)",
                int(xmax - xmin), xmin, xmax,
                int(ymax - ymin) * 4, ymin, ymax
            )
hist2d_HG_average_Cer_3mm.SetMarkerSize(0.55)

hist2d_HG_average_Cer_6mm = ROOT.TH2F(
                f"HG_average_Cer_6mm",
                f"FERS HG average ({FERS_energe_sum_range[0]}<HG sum<{FERS_energe_sum_range[1]});X;Y (Cherenkov)",
                int(xmax - xmin), xmin, xmax,
                int(ymax - ymin), ymin, ymax
            )
hist2d_HG_average_Cer_6mm.SetMarkerSize(0.55)

hist2d_LG_average_Cer_3mm = ROOT.TH2F(
                f"LG_average_Cer_3mm",
                f"FERS LG average ({FERS_energe_sum_range[0]}<HG sum<{FERS_energe_sum_range[1]});X;Y (Cherenkov)",
                int(xmax - xmin), xmin, xmax,
                int(ymax - ymin) * 4, ymin, ymax
            )
hist2d_LG_average_Cer_3mm.SetMarkerSize(0.55)

hist2d_LG_average_Cer_6mm = ROOT.TH2F(
                f"LG_average_Cer_6mm",
                f"FERS LG average ({FERS_energe_sum_range[0]}<HG sum<{FERS_energe_sum_range[1]});X;Y (Cherenkov)",
                int(xmax - xmin), xmin, xmax,
                int(ymax - ymin), ymin, ymax
            )
hist2d_LG_average_Cer_6mm.SetMarkerSize(0.55)

hist2d_HG_saturateratio_Cer_3mm = ROOT.TH2F(
                f"HG_saturateratio_Cer_3mm",
                f"FERS HG saturate ratio ({FERS_energe_sum_range[0]}<HG sum<{FERS_energe_sum_range[1]});X;Y (Cherenkov)",
                int(xmax - xmin), xmin, xmax,
                int(ymax - ymin) * 4, ymin, ymax
            )
hist2d_HG_saturateratio_Cer_3mm.SetMarkerSize(0.55)

hist2d_HG_saturateratio_Cer_6mm = ROOT.TH2F(
                f"HG_saturateratio_Cer_6mm",
                f"FERS HG saturate ratio ({FERS_energe_sum_range[0]}<HG sum<{FERS_energe_sum_range[1]});X;Y (Cherenkov)",
                int(xmax - xmin), xmin, xmax,
                int(ymax - ymin), ymin, ymax
            )
hist2d_HG_saturateratio_Cer_6mm.SetMarkerSize(0.55)

hist2d_LG_saturateratio_Cer_3mm = ROOT.TH2F(
                f"LG_saturateratio_Cer_3mm",
                f"FERS LG saturate ratio ({FERS_energe_sum_range[0]}<HG sum<{FERS_energe_sum_range[1]});X;Y (Cherenkov)",
                int(xmax - xmin), xmin, xmax,
                int(ymax - ymin) * 4, ymin, ymax
            )
hist2d_LG_saturateratio_Cer_3mm.SetMarkerSize(0.55)

hist2d_LG_saturateratio_Cer_6mm = ROOT.TH2F(
                f"LG_saturateratio_Cer_6mm",
                f"FERS LG saturate ratio ({FERS_energe_sum_range[0]}<HG sum<{FERS_energe_sum_range[1]});X;Y (Cherenkov)",
                int(xmax - xmin), xmin, xmax,
                int(ymax - ymin), ymin, ymax
            )
hist2d_LG_saturateratio_Cer_6mm.SetMarkerSize(0.55)

rdf =  rdf.Filter(f"(FERS_CerEnergyHG_subtracted_saturationcorrected > {FERS_energe_sum_range[0]}) && (FERS_CerEnergyHG_subtracted_saturationcorrected < {FERS_energe_sum_range[1]})","filter Cer energy")
FERSChannels_Cer = []
for _, FERSBoard in FERSBoards.items():
    boardNo = FERSBoard.boardNo
    for iTowerX, iTowerY in FERSBoard.GetListOfTowers():
        chan_Cer = FERSBoards[f"Board{boardNo}"].GetChannelByTower(
            iTowerX, iTowerY, isCer=True)
        channelNo_Cer = chan_Cer.channelNo
        FERS_branch_HG = f"FERS_Board{boardNo}_energyHG_{channelNo_Cer}_subtracted_saturationcorrected"
        FERS_branch_LG = f"FERS_Board{boardNo}_energyLG_{channelNo_Cer}_subtracted"
        FERSChannels_Cer.append(FERS_branch_HG)
        FERSChannels_Cer.append(FERS_branch_LG)

FERSData = rdf.AsNumpy(columns = FERSChannels_Cer)
for _, FERSBoard in FERSBoards.items():
    boardNo = FERSBoard.boardNo
    for iTowerX, iTowerY in FERSBoard.GetListOfTowers():
        chan_Cer = FERSBoards[f"Board{boardNo}"].GetChannelByTower(
            iTowerX, iTowerY, isCer=True)
        channelNo_Cer = chan_Cer.channelNo
        FERS_branch_HG = f"FERS_Board{boardNo}_energyHG_{channelNo_Cer}_subtracted_saturationcorrected"
        FERS_branch_LG = f"FERS_Board{boardNo}_energyLG_{channelNo_Cer}_subtracted"
        if FERSBoards[f"Board{boardNo}"].Is3mm():
            hist2d_HG_average_Cer_3mm.Fill(iTowerX, iTowerY, np.mean(FERSData[FERS_branch_HG]))
            hist2d_LG_average_Cer_3mm.Fill(iTowerX, iTowerY, np.mean(FERSData[FERS_branch_LG]))
            hist2d_HG_saturateratio_Cer_3mm.Fill(iTowerX, iTowerY, float(np.sum(FERSData[FERS_branch_HG]>7500))/len(FERSData[FERS_branch_HG]))
            hist2d_LG_saturateratio_Cer_3mm.Fill(iTowerX, iTowerY, float(np.sum(FERSData[FERS_branch_LG]>7500))/len(FERSData[FERS_branch_LG]))
        else:
            hist2d_HG_average_Cer_6mm.Fill(iTowerX, iTowerY, np.mean(FERSData[FERS_branch_HG]))
            hist2d_LG_average_Cer_6mm.Fill(iTowerX, iTowerY, np.mean(FERSData[FERS_branch_LG]))
            hist2d_HG_saturateratio_Cer_6mm.Fill(iTowerX, iTowerY, float(np.sum(FERSData[FERS_branch_HG]>7500))/len(FERSData[FERS_branch_HG]))
            hist2d_LG_saturateratio_Cer_6mm.Fill(iTowerX, iTowerY, float(np.sum(FERSData[FERS_branch_LG]>7500))/len(FERSData[FERS_branch_LG]))

plotdir = f"results/plots/Run{runNumber}/"
DrawHistos([hist2d_HG_average_Cer_3mm,hist2d_HG_average_Cer_6mm], f"", xmin, xmax, "iX",
                       ymin, ymax, "iY", f"FERS_mean_ADC_HG_filter_HGsum_{FERS_energe_sum_range[0]}_{FERS_energe_sum_range[1]}",
                       dology=False, drawoptions=["COL,text","COL,text"],
                       zmin=0.0, zmax=max(hist2d_HG_average_Cer_3mm.GetMaximum(),hist2d_HG_average_Cer_6mm.GetMaximum())*1.01, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText="Cer", runNumber=runNumber,
                       outdir=plotdir,nTextDigits=0)
DrawHistos([hist2d_LG_average_Cer_3mm,hist2d_LG_average_Cer_6mm], f"", xmin, xmax, "iX",
                       ymin, ymax, "iY", f"FERS_mean_ADC_LG_filter_HGsum_{FERS_energe_sum_range[0]}_{FERS_energe_sum_range[1]}",
                       dology=False, drawoptions=["COL,text","COL,text"],
                       zmin=0.0, zmax=max(hist2d_LG_average_Cer_3mm.GetMaximum(),hist2d_LG_average_Cer_6mm.GetMaximum())*1.01, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText="Cer", runNumber=runNumber,
                       outdir=plotdir,nTextDigits=0)
DrawHistos([hist2d_HG_saturateratio_Cer_3mm,hist2d_HG_saturateratio_Cer_6mm], f"", xmin, xmax, "iX",
                       ymin, ymax, "iY", f"FERS_saturationratio_HG_filter_HGsum_{FERS_energe_sum_range[0]}_{FERS_energe_sum_range[1]}",
                       dology=False, drawoptions=["COL,text","COL,text"],
                       zmin=0.0, zmax=max(hist2d_HG_saturateratio_Cer_3mm.GetMaximum(),hist2d_HG_saturateratio_Cer_6mm.GetMaximum())*1.01, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText="Cer", runNumber=runNumber,
                       outdir=plotdir,nTextDigits=2)
DrawHistos([hist2d_LG_saturateratio_Cer_3mm,hist2d_LG_saturateratio_Cer_6mm], f"", xmin, xmax, "iX",
                       ymin, ymax, "iY", f"FERS_saturationratio_LG_filter_HGsum_{FERS_energe_sum_range[0]}_{FERS_energe_sum_range[1]}",
                       dology=False, drawoptions=["COL,text","COL,text"],
                       zmin=0.0, zmax=max(hist2d_LG_saturateratio_Cer_3mm.GetMaximum(),hist2d_LG_saturateratio_Cer_6mm.GetMaximum())*1.01, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText="Cer", runNumber=runNumber,
                       outdir=plotdir,nTextDigits=2)                       

