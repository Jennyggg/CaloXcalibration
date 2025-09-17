import os
import sys
import ROOT
import uproot
from utils.channel_map import buildFERSBoards
from utils.utils import filterPrefireEvents, loadRDF, calculateEnergySumFERS, vectorizeFERS, calibrateFERSChannels,calibrateFERSChannelsBeam,testFERSBeamCalibration
from utils.html_generator import generate_html
from utils.fitter import eventFit
from utils.colors import colors
from configs.plotranges import getRangesForFERSEnergySums, getBoardEnergyFitParameters, getEventEnergyFitParameters
from runconfig import runNumber, firstEvent, lastEvent, beamEnergy
import json
#sys.path.append("CMSPLOTS")  # noqa
import numpy as np
ifile = "FERS_sim_testbeam.root"
tfile = "FERS_sim_testbeam_1.root"
ijson = "FERS_sim_testbeam.json"
beamEnergy = 50
infile = ROOT.TFile(ifile, "READ")
rdf = ROOT.RDataFrame("EventTree", infile)
#rdf = rdf.Filter("energysum > 49")
with open(ijson, 'r') as f:
        factors = json.load(f)
FERSBoards = buildFERSBoards(run=runNumber)
caliFactors = calibrateFERSChannelsBeam(rdf, FERSBoards, beamEnergy,rangeLow = 0, rangeHigh=-1, refRangeLow=0,refRangeHigh = 500, centers_x=rdf.AsNumpy(columns=["center_x"])["center_x"], centers_y=rdf.AsNumpy(columns=["center_y"])["center_y"])


testfile = ROOT.TFile(tfile, "READ")
rdf_test = ROOT.RDataFrame("EventTree", testfile)
#rdf_test = rdf_test.Filter("energysum > 49")
caliFactors_1 = calibrateFERSChannelsBeam(rdf_test, FERSBoards, beamEnergy,rangeLow = 0, rangeHigh=-1, refRangeLow=0,refRangeHigh = 500, centers_x=rdf_test.AsNumpy(columns=["center_x"])["center_x"], centers_y=rdf_test.AsNumpy(columns=["center_y"])["center_y"])
beamEnergyTruth_test = rdf_test.AsNumpy(columns=["energysum"])["energysum"]
for b in factors.keys():
        factor_true = factors[b]
        factor_est = 1./caliFactors[b]
        factor_est_1 = 1./caliFactors_1[b]
        print("branch: ",b, ", factor_true: ",factor_true, ", factor_est: ",factor_est,", factor_est_1: ",factor_est_1)

res_Cer, res_Sci = testFERSBeamCalibration(rdf_test,FERSBoards,beamEnergy,beamEnergyTruth_test,caliFactors,rangeLow = 0, rangeHigh=-1)
print("res_Cer ", res_Cer, "res_Sci ",res_Sci)