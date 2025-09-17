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

print("Start running FERS beam energy simulation")
energy = 50
centers_x = [0,0,0,0,0,-4,-4,-4,-4,4,4,4,4,-8,-8,-8,-8,8,8,8,8,-12,-12,12,12]
centers_y = [0,-6,-4,4,6,-4,-2,1,4,-4,-2,1,4,-5,-2,1,4,-5,-2,1,4,-1.5,1.5,-1.5,1.5]
width = 2
nevt = 500
# multi-threading support
ROOT.ROOT.EnableImplicitMT(10)
ROOT.gROOT.SetBatch(True)  # Disable interactive mode for batch processing
ROOT.gSystem.Load("utils/functions_cc.so")  # Load the compiled C++ functions

FERSBoards = buildFERSBoards(run=runNumber)
genFactors = {}

import numpy as np
from math import erf, sqrt

def gaussian_square_integral(x, y, center_x, center_y, w, size_x,size_y):
    """
    Compute the integral of a normalized 2D Gaussian over the square
    [x-0.5*size_x, x+0.5*size_x] × [y-0.5*size_y, y+0.5*size_y].
    
    Parameters
    ----------
    x, y : float
        Center of the integration square.
    center_x, center_y : float
        Gaussian mean coordinates.
    w : float
        Standard deviation (same for both axes).
        
    Returns
    -------
    float
        The integrated probability over the square.
    """
    def erf_part(a, b, mu):
        return 0.5 * (erf((b - mu) / (sqrt(2) * w)) -
                      erf((a - mu) / (sqrt(2) * w)))
    
    x_int = erf_part(x - 0.5*size_x, x + 0.5*size_x, center_x)
    y_int = erf_part(y - 0.5*size_y, y + 0.5*size_y, center_y)
    
    return x_int * y_int
#print("debugging ", gaussian_square_integral(0, 0, 0, 0, 2, 20,20))
np.random.seed(0)
area = 0
for _, FERSBoard in FERSBoards.items():
        boardNo = FERSBoard.boardNo
        for iTowerX, iTowerY in FERSBoard.GetListOfTowers():
            chan_Cer = FERSBoard.GetChannelByTower(
                iTowerX, iTowerY, isCer=True)
            chan_Sci = FERSBoard.GetChannelByTower(
                iTowerX, iTowerY, isCer=False)
            #print("iTowerX=",iTowerX, "iTowerY=",iTowerY)
            channelNo_Cer = chan_Cer.channelNo
            channelNo_Sci = chan_Sci.channelNo
            branchName_Cer = f"FERS_Board{boardNo}_energyHG_{channelNo_Cer}_subtracted_calibrated_clipped"
            branchName_Sci = f"FERS_Board{boardNo}_energyHG_{channelNo_Sci}_subtracted_calibrated_clipped"
            genFactors[branchName_Cer] = np.random.uniform(0.9,1)
            genFactors[branchName_Sci] = np.random.uniform(0.9,1)
            if FERSBoard.Is3mm():
                area += 0.48
            else:
                area += 1.92
#print("area ", area)
np.random.seed(0)
sim_signals = {b:[] for b in genFactors.keys()}
sim_signals["energysum"] = []
sim_signals["signalsum_Cer"] = []
sim_signals["signalsum_Sci"] = []
sim_signals["center_x"] = []
sim_signals["center_y"] = []
for beam_x, beam_y in zip(centers_x,centers_y):
    for ievt in range(nevt):
        center_x = np.random.normal(beam_x,2)
        center_y = np.random.normal(beam_y,2)
        energysum = 0
        signalsum_Cer = 0
        signalsum_Sci = 0
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
                if FERSBoard.Is3mm():
                    size_x = 1.2
                    size_y = 0.4
                else:
                    size_x = 1.2
                    size_y = 1.6
                energy_fraction = gaussian_square_integral(iTowerX*1.2, iTowerY*1.6, center_x, center_y, width, size_x,size_y)
                signal_Cer = np.random.poisson(genFactors[branchName_Cer] * energy_fraction * energy * 100) / 100
                signal_Sci = np.random.poisson(genFactors[branchName_Sci] * energy_fraction * energy * 100) / 100
                energysum += energy_fraction * energy
                signalsum_Cer += signal_Cer
                signalsum_Sci += signal_Sci
                sim_signals[branchName_Cer].append(signal_Cer)
                sim_signals[branchName_Sci].append(signal_Sci)
        sim_signals["energysum"].append(energysum)
        sim_signals["signalsum_Cer"].append(signalsum_Cer)
        sim_signals["signalsum_Sci"].append(signalsum_Sci)
        sim_signals["center_x"].append(center_x)
        sim_signals["center_y"].append(center_y)
sim_signals = {b:np.array(v,dtype=np.float32) for b,v in sim_signals.items()}

with uproot.recreate("FERS_sim_testbeam.root") as f:
    f["EventTree"] = sim_signals
with open(f"FERS_sim_testbeam.json", "w") as f:
    json.dump(genFactors, f, indent=4)


