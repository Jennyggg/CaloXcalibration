import os
import sys
import ROOT
from utils.channel_map import buildFERSBoards
from utils.utils import loadRDF,vectorizeFERS
from runconfig import runNumber, firstEvent, lastEvent, beamEnergy
import json
sys.path.append("CMSPLOTS")  # noqa

print("Start running FERS beam energy calibration. Pedestal estimation")
plotdir = f"results/plots/Run{runNumber}/"
output_json_dir = f"results/root/Run{runNumber}/"
# multi-threading support
ROOT.ROOT.EnableImplicitMT(10)
ROOT.gROOT.SetBatch(True)  # Disable interactive mode for batch processing
ROOT.gSystem.Load("utils/functions_cc.so")  # Load the compiled C++ functions
rdf, rdf_org = loadRDF(runNumber, firstEvent, lastEvent)
FERSBoards = buildFERSBoards(run=runNumber)
rdf = vectorizeFERS(rdf, FERSBoards)
pedestal = {}
for _, FERSBoard in FERSBoards.items():
        boardNo = FERSBoard.boardNo
        for channel in FERSBoard:
                var_HG = f"FERS_Board{boardNo}_energyHG_{channel.channelNo}"
                var_LG = f"FERS_Board{boardNo}_energyLG_{channel.channelNo}"
                mean_HG = rdf.Mean(var_HG).GetValue()
                mean_LG = rdf.Mean(var_LG).GetValue()
                pedestal[var_HG] = mean_HG
                pedestal[var_LG] = mean_LG

if not os.path.exists(output_json_dir):
    os.makedirs(output_json_dir)
with open(f"{output_json_dir}/testbeam_pedestal.json", "w") as f:
    json.dump(pedestal, f, indent=4)
