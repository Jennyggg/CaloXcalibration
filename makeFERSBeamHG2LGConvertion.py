import os
import sys
import ROOT
from utils.channel_map import buildFERSBoards
from utils.utils import loadRDF, vectorizeFERS,subtractFERSBeamPedestal
import json
import numpy as np
# test beam 1
#positions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
#runNumbers = [1234,1231,1235,1233,1236,1237,1238,1239,1240,1241,1243,1245,1246,1248,1249,1250,1252]
#positions_boards = {
#    0: [11,12],
#    1: [10,11],
#    2: [10,3],
#    3: [2,9],
#    4: [7,8],
#    5: [17],
#    6: [17,16],
#    7: [1],
#    8: [1],
#    9: [12,13],
#    10: [4,13],
#    11: [5,14],
#    12: [6,15],
#    13: [15,16]
#}

# test beam 2
positions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20.5,22,23,24.5]
runNumbers = [1355,1358,1359,1360,1361,1362,1363,1364,1365,1366,1367,1368,1369,1370,1372,1373,1389,1390,1398,1408,1404,1405,1406]
positions_boards = {
    2: [20.5,11,12],
    3: [11,19],
    4: [3,10],
    5: [2,9],
    6: [7,8],
    7: [17,18],
    8: [24.5,16,17],
    9: [1,2,3,4,5],
    10: [1,2,7,6,5],
    11: [12,22],
    12: [4,13],
    13: [5,14],
    14: [6,15],
    15: [16,23]
}

def TuneOneBoard(FERSBoard,factors_HG_to_LG):
    boardNo = FERSBoard.boardNo
    #if boardNo not in [3]: continue
    rdf_lists = []
    FERSBoards_subdict = {f"Board{boardNo}": FERSBoards[f"Board{boardNo}"]}
    for pos in positions_boards[boardNo]:
        runNumber = runNumbers[positions.index(pos)]
        rdf, _ = loadRDF(runNumber, firstEvent, lastEvent)
        rdf = vectorizeFERS(rdf, FERSBoards_subdict)
        rdf = subtractFERSBeamPedestal(rdf, FERSBoards_subdict, pedestal)
        print("loaded", len(rdf.Take["unsigned int"]("event_n").GetValue()))
        rdf_lists.append(rdf)
    for channel in FERSBoard:
        var_HG = f"FERS_Board{boardNo}_energyHG_{channel.channelNo}"
        var_LG = f"FERS_Board{boardNo}_energyLG_{channel.channelNo}"
        var_HG_subtract = f"FERS_Board{boardNo}_energyHG_{channel.channelNo}_subtracted"
        var_LG_subtract = f"FERS_Board{boardNo}_energyLG_{channel.channelNo}_subtracted"
        requirement1 = f"({var_HG_subtract} > 2000) && ({var_HG_subtract} < 7500) && ({var_LG_subtract} > 20) && ({var_LG_subtract} < 800)"
        HG_vals = []
        LG_vals = []
        for rdf in rdf_lists:
            values = rdf.Filter(requirement1,"filter pedestal").AsNumpy([f"{var_HG_subtract}",f"{var_LG_subtract}"])
            HG_vals.append(values[f"{var_HG_subtract}"])
            LG_vals.append(values[f"{var_LG_subtract}"])
        HG_vals = np.concatenate(HG_vals)
        LG_vals = np.concatenate(LG_vals)
        #ratios = HG_vals/LG_vals
        #HG_vals = HG_vals[(ratios> 7.0)&(ratios<15.0)]
        #LG_vals = LG_vals[(ratios>7.0)&(ratios<15.0)]
        #print("after filtering ratios: ",len(HG_vals))
        #print("HG_vals",HG_vals)
        #print("LG_vals",LG_vals)
        if len(HG_vals) < 4:
            print(f"Warning: Board{boardNo}_{channel.channelNo} do not have enough signals to fit")
            factors_HG_to_LG[f"FERS_Board{boardNo}_{channel.channelNo}"] = [0,0.1]
            continue
        g = ROOT.TGraph(len(HG_vals), HG_vals, LG_vals)
        print("len(HG_vals): ",len(HG_vals))
        print("len(LG_vals): ",len(LG_vals))
        print("HG_vals: ",HG_vals)
        print("LG_vals: ",LG_vals)
        print("HG_vals/LG_vals: ",HG_vals/LG_vals)
        fit_result = g.Fit("pol1", "S","",200,7500)

        intercept = fit_result.Parameter(0)  # intercept
        slope = fit_result.Parameter(1)  # slope
        factors_HG_to_LG[f"FERS_Board{boardNo}_{channel.channelNo}"] = [intercept,slope]
        if PLOTFIT:
            c = ROOT.TCanvas()
            g.SetTitle(f"Board{boardNo}_{channel.channelNo} LG vs HG; High Gain; Low Gain")
            g.SetMarkerStyle(1)
            g.Draw("AP")
            c.SaveAs(f"{plot_dir}/linear_fit_Board{boardNo}_{channel.channelNo}_LG_vs_HG.png")
    del rdf_lists
    return factors_HG_to_LG
        

rdf_lists = []
firstEvent = 0
lastEvent = -1
runPedestal = 1374
PLOTFIT = True

ROOT.ROOT.EnableImplicitMT(10)
ROOT.gROOT.SetBatch(True)

with open(f"results/root/Run{runPedestal}/testbeam_pedestal.json", "r") as f:
    pedestal = json.load(f)
plot_dir = "results/plots/positroncali_round2"
output_json_dir = f"results/root/positroncali_round2/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
FERSBoards = buildFERSBoards(run=1355)

factors_HG_to_LG = {}
for _, FERSBoard in FERSBoards.items():
    boardNo = FERSBoard.boardNo
    if boardNo != 10: continue
    factors_HG_to_LG = TuneOneBoard(FERSBoard,factors_HG_to_LG)
if not os.path.exists(output_json_dir):
    os.makedirs(output_json_dir)
with open(f"{output_json_dir}/testbeam_FERS_HG_to_LG_factors_board10.json", "w") as f:
    json.dump(factors_HG_to_LG, f, indent=4)
