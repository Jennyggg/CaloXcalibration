import sys
sys.path.append("CMSPLOTS")  # noqa
import ROOT
from CMSPLOTS.myFunction import DrawHistos
from utils.channel_map import buildDRSBoards, buildFERSBoards, buildHodoPosChannels
from utils.utils import number2string, getDataFile, processDRSBoards, processHodoPeaks
from utils.html_generator import generate_html
from runNumber import runNumber
import time
import numpy as np
from sklearn.linear_model import LinearRegression

ROOT.gSystem.Load("utils/functions_cc.so")

ROOT.gROOT.SetBatch(True)  # Run in batch mode

outdir = f"plots/Run{runNumber}/"

xmax = 14
xmin = -14
ymax = 10
ymin = -10
W_ref = 1000
H_ref = 1100

TopY_hodo = [10,15,20,25]
BottomY_hodo = [-25,-20,-15,-10]

def fitMuonTrack(infilename):
    start_time = time.time()
    infile = ROOT.TFile(infilename, "READ")
    if not infile or infile.IsZombie():
        raise RuntimeError(f"Failed to open input file: {infile}")

    # Create an RDataFrame from the EventTree
    rdf = ROOT.RDataFrame("EventTree", infile)
    DRSBoards = buildDRSBoards(run=runNumber)
    FERSBoards = buildFERSBoards(run=runNumber)

    rdf = processDRSBoards(rdf)
    evtNumbers = np.array(rdf.Take["unsigned int"]("event_n").GetValue())
    print("evtNumbers ", evtNumbers,type(evtNumbers))
    pos_x = []
    pos_y = []
    weights_Cer = []
    weights_Sci = []
    fit_coeff = []
    fit_intercept = []
    fit_score = []
    for _, FERSBoard in FERSBoards.items():
        for iTowerX, iTowerY in FERSBoard.GetListOfTowers():
            #chan_Cer = FERSBoard.GetChannelByTower(
            #    iTowerX, iTowerY, isCer=True)
            chan_Sci = FERSBoard.GetChannelByTower(
                iTowerX, iTowerY, isCer=False)
            pos_x.append(iTowerX)
            pos_y.append(iTowerY)
            #var_Cer = chan_Cer.GetHGChannelName()
            var_Sci = chan_Sci.GetHGChannelName()
            #weights_Cer.append(np.array(rdf.Take["unsigned short"](var_Cer).GetValue()))
            weights_Sci.append(np.array(rdf.Take["unsigned short"](var_Sci).GetValue()))
    pos_x = np.array(pos_x)
    pos_y = np.array(pos_y)
    #weights_Cer = np.array(weights_Cer).T
    weights_Sci = np.array(weights_Sci).T
    events_filtered = []
    for event, weight in zip(evtNumbers,weights_Sci):
        mask = (weight>500)
        model = LinearRegression()
        model.fit(pos_x[mask].reshape(sum(mask),1), pos_y[mask], sample_weight = weight[mask])
        score = model.score(pos_x[mask].reshape(sum(mask),1), pos_y[mask], sample_weight = weight[mask])
        if score > 0.3:
            events_filtered.append(event)
            fit_coeff.append(model.coef_[0])
            fit_intercept.append(model.intercept_)
            fit_score.append(score)
    return np.array(events_filtered), np.array(fit_coeff), np.array(fit_intercept), np.array(fit_score)


def findHodoX(infilename,events_filtered,coeff,intercept):
    rdf = ROOT.RDataFrame("EventTree", infilename)
    rdf = processDRSBoards(rdf)
    hodo_pos_channels = buildHodoPosChannels(run=runNumber)
    rdf = processHodoPeaks(rdf,hodo_pos_channels)
    evtNumbers = np.array(rdf.Take["unsigned int"]("event_n").GetValue())
    hodoTopX = np.array(rdf.Take["int"]("TopX_delta_hodopeak").GetValue())
    hodoBottomX = np.array(rdf.Take["int"]("BottomX_delta_hodopeak").GetValue())
    _,indices_filtered,_ = np.intersect1d(evtNumbers, events_filtered, return_indices=True)
    hists_compare = []
    for pos,hodoX,hodoYScan in zip(["TopX","BottomX"],[hodoTopX, hodoBottomX],[TopY_hodo,BottomY_hodo]):
        for hodoY in hodoYScan:
            hodoX_extrapolate = (-intercept+hodoY)/coeff
            hodoX_measure = hodoX[indices_filtered]
            hists_compare.append(ROOT.TH2F(f"hist_hodoX_extrapolation_{hodoY}", f"hodo {pos} versus track at Y={hodoY};track X at Y={hodoY};{pos} peak deltaTS", 50, -25, 25,50, -200, 200))
            for X_extra, X_measure in zip(hodoX_extrapolate,hodoX_measure):
                hists_compare[-1].Fill(X_extra, X_measure)
    return hists_compare
            


if __name__ == "__main__":
    input_file = f"root/Run{runNumber}/filtered_track_12.root"
    print(f"Processing file: {input_file}")
    events, coeff, intercept, score = fitMuonTrack(input_file)
    np.savez_compressed(f"root/Run{runNumber}/trackfit_filteredtrack_12.npz",
                        event_n=events,
                        coeff=coeff,
                        intercept=intercept,
                        score=score)
    print("tracks fitted")
    hists_compare = findHodoX(input_file,events,coeff,intercept)
    outfile_name = f"root/Run{runNumber}/hodoX_trackX.root"
    outfile = ROOT.TFile(outfile_name, "recreate")
    for hist in hists_compare:
        hist.Write()
    outfile.Close()

