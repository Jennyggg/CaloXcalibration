import os
import ROOT
import json

from utils.channel_map import buildDRSBoards, buildFERSBoards, buildTimeReferenceChannels, buildHodoTriggerChannels, buildHodoPosChannels, mapDRSChannel2TriggerChannel,buildDRSChannels,buildDRSChannelsAmplified
from utils.utils import number2string, preProcessDRSBoards, filterPrefireEvents, loadRDF, vectorizeFERS, prepareDRSStats, processDRSPeaks, getBranchStats, readTSconfig, processHodoPeaks
from configs.plotranges import getDRSPlotRanges
from runconfig import runNumber, firstEvent, lastEvent
import time
import re

print("Start running prepareDQMPlots.py")


# multi-threading support
ROOT.ROOT.EnableImplicitMT(10)
ROOT.gSystem.Load("utils/functions_cc.so")  # Load the compiled C++ functions

debugDRS = False

rdf, rdf_org = loadRDF(runNumber, firstEvent, lastEvent, filter=False)
#rdf, rdf_prefilter = filterPrefireEvents(rdf, runNumber)

DRSBoards = buildDRSBoards(run=runNumber)
print("DRSBoards ",DRSBoards)
FERSBoards = buildFERSBoards(run=runNumber)
trigger_channels = buildTimeReferenceChannels(run=runNumber)
drs_channels = buildDRSChannels(DRSBoards)
#TSconfig = readTSconfig(run = runNumber)
# Get total number of entries
n_entries = rdf.Count().GetValue()
nEvents = int(n_entries)
nbins_Event = min(max(int(nEvents / 100), 1), 500)
print(f"Total number of events to process: {nEvents} in run {runNumber}")
#rdf = processDRSPeaks(rdf,drs_channels,trigger_channels,bl_start = TSconfig["noiseBaselineTS"][0], bl_end = TSconfig["noiseBaselineTS"][1],window_start = TSconfig["peakTSWindow"][0], window_end =  TSconfig["peakTSWindow"][1],drs_amplified = [])
rdf = vectorizeFERS(rdf, FERSBoards)

rdf = preProcessDRSBoards(rdf, debug=debugDRS)
rdf = prepareDRSStats(rdf, DRSBoards, 0, 400, 9)

def monitorConditions():
    # monitor the V, I, T conditions
    hists2d_Condition_vs_Event = []
    for _, FERSBoard in FERSBoards.items():
        boardNo = FERSBoard.boardNo
        hist_SipmHV = rdf.Histo2D((
            f"hist_FERS_Board{boardNo}_SipmHV_vs_Event",
            f"FERS Board {boardNo} - SipmHV vs Event;Event;SipmHV (V)",
            nbins_Event, 0, nEvents, 40, 26, 29),
            "event_n", f"FERS_Board{boardNo}_SipmHV"
        )
        hist_SipmI = rdf.Histo2D((
            f"hist_FERS_Board{boardNo}_SipmI_vs_Event",
            f"FERS Board {boardNo} - SipmI vs Event;Event;SipmI (mA)",
            nbins_Event, 0, nEvents, 50, 0.02, 0.2),
            "event_n", f"FERS_Board{boardNo}_SipmI"
        )
        hist_TempDET = rdf.Histo2D((
            f"hist_FERS_Board{boardNo}_TempDET_vs_Event",
            f"FERS Board {boardNo} - TempDET vs Event;Event;TempDET (C)",
            nbins_Event, 0, nEvents, 100, 10, 30),
            "event_n", f"FERS_Board{boardNo}_TempDET"
        )
        hist_TempFPGA = rdf.Histo2D((
            f"hist_FERS_Board{boardNo}_TempFPGA_vs_Event",
            f"FERS Board {boardNo} - TempFPGA vs Event;Event;TempFPGA (C)",
            nbins_Event, 0, nEvents, 100, 30, 50),
            "event_n", f"FERS_Board{boardNo}_TempFPGA"
        )
        hists2d_Condition_vs_Event.append(hist_SipmHV)
        hists2d_Condition_vs_Event.append(hist_SipmI)
        hists2d_Condition_vs_Event.append(hist_TempDET)
        hists2d_Condition_vs_Event.append(hist_TempFPGA)

    return hists2d_Condition_vs_Event


hodo_trigger_channels = buildHodoTriggerChannels(run=runNumber)
# FRES board outputs
# define variables as RDF does not support reading vectors
# with indices directly
#for _, FERSBoard in FERSBoards.items():
#    boardNo = FERSBoard.boardNo
#    for channel in FERSBoard:
#        rdf = rdf.Define(
#            f"FERS_Board{boardNo}_energyHG_{channel.channelNo}",
#            f"FERS_Board{boardNo}_energyHG[{channel.channelNo}]")
#        rdf = rdf.Define(
#            f"FERS_Board{boardNo}_energyLG_{channel.channelNo}",
#            f"FERS_Board{boardNo}_energyLG[{channel.channelNo}]"
#        )

#branches = [str(b) for b in rdf.GetColumnNames()]
#pattern = re.compile(r"DRS.*Group.*Channel.*")
#drs_branches = [b for b in branches if pattern.search(b)]
drs_amplified = buildDRSChannelsAmplified(runNumber)
#TSconfig = readTSconfig(run=runNumber)
#rdf = processDRSPeaks(rdf, buildDRSChannels(DRSBoards), trigger_channels,TSconfig["noiseBaselineTS"][0],TSconfig["noiseBaselineTS"][1],TSconfig["peakTSWindow"][0],TSconfig["peakTSWindow"][1],drs_amplified)
hodo_pos_channels = buildHodoPosChannels(run=runNumber)
rdf = processHodoPeaks(rdf,hodo_pos_channels)
drs_Cer_channels_time_check = [["DRS_Board1_Group0_Channel0","DRS_Board1_Group0_Channel1"],
                           ["DRS_Board1_Group0_Channel2","DRS_Board1_Group0_Channel3"],
                           ["DRS_Board1_Group0_Channel3","DRS_Board1_Group1_Channel3"],
                           ["DRS_Board1_Group1_Channel2","DRS_Board1_Group2_Channel2"],
                           ["DRS_Board1_Group3_Channel1","DRS_Board2_Group0_Channel1"],
                           ["DRS_Board1_Group3_Channel2","DRS_Board2_Group0_Channel2"],
                           ["DRS_Board2_Group0_Channel1","DRS_Board2_Group0_Channel2"],
                           ["DRS_Board2_Group0_Channel2","DRS_Board2_Group1_Channel2"],
                           ["DRS_Board2_Group1_Channel2","DRS_Board2_Group2_Channel2"],
                           ["DRS_Board3_Group0_Channel1","DRS_Board3_Group0_Channel2"],
                           ["DRS_Board3_Group0_Channel3","DRS_Board3_Group1_Channel3"],
                           ["DRS_Board3_Group1_Channel0","DRS_Board3_Group2_Channel0"],
                           ["DRS_Board2_Group3_Channel3","DRS_Board3_Group0_Channel3"],
                           ["DRS_Board2_Group3_Channel0","DRS_Board3_Group0_Channel0"]
                           ]

drs_Sci_channels_time_check = [["DRS_Board1_Group0_Channel4","DRS_Board1_Group0_Channel5"],
                               ["DRS_Board1_Group0_Channel6","DRS_Board1_Group0_Channel7"],
                               ["DRS_Board1_Group0_Channel7","DRS_Board1_Group1_Channel7"],
                               ["DRS_Board1_Group1_Channel6","DRS_Board1_Group2_Channel6"],
                               ["DRS_Board1_Group3_Channel4","DRS_Board2_Group0_Channel4"],
                               ["DRS_Board1_Group3_Channel5","DRS_Board2_Group0_Channel5"],
                               ["DRS_Board2_Group0_Channel4","DRS_Board2_Group0_Channel5"],
                               ["DRS_Board2_Group0_Channel5","DRS_Board2_Group1_Channel5"],
                               ["DRS_Board2_Group1_Channel5","DRS_Board2_Group2_Channel5"],
                               ["DRS_Board3_Group0_Channel4","DRS_Board3_Group0_Channel5"],
                               ["DRS_Board3_Group0_Channel6","DRS_Board3_Group1_Channel6"],
                               ["DRS_Board3_Group1_Channel3","DRS_Board3_Group2_Channel3"],
                               ["DRS_Board2_Group3_Channel6","DRS_Board3_Group0_Channel6"],
                               ["DRS_Board2_Group3_Channel3","DRS_Board3_Group0_Channel3"]]

drs_triggers_time_check = [["DRS_Board1_Group0_Channel8","DRS_Board1_Group1_Channel8"],
                           ["DRS_Board1_Group0_Channel8","DRS_Board1_Group2_Channel8"],
                           ["DRS_Board1_Group0_Channel8","DRS_Board1_Group3_Channel8"],
                           ["DRS_Board1_Group2_Channel8","DRS_Board1_Group3_Channel8"],
                           ["DRS_Board1_Group0_Channel8","DRS_Board2_Group0_Channel8"],
                           ["DRS_Board1_Group0_Channel8","DRS_Board2_Group1_Channel8"],
                           ["DRS_Board1_Group0_Channel8","DRS_Board2_Group2_Channel8"],
                           ["DRS_Board1_Group0_Channel8","DRS_Board2_Group3_Channel8"],
                           ["DRS_Board2_Group0_Channel8","DRS_Board2_Group1_Channel8"],
                           ["DRS_Board2_Group0_Channel8","DRS_Board2_Group2_Channel8"],
                           ["DRS_Board2_Group0_Channel8","DRS_Board2_Group3_Channel8"],
                           ["DRS_Board1_Group0_Channel8","DRS_Board3_Group0_Channel8"],
                           ["DRS_Board1_Group0_Channel8","DRS_Board3_Group1_Channel8"],
                           ["DRS_Board1_Group0_Channel8","DRS_Board3_Group2_Channel8"],
                           ["DRS_Board1_Group0_Channel8","DRS_Board3_Group3_Channel8"],
                           ["DRS_Board3_Group0_Channel8","DRS_Board3_Group1_Channel8"],
                           ["DRS_Board3_Group0_Channel8","DRS_Board3_Group2_Channel8"]]


def makeFERS1DPlots():
    hists1d_FERS = []
    for _, FERSBoard in FERSBoards.items():
        boardNo = FERSBoard.boardNo
        for iTowerX, iTowerY in FERSBoard.GetListOfTowers():
            sTowerX = number2string(iTowerX)
            sTowerY = number2string(iTowerY)

            for var in ["Cer", "Sci"]:
                # Get the channel for CER or SCI
                chan = FERSBoard.GetChannelByTower(
                    iTowerX, iTowerY, isCer=(var == "Cer"))
                hist = rdf.Histo1D((
                    f"hist_FERS_Board{boardNo}_{var}_{sTowerX}_{sTowerY}_HG",
                    f"FERS Board {boardNo} - {var} iTowerX {sTowerX} iTowerY {sTowerY};{var} Energy HG;Counts",
                    500, 0, 9000),
                    chan.GetHGChannelName()
                )
                hists1d_FERS.append(hist)
                hist_LG = rdf.Histo1D((
                    f"hist_FERS_Board{boardNo}_{var}_{sTowerX}_{sTowerY}_LG",
                    f"FERS Board {boardNo} - {var} iTowerX {sTowerX} iTowerY {sTowerY};{var} Energy LG;Counts",
                    500, 0, 9000),
                    chan.GetLGChannelName()
                )
                hists1d_FERS.append(hist_LG)

    return hists1d_FERS


def collectFERSPedestals(hists1d_FERS):
    pedestals = {}
    for _, FERSBoard in FERSBoards.items():
        boardNo = FERSBoard.boardNo
        for iTowerX, iTowerY in FERSBoard.GetListOfTowers():
            sTowerX = number2string(iTowerX)
            sTowerY = number2string(iTowerY)

            for var in ["Cer", "Sci"]:
                chan = FERSBoard.GetChannelByTower(
                    iTowerX, iTowerY, isCer=(var == "Cer"))
                channelName_HG = chan.GetHGChannelName()

                hname = f"hist_FERS_Board{boardNo}_{var}_{sTowerX}_{sTowerY}"
                hist = None
                for h in hists1d_FERS:
                    if h.GetName() == hname:
                        hist = h
                        break
                if hist is None:
                    print(
                        f"Warning: Histogram {hname} not found in hists1d_FERS")
                    pedestals[var][channelName_HG] = None
                    continue
                pedestal = hist.GetXaxis().GetBinCenter(hist.GetMaximumBin())
                pedestals[channelName_HG] = pedestal
    return pedestals


def collectFERSStats():
    stats = {}
    for _, FERSBoard in FERSBoards.items():
        for chan in FERSBoard:
            channelName_HG = chan.GetHGChannelName()
            stats[channelName_HG] = (
                rdf.Mean(channelName_HG),
                rdf.Max(channelName_HG),
            )
            channelName_LG = chan.GetLGChannelName()
            stats[channelName_LG] = (
                rdf.Mean(channelName_LG),
                rdf.Max(channelName_LG),
            )

    return stats


def trackFERSPlots():
    hists2d_FERS_vs_Event = []
    for _, FERSBoard in FERSBoards.items():
        boardNo = FERSBoard.boardNo
        for iTowerX, iTowerY in FERSBoard.GetListOfTowers():
            sTowerX = number2string(iTowerX)
            sTowerY = number2string(iTowerY)
            for var in ["Cer", "Sci"]:
                # Get the channel for CER or SCI
                chan = FERSBoard.GetChannelByTower(
                    iTowerX, iTowerY, isCer=(var == "Cer"))
                hist = rdf.Histo2D((
                    f"hist_FERS_Board{boardNo}_{var}_vs_Event_{sTowerX}_{sTowerY}",
                    f"FERS Board {boardNo} - Event vs {var} {chan.channelNo} in iTowerX {sTowerX} iTowerY {sTowerY};Event;{var} Energy HG",
                    nbins_Event, 0, nEvents, 1000, 0, 9000),
                    "event_n", chan.GetHGChannelName()
                )
                hists2d_FERS_vs_Event.append(hist)
    return hists2d_FERS_vs_Event


def makeFERS2DPlots():
    hists2d_FERS = []
    for _, FERSBoard in FERSBoards.items():
        boardNo = FERSBoard.boardNo
        for iTowerX, iTowerY in FERSBoard.GetListOfTowers():
            sTowerX = number2string(iTowerX)
            sTowerY = number2string(iTowerY)
            chan_Cer = FERSBoard.GetChannelByTower(
                iTowerX, iTowerY, isCer=True)
            chan_Sci = FERSBoard.GetChannelByTower(
                iTowerX, iTowerY, isCer=False)

            iCer = chan_Cer.channelNo
            iSci = chan_Sci.channelNo
            # high gain
            hist = rdf.Histo2D((
                f"hist_FERS_Board{boardNo}_Cer_vs_Sci_{sTowerX}_{sTowerY}",
                f"CER {iCer} vs SCI {iSci} in iTowerX {sTowerX} iTowerY {sTowerY};CER Energy HG;SCI Energy HG",
                300, 0, 9000, 300, 0, 9000),
                chan_Cer.GetHGChannelName(),
                chan_Sci.GetHGChannelName()
            )
            hist_zoomed = rdf.Histo2D((
                f"hist_FERS_Board{boardNo}_Cer_vs_Sci_{sTowerX}_{sTowerY}_zoom",
                f"CER {iCer} vs SCI {iSci} in iTowerX {sTowerX} iTowerY {sTowerY} (zoomed);CER Energy HG;SCI Energy HG",
                300, 0, 1000, 200, 0, 2000),
                chan_Cer.GetHGChannelName(),
                chan_Sci.GetHGChannelName()
            )
            hists2d_FERS.append(hist)
            hists2d_FERS.append(hist_zoomed)

            # high gain vs low gain for Sci
            hist_sci_hg_vs_lg = rdf.Histo2D((
                f"hist_FERS_Board{boardNo}_Sci_{sTowerX}_{sTowerY}_hg_vs_lg",
                f"SCI {iSci} HG vs LG;SCI Energy HG;SCI Energy LG",
                300, 0, 9000, 300, 0, 3000),
                chan_Sci.GetHGChannelName(),
                chan_Sci.GetLGChannelName()
            )
            hists2d_FERS.append(hist_sci_hg_vs_lg)
            hist_cer_hg_vs_lg = rdf.Histo2D((
                f"hist_FERS_Board{boardNo}_Cer_{sTowerX}_{sTowerY}_hg_vs_lg",
                f"CER {iCer} HG vs LG;CER Energy HG;CER Energy LG",
                300, 0, 9000, 300, 0, 3000),
                chan_Cer.GetHGChannelName(),
                chan_Cer.GetLGChannelName()
            )
            hists2d_FERS.append(hist_cer_hg_vs_lg)
    return hists2d_FERS


def makeDRS1DPlots():
    hists1d_DRS = []
    for _, DRSBoard in DRSBoards.items():
        boardNo = DRSBoard.boardNo
        for iTowerX, iTowerY in DRSBoard.GetListOfTowers():
            sTowerX = number2string(iTowerX)
            sTowerY = number2string(iTowerY)

            for var in ["Cer", "Sci"]:
                chan = DRSBoard.GetChannelByTower(
                    iTowerX, iTowerY, isCer=(var == "Cer"))

                if chan is None:
                    continue
                channelName = chan.GetChannelName()
                #print("channelName: ", channelName, ", boardNo: ",boardNo, ", groupNo: ",chan.groupNo, ", channelNo: ",chan.channelNo)
                threBins = TSconfig["threBinsCer"] if var == "Cer" else TSconfig["threBinsSci"]
                for i in range(len(threBins)-1):
                    thre_low = threBins[i]
                    thre_high = threBins[i+1]
                    if channelName in drs_amplified:
                        thre_low = thre_low * 10
                        thre_high = thre_high * 10
                    hist_peakT = rdf.Filter(f"{channelName}_peakA > {thre_low} && {channelName}_peakA <= {thre_high}",f"{channelName} amplitude in [{thre_low}, {thre_high}]").Histo1D((
                        f"hist_DRS_Board{boardNo}_{var}_{sTowerX}_{sTowerY}_peakT_amp{thre_low}-{thre_high}",
                        f"DRS Board {boardNo} - {var} iTowerX {sTowerX} iTowerY {sTowerY} (peak T, amp {thre_low}-{thre_high});delta TS;Counts",
                        10240, 0, 1024),
                        f"{channelName}_peakT"
                        )
                    hist_deltaT = rdf.Filter(f"{channelName}_peakA > {thre_low} && {channelName}_peakA <= {thre_high}",f"{channelName} amplitude in [{thre_low}, {thre_high}]").Histo1D((
                        f"hist_DRS_Board{boardNo}_{var}_{sTowerX}_{sTowerY}_deltaT_amp{thre_low}-{thre_high}",
                        f"DRS Board {boardNo} - {var} iTowerX {sTowerX} iTowerY {sTowerY} (delta T + 1024, amp {thre_low}-{thre_high});peak TS - trigger TS + 1024;Counts",
                        81920, 0, 2048),
                        f"{channelName}_deltaT"
                    )
                    hists1d_DRS.append(hist_peakT)
                    hists1d_DRS.append(hist_deltaT)
    for var,drs_channels_time_check in zip(["Cer","Sci"],[drs_Cer_channels_time_check,drs_Sci_channels_time_check]):
        for ch1, ch2 in drs_channels_time_check:
            thre_low = 5
            if ch1 in drs_amplified and ch2 in drs_amplified:
                thre_low = 100
            rdf_both_filtered = rdf.Filter(f"({ch1}_peakA > {thre_low})&&({ch2}_peakA > {thre_low})&&({ch1}_peakT>{TSconfig['peakTSWindow'][0]})&&({ch1}_peakT<{TSconfig['peakTSWindow'][1]})&&({ch2}_peakT>{TSconfig['peakTSWindow'][0]})&&({ch2}_peakT<{TSconfig['peakTSWindow'][1]})")
            rdf_both_filtered = rdf_both_filtered.Define(f"{ch1}_{ch2}_deltaT",
                                                     f"{ch1}_deltaT - {ch2}_deltaT")
            hist_deltaT_channels = rdf_both_filtered.Histo1D((
                        f"hist_{var}_DRS_{ch1}_{ch2}_deltaT",
                        f"DRS channel {ch1} {ch2} delta TS;delta TS({ch1},{ch2});Counts",
                        8000, -100, 100),
                        f"{ch1}_{ch2}_deltaT"
                    )
            rdf_both_filtered = rdf_both_filtered.Define(f"{ch1}_{ch2}_peakT",
                                                     f"{ch1}_peakT - {ch2}_peakT")
            hist_peakT_channels = rdf_both_filtered.Histo1D((
                        f"hist_{var}_DRS_{ch1}_{ch2}_peakT",
                        f"DRS channel {ch1} {ch2} delta TS (without trigger corr);delta TS({ch1},{ch2});Counts",
                        8000, -100, 100),
                        f"{ch1}_{ch2}_peakT"
                    )
            hists1d_DRS.append(hist_deltaT_channels)
            hists1d_DRS.append(hist_peakT_channels)
    
    for ch1, ch2 in drs_triggers_time_check:
        hist_deltaT_triggers = rdf.Define(f"{ch1}_{ch2}_triggerdeltaT",
                                         f"{ch1}_triggerT - {ch2}_triggerT").Histo1D((
                                         f"hist_DRS_{ch1}_{ch2}_triggerdeltaT",
                                         f"DRS channel {ch1} {ch2} delta TS;delta TS({ch1},{ch2});Counts",
                        8000, -100, 100),
                        f"{ch1}_{ch2}_triggerdeltaT"
                    )
        hists1d_DRS.append(hist_deltaT_triggers)
    return hists1d_DRS


def makeDRS2DPlots(debug=False):
    hists2d_DRS_vs_TS = []
    if debug:
        hists2d_DRS_vs_RTSpos = []
        hists2d_DRS_vs_RTSneg = []
    for _, DRSBoard in DRSBoards.items():
        boardNo = DRSBoard.boardNo
        for iTowerX, iTowerY in DRSBoard.GetListOfTowers():
            sTowerX = number2string(iTowerX)
            sTowerY = number2string(iTowerY)

            for var in ["Cer", "Sci"]:
                chan = DRSBoard.GetChannelByTower(
                    iTowerX, iTowerY, isCer=(var == "Cer"))
                if chan is None:
                    continue
                channelName = chan.GetChannelName()
                ymin, ymax = getDRSPlotRanges(
                    subtractMedian=True, isAmplified=chan.isAmplified)
                hist_subtractMedian = rdf.Histo2D((
                    f"hist_DRS_Board{boardNo}_{var}_vs_TS_{sTowerX}_{sTowerY}_subtractMedian",
                    f"DRS Board {boardNo} - {var} {chan.channelNo} in iTowerX {sTowerX} iTowerY {sTowerY} (subtract median);TS;{var} Variable",
                    1024, 0, 1024, 50, ymin, ymax),
                    "TS", channelName + "_subtractMedian"
                )
                hists2d_DRS_vs_TS.append(hist_subtractMedian)

                if debug:
                    hist_subtractMedian_RTSpos = rdf.Histo2D((
                        f"hist_DRS_Board{boardNo}_{var}_vs_RTSpos_{sTowerX}_{sTowerY}_subtractMedian",
                        f"DRS Board {boardNo} - {var} {chan.channelNo} in iTowerX {sTowerX} iTowerY {sTowerY} (subtract median);RTS pos;{var} Variable",
                        1024, 0, 1024, 50, ymin, ymax),
                        f"RTS_pos_{channelName}", channelName +
                        "_subtractMedian"
                    )
                    hist_subtractMedian_RTSneg = rdf.Histo2D((
                        f"hist_DRS_Board{boardNo}_{var}_vs_RTSneg_{sTowerX}_{sTowerY}_subtractMedian",
                        f"DRS Board {boardNo} - {var} {chan.channelNo} in iTowerX {sTowerX} iTowerY {sTowerY} (subtract median);RTS neg;{var} Variable",
                        1024, 0, 1024, 50, ymin, ymax),
                        f"RTS_neg_{channelName}", channelName +
                        "_subtractMedian"
                    )
                    hists2d_DRS_vs_RTSpos.append(hist_subtractMedian_RTSpos)
                    hists2d_DRS_vs_RTSneg.append(hist_subtractMedian_RTSneg)
    if debug:
        return hists2d_DRS_vs_TS, hists2d_DRS_vs_RTSpos, hists2d_DRS_vs_RTSneg
    return hists2d_DRS_vs_TS


def makeDRSHodoPlots(rdf,hodoscope_channels):
    hists2d_DRSHodo = []
    for _, DRSBoard in DRSBoards.items():
        boardNo = DRSBoard.boardNo
        for iTowerX, iTowerY in DRSBoard.GetListOfTowers():
            sTowerX = number2string(iTowerX)
            sTowerY = number2string(iTowerY)

            for var in ["Cer", "Sci"]:
                chan = DRSBoard.GetChannelByTower(
                    iTowerX, iTowerY, isCer=(var == "Cer"))

                if chan is None:
                    continue
                channelName = chan.GetChannelName()
                thre_low = 280 if channelName in drs_amplified else 15
                for axis in hodoscope_channels.keys():
                    hist_DRSHodo = rdf.Filter(
                        f"({hodoscope_channels[f'{axis}'][0]}_hodopeak_value < -100.0 ) && ({hodoscope_channels[f'{axis}'][1]}_hodopeak_value < -100.0 ) && ({channelName}_peakA > {thre_low})"
                        ).Histo2D((
                        f"hist_DRS_Board{boardNo}_{var}_peak_vs_hodopos_{axis}_{sTowerX}_{sTowerY}",
                        f"DRS Board {boardNo} - {var} {chan.channelNo} in iTowerX {sTowerX} iTowerY {sTowerY} (Hodo DeltaT vs DRS peak DeltaT);DRS peak TS - trigger TS + 1024;{axis} Hodo DeltaTS",
                        1024, 0, 1024, 2048, -1024, 1024),
                        f"{channelName}_deltaT", f"{axis}_delta_hodopeak"
                    )
                    hists2d_DRSHodo.append(hist_DRSHodo)
    return hists2d_DRSHodo

def trackDRSPlots():
    hists2d_DRS_vs_Event = []
    for _, DRSBoard in DRSBoards.items():
        boardNo = DRSBoard.boardNo
        for iTowerX, iTowerY in DRSBoard.GetListOfTowers():
            sTowerX = number2string(iTowerX)
            sTowerY = number2string(iTowerY)
            for var in ["Cer", "Sci"]:
                # Get the channel for CER or SCI
                chan = DRSBoard.GetChannelByTower(
                    iTowerX, iTowerY, isCer=(var == "Cer"))
                if chan is None:
                    continue
                channelName = chan.GetChannelName()
                mean_value = stats[channelName]['mean']
                hist = rdf.Histo2D((
                    f"hist_DRS_Board{boardNo}_{var}_vs_Event_{sTowerX}_{sTowerY}",
                    f"DRS Board {boardNo} Mean - Event vs {var} {chan.channelNo} in iTowerX {sTowerX} iTowerY {sTowerY};Event;{var} Variable",
                    nbins_Event, 0, nEvents, 200, mean_value - 100, mean_value + 100),
                    "event_n", channelName + "_mean"
                )
                hists2d_DRS_vs_Event.append(hist)
    return hists2d_DRS_vs_Event


def compareDRSChannels(channels_to_compare):
    hists_trigger = []
    for chan_name in channels_to_compare:
        hist_subtractMedian = rdf.Histo2D((
            f"hist_{chan_name}_subtractMedian",
            f"{chan_name} (subtract median);TS;DRS values",
            1024, 0, 1024,
            300, -2500, 500),
            "TS", chan_name + "_subtractMedian"
        )
        hists_trigger.append(hist_subtractMedian)
    return hists_trigger


def checkFERSvsDRSSum():
    xymax = {
        "Cer": (1000, 4000),
        "Sci": (9000, 9000)
    }
    xymax_LG = {
        "Cer": (1000, 1000),
        "Sci": (9000, 9000)
    }

    # correlate  FERS and DRS outputs
    h2s_FERS_VS_DRS = []
    h2s_FERSLG_VS_DRS = []
    for _, DRSBoard in DRSBoards.items():
        boardNo = DRSBoard.boardNo
        for iTowerX, iTowerY in DRSBoard.GetListOfTowers():
            sTowerX = number2string(iTowerX)
            sTowerY = number2string(iTowerY)

            for var in ["Cer", "Sci"]:
                chan_DRS = DRSBoard.GetChannelByTower(
                    iTowerX, iTowerY, isCer=(var == "Cer"))
                if chan_DRS is None:
                    print(
                        f"Warning: DRS Channel not found for Board{boardNo}, Tower({sTowerX}, {sTowerY}), {var}")
                    continue
                chan_FERS = None
                for _, FERSBoard in FERSBoards.items():
                    chan_FERS = FERSBoard.GetChannelByTower(
                        iTowerX, iTowerY, isCer=(var == "Cer"))
                    if chan_FERS is not None:
                        break
                if chan_FERS is None:
                    print(
                        f"Warning: FERS Channel not found for Board{boardNo}, Tower({sTowerX}, {sTowerY}), {var}")
                    continue

                h2_FERS_VS_DRS = rdf.Histo2D((
                    f"hist_FERS_VS_DRS_Board{boardNo}_{var}_{sTowerX}_{sTowerY}",
                    f"FERS vs DRS energy correlation for Board{boardNo}, Tower({sTowerX}, {sTowerY}), {var}",
                    100, 0, xymax[var][0], 100, 0, xymax[var][1]
                ),
                    f"{chan_DRS.GetChannelName()}_sum",
                    chan_FERS.GetHGChannelName(),
                )
                h2s_FERS_VS_DRS.append(h2_FERS_VS_DRS)

                h2_FERSLG_VS_DRS = rdf.Histo2D((
                    f"hist_FERSLG_VS_DRS_Board{boardNo}_{var}_{sTowerX}_{sTowerY}",
                    f"FERS LG vs DRS energy correlation for Board{boardNo}, Tower({sTowerX}, {sTowerY}), {var}",
                    100, 0, xymax_LG[var][0], 100, 0, xymax_LG[var][1]
                ),
                    f"{chan_DRS.GetChannelName()}_sum",
                    chan_FERS.GetLGChannelName(),
                )
                h2s_FERSLG_VS_DRS.append(h2_FERSLG_VS_DRS)

    # sum of FERS and DRS outputs
    h2s_FERS_VS_DRS_sum = []
    h2s_FERSLG_VS_DRS_sum = []
    for _, DRSBoard in DRSBoards.items():
        boardNo = DRSBoard.boardNo
        for var in ["Cer", "Sci"]:
            h2sum = ROOT.TH2F(
                f"hist_FERS_VS_DRS_Board{boardNo}_{var}_sum",
                f"FERS vs DRS energy correlation for Board{boardNo}, {var}",
                100, 0, xymax[var][0], 100, 0, xymax[var][1]
            )
            for h2 in h2s_FERS_VS_DRS:
                if f"Board{boardNo}_{var}" in h2.GetName():
                    h2sum.Add(h2.GetValue())
            h2s_FERS_VS_DRS_sum.append(h2sum)

            h2sum_LG = ROOT.TH2F(
                f"hist_FERSLG_VS_DRS_Board{boardNo}_{var}_sum",
                f"FERS LG vs DRS energy correlation for Board{boardNo}, {var}",
                100, 0, xymax_LG[var][0], 100, 0, xymax_LG[var][1]
            )
            for h2 in h2s_FERSLG_VS_DRS:
                if f"Board{boardNo}_{var}" in h2.GetName():
                    h2sum_LG.Add(h2.GetValue())
            h2s_FERSLG_VS_DRS_sum.append(h2sum_LG)
    return h2s_FERS_VS_DRS_sum + h2s_FERSLG_VS_DRS_sum + h2s_FERS_VS_DRS + h2s_FERSLG_VS_DRS


def checkDRSPeakvsFERS():
    h2s_DRSPeak_vs_FERS = []
    for _, DRSBoard in DRSBoards.items():
        boardNo = DRSBoard.boardNo
        for iTowerX, iTowerY in DRSBoard.GetListOfTowers():
            sTowerX = number2string(iTowerX)
            sTowerY = number2string(iTowerY)

            for var in ["Cer", "Sci"]:
                chan_DRS = DRSBoard.GetChannelByTower(
                    iTowerX, iTowerY, isCer=(var == "Cer"))
                if chan_DRS is None:
                    print(
                        f"Warning: DRS Channel not found for Board{boardNo}, Tower({sTowerX}, {sTowerY}), {var}")
                    continue
                chan_FERS = None
                for _, FERSBoard in FERSBoards.items():
                    chan_FERS = FERSBoard.GetChannelByTower(
                        iTowerX, iTowerY, isCer=(var == "Cer"))
                    if chan_FERS is not None:
                        break
                if chan_FERS is None:
                    print(
                        f"Warning: FERS Channel not found for Board{boardNo}, Tower({sTowerX}, {sTowerY}), {var}")
                    continue

                _, ymax = getDRSPlotRanges(
                    subtractMedian=True, isAmplified=chan_DRS.isAmplified)
                h2_DRSPeak_vs_FERS = rdf.Histo2D((
                    f"hist_DRSPeak_VS_FERS_Board{boardNo}_{var}_{sTowerX}_{sTowerY}",
                    f"DRS Peak vs FERS energy correlation for Board{boardNo}, Tower({sTowerX}, {sTowerY}), {var}",
                    100, 0, 9000, 100, 0, ymax
                ),
                    chan_FERS.GetHGChannelName(),
                    f"{chan_DRS.GetChannelName()}_peak",
                )
                h2s_DRSPeak_vs_FERS.append(h2_DRSPeak_vs_FERS)

    return h2s_DRSPeak_vs_FERS


def checkDRSPeakTS():
    h1s_DRSPeakTS = {}
    h1s_DRSPeakTS["Cer"] = []
    h1s_DRSPeakTS["Sci"] = []
    h2s_DRSPeakTS_Cer_vs_Sci = []

    for _, DRSBoard in DRSBoards.items():
        boardNo = DRSBoard.boardNo
        for iTowerX, iTowerY in DRSBoard.GetListOfTowers():
            sTowerX = number2string(iTowerX)
            sTowerY = number2string(iTowerY)

            channelNames = {}
            for var in ["Cer", "Sci"]:
                chan_DRS = DRSBoard.GetChannelByTower(
                    iTowerX, iTowerY, isCer=(var == "Cer"))
                if chan_DRS is None:
                    print(
                        f"Warning: DRS Channel not found for Board{boardNo}, Tower({sTowerX}, {sTowerY}), {var}")
                    continue

                channelName = chan_DRS.GetChannelName()
                channelNames[var] = channelName

                h1_DRS_PeakTS = rdf.Histo1D((
                    f"hist_DRS_PeakTS_Board{boardNo}_peakTS_{sTowerX}_{sTowerY}_{var}",
                    f"DRS Peak TS for Board{boardNo}, Tower({sTowerX}, {sTowerY}), {var};Peak TS;Counts",
                    400, 0, 400),
                    channelName + "_peakTS"
                )
                h1s_DRSPeakTS[var].append(h1_DRS_PeakTS)

            if len(channelNames) < 2:
                print(
                    f"Warning: Not enough channels found for Board{boardNo}, Tower({sTowerX}, {sTowerY})")
                continue

            h2_DRSPeak_Cer_vs_Sci = rdf.Histo2D((
                f"hist_DRSPeak_Cer_vs_Sci_Board{boardNo}_{sTowerX}_{sTowerY}",
                f"DRS Peak TS - CER vs SCI for Board{boardNo}, Tower({sTowerX}, {sTowerY});CER Peak TS;SCI Peak TS",
                400, 0, 400, 400, 0, 400),
                channelNames["Cer"] + "_peakTS",
                channelNames["Sci"] + "_peakTS"
            )
            h2s_DRSPeakTS_Cer_vs_Sci.append(h2_DRSPeak_Cer_vs_Sci)
    return h1s_DRSPeakTS["Cer"], h1s_DRSPeakTS["Sci"], h2s_DRSPeakTS_Cer_vs_Sci


def compareDRSTime(channels_to_compare):
    hists_trigger = []
    for chan_name in channels_to_compare:
        hist_TSmerge = rdf.Histo1D((
                    f"hist_{chan_name}_TSmerge",
                    f"{chan_name} (TS merge);TS edges;Counts",
                    1024, 0, 1024),
                    chan_name + "_triggerT"
                )
        hists_trigger.append(hist_TSmerge)
    return hists_trigger

if __name__ == "__main__":
    start_time = time.time()
    rootdir = f"results/root/Run{runNumber}"
    if not os.path.exists(rootdir):
        os.makedirs(rootdir)
    print("dir made")
    stats = collectFERSStats()
    rootdir = f"results/root/Run{runNumber}"
    os.makedirs(rootdir, exist_ok=True)

    stats_results = {}
    for channelName, (mean, max_value) in stats.items():
        stats_results[channelName] = (mean.GetValue(), max_value.GetValue())
    with open(f"{rootdir}/fers_stats.json", "w") as f:
        json.dump(stats_results, f, indent=4)
    print("fers_stats made")
    hists_conditions = monitorConditions()
    outfile = ROOT.TFile(f"{rootdir}/conditions_vs_event.root", "RECREATE")
    for hist in hists_conditions:
        hist.SetDirectory(outfile)
        hist.Write()
    outfile.Close()
    print("conditions_vs_event made")
    hists1d_FERS = makeFERS1DPlots()

    # hists2d_FERS = makeFERS2DPlots()
    # hists2d_FERS_vs_Event = trackFERSPlots()

        #hists1d_DRS = makeDRS1DPlots()
    hists2d_DRS_vs_RTSpos = None
    hists2d_DRS_vs_RTSneg = None
    if debugDRS:
        hists2d_DRS_vs_TS, hists2d_DRS_vs_RTSpos, hists2d_DRS_vs_RTSneg = makeDRS2DPlots(
            debug=True)
    else:
        hists2d_DRS_vs_TS = makeDRS2DPlots(debug=False)
    # hists2d_DRS_vs_Event = trackDRSPlots()

    hists2d_DRSPeak_vs_FERS = checkDRSPeakvsFERS()

    hists1d_DRSPeakTS_Cer, hists1d_DRSPeakTS_Sci, hists2d_DRSPeakTS_Cer_vs_Sci = checkDRSPeakTS()

    time_reference_channels = buildTimeReferenceChannels(run=runNumber)
    hists2d_time_reference = compareDRSChannels(time_reference_channels)

    hodo_trigger_channels = buildHodoTriggerChannels(run=runNumber)
    hists2d_hodo_trigger = compareDRSChannels(hodo_trigger_channels)

    hodo_pos_channels = buildHodoPosChannels(run=runNumber)
    channels = [channel for channels in hodo_pos_channels.values()
                    for channel in channels]
    hists2d_hodo_pos = compareDRSChannels(channels)

    hists2d_FERS_vs_DRSs = checkFERSvsDRSSum()
    #rdf = processHodoPeaks(rdf,hodo_pos_channels)
    #hists2d_DRS_vs_Hodo = makeDRSHodoPlots(rdf, hodo_pos_channels)
    print("Save histograms")

    print("\033[94mSave results\033[0m")
    # dump stats into a json file

    # Save histograms to an output ROOT file
    
    
    outfile = ROOT.TFile(f"{rootdir}/fers_all_channels_1D.root", "RECREATE")
    for hist in hists1d_FERS:
        hist.Write()
    outfile.Close()
    print("fers_all_channels made")
    # outfile = ROOT.TFile(f"{rootdir}/fers_all_channels_2D.root", "RECREATE")
    # for hist in hists2d_FERS:
    #    hist.Write()
    # outfile.Close()
    # outfile = ROOT.TFile(
    #    f"{rootdir}/fers_all_channels_2D_vs_event.root", "RECREATE")
    # for hist in hists2d_FERS_vs_Event:
    #    hist.Write()
    # outfile.Close()
    #

    #outfile_DRS = ROOT.TFile(f"{rootdir}/drs_all_channels_1D.root", "RECREATE")
    #for hist in hists1d_DRS:
    #    hist.SetDirectory(outfile_DRS)
    #    hist.Write()
    #outfile_DRS.Close()
    outfile_DRS = ROOT.TFile(f"{rootdir}/drs_vs_TS.root", "RECREATE")
    for hist in hists2d_DRS_vs_TS:
        hist.SetDirectory(outfile_DRS)
        hist.Write()
    outfile_DRS.Close()
    print("drs_vs_TS made")
    if debugDRS:
        outfile_DRS_RTSpos = ROOT.TFile(
            f"{rootdir}/drs_vs_RTSpos.root", "RECREATE")
        for hist in hists2d_DRS_vs_RTSpos:
            hist.SetDirectory(outfile_DRS_RTSpos)
            hist.Write()
        outfile_DRS_RTSpos.Close()

        outfile_DRS_RTSneg = ROOT.TFile(
            f"{rootdir}/drs_vs_RTSneg.root", "RECREATE")
        for hist in hists2d_DRS_vs_RTSneg:
            hist.SetDirectory(outfile_DRS_RTSneg)
            hist.Write()
        outfile_DRS_RTSneg.Close()
        # outfile_DRS = ROOT.TFile(
        #    f"{rootdir}/drs_all_channels_2D_vs_event.root", "RECREATE")
        # for hist in hists2d_DRS_vs_Event:
        #    hist.SetDirectory(outfile_DRS)
        #    hist.Write()
        # outfile_DRS.Close()
        #outfile_DRS_Hodo = ROOT.TFile(f"{rootdir}/drs_all_channels_peak_vs_hodo.root", "RECREATE")
        #for hist in hists2d_DRS_vs_Hodo:
        #    hist.SetDirectory(outfile_DRS_Hodo)
        #    hist.Write()
        #outfile_DRS_Hodo.Close()

        outfile_DRSPeak_vs_FERS = ROOT.TFile(
            f"{rootdir}/drs_peak_vs_fers.root", "RECREATE")
        for hist in hists2d_DRSPeak_vs_FERS:
            hist.SetDirectory(outfile_DRSPeak_vs_FERS)
            hist.Write()
        outfile_DRSPeak_vs_FERS.Close()
        print("drs_peak_vs_fers made")
        outfile_DRSPeakTS = ROOT.TFile(f"{rootdir}/drs_peak_ts.root", "RECREATE")
        for hist in hists1d_DRSPeakTS_Cer:
            hist.SetDirectory(outfile_DRSPeakTS)
            hist.Write()
        for hist in hists1d_DRSPeakTS_Sci:
            hist.SetDirectory(outfile_DRSPeakTS)
            hist.Write()
        for hist in hists2d_DRSPeakTS_Cer_vs_Sci:
            hist.SetDirectory(outfile_DRSPeakTS)
            hist.Write()
        outfile_DRSPeakTS.Close()
        print("drs_peak_ts made")
        outfile_time_reference = ROOT.TFile(
            f"{rootdir}/time_reference_channels.root", "RECREATE")
        for hist in hists2d_time_reference:
            hist.SetDirectory(outfile_time_reference)
            hist.Write()
        outfile_time_reference.Close()
        print("time_reference_channels made")
        #outfile_hodo_trigger = ROOT.TFile(
        #    f"{rootdir}/hodo_trigger_channels.root", "RECREATE")
        #for hist in hists2d_hodo_trigger:
        #    hist.SetDirectory(outfile_hodo_trigger)
        #    hist.Write()
        #outfile_hodo_trigger.Close()

        #outfile_hodo_pos = ROOT.TFile(
        #    f"{rootdir}/hodo_pos_channels.root", "RECREATE")
        #for hist in hists2d_hodo_pos:
        #    hist.SetDirectory(outfile_hodo_pos)
        #    hist.Write()
        #outfile_hodo_pos.Close()

        outfile_FERS_DRS = ROOT.TFile(
        f"{rootdir}/fers_vs_drs.root", "RECREATE")
        for hist in hists2d_FERS_vs_DRSs:
            hist.SetDirectory(outfile_FERS_DRS)
            hist.Write()
        outfile_FERS_DRS.Close()

    time_taken = time.time() - start_time
    print(f"Finished running script in {time_taken:.2f} seconds")
