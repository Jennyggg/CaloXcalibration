import sys
sys.path.append("CMSPLOTS")  # noqa
import ROOT
from myFunction import DrawHistos
from utils.channel_map import buildDRSBoards, buildFERSBoards, buildTimeReferenceChannels, buildHodoTriggerChannels, buildHodoPosChannels, buildDRSChannelsAmplified
from utils.utils import number2string, round_up_to_1eN, readTSconfig

from utils.html_generator import generate_html
from utils.visualization import visualizeFERSBoards
from utils.validateMap import DrawFERSBoards, DrawDRSBoards
from utils.colors import colors
from configs.plotranges import getDRSPlotRanges
from runconfig import runNumber
import numpy as np


print("Start running script")
ROOT.gROOT.SetBatch(True)
#TSconfig = readTSconfig(run = runNumber)
DRSBoards = buildDRSBoards(run=runNumber)
FERSBoards = buildFERSBoards(run=runNumber)
time_reference_channels = buildTimeReferenceChannels(run=runNumber)
hodo_trigger_channels = buildHodoTriggerChannels(run=runNumber)
hodo_pos_channels = buildHodoPosChannels(run=runNumber)
drs_amplified = buildDRSChannelsAmplified(runNumber)

rootdir = f"results/root/Run{runNumber}/"
plotdir = f"results/plots/Run{runNumber}/"
htmldir = f"results/html/Run{runNumber}/"


def makeConditionsPlots():
    plots = []
    outdir_plots = f"{plotdir}/Conditions_vs_Event"
    infile_name = f"{rootdir}/conditions_vs_event.root"
    infile = ROOT.TFile(infile_name, "READ")

    hprofiles_SipmHV = []
    hprofiles_SipmI = []
    hprofiles_TempDET = []
    hprofiles_TempFPGA = []

    legends = []

    for _, FERSBoard in FERSBoards.items():
        boardNo = FERSBoard.boardNo
        hist_SipmHV_name = f"hist_FERS_Board{boardNo}_SipmHV_vs_Event"
        hist_SipmI_name = f"hist_FERS_Board{boardNo}_SipmI_vs_Event"
        hist_TempDET_name = f"hist_FERS_Board{boardNo}_TempDET_vs_Event"
        hist_TempFPGA_name = f"hist_FERS_Board{boardNo}_TempFPGA_vs_Event"

        hist_SipmHV = infile.Get(hist_SipmHV_name)
        hist_SipmI = infile.Get(hist_SipmI_name)
        hist_TempDET = infile.Get(hist_TempDET_name)
        hist_TempFPGA = infile.Get(hist_TempFPGA_name)

        if not (hist_SipmHV and hist_SipmI and hist_TempDET and hist_TempFPGA):
            print(
                f"Warning: Histograms {hist_SipmHV_name}, {hist_SipmI_name}, {hist_TempDET_name}, or {hist_TempFPGA_name} not found in {infile_name}")
            continue

        hprofile_SipmHV = hist_SipmHV.ProfileX(
            f"hprof_FERS_Board{boardNo}_SipmHV_vs_Event")
        hprofiles_SipmHV.append(hprofile_SipmHV)

        hprofile_SipmI = hist_SipmI.ProfileX(
            f"hprof_FERS_Board{boardNo}_SipmI_vs_Event")
        hprofiles_SipmI.append(hprofile_SipmI)

        hprofile_TempDET = hist_TempDET.ProfileX(
            f"hprof_FERS_Board{boardNo}_TempDET_vs_Event")
        hprofiles_TempDET.append(hprofile_TempDET)

        hprofile_TempFPGA = hist_TempFPGA.ProfileX(
            f"hprof_FERS_Board{boardNo}_TempFPGA_vs_Event")
        hprofiles_TempFPGA.append(hprofile_TempFPGA)

        legends.append(str(boardNo))

        nEvents = hist_SipmHV.GetXaxis().GetXmax()
        zmax = nEvents * 10

        extraToDraw = ROOT.TPaveText(0.20, 0.80, 0.60, 0.90, "NDC")
        extraToDraw.SetTextAlign(11)
        extraToDraw.SetFillColorAlpha(0, 0)
        extraToDraw.SetBorderSize(0)
        extraToDraw.SetTextFont(42)
        extraToDraw.SetTextSize(0.04)
        extraToDraw.AddText(f"Board: {FERSBoard.boardNo}")

        # output_name = f"Conditions_Board{boardNo}_SipmHV_vs_Event"
        # DrawHistos([hist_SipmHV], f"", 0, nEvents, "Event", 26, 29, "Voltage (V)",
        #           output_name,
        #           dology=False, drawoptions="COLZ", doth2=True, zmin=1, zmax=zmax, dologz=True, extraToDraw=extraToDraw,
        #           outdir=outdir_plots, addOverflow=True, runNumber=runNumber)
        # plots.append(output_name + ".png")

        # output_name = f"Conditions_Board{boardNo}_SipmI_vs_Event"
        # DrawHistos([hist_SipmI], f"", 0, nEvents, "Event", 0.02, 0.2, "Current (mA)",
        #           output_name,
        #           dology=False, drawoptions="COLZ", doth2=True, zmin=1, zmax=zmax, dologz=True, extraToDraw=extraToDraw,
        #           outdir=outdir_plots, addOverflow=True, runNumber=runNumber)
        # plots.append(output_name + ".png")

        # output_name = f"Conditions_Board{boardNo}_TempDET_vs_Event"
        # DrawHistos([hist_TempDET], f"", 0, nEvents, "Event", 10, 30, "Temperature (C)",
        #           output_name,
        #           dology=False, drawoptions="COLZ", doth2=True, zmin=1, zmax=zmax, dologz=True, extraToDraw=extraToDraw,
        #           outdir=outdir_plots, addOverflow=True, runNumber=runNumber)
        # plots.append(output_name + ".png")

        # output_name = f"Conditions_Board{boardNo}_TempFPGA_vs_Event"
        # DrawHistos([hist_TempFPGA], f"", 0, nEvents, "Event", 30, 50, "Temperature (C)",
        #           output_name,
        #           dology=False, drawoptions="COLZ", doth2=True, zmin=1, zmax=zmax, dologz=True, extraToDraw=extraToDraw,
        #           outdir=outdir_plots, addOverflow=True, runNumber=runNumber)
        # plots.append(output_name + ".png")

    # Draw the profiles
    legendPos = [0.3, 0.7, 0.9, 0.9]
    output_name = "Conditions_SipmHV_vs_Event"
    DrawHistos(hprofiles_SipmHV, legends, 0, nEvents, "Event", 26, 29, "Voltage (V)",
               output_name,
               dology=False, drawoptions="HIST", mycolors=colors, addOverflow=True, addUnderflow=True,
               outdir=outdir_plots, runNumber=runNumber, legendNCols=3, legendPos=legendPos)
    plots.insert(0, output_name + ".png")

    output_name = "Conditions_SipmI_vs_Event"
    DrawHistos(hprofiles_SipmI, legends, 0, nEvents, "Event", 0.02, 0.2, "Current (mA)",
               output_name,
               dology=False, drawoptions="HIST", mycolors=colors, addOverflow=True, addUnderflow=True,
               outdir=outdir_plots, runNumber=runNumber, legendNCols=3, legendPos=legendPos)
    plots.insert(1, output_name + ".png")

    output_name = "Conditions_TempDET_vs_Event"
    DrawHistos(hprofiles_TempDET, legends, 0, nEvents, "Event", 15, 30, "Temperature (C)",
               output_name,
               dology=False, drawoptions="HIST", mycolors=colors, addOverflow=True, addUnderflow=True,
               outdir=outdir_plots, runNumber=runNumber, legendNCols=3, legendPos=legendPos)
    plots.insert(2, output_name + ".png")

    output_name = "Conditions_TempFPGA_vs_Event"
    DrawHistos(hprofiles_TempFPGA, legends, 0, nEvents, "Event", 30, 50, "Temperature (C)",
               output_name,
               dology=False, drawoptions="HIST", mycolors=colors, addOverflow=True, addUnderflow=True,
               outdir=outdir_plots, runNumber=runNumber, legendNCols=3, legendPos=legendPos)
    plots.insert(3, output_name + ".png")

    output_html = f"{htmldir}/Conditions_vs_Event/index.html"
    generate_html(plots, outdir_plots, plots_per_row=4,
                  output_html=output_html)
    return output_html


def makeFERS1DPlots():
    plots = []

    infile_name = f"{rootdir}fers_all_channels_1D.root"
    infile = ROOT.TFile(infile_name, "READ")
    outdir_plots = f"{plotdir}/FERS_1D"
    for _, FERSBoard in FERSBoards.items():
        boardNo = FERSBoard.boardNo
        for iTowerX, iTowerY in FERSBoard.GetListOfTowers():
            sTowerX = number2string(iTowerX)
            sTowerY = number2string(iTowerY)
            for gain in ["HG","LG"]:
                hist_C_name = f"hist_FERS_Board{boardNo}_Cer_{sTowerX}_{sTowerY}_{gain}"
                hist_S_name = f"hist_FERS_Board{boardNo}_Sci_{sTowerX}_{sTowerY}_{gain}"
                hist_C = infile.Get(hist_C_name)
                hist_S = infile.Get(hist_S_name)
                if not hist_C or not hist_S:
                    print(
                        f"Warning: Histograms {hist_C_name} or {hist_S_name} not found in {infile_name}")
                    continue

                extraToDraw = ROOT.TPaveText(0.20, 0.65, 0.60, 0.90, "NDC")
                extraToDraw.SetTextAlign(11)
                extraToDraw.SetFillColorAlpha(0, 0)
                extraToDraw.SetBorderSize(0)
                extraToDraw.SetTextFont(42)
                extraToDraw.SetTextSize(0.04)
                extraToDraw.AddText(
                    f"Board: {FERSBoard.boardNo}")
                extraToDraw.AddText(f"Tower X: {iTowerX}")
                extraToDraw.AddText(f"Tower Y: {iTowerY}")
                extraToDraw.AddText(
                    f"Cer Channel: {FERSBoard.GetChannelByTower(iTowerX, iTowerY, isCer=True).channelNo}")
                extraToDraw.AddText(
                    f"Sci Channel: {FERSBoard.GetChannelByTower(iTowerX, iTowerY, isCer=False).channelNo}")

                output_name = f"Energy_Board{boardNo}_iTowerX{sTowerX}_iTowerY{sTowerY}_{gain}"
                DrawHistos([hist_S, hist_C], ["Sci", "Cer"], 0, 8000, f"Energy {gain}", 1, 1e5, "Counts",
                       output_name,
                       dology=True, drawoptions="HIST", mycolors=[4,2], addOverflow=True, addUnderflow=True, extraToDraw=extraToDraw,
                       outdir=outdir_plots, runNumber=runNumber)

                plots.append(output_name + ".png")

    output_html = f"{htmldir}/FERS_1D/index.html"
    generate_html(plots, outdir_plots,
                  output_html=output_html)
    return output_html


def makeFERSStatsPlots():
    plots = []
    outdir_plots = f"{plotdir}/FERS_Stats"
    # load the json file
    import json
    infile_name = f"{rootdir}/fers_stats.json"
    with open(infile_name, "r") as f:
        stats = json.load(f)

    xmax = 14
    xmin = -14
    ymax = 10
    ymin = -10
    W_ref = 1000
    H_ref = 1100
    valuemaps_HG_mean = {}
    valuemaps_HG_max = {}
    valuemaps_LG_mean = {}
    valuemaps_LG_max = {}

    for channelName, (vmean, vmax) in stats.items():
        if "energyHG" in channelName:
            valuemaps_HG_mean[channelName] = vmean
            valuemaps_HG_max[channelName] = vmax
        elif "energyLG" in channelName:
            valuemaps_LG_mean[channelName] = vmean
            valuemaps_LG_max[channelName] = vmax

    [h2_Cer_HG_mean, h2_Cer_3mm_HG_mean], [h2_Sci_HG_mean, h2_Sci_3mm_HG_mean] = visualizeFERSBoards(
        FERSBoards, valuemaps_HG_mean, suffix=f"Run{runNumber}_HG_mean", useHG=True)
    [h2_Cer_HG_max, h2_Cer_3mm_HG_max], [h2_Sci_HG_max, h2_Sci_3mm_HG_max] = visualizeFERSBoards(
        FERSBoards, valuemaps_HG_max, suffix=f"Run{runNumber}_HG_max", useHG=True)
    [h2_Cer_LG_mean, h2_Cer_3mm_LG_mean], [h2_Sci_LG_mean, h2_Sci_3mm_LG_mean] = visualizeFERSBoards(
        FERSBoards, valuemaps_LG_mean, suffix=f"Run{runNumber}_LG_mean", useHG=False)
    [h2_Cer_LG_max, h2_Cer_3mm_LG_max], [h2_Sci_LG_max, h2_Sci_3mm_LG_max] = visualizeFERSBoards(
        FERSBoards, valuemaps_LG_max, suffix=f"Run{runNumber}_LG_max", useHG=False)

    output_name = f"FERS_Boards_Run{runNumber}_Stats_HG_mean"
    DrawHistos([h2_Cer_HG_mean, h2_Cer_3mm_HG_mean], "", xmin, xmax, "iX", ymin,
               ymax, "iY", output_name + "_Cer", dology=False, drawoptions=["col,text", "col,text"],
               outdir=outdir_plots, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText="Cer", runNumber=runNumber, zmin=0, zmax=8000)
    plots.append(output_name + "_Cer.png")
    DrawHistos([h2_Sci_HG_mean, h2_Sci_3mm_HG_mean], "", xmin, xmax, "iX", ymin,
               ymax, "iY", output_name + "_Sci", dology=False, drawoptions=["col,text", "col,text"],
               outdir=outdir_plots, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText="Sci", runNumber=runNumber, zmin=0, zmax=8000)
    plots.append(output_name + "_Sci.png")

    output_name = f"FERS_Boards_Run{runNumber}_Stats_HG_max"
    DrawHistos([h2_Cer_HG_max, h2_Cer_3mm_HG_max], "", xmin, xmax, "iX", ymin,
               ymax, "iY", output_name + "_Cer", dology=False, drawoptions=["col,text", "col,text"],
               outdir=outdir_plots, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText="Cer", runNumber=runNumber, zmin=0, zmax=8000)
    plots.append(output_name + "_Cer.png")
    DrawHistos([h2_Sci_HG_max, h2_Sci_3mm_HG_max], "", xmin, xmax, "iX", ymin,
               ymax, "iY", output_name + "_Sci", dology=False, drawoptions=["col,text", "col,text"],
               outdir=outdir_plots, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText="Sci", runNumber=runNumber, zmin=0, zmax=8000)
    plots.append(output_name + "_Sci.png")

    output_name = f"FERS_Boards_Run{runNumber}_Stats_LG_mean"
    DrawHistos([h2_Cer_LG_mean, h2_Cer_3mm_LG_mean], "", xmin, xmax, "iX", ymin,
               ymax, "iY", output_name + "_Cer", dology=False, drawoptions=["col,text", "col,text"],
               outdir=outdir_plots, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText="Cer", runNumber=runNumber, zmin=0, zmax=8000)
    plots.append(output_name + "_Cer.png")
    DrawHistos([h2_Sci_LG_mean, h2_Sci_3mm_LG_mean], "", xmin, xmax, "iX", ymin,
               ymax, "iY", output_name + "_Sci", dology=False, drawoptions=["col,text", "col,text"],
               outdir=outdir_plots, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText="Sci", runNumber=runNumber, zmin=0, zmax=8000)
    plots.append(output_name + "_Sci.png")

    output_name = f"FERS_Boards_Run{runNumber}_Stats_LG_max"
    DrawHistos([h2_Cer_LG_max, h2_Cer_3mm_LG_max], "", xmin, xmax, "iX", ymin,
               ymax, "iY", output_name + "_Cer", dology=False, drawoptions=["col,text", "col,text"],
               outdir=outdir_plots, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText="Cer", runNumber=runNumber, zmin=0, zmax=8000)
    plots.append(output_name + "_Cer.png")
    DrawHistos([h2_Sci_LG_max, h2_Sci_3mm_LG_max], "", xmin, xmax, "iX", ymin,
               ymax, "iY", output_name + "_Sci", dology=False, drawoptions=["col,text", "col,text"],
               outdir=outdir_plots, doth2=True, W_ref=W_ref, H_ref=H_ref, extraText="Sci", runNumber=runNumber, zmin=0, zmax=8000)
    plots.append(output_name + "_Sci.png")

    output_html = f"{htmldir}/FERS_Stats/index.html"
    generate_html(plots, outdir_plots, plots_per_row=2,
                  output_html=output_html)
    return output_html


# 2D FERS histograms, hg vs lg
def makeFERS2DPlots():
    plots = []
    outdir_plots = f"{plotdir}/FERS_2D"
    infile_name = f"{rootdir}/fers_all_channels_2D.root"
    infile = ROOT.TFile(infile_name, "READ")
    for _, FERSBoard in FERSBoards.items():
        boardNo = FERSBoard.boardNo
        for iTowerX, iTowerY in FERSBoard.GetListOfTowers():
            sTowerX = number2string(iTowerX)
            sTowerY = number2string(iTowerY)
            for var in ["Cer", "Sci"]:
                chan = FERSBoard.GetChannelByTower(
                    iTowerX, iTowerY, isCer=(var == "Cer"))
                hist_name = f"hist_FERS_Board{boardNo}_{var}_{sTowerX}_{sTowerY}_hg_vs_lg"
                hist = infile.Get(hist_name)

                extraToDraw = ROOT.TPaveText(0.20, 0.70, 0.60, 0.90, "NDC")
                extraToDraw.SetTextAlign(11)
                extraToDraw.SetFillColorAlpha(0, 0)
                extraToDraw.SetBorderSize(0)
                extraToDraw.SetTextFont(42)
                extraToDraw.SetTextSize(0.04)
                extraToDraw.AddText(
                    f"Board: {FERSBoard.boardNo}")
                extraToDraw.AddText(f"Tower X: {iTowerX}")
                extraToDraw.AddText(f"Tower Y: {iTowerY}")
                extraToDraw.AddText(f"{var} Channel: {chan.channelNo}")

                output_name = f"FERS_Board{boardNo}_{var}_{sTowerX}_{sTowerY}_hg_vs_lg"
                DrawHistos([hist], f"", 0, 9000, "HG", 0, 1500, "LG",
                           output_name,
                           dology=False, drawoptions="COLZ", doth2=True, zmin=1, zmax=1e4, dologz=True, extraToDraw=extraToDraw,
                           outdir=outdir_plots, addOverflow=True, runNumber=runNumber)
                plots.append(output_name + ".png")
    output_html = f"{htmldir}/FERS_2D/index.html"
    generate_html(plots, outdir_plots, plots_per_row=4,
                  output_html=output_html)
    return output_html


# FERS output vs event
def trackFERSPlots():
    plots = []
    outdir_plots = f"{plotdir}/FERS_vs_Event"
    infile_name = f"{rootdir}/fers_all_channels_2D_vs_event.root"
    infile = ROOT.TFile(infile_name, "READ")
    for _, FERSBoard in FERSBoards.items():
        boardNo = FERSBoard.boardNo
        for iTowerX, iTowerY in FERSBoard.GetListOfTowers():
            sTowerX = number2string(iTowerX)
            sTowerY = number2string(iTowerY)
            for var in ["Cer", "Sci"]:
                chan = FERSBoard.GetChannelByTower(
                    iTowerX, iTowerY, isCer=(var == "Cer"))
                hist_name = f"hist_FERS_Board{boardNo}_{var}_vs_Event_{sTowerX}_{sTowerY}"
                hist = infile.Get(hist_name)

                if not hist:
                    print(
                        f"Warning: Histogram {hist_name} not found in {infile_name}")
                    continue

                extraToDraw = ROOT.TPaveText(0.20, 0.70, 0.60, 0.90, "NDC")
                extraToDraw.SetTextAlign(11)
                extraToDraw.SetFillColorAlpha(0, 0)
                extraToDraw.SetBorderSize(0)
                extraToDraw.SetTextFont(42)
                extraToDraw.SetTextSize(0.04)
                extraToDraw.AddText(
                    f"Board: {FERSBoard.boardNo}")
                extraToDraw.AddText(f"Tower X: {iTowerX}")
                extraToDraw.AddText(f"Tower Y: {iTowerY}")
                extraToDraw.AddText(f"{var} Channel: {chan.channelNo}")

                nEvents = hist.GetXaxis().GetXmax()

                output_name = f"FERS_Board{boardNo}_{var}_{sTowerX}_{sTowerY}_vs_Event"
                DrawHistos([hist], "", 0, nEvents, "Event", 1, 1e5, f"{var} Energy HG",
                           output_name,
                           dology=True, drawoptions="COLZ", doth2=True, zmin=1, zmax=1e4, dologz=True,
                           extraToDraw=extraToDraw,
                           outdir=outdir_plots, addOverflow=True, runNumber=runNumber)
                plots.append(output_name + ".png")
    output_html = f"{htmldir}/FERS_vs_Event/index.html"
    generate_html(plots, outdir_plots, plots_per_row=4,
                  output_html=output_html)
    return output_html


# 1D histograms for DRS variables
def makeDRS1DPlots():
    # PeakT distributions of DRS peaks passing certain thresholds
    plots = []
    infile_name = f"{rootdir}/drs_all_channels_1D.root"
    infile = ROOT.TFile(infile_name, "READ")
    hist_peak_merge = {
        "Cer": [None]*(len(TSconfig["threBinsCer"])-1),
        "Sci": [None]*(len(TSconfig["threBinsSci"])-1)
    }
    hist_deltaT_merge = {
        "Cer": [None]*(len(TSconfig["threBinsCer"])-1),
        "Sci": [None]*(len(TSconfig["threBinsSci"])-1)
    }
    hist_deltaT_merge_board = {
        "Cer": {},
        "Sci": {},
    }
    # Get the histrograms and merge on level of boards or all.
    for _, DRSBoard in DRSBoards.items():
        boardNo = DRSBoard.boardNo
        hist_deltaT_merge_board["Cer"][boardNo] = [None]*(len(TSconfig["threBinsCer"])-1)
        hist_deltaT_merge_board["Sci"][boardNo] = [None]*(len(TSconfig["threBinsSci"])-1)
        for iTowerX, iTowerY in DRSBoard.GetListOfTowers():
            sTowerX = number2string(iTowerX)
            sTowerY = number2string(iTowerY)
            for var in ["Cer", "Sci"]:
                chan = DRSBoard.GetChannelByTower(
                        iTowerX, iTowerY, isCer=(var == "Cer"))
                channelName = chan.GetChannelName()
                amp_factor = 1
                if channelName in drs_amplified:
                     amp_factor = 10
                for ithr in range(len(TSconfig[f"threBins{var}"])-1):
                    hist_peak_name = f"hist_DRS_Board{boardNo}_{var}_{sTowerX}_{sTowerY}_peakT_amp{TSconfig[f'threBins{var}'][ithr]*amp_factor}-{TSconfig[f'threBins{var}'][ithr+1]*amp_factor}"
                    hist_peak = infile.Get(hist_peak_name)
                    hist_deltaT_name = f"hist_DRS_Board{boardNo}_{var}_{sTowerX}_{sTowerY}_deltaT_amp{TSconfig[f'threBins{var}'][ithr]*amp_factor}-{TSconfig[f'threBins{var}'][ithr+1]*amp_factor}"
                    hist_deltaT = infile.Get(hist_deltaT_name)
                    if hist_peak:
                        if hist_peak_merge[var][ithr]:
                           hist_peak_merge[var][ithr].Add(hist_peak)
                        else:
                            hist_peak_merge[var][ithr] = hist_peak.Clone(f"hist_DRS_{var}_peakT_merge_amp{TSconfig[f'threBins{var}'][ithr]*amp_factor}-{TSconfig[f'threBins{var}'][ithr+1]*amp_factor}")
                    if hist_deltaT:
                        if hist_deltaT_merge[var][ithr]:
                           hist_deltaT_merge[var][ithr].Add(hist_deltaT)
                        else:
                            hist_deltaT_merge[var][ithr] = hist_deltaT.Clone(f"hist_DRS_{var}_deltaT_merge_amp{TSconfig[f'threBins{var}'][ithr]*amp_factor}-{TSconfig[f'threBins{var}'][ithr+1]*amp_factor}")
                        if channelName in drs_amplified:
                            if "amplified" not in hist_deltaT_merge_board[var].keys():
                                hist_deltaT_merge_board[var]["amplified"] = [None]*(len(TSconfig[f"threBins{var}"])-1)
                            if hist_deltaT_merge_board[var]["amplified"][ithr]:
                                hist_deltaT_merge_board[var]["amplified"][ithr].Add(hist_deltaT)    
                            else:
                                hist_deltaT_merge_board[var]["amplified"][ithr] = hist_deltaT.Clone(f"hist_DRS_board{boardNo}_amplified_{var}_deltaT_merge_amp{TSconfig[f'threBins{var}'][ithr]*amp_factor}-{TSconfig[f'threBins{var}'][ithr+1]*amp_factor}")
                        else:
                            if hist_deltaT_merge_board[var][boardNo][ithr]:
                                hist_deltaT_merge_board[var][boardNo][ithr].Add(hist_deltaT)
                            else:
                                hist_deltaT_merge_board[var][boardNo][ithr] = hist_deltaT.Clone(f"hist_DRS_board{boardNo}_{var}_deltaT_merge_amp{TSconfig[f'threBins{var}'][ithr]*amp_factor}-{TSconfig[f'threBins{var}'][ithr+1]*amp_factor}")

    # Plot merged peak TS for all Cer/Sci channels
    for hist_Ttype, window_shift, xlabel, hist_merge in zip(["peakT","deltaT"],[0,1024], ["peak TS","peak TS - trigger TS + 1024"],[hist_peak_merge, hist_deltaT_merge]):
        if None in  hist_merge["Cer"] or None in hist_merge["Sci"]:
            print(
                f"Warning: Histograms hist_DRS_{var}_{hist_Ttype}_merge has None")
        else:
            extraToDraw = ROOT.TPaveText(0.15, 0.65, 0.9, 0.9, "NDC")
            extraToDraw.SetTextAlign(11)
            extraToDraw.SetFillColorAlpha(0, 0)
            extraToDraw.SetBorderSize(0)
            extraToDraw.SetTextFont(42)
            extraToDraw.SetTextSize(0.04)
            extraToDraw.AddText(f"Merged {hist_Ttype} for all broads")
            extraToDraw.AddText(f"TS window: [{TSconfig[f'{hist_Ttype}SWindow'][0]+window_shift}, {TSconfig[f'{hist_Ttype}SWindow'][1]+window_shift}]")
            label_merge = {
                "Cer": [],
                "Sci": []
            }
            for var in ["Cer", "Sci"]:
                bin_centers = (np.arange(TSconfig[f"{hist_Ttype}SWindow"][0]+window_shift,TSconfig[f"{hist_Ttype}SWindow"][1]+window_shift)+np.arange(TSconfig[f"{hist_Ttype}SWindow"][0]+1+window_shift,TSconfig[f"{hist_Ttype}SWindow"][1]+1+window_shift))/2
                var_mean = []
                var_std = []
                for ithr in range(len(TSconfig[f"threBins{var}"])-1):
                    hist_merge_window_y = np.array([hist_merge[var][ithr].GetBinContent(i) for i in range(TSconfig[f"{hist_Ttype}SWindow"][0]+1+window_shift,TSconfig[f"{hist_Ttype}SWindow"][1]+1+window_shift)])
                    if sum(hist_merge_window_y):
                        var_mean.append(np.average(bin_centers, weights = hist_merge_window_y))
                        var_std.append(np.sqrt(np.average((bin_centers-var_mean[ithr])**2, weights = hist_merge_window_y)))
                    else:
                        var_mean.append(np.nan)
                        var_std.append(np.nan)
                    hist_merge[var][ithr].Rebin(10)
                    label_merge[var].append(f"{var} amp: {TSconfig[f'threBins{var}'][ithr]*amp_factor}-{TSconfig[f'threBins{var}'][ithr+1]*amp_factor}")
                    extraToDraw.AddText(f"{var} std: "+", ".join([f"{var_std[ithr]:.2f}" for ithr in range(len(TSconfig[f"threBins{var}"])-1)]))


            output_name = f"DRS_{hist_Ttype}_Merged"
            outdir_plots = outdir + "/DRS_1D"
            DrawHistos(hist_merge["Sci"] + hist_merge["Cer"], label_merge["Sci"]+label_merge["Cer"],TSconfig[f"{hist_Ttype}SWindow"][0]+window_shift, TSconfig[f"{hist_Ttype}SWindow"][1]+window_shift, xlabel, 0.5, 1e6, "Counts",
            output_name,
            dology=True, drawoptions="HIST", mycolors=
            np.concatenate((TSconfig["threColorsSci"], TSconfig["threColorsCer"])).tolist(), addOverflow=True, extraToDraw=extraToDraw,
            legendPos=(0.18, 0.5, 0.90, 0.65),
            legendNCols = len(hist_merge["Cer"]),
            outdir=outdir_plots)
            plots.append(output_name + ".png")

    # Plot merged peak TS corrected by time reference for all Cer/Sci channels in each board

    bin_centers = (np.arange(TSconfig["deltaTSWindow"][0]+1024,TSconfig["deltaTSWindow"][1]+1024)+np.arange(TSconfig["deltaTSWindow"][0]+1025,TSconfig["deltaTSWindow"][1]+1025))/2
    None_in_board = False
    for var in ["Cer","Sci"]:
        for boardNo in hist_deltaT_merge_board[var].keys():
            if None in hist_deltaT_merge_board[var][boardNo]:
                print(
                f"Warning: Histograms hist_DRS_board{boardNo}_{var}_deltaT_merge has None")
            None_in_board = None_in_board + (None in hist_deltaT_merge_board[var][boardNo])
    if not None_in_board:
        hists_plot = []
        extraToDraw = ROOT.TPaveText(0.15, 0.65, 0.9, 0.9, "NDC")
        extraToDraw.SetTextAlign(11)
        extraToDraw.SetFillColorAlpha(0, 0)
        extraToDraw.SetBorderSize(0)
        extraToDraw.SetTextFont(42)
        extraToDraw.SetTextSize(0.03)
        extraToDraw.AddText(f"Merged delta TS")
        extraToDraw.AddText(f"TS window: [{TSconfig['deltaTSWindow'][0]+1024}, {TSconfig['deltaTSWindow'][1]+1024}]")
        label_merge = {
            "Cer": [],
            "Sci": []
        }
        for var in ["Sci","Cer"]:
            var_mean = []
            var_std = []
            for ib, boardNo in enumerate(hist_deltaT_merge_board[var].keys()):
                amp_factor = 1
                if boardNo == "amplified": amp_factor = 10
                for ithr in range(len(TSconfig[f"threBins{var}"])-1):
                    hist_merge_window_y = np.array([hist_deltaT_merge_board[var][boardNo][ithr].GetBinContent(i) for i in range(TSconfig["deltaTSWindow"][0]+1025,TSconfig["deltaTSWindow"][1]+1025)])
                    if sum(hist_merge_window_y):
                        var_mean.append(np.average(bin_centers, weights = hist_merge_window_y))
                        var_std.append(np.sqrt(np.average((bin_centers-var_mean[ithr])**2, weights = hist_merge_window_y)))
                    else:
                        var_mean.append(np.nan)
                        var_std.append(np.nan)
                    hist_deltaT_merge_board[var][boardNo][ithr].Rebin(4)
                    label_merge[var].append(f"Board {boardNo}, {var} amp: {TSconfig[f'threBins{var}'][ithr]*amp_factor}-{TSconfig[f'threBins{var}'][ithr+1]*amp_factor}")
                    hists_plot.append(hist_deltaT_merge_board[var][boardNo][ithr])
                extraToDraw.AddText(f"Board {boardNo} {var} amp: {TSconfig[f'threBins{var}'] * amp_factor}, " + "mean: "+", ".join([f"{var_mean[ib*(len(TSconfig[f'threBins{var}'])-1)+ithr]:.2f}" for ithr in range(len(TSconfig[f"threBins{var}"])-1)]) + ", std: "+", ".join([f"{var_std[ib*(len(TSconfig[f'threBins{var}'])-1)+ithr]:.2f}" for ithr in range(len(TSconfig[f"threBins{var}"])-1)]))

        output_name = f"DRS_DeltaT_board_Merged"
        DrawHistos(hists_plot, label_merge["Sci"]+label_merge["Cer"], TSconfig["deltaTSWindow"][0]+1024, TSconfig["deltaTSWindow"][1]+1024, "peak TS - trigger TS + 1024", 0.5, 1e6, "Counts",
            output_name,
            dology=True, drawoptions="HIST", 
            #mycolors=TSconfig["threBoardColorCer"].tolist(),
            mycolors=np.concatenate((TSconfig["threBoardColorSci"],TSconfig["threBoardColorCer"])).tolist(),
            addOverflow=True, extraToDraw=extraToDraw,
            legendPos=(0.18, 0.4, 0.90, 0.65),
            legendNCols = 1,
            outdir=outdir_plots)
        plots.append(output_name + ".png")
    outdir_plots = outdir + "/DRS_1D"
    output_html = f"html/Run{runNumber}/DRS_deltaT/index.html"
    generate_html(plots, outdir_plots,
                output_html=f"html/Run{runNumber}/DRS_deltaT/viewer.html")
    return output_html

# DRS vs TS
def makeDRS2DPlots(doSubtractMedian=False, doRTS=0):
    suffix = ""
    if doSubtractMedian:
        suffix = "_subtractMedian"
    varTS = "TS"
    if doRTS == 1:
        varTS = "RTSpos"
    elif doRTS == 2:
        varTS = "RTSneg"

    plots = []
    outdir_plots = f"{plotdir}/DRS_vs_{varTS}"
    infile_name = f"{rootdir}/drs_vs_{varTS}.root"
    infile = ROOT.TFile(infile_name, "READ")
    for _, DRSBoard in DRSBoards.items():
        boardNo = DRSBoard.boardNo
        for iTowerX, iTowerY in DRSBoard.GetListOfTowers():
            sTowerX = number2string(iTowerX)
            sTowerY = number2string(iTowerY)
            for var in ["Cer", "Sci"]:
                chan = DRSBoard.GetChannelByTower(
                    iTowerX, iTowerY, isCer=(var == "Cer"))
                hist_name = f"hist_DRS_Board{boardNo}_{var}_vs_{varTS}_{sTowerX}_{sTowerY}{suffix}"
                hist = infile.Get(hist_name)
                if not hist:
                    print(
                        f"Warning: Histogram {hist_name} not found in {infile_name}")
                    continue

                output_name = f"DRS_{var}_vs_TS_{sTowerX}_{sTowerY}{suffix}"
                plots.append(output_name + ".png")
                ymin_tmp, ymax_tmp = getDRSPlotRanges(
                    subtractMedian=doSubtractMedian, isAmplified=chan.isAmplified)

                # value_mean = hist.GetMean(2)

                extraToDraw = ROOT.TPaveText(0.20, 0.75, 0.60, 0.90, "NDC")
                extraToDraw.SetTextAlign(11)
                extraToDraw.SetFillColorAlpha(0, 0)
                extraToDraw.SetBorderSize(0)
                extraToDraw.SetTextFont(42)
                extraToDraw.SetTextSize(0.04)
                extraToDraw.AddText(
                    f"B: {DRSBoard.boardNo}, G: {chan.groupNo}, C: {chan.channelNo}")
                extraToDraw.AddText(f"iTowerX: {iTowerX}")
                extraToDraw.AddText(f"iTowerY: {iTowerY}")


                DrawHistos([hist], "", 0, 1024, "Time Slice", ymin_tmp, ymax_tmp, f"DRS Output",
                           output_name,
                           dology=False, drawoptions="COLZ", doth2=True, zmin=1, zmax=1e4, dologz=True,
                           extraToDraw=extraToDraw,
                           outdir=outdir_plots, extraText=var, runNumber=runNumber, addOverflow=True)
    output_html = f"{htmldir}/DRS_vs_{varTS}{suffix}/index.html"
    generate_html(plots, outdir_plots, plots_per_row=2,
                  output_html=output_html)
    return output_html


def makeDRSPeakTSPlots():
    plots = []
    outdir_plots = f"{plotdir}/DRSPeakTS"
    infile_name = f"{rootdir}/drs_peak_ts.root"
    infile = ROOT.TFile(infile_name, "READ")
    hists_Cer = []
    hists_Sci = []
    for _, DRSBoard in DRSBoards.items():
        boardNo = DRSBoard.boardNo
        for iTowerX, iTowerY in DRSBoard.GetListOfTowers():
            sTowerX = number2string(iTowerX)
            sTowerY = number2string(iTowerY)

            hists = {}
            channelNos = {}
            for var in ["Cer", "Sci"]:
                chan = DRSBoard.GetChannelByTower(
                    iTowerX, iTowerY, isCer=(var == "Cer"))
                hist_name = f"hist_DRS_PeakTS_Board{boardNo}_peakTS_{sTowerX}_{sTowerY}_{var}"
                hist = infile.Get(hist_name)
                output_name = hist_name[5:-4]

                if not hist:
                    print(
                        f"Warning: Histogram {hist_name} not found in {infile_name}")
                    hists[var] = None
                    channelNos[var] = -1
                else:
                    hists[var] = hist
                    channelNos[var] = chan.channelNo

            if not hists["Cer"] or not hists["Sci"]:
                print(
                    f"Warning: Histograms for Cer or Sci not found for Board {boardNo}, Tower ({iTowerX}, {iTowerY})")
                continue

            extraToDraw = ROOT.TPaveText(0.20, 0.65, 0.60, 0.90, "NDC")
            extraToDraw.SetTextAlign(11)
            extraToDraw.SetFillColorAlpha(0, 0)
            extraToDraw.SetBorderSize(0)
            extraToDraw.SetTextFont(42)
            extraToDraw.SetTextSize(0.04)
            extraToDraw.AddText(
                f"B: {DRSBoard.boardNo}, G: {chan.groupNo}")
            extraToDraw.AddText(f"Tower X: {iTowerX}")
            extraToDraw.AddText(f"Tower Y: {iTowerY}")
            extraToDraw.AddText(
                f"Cer Channel: {channelNos['Cer']}")
            extraToDraw.AddText(
                f"Sci Channel: {channelNos['Sci']}")

            hists_Cer.append(hists["Cer"])
            hists_Sci.append(hists["Sci"])

            DrawHistos([hists["Cer"], hists["Sci"]], ["Cer", "Sci"], 0, 400, "Peak TS", 1, None, "Counts",
                       output_name,
                       dology=False, drawoptions="HIST", mycolors=[2, 4], addOverflow=True, addUnderflow=False, extraToDraw=extraToDraw,
                       outdir=outdir_plots, runNumber=runNumber)
            plots.append(output_name + ".png")

    # summary plots
    hist_Cer_Sum = ROOT.TH1F("hist_DRS_PeakTS_Cer_Sum",
                             "DRS Peak TS Cer Sum", 400, 0, 400)
    hist_Sci_Sum = ROOT.TH1F("hist_DRS_PeakTS_Sci_Sum",
                             "DRS Peak TS Sci Sum", 400, 0, 400)
    for hist in hists_Cer:
        if hist:
            hist_Cer_Sum.Add(hist)
    for hist in hists_Sci:
        if hist:
            hist_Sci_Sum.Add(hist)
    DrawHistos([hist_Cer_Sum, hist_Sci_Sum], ["Cer", "Sci"], 0, 400, "Peak TS", 1, None, "Counts",
               "DRS_PeakTS_Sum",
               dology=False, drawoptions="HIST", mycolors=[2, 4], addOverflow=True, addUnderflow=False,
               outdir=outdir_plots, runNumber=runNumber)
    plots.insert(0, "DRS_PeakTS_Sum.png")

    output_html = f"{htmldir}/DRSPeakTS/index.html"
    generate_html(plots, outdir_plots, plots_per_row=4,
                  output_html=output_html)
    return output_html


def makeDRSPeakTS2DPlots():
    plots = []
    hists = []
    outdir_plots = f"{plotdir}/DRSPeakTS2D"
    infile_name = f"{rootdir}/drs_peak_ts.root"
    infile = ROOT.TFile(infile_name, "READ")

    # Create a dashed diagonal line from (0,0) to (400,400)
    diagonal_line = ROOT.TLine(0, 0, 400, 400)
    diagonal_line.SetLineStyle(2)  # 2 = dashed
    diagonal_line.SetLineWidth(1)
    diagonal_line.SetLineColor(ROOT.kRed)
    extraToDraw = diagonal_line

    for _, DRSBoard in DRSBoards.items():
        boardNo = DRSBoard.boardNo
        for iTowerX, iTowerY in DRSBoard.GetListOfTowers():
            sTowerX = number2string(iTowerX)
            sTowerY = number2string(iTowerY)

            hist_name = f"hist_DRSPeak_Cer_vs_Sci_Board{boardNo}_{sTowerX}_{sTowerY}"
            hist = infile.Get(hist_name)
            output_name = hist_name[5:]

            if not hist:
                print(
                    f"Warning: Histogram {hist_name} not found in {infile_name}")
                continue

            hists.append(hist)

            DrawHistos([hist], "", 0, 400, "Cer Peak TS", 0, 400, f"Sci Peak TS",
                       output_name,
                       dology=False, drawoptions="COLZ", doth2=True, zmin=1, zmax=1e2, dologz=True,
                       outdir=outdir_plots, addOverflow=False, runNumber=runNumber, extraToDraw=extraToDraw)

            plots.append(output_name + ".png")

    # summary plots
    h2 = ROOT.TH2F("hist_DRSPeak_Cer_vs_Sci_Sum",
                   "DRS Peak TS Cer vs Sci Sum", 400, 0, 400, 400, 0, 400)
    for hist in hists:
        if hist:
            h2.Add(hist)

    output_name = "DRS_PeakTS_Cer_vs_Sci_Sum"
    DrawHistos([h2], "", 0, 400, "Cer Peak TS", 0, 400, f"Sci Peak TS",
               output_name,
               dology=False, drawoptions="COLZ", doth2=True, zmin=1, zmax=1e2, dologz=True,
               outdir=outdir_plots, addOverflow=False, runNumber=runNumber, extraToDraw=extraToDraw)
    plots.insert(0, output_name + ".png")

    output_html = f"{htmldir}/DRSPeakTS2D/index.html"
    generate_html(plots, outdir_plots, plots_per_row=4,
                  output_html=output_html)
    return output_html

# DRS mean vs event
def trackDRSPlots():
    plots = []
    infile_name = f"{rootdir}/drs_all_channels_2D_vs_event.root"
    infile = ROOT.TFile(infile_name, "READ")
    outdir_plots = f"{plotdir}/DRS_vs_Event"
    for _, DRSBoard in DRSBoards.items():
        boardNo = DRSBoard.boardNo
        for iTowerX, iTowerY in DRSBoard.GetListOfTowers():
            sTowerX = number2string(iTowerX)
            sTowerY = number2string(iTowerY)
            for var in ["Cer", "Sci"]:
                chan = DRSBoard.GetChannelByTower(
                    iTowerX, iTowerY, isCer=(var == "Cer"))
                hist_name = f"hist_DRS_Board{boardNo}_{var}_vs_Event_{sTowerX}_{sTowerY}"
                hist = infile.Get(hist_name)

                if not hist:
                    print(
                        f"Warning: Histogram {hist_name} not found in {infile_name}")
                    continue

                extraToDraw = ROOT.TPaveText(0.20, 0.70, 0.60, 0.90, "NDC")
                extraToDraw.SetTextAlign(11)
                extraToDraw.SetFillColorAlpha(0, 0)
                extraToDraw.SetBorderSize(0)
                extraToDraw.SetTextFont(42)
                extraToDraw.SetTextSize(0.04)
                extraToDraw.AddText(f"Board: {DRSBoard.boardNo}")
                extraToDraw.AddText(f"iTowerX: {iTowerX}")
                extraToDraw.AddText(f"iTowerY: {iTowerY}")
                extraToDraw.AddText(f"{var} Group: {chan.groupNo}")
                extraToDraw.AddText(f"{var} Channel: {chan.channelNo}")

                nEvents = hist.GetXaxis().GetXmax()

                output_name = f"DRS_Board{boardNo}_{var}_{sTowerX}_{sTowerY}_vs_Event"
                DrawHistos([hist], "", 0, nEvents, "Event", 1400, 2300, f"{var} Mean",
                           output_name,
                           dology=False, drawoptions="COLZ", doth2=True, zmin=1, zmax=1e4, dologz=True,
                           extraToDraw=extraToDraw,
                           outdir=outdir_plots, addOverflow=True, runNumber=runNumber)
                plots.append(output_name + ".png")
    output_html = f"{htmldir}/DRS_vs_Event/index.html"
    generate_html(plots, outdir_plots, plots_per_row=4,
                  output_html=output_html)
    return output_html


# time reference
def compareTimeReferencePlots(doSubtractMedian=False):
    suffix = ""
    ymin = 500
    ymax = 2500
    if doSubtractMedian:
        suffix = "_subtractMedian"
        ymin = -2500
        ymax = 500
    plots = []
    infile_name = f"{rootdir}/time_reference_channels.root"
    infile = ROOT.TFile(infile_name, "READ")
    outdir_plots = f"{plotdir}/TimeReference"
    for chan_name in time_reference_channels:
        hist_name = f"hist_{chan_name}{suffix}"
        hist = infile.Get(hist_name)
        if not hist:
            print(f"Warning: Histogram {hist_name} not found in {infile_name}")
            continue
        extraToDraw = ROOT.TPaveText(0.20, 0.70, 0.60, 0.90, "NDC")
        extraToDraw.SetTextAlign(11)
        extraToDraw.SetFillColorAlpha(0, 0)
        extraToDraw.SetBorderSize(0)
        extraToDraw.SetTextFont(42)
        extraToDraw.SetTextSize(0.04)
        extraToDraw.AddText(f"{chan_name}")
        output_name = f"TimeReference_{chan_name}{suffix}"
        DrawHistos([hist], "", 0, 1024, "Time Slice", ymin, ymax, "Counts",
                   output_name,
                   dology=False, drawoptions="COLZ", doth2=True, zmin=1, zmax=1e4, dologz=True,
                   extraToDraw=extraToDraw,
                   outdir=outdir_plots, addOverflow=True, runNumber=runNumber)
        plots.append(output_name + ".png")

    output_html = f"{htmldir}/TimeReference{suffix}/index.html"
    generate_html(plots, outdir_plots, plots_per_row=2,
                  output_html=output_html)
    return output_html


# trigger
def compareHodoTriggerPlots(doSubtractMedian=False):
    suffix = ""
    ymin = 500
    ymax = 2500
    if doSubtractMedian:
        suffix = "_subtractMedian"
        ymin = -1500
        ymax = 500
    plots = []
    infile_name = f"{rootdir}/hodo_trigger_channels.root"
    infile = ROOT.TFile(infile_name, "READ")
    outdir_plots = f"{plotdir}/HodoTrigger"
    for chan_name in hodo_trigger_channels:
        hist_name = f"hist_{chan_name}{suffix}"
        hist = infile.Get(hist_name)
        if not hist:
            print(f"Warning: Histogram {hist_name} not found in {infile_name}")
            continue
        extraToDraw = ROOT.TPaveText(0.20, 0.70, 0.60, 0.90, "NDC")
        extraToDraw.SetTextAlign(11)
        extraToDraw.SetFillColorAlpha(0, 0)
        extraToDraw.SetBorderSize(0)
        extraToDraw.SetTextFont(42)
        extraToDraw.SetTextSize(0.04)
        extraToDraw.AddText(f"{chan_name}")
        output_name = f"HodoTrigger_{chan_name}{suffix}"
        DrawHistos([hist], "", 0, 1024, "Time Slice", ymin, ymax, "Counts",
                   output_name,
                   dology=False, drawoptions="COLZ", doth2=True, zmin=1, zmax=1e4, dologz=True,
                   extraToDraw=extraToDraw,
                   outdir=outdir_plots, addOverflow=True, runNumber=runNumber)
        plots.append(output_name + ".png")

    output_html = f"{htmldir}/HodoTrigger{suffix}/index.html"
    generate_html(plots, outdir_plots, plots_per_row=2,
                  output_html=output_html)
    return output_html


# hodo position
def compareHodoPosPlots(doSubtractMedian=False):
    suffix = ""
    ymin = 500
    ymax = 2500
    if doSubtractMedian:
        suffix = "_subtractMedian"
        ymin = -1500
        ymax = 500
    plots = []
    infile_name = f"{rootdir}/hodo_pos_channels.root"
    infile = ROOT.TFile(infile_name, "READ")
    outdir_plots = f"{plotdir}/HodoPos"
    for board, channels in hodo_pos_channels.items():
        for chan_name in channels:
            hist_name = f"hist_{chan_name}{suffix}"
            hist = infile.Get(hist_name)
            if not hist:
                print(
                    f"Warning: Histogram {hist_name} not found in {infile_name}")
                continue
            extraToDraw = ROOT.TPaveText(0.20, 0.70, 0.60, 0.90, "NDC")
            extraToDraw.SetTextAlign(11)
            extraToDraw.SetFillColorAlpha(0, 0)
            extraToDraw.SetBorderSize(0)
            extraToDraw.SetTextFont(42)
            extraToDraw.SetTextSize(0.04)
            extraToDraw.AddText(f"{chan_name}")
            output_name = f"HodoPos_{chan_name}{suffix}"
            DrawHistos([hist], "", 0, 1024, "Time Slice", ymin, ymax, "Counts",
                       output_name,
                       dology=False, drawoptions="COLZ", doth2=True, zmin=1, zmax=1e4, dologz=True,
                       extraToDraw=extraToDraw,
                       outdir=outdir_plots, addOverflow=True, runNumber=runNumber)
            plots.append(output_name + ".png")

    output_html = f"{htmldir}/HodoPos{suffix}/index.html"
    generate_html(plots, outdir_plots, plots_per_row=2,
                  output_html=output_html)
    return output_html


def checkFERSvsDRSSum():
    """
    Check if the sum of FERS and DRS energies are consistent.
    """
    plots = []
    outdir_plots = f"{plotdir}/FERS_vs_DRS_Sum"
    infile_name = f"{rootdir}/fers_vs_drs.root"
    infile = ROOT.TFile(infile_name, "READ")

    xymax = {
        "Cer": (1000, 4000),
        "Sci": (9000, 9000)
    }
    xymax_LG = {
        "Cer": (1000, 1000),
        "Sci": (9000, 9000)
    }

    # for _, DRSBoard in DRSBoards.items():
    #    boardNo = DRSBoard.boardNo
    #    for iTowerX, iTowerY in DRSBoard.GetListOfTowers():
    #        sTowerX = number2string(iTowerX)
    #        sTowerY = number2string(iTowerY)

    #        for var in ["Cer", "Sci"]:
    #            for gain in ["FERS", "FERSLG"]:
    #                histname = f"hist_{gain}_VS_DRS_Board{boardNo}_{var}_{sTowerX}_{sTowerY}"
    #                output_name = histname.replace("hist_", "")
    #                plots.append(output_name + ".png")

    #                hist = infile.Get(histname)
    #                if not hist:
    #                    print(
    #                        f"Warning: Histogram {histname} not found in {infile_name}")
    #                    continue

    #                zmax = hist.Integral(0, 10000, 0, 10000)
    #                zmax = round_up_to_1eN(zmax)

    #                tmp = xymax[var] if gain == "FERS" else xymax_LG[var]

    #                DrawHistos([hist], "", 0, tmp[0], "DRS Energy", 0, tmp[1], gain,
    #                           output_name,
    #                           dology=False, drawoptions="COLZ", doth2=True, zmin=1, zmax=zmax, dologz=True,
    #                           outdir=outdir_plots, addOverflow=True, runNumber=runNumber, extraText=f"{var}")

    # summary plots
    for _, DRSBoard in DRSBoards.items():
        boardNo = DRSBoard.boardNo
        for var in ["Cer", "Sci"]:
            for gain in ["FERS", "FERSLG"]:
                histname = f"hist_{gain}_VS_DRS_Board{boardNo}_{var}_sum"
                output_name = histname.replace("hist_", "")
                # append to plots no matter what
                plots.append(output_name + ".png")

                hist = infile.Get(histname)
                if not hist:
                    print(
                        f"Warning: Histogram {histname} not found in {infile_name}")
                    continue

                zmax = hist.Integral(0, 10000, 0, 10000)
                zmax = round_up_to_1eN(zmax)

                output_name = histname.replace("hist_", "")

                tmp = xymax[var] if gain == "FERS" else xymax_LG[var]
                DrawHistos([hist], "", 0, tmp[0], "DRS Energy", 0, tmp[1], gain,
                           output_name,
                           dology=False, drawoptions="COLZ", doth2=True, zmin=1, zmax=zmax, dologz=True,
                           outdir=outdir_plots, addOverflow=True, runNumber=runNumber, extraText=f"{var}")

    output_html = f"{htmldir}/FERS_vs_DRS_Sum/index.html"
    generate_html(plots, outdir_plots, plots_per_row=4,
                  output_html=output_html)
    return output_html


def checkDRSPeakvsFERS():
    plots = []
    outdir_plots = f"{plotdir}/DRSPeak_vs_FERS"
    infile_name = f"{rootdir}/drs_peak_vs_fers.root"
    infile = ROOT.TFile(infile_name, "READ")

    for _, DRSBoard in DRSBoards.items():
        boardNo = DRSBoard.boardNo
        for iTowerX, iTowerY in DRSBoard.GetListOfTowers():
            sTowerX = number2string(iTowerX)
            sTowerY = number2string(iTowerY)

            for var in ["Cer", "Sci"]:
                chan = DRSBoard.GetChannelByTower(
                    iTowerX, iTowerY, isCer=(var == "Cer"))
                if not chan:
                    print(
                        f"Warning: Channel not found for Board {boardNo}, Tower ({iTowerX}, {iTowerY}), Var {var}")
                    continue
                _, ymax = getDRSPlotRanges(
                    subtractMedian=True, isAmplified=chan.isAmplified)
                histname = f"hist_DRSPeak_VS_FERS_Board{boardNo}_{var}_{sTowerX}_{sTowerY}"
                output_name = histname.replace("hist_", "")
                plots.append(output_name + ".png")

                hist = infile.Get(histname)
                if not hist:
                    print(
                        f"Warning: Histogram {histname} not found in {infile_name}")
                    continue

                DrawHistos([hist], "", 0, 9000, "FERS ADC", 0, ymax, "DRS Peak",
                           output_name,
                           dology=False, drawoptions="COLZ", doth2=True, zmin=1, zmax=None, dologz=True,
                           outdir=outdir_plots, addOverflow=True, runNumber=runNumber, extraText=f"{var}")

    output_html = f"{htmldir}/DRSPeak_vs_FERS/index.html"
    generate_html(plots, outdir_plots, plots_per_row=4,
                  output_html=output_html)
    return output_html


if __name__ == "__main__":
    output_htmls = {}

    output_htmls["conditions plots"] = makeConditionsPlots()

    # validate DRS and FERS boards
    output_htmls["fers mapping"] = DrawFERSBoards(run=runNumber)
    output_htmls["drs mapping"] = DrawDRSBoards(run=runNumber)

    output_htmls["fers 1D"] = makeFERS1DPlots()
    output_htmls["fers stats"] = makeFERSStatsPlots()
    # makeDRS2DPlots()
    #output_htmls["drs 2D"] = makeDRS2DPlots(doSubtractMedian=True)
    #output_htmls["drs peak ts"] = makeDRSPeakTSPlots()
    #output_htmls["drs peak ts 2D"] = makeDRSPeakTS2DPlots()
    #output_htmls["drs 1D"] = makeDRS1DPlots()


    #output_htmls["time reference"] = compareTimeReferencePlots(True)
    #output_htmls["hodo trigger"] = compareHodoTriggerPlots(True)
    #output_htmls["hodo pos"] = compareHodoPosPlots(True)

    #output_htmls["fers vs drs sum"] = checkFERSvsDRSSum()

    #output_htmls["drs peak vs fers"] = checkDRSPeakvsFERS()

    print("\n\n\n")
    print("*" * 30)
    for key, value in output_htmls.items():
        print(f"✅ {key} plots can be viewed at: {value}")

    print("All plots generated successfully.")
