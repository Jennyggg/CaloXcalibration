import os
import ROOT
from utils.channel_map import buildDRSBoards, buildFERSBoards, buildDRSChannels,buildDRSChannelsAmplified,buildTimeReferenceChannels
from utils.utils import number2string, getDataFile,processDRSPeaks,readTSconfig,filterPrefireEvents, loadRDF, calculateEnergySumFERS, vectorizeFERS, calibrateFERSChannels
from runconfig import runNumber, firstEvent, lastEvent
import time

start_time = time.time()

print("Start running prepareDQMPlots.py")

# multi-threading support
ROOT.ROOT.EnableImplicitMT(5)
ROOT.gSystem.Load("utils/functions_cc.so")
TSconfig = readTSconfig(run = runNumber)
# Open the input ROOT file
#ifile = getDataFile(runNumber)

#infile = ROOT.TFile(ifile, "READ")
#rdf = ROOT.RDataFrame("EventTree", infile)
#DRSBoards = buildDRSBoards(run=runNumber)
#FERSBoards = buildFERSBoards(run=runNumber)




file_gains = f"results/root/Run{runNumber}/valuemaps_gain.json"
file_pedestals = f"results/root/Run{runNumber}/valuemaps_pedestal.json"

rdf, rdf_org = loadRDF(runNumber, firstEvent, lastEvent)
rdf, rdf_prefilter = filterPrefireEvents(rdf, runNumber)

FERSBoards = buildFERSBoards(run=runNumber)
DRSBoards = buildDRSBoards(run=runNumber)
rdf = vectorizeFERS(rdf, FERSBoards)
# define energy sums with different configurations
rdf = calibrateFERSChannels(
    rdf, FERSBoards, file_gains=file_gains, file_pedestals=file_pedestals)
rdf = calculateEnergySumFERS(
    rdf, FERSBoards, subtractPedestal=False, calibrate=False, clip=False)
rdf = calculateEnergySumFERS(
    rdf, FERSBoards, subtractPedestal=True, calibrate=False, clip=False)
rdf = calculateEnergySumFERS(
    rdf, FERSBoards, subtractPedestal=True, calibrate=True, clip=False)
rdf = calculateEnergySumFERS(
    rdf, FERSBoards, subtractPedestal=True, calibrate=True, clip=True)

trigger_channels = buildTimeReferenceChannels(run=runNumber)
drs_amplified = buildDRSChannelsAmplified(runNumber)
rdf = processDRSPeaks(rdf, buildDRSChannels(DRSBoards), trigger_channels,TSconfig["noiseBaselineTS"][0],TSconfig["noiseBaselineTS"][1],TSconfig["peakTSWindow"][0],TSconfig["peakTSWindow"][1],drs_amplified)

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


#fers_requirements = ""
#for _, FERSBoard in FERSBoards.items():
#   boardNo = FERSBoard.boardNo
#    if boardNo not in [4]:
#        continue
    # require on the 3mm boards
#    fers_requirements += "("
#    for iTowerX, iTowerY in FERSBoard.GetListOfTowers():
#        chan_Sci = FERSBoard.GetChannelByTower(
#            iTowerX, iTowerY, isCer=False)
#        chan_Cer = FERSBoard.GetChannelByTower(
#            iTowerX, iTowerY, isCer=True)
#        var_Sci = chan_Sci.GetHGChannelName()
#        var_Cer = chan_Cer.GetHGChannelName()
#        fers_requirements += f"({var_Sci} > 400) || "

#    fers_requirements = fers_requirements[:-4]
#    fers_requirements += ") && "
#fers_requirements = fers_requirements[:-4]

#print(f"Requirements: {fers_requirements}")
#rdf_filtered_fers = rdf.Filter(
#    fers_requirements, "Filter FERS boards with energy > 400")
#print(f"Number of events after filtering: {rdf_filtered_fers.Count().GetValue()}")
#drs_requirements_board = {}

#for _, DRSBoard in DRSBoards.items():
#        boardNo = DRSBoard.boardNo
#        drs_requirements_board[boardNo] = "("

#        for iTowerX, iTowerY in DRSBoard.GetListOfTowers():
#            sTowerX = number2string(iTowerX)
#            sTowerY = number2string(iTowerY)
#            var = "Sci"
#            chan = DRSBoard.GetChannelByTower(
#                    iTowerX, iTowerY, isCer=False)
#            if chan is None:
#                continue
#            channelName = chan.GetChannelName()
#            thre = 7
#            if channelName in drs_amplified:
#                thre = 70
#            drs_requirements_board[boardNo] += f"({channelName}_peakA > {thre} && {channelName}_peakT > {TSconfig["peakTSWindow"][0]} &&  {channelName}_peakT < {TSconfig["peakTSWindow"][1]} ) || "
#        drs_requirements_board[boardNo] = drs_requirements_board[boardNo][:-4]
#        drs_requirements_board[boardNo] += ")"

#drs_requirements = f"{drs_requirements_board[3]} && ({drs_requirements_board[1]} || {drs_requirements_board[2]})"


#rdf_filtered_drs = rdf_filtered_fers.Filter(
#    f"{drs_requirements_board[3]}", "Filter DRS boards peaks on Boad 3")
#print(f"Requirements: {drs_requirements_board[3]}")

#rdf_filtered_drs = rdf_filtered_drs.Filter(
#    f"{drs_requirements_board[1]} || {drs_requirements_board[2]}", "Filter DRS boards peaks on Boad 1 or 2")
#print(f"Requirements: {drs_requirements_board[1]} || {drs_requirements_board[2]}")


#rdf_filtered_drs = rdf_filtered_fers.Filter(
#    f"{drs_requirements_board[2]}", "Filter DRS boards peaks on Boad 2")
#rdf_filtered_drs = rdf_filtered_drs.Filter(
#    f"{drs_requirements_board[1]}", "Filter DRS boards peaks on Boad 1")
#print(f"Requirements: {drs_requirements_board[1]}")
#print(f"Number of events after filtering: {rdf_filtered_drs.Count().GetValue()}")
# snapshot the filtered RDF


requirement = "(FERS_SciEnergyHG_subtracted_calibrated > 200)&&(FERS_SciEnergyHG_subtracted_calibrated < 400)"
rdf_filtered_drs = rdf.Filter(
    requirement, "Filter FERS total energy")
rootdir = f"results/root/Run{runNumber}/"
outfile_name = f"{rootdir}/filtered_track_FERS_nofanoutcorr.root"
if not os.path.exists(os.path.dirname(outfile_name)):
    os.makedirs(os.path.dirname(outfile_name))
rdf_filtered_drs.Snapshot("EventTree", outfile_name)
print(f"Filtered data saved to {outfile_name}")

print(f"Time taken: {time.time() - start_time:.2f} seconds")
