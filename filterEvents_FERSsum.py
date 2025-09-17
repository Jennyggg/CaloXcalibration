import os
import ROOT
from utils.channel_map import buildDRSBoards, buildFERSBoards
from utils.utils import number2string, getDataFile,calculateEnergySumFERS,vectorizeFERS
from runconfig import runNumber
import time

start_time = time.time()

print("Start running filterEvents_FERSsum.py")

# multi-threading support
ROOT.ROOT.EnableImplicitMT(5)

# Open the input ROOT file
ifile = getDataFile(runNumber)

suffix = f"run{runNumber}"
infile = ROOT.TFile(ifile, "READ")
rdf = ROOT.RDataFrame("EventTree", infile)


DRSBoards = buildDRSBoards(run=runNumber)
FERSBoards = buildFERSBoards(run=runNumber)
rdf = vectorizeFERS(rdf, FERSBoards)
rdf = calculateEnergySumFERS(
    rdf, FERSBoards, subtractPedestal=False, calibrate=False, clip=False)
# FRES board outputs
# define variables as RDF does not support reading vectors
# with indices directly



#requirements = "(FERS_CerEnergyHG > 125000) && (FERS_SciEnergyHG > 330000) && (FERS_CerEnergyHG < 175000) && (FERS_SciEnergyHG < 500000)"

requirements = "(FERS_SciEnergyHG > 700000)"
print(f"Requirements: {requirements}")

rdf_filtered = rdf.Filter(
    requirements, "Filter FERS boards with Sci energy > 700000")

print(f"Number of events after filtering: {rdf_filtered.Count().GetValue()}")
# snapshot the filtered RDF
outfile_name = f"root/Run{runNumber}/filtered_highE.root"
if not os.path.exists(os.path.dirname(outfile_name)):
    os.makedirs(os.path.dirname(outfile_name))
rdf_filtered.Snapshot("EventTree", outfile_name)
print(f"Filtered data saved to {outfile_name}")

print(f"Time taken: {time.time() - start_time:.2f} seconds")
