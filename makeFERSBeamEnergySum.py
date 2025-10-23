import os
import sys
import ROOT
from utils.channel_map import buildFERSBoards
from utils.utils import loadRDF, vectorizeFERS,calculateEnergySumFERS,subtractFERSBeamPedestal,correctFERSSaturation,calibrateFERSChannelsBeam,testFERSBeamCalibration,preProcessDRSBoards
from utils.html_generator import generate_html
from utils.fitter import eventFit
from utils.colors import colors
from runconfig import runNumber, firstEvent, lastEvent, beamEnergy
from selections.selections import vetoMuonCounter, applyUpstreamVeto, PSDSelection,applyCC1Selection,applyCC2Selection,applyCC3Selection,applyPSDSelection
import json

sys.path.append("CMSPLOTS")  # noqa
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--muonveto',
                    action='store_true')          
parser.add_argument('--holeveto',
                    action='store_true')
parser.add_argument('--PSD',
                    action='store_true')
parser.add_argument('--CC1',
                    action='store_true')
parser.add_argument('--CC2',
                    action='store_true')
parser.add_argument('--CC3',
                    action='store_true')                                                     
parser.add_argument('--isHadron',
                    action='store_true') 
args = parser.parse_args()
isHadron = args.isHadron
filter_suffix = ""
if args.muonveto:
    filter_suffix += "_muonveto"
if args.holeveto:
    filter_suffix += "_holeveto"
if args.PSD:
    filter_suffix += "_passPSD"
if args.CC1:
    if isHadron:
        filter_suffix += "_failCC1"
    else:
        filter_suffix += "_passCC1"
if args.CC2:
    if isHadron:
        filter_suffix += "_failCC2"
    else:
        filter_suffix += "_passCC2"
if args.CC3:
    if isHadron:
        filter_suffix += "_failCC3"
    else:
        filter_suffix += "_passCC3"        
if filter_suffix == "": filter_suffix = "_wofilter"

runPedestal = 1374
print("Start running FERS beam energy calibration")
plotdir = f"results/plots/Run{runNumber}/"
output_json_dir = f"results/root/Run{runNumber}/"
if not os.path.exists(plotdir):
    os.makedirs(plotdir)
PLOTFIT = True

# multi-threading support
ROOT.ROOT.EnableImplicitMT(10)
ROOT.gROOT.SetBatch(True)  # Disable interactive mode for batch processing
ROOT.gSystem.Load("utils/functions_cc.so")  # Load the compiled C++ functions

with open(f"results/root/Run{runPedestal}/testbeam_pedestal.json", "r") as f:
    pedestal = json.load(f)
HGLG_json_dir = f"results/root/positroncali_round2/"
with open(f"{HGLG_json_dir}/testbeam_FERS_HG_to_LG_factors.json", "r") as f:
    factors_HG_to_LG = json.load(f)

FERSBoards = buildFERSBoards(run=runNumber)
rdf, _ = loadRDF(runNumber, firstEvent, lastEvent)
rdf = preProcessDRSBoards(rdf)

if args.muonveto:
    rdf, rdf_prefilter = vetoMuonCounter(rdf, TSmin=400, TSmax=700, cut=-30)
if args.holeveto:
    rdf, rdf_filterveto = applyUpstreamVeto(rdf, runNumber)
if args.PSD:
    rdf = applyPSDSelection(rdf, runNumber, applyCut=False)
    if isHadron:
        str_psd = "pass_PSDEle_selection == 0"
    else:
        str_psd = "pass_PSDEle_selection == 1"

    rdf = rdf.Filter(str_psd)
    print("select events "+str_psd)
    print(
    f"Events after PSD selection: {rdf.Count().GetValue()}")
if args.CC1:
    rdf = applyCC1Selection(rdf, runNumber, applyCut=False)
    if isHadron:
        str_cc1 = "pass_CC1Ele_selection == 0"
    else:
        str_cc1 = "pass_CC1Ele_selection == 1"
    rdf = rdf.Filter(str_cc1)
    print("select events "+str_cc1)
    print(
    f"Events after CC1 selection: {rdf.Count().GetValue()}")
if args.CC2:
    rdf = applyCC2Selection(rdf, runNumber, applyCut=False)
    if isHadron:
        str_cc2 = "pass_CC2Ele_selection == 0"
    else:
        str_cc2 = "pass_CC2Ele_selection == 1"
    rdf = rdf.Filter(str_cc2)
    print("select events "+str_cc2)
    print(
    f"Events after CC2 selection: {rdf.Count().GetValue()}")
if args.CC3:
    rdf = applyCC3Selection(rdf, runNumber, applyCut=False)
    if isHadron:
        str_cc3 = "pass_CC3Ele_selection == 0"
    else:
        str_cc3 = "pass_CC3Ele_selection == 1"
    rdf = rdf.Filter(str_cc3)
    print("select events "+str_cc3)
    print(
    f"Events after CC3 selection: {rdf.Count().GetValue()}")



rdf = vectorizeFERS(rdf, FERSBoards)
rdf = subtractFERSBeamPedestal(rdf, FERSBoards, pedestal)
rdf = correctFERSSaturation(rdf,FERSBoards, factors_HG_to_LG)



rdf = calculateEnergySumFERS(rdf, FERSBoards, subtractPedestal=True, calibrate=False, clip=False, saturationCorrected=True)
rdf = calculateEnergySumFERS(rdf, FERSBoards, subtractPedestal=True, calibrate=False, clip=False, saturationCorrected=False, lowGain = True)

hist_CerEnergyHG = rdf.Histo1D((
    f"hist_FERS_CerEnergyHG_subtracted_saturationcorrected",
    "FERS - CER Energy HG;CER Energy HG;Counts",
    500, -10000, 3e5),
    f"FERS_CerEnergyHG_subtracted_saturationcorrected"
    )
hist_SciEnergyHG = rdf.Histo1D((
    f"hist_FERS_SciEnergyHG_subtracted_saturationcorrected",
    "FERS - SCI Energy HG;SCI Energy HG;Counts",
    500, -10000, 5e5),
    f"FERS_SciEnergyHG_subtracted_saturationcorrected"
    )
hist_CerEnergyLG = rdf.Histo1D((
    f"hist_FERS_CerEnergyLG_subtracted",
    "FERS - CER Energy LG;CER Energy LG;Counts",
    500, -2000, 5000),
    f"FERS_CerEnergyLG_subtracted"
    )
hist_SciEnergyLG = rdf.Histo1D((
    f"hist_FERS_SciEnergyLG_subtracted",
    "FERS - SCI Energy LG;SCI Energy LG;Counts",
    500, -1000, 2e4),
    f"FERS_SciEnergyLG_subtracted"
    )


#HG_fitrange_Cer = [80000,130000]
#HG_fitrange_Sci = [1200000,2000000]
#HG_fit_amp_Cer = 350
#HG_fit_mean_Cer = 100000
#HG_fit_width_Cer = 10000
#HG_fit_amp_Sci = 60
#HG_fit_mean_Sci = 1500000
#HG_fit_width_Sci = 100000

#LG_fitrange_Cer = [6000,14000]
#LG_fitrange_Sci = [20000,40000]
#LG_fit_amp_Cer = 350
#LG_fit_mean_Cer = 10000
#LG_fit_amp_Sci = 60
#LG_fit_mean_Sci = 30000
#LG_fit_width_Sci = 5000


#HG_fitrange_Cer = [20000,40000]
#HG_fitrange_Sci = [300000,500000]
#HG_fit_amp_Cer = 350
#HG_fit_mean_Cer = 30000
#HG_fit_width_Cer = 5000
#HG_fit_amp_Sci = 60
#HG_fit_mean_Sci = 400000
#HG_fit_width_Sci = 50000

#LG_fitrange_Cer = [1000,4000]
#LG_fitrange_Sci = [23000,30000]
#LG_fit_amp_Cer = 350
#LG_fit_mean_Cer = 3000
#LG_fit_width_Cer = 1000
#LG_fit_amp_Sci = 60
#LG_fit_mean_Sci = 25000
#LG_fit_width_Sci = 2500


HG_fitrange_Cer = [20000,50000]
HG_fitrange_Sci = [60000,150000]
HG_fit_amp_Cer = 350
HG_fit_mean_Cer = 40000
HG_fit_width_Cer = 5000
HG_fit_amp_Sci = 60
HG_fit_mean_Sci = 10000
HG_fit_width_Sci = 5000

LG_fitrange_Cer = [1000,2000]
LG_fitrange_Sci = [1000,5000]
LG_fit_amp_Cer = 350
LG_fit_mean_Cer = 1500
LG_fit_width_Cer = 100
LG_fit_amp_Sci = 60
LG_fit_mean_Sci = 4000
LG_fit_width_Sci = 100

#fitrange_Cer = [30000,40000]
#fitrange_Sci = [500000,700000]
#fit_amp_Cer = 350
#fit_mean_Cer = 35000
#fit_width_Cer = 5000
#fit_amp_Sci = 60
#fit_mean_Sci = 580000
#fit_width_Sci = 80000
fit_dic_HG = {}
fit_dic_LG = {}
for gain,var,fitrange,amp,mean,width,hist in zip(["HG","HG","LG","LG"],["Cer","Sci","Cer","Sci"],[HG_fitrange_Cer,HG_fitrange_Sci,LG_fitrange_Cer,LG_fitrange_Sci],[HG_fit_amp_Cer,HG_fit_amp_Sci,LG_fit_amp_Cer,LG_fit_amp_Sci],[HG_fit_mean_Cer,HG_fit_mean_Sci,LG_fit_mean_Cer,LG_fit_mean_Sci],[HG_fit_width_Cer,HG_fit_width_Sci,LG_fit_width_Cer,LG_fit_width_Sci],[hist_CerEnergyHG,hist_SciEnergyHG,hist_CerEnergyLG,hist_SciEnergyLG]):
    fit_func = ROOT.TF1("fit_func", "gaus(0)", fitrange[0], fitrange[1])
    fit_func.SetParameters(amp, mean, width)  # (ampl1, mean1, sigma1, ampl2, mean2, sigma2)
    # Fit histogram
    hist.Fit(fit_func, "R")
    print(f"fit {var}", fit_func.GetParameter(0),fit_func.GetParameter(1),fit_func.GetParameter(2))
    c = ROOT.TCanvas()
    hist.Draw()
    fit_func.Draw("same")
    c.SaveAs(f"{plotdir}/fit_energysum_{gain}_{var}{filter_suffix}.pdf")
    if gain == "HG":
        fit_dic_HG[f"amp_s_{var}"] = fit_func.GetParameter(0)
        fit_dic_HG[f"mean_s_{var}"] = fit_func.GetParameter(1)
        fit_dic_HG[f"width_s_{var}"] = abs(fit_func.GetParameter(2))
        fit_dic_HG[f"mean_response_s_{var}"] = fit_func.GetParameter(1)/beamEnergy
    else:
        fit_dic_LG[f"amp_s_{var}"] = fit_func.GetParameter(0)
        fit_dic_LG[f"mean_s_{var}"] = fit_func.GetParameter(1)
        fit_dic_LG[f"width_s_{var}"] = abs(fit_func.GetParameter(2))
        fit_dic_LG[f"mean_response_s_{var}"] = fit_func.GetParameter(1)/beamEnergy

if not os.path.exists(output_json_dir):
    os.makedirs(output_json_dir)
with open(f"{output_json_dir}/testbeam_energy_sum_fit{filter_suffix}.json", "w") as f:
    json.dump(fit_dic_HG, f, indent=4)
with open(f"{output_json_dir}/testbeam_energy_sum_fit_LG{filter_suffix}.json", "w") as f:
    json.dump(fit_dic_LG, f, indent=4)

#caliFactors,_ = calibrateFERSChannelsBeam(rdf, FERSBoards, beamEnergy,rangeLow = firstEvent, rangeHigh=lastEvent)
#output_json_dir = f"results/root/Run{runNumber}/"
#if not os.path.exists(output_json_dir):
#    os.makedirs(output_json_dir)
#with open(f"{output_json_dir}/testbeam_energy_calibration.json", "w") as f:
#    json.dump(caliFactors, f, indent=4)
