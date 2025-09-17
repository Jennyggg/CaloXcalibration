import os
import sys
sys.path.append("CMSPLOTS")
import ROOT
from CMSPLOTS.myFunction import DrawHistos
from utils.channel_map import buildFERSBoards
from utils.utils import loadRDF, vectorizeFERS,calculateEnergySumFERS,subtractFERSBeamPedestal,correctFERSSaturation,calibrateFERSChannelsBeam,testFERSBeamCalibration,preProcessDRSBoards
from utils.html_generator import generate_html
from utils.fitter import eventFit
from utils.colors import colors
from sklearn.linear_model import LinearRegression
from selections.selections import vetoMuonCounter, applyUpstreamVeto, PSDSelection
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from scipy.optimize import curve_fit


isHG=True
textgain = "HG" if isHG else "LG"
isCer = True
var = "Cer" if isCer else "Sci"
calibration_file_Cer = "FERS_response/FERS_response_runNumber_1234-1252_1267-1275_energy_80_40_board7_8_2_3_4_10_11_12_0_1_5_6_9_13_Cer_HG.json"
calibration_file_Sci = "FERS_response/FERS_response_runNumber1267_1268_1270_1271_1272_1273_1274_1275_energy40_40_40_40_40_40_40_40_board7_8_2_3_4_10_11_12_0_1_5_6_9_13_Sci_HG.json"
runNumbers = [1292,1262,1266,1264,1290,1288]
energies = [60,80,100,120,140,160]
firstEvents = [0]*6
lastEvents = [-1]*6
particle = "pion"
isFilter = True
filter_suffix = "_addCC1"
if not isFilter: filter_suffix = "_wofilter"

#runNumbers = [1261,1267,1260,1234]
#energies = [30,40,60,80]
#firstEvents = [0]*4
#lastEvents = [-1]*4
#particle = "positron"
xmax = 14
xmin = -14
ymax = 10
ymin = -10
W_ref = 1400
H_ref = 1600

def ReadADC2EnergyConversion(conversion_file):
    branch_names = []
    conversion_factors = []
    with open(conversion_file, "r") as f: 
        response_dic = json.load(f)
    for key, value in response_dic.items():
        branch_names.append(key)
        conversion_factors.append(value["response"])
    return branch_names,np.array(conversion_factors)

def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def PlotEnergyRaw(array, energy, var, runNumber,bins = 50):
    mean = np.mean(array)
    std =  np.std(array)
    range_min, range_max = max(0,mean-5*std),mean+5*std
    hist_fit, bin_edges = np.histogram(array, bins=bins, range=(range_min, range_max))
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    p0 = [hist_fit.max(), bin_centers[np.argmax(hist_fit)], 15000.0]
    try:
        popt, pcov = curve_fit(gaussian, bin_centers, hist_fit, p0=p0)
        A_fit, mu_fit, sigma_fit = popt
        perr = np.sqrt(np.diag(pcov))  # uncertainties
        A_err, mu_err, sigma_err = perr
    except RuntimeError:
        A_fit, mu_fit, sigma_fit = np.nan, np.nan, np.nan
        mu_err, sigma_err = np.nan, np.nan
    plt.figure(figsize=(8,6))
    plt.step(bin_centers, hist_fit, where='mid', label="sum #ADC")
    if not np.isnan(mu_fit):
        x_fit = np.linspace(range_min, range_max, 1000)
        plt.plot(x_fit, gaussian(x_fit, *popt), 'r-', linewidth=2, label="Gaussian Fit")
        fit_text = (f"μ = {mu_fit:.2f} ± {mu_err:.2f}\n"
                f"σ = {sigma_fit:.2f} ± {sigma_err:.2f}")
        plt.gca().text(
            0.95, 0.8, fit_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray")
        )
    # Labels and legend
    plt.xlabel("sum #ADC")
    plt.ylabel("Counts")
    plt.legend()
    plt.title(f"sum #ADC {var} {textgain}, Run{runNumber}, E={energy} GeV")

    # Save to PNG
    plt.tight_layout()
    os.system(f"mkdir -p results/plots/Run{runNumber}")
    plt.savefig(f"results/plots/Run{runNumber}/ADC_sum_fit_run{runNumber}_{var}_{textgain}_energy{energy}{filter_suffix}.png", dpi=300)
    plt.close()
    return mu_fit,mu_err,sigma_fit,sigma_err



def PlotEnergyMeasured(array, energy, var, runNumber,bins = 50):
    range_min, range_max = max(0,energy*0.8-40),energy+40
    hist_fit, bin_edges = np.histogram(array, bins=bins, range=(range_min, range_max))
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    p0 = [hist_fit.max(), bin_centers[np.argmax(hist_fit)], 10.0]
    try:
        popt, pcov = curve_fit(gaussian, bin_centers, hist_fit, p0=p0)
        A_fit, mu_fit, sigma_fit = popt
        perr = np.sqrt(np.diag(pcov))  # uncertainties
        A_err, mu_err, sigma_err = perr
    except RuntimeError:
        A_fit, mu_fit, sigma_fit = np.nan, np.nan, np.nan
        mu_err, sigma_err = np.nan, np.nan
    plt.figure(figsize=(8,6))
    plt.step(bin_centers, hist_fit, where='mid', label="energy sum")
    if not np.isnan(mu_fit):
        x_fit = np.linspace(range_min, range_max, 1000)
        plt.plot(x_fit, gaussian(x_fit, *popt), 'r-', linewidth=2, label="Gaussian Fit")
        fit_text = (f"μ = {mu_fit:.2f} ± {mu_err:.2f}\n"
                f"σ = {sigma_fit:.2f} ± {sigma_err:.2f}")
        plt.gca().text(
            0.95, 0.8, fit_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray")
        )
    # Labels and legend
    plt.xlabel("Energy [GeV]")
    plt.ylabel("Counts")
    plt.legend()
    plt.title(f"measured energy {var} {textgain}, Run{runNumber}, E={energy} GeV")

    # Save to PNG
    plt.tight_layout()
    os.system(f"mkdir -p results/plots/Run{runNumber}")
    plt.savefig(f"results/plots/Run{runNumber}/measured_energy_fit_run{runNumber}_{var}_{textgain}_energy{energy}{filter_suffix}.png", dpi=300)
    plt.close()
    return mu_fit,mu_err,sigma_fit,sigma_err


def PlotEnergyRatioMeasured(array_Cer,array_Sci, energy, runNumber,bins = 50):
    range_min, range_max = 0.3,1.8
    hist_fit, bin_edges = np.histogram(array_Cer/array_Sci, bins=bins, range=(range_min, range_max))
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    p0 = [hist_fit.max(), bin_centers[np.argmax(hist_fit)], 0.1]
    try:
        popt, pcov = curve_fit(gaussian, bin_centers, hist_fit, p0=p0)
        A_fit, mu_fit, sigma_fit = popt
        perr = np.sqrt(np.diag(pcov))  # uncertainties
        A_err, mu_err, sigma_err = perr
    except RuntimeError:
        A_fit, mu_fit, sigma_fit = np.nan, np.nan, np.nan
        mu_err, sigma_err = np.nan, np.nan
    plt.figure(figsize=(8,6))
    plt.step(bin_centers, hist_fit, where='mid', label="Cer./Sci.")
    if not np.isnan(mu_fit):
        x_fit = np.linspace(range_min, range_max, 1000)
        plt.plot(x_fit, gaussian(x_fit, *popt), 'r-', linewidth=2, label="Gaussian Fit")
        fit_text = (f"μ = {mu_fit:.2f} ± {mu_err:.2f}\n"
                f"σ = {sigma_fit:.2f} ± {sigma_err:.2f}")
        plt.gca().text(
            0.95, 0.8, fit_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray")
        )
    # Labels and legend
    plt.xlabel("Cer./Sci. ratio")
    plt.ylabel("Counts")
    plt.legend()
    plt.title(f"measured energy ratios between Cer. and Sci. {textgain}, Run{runNumber}, E={energy} GeV")

    # Save to PNG
    plt.tight_layout()
    os.system(f"mkdir -p results/plots/Run{runNumber}")
    plt.savefig(f"results/plots/Run{runNumber}/CSratio_run{runNumber}_{textgain}_energy{energy}{filter_suffix}.png", dpi=300)
    plt.close()
    return mu_fit,mu_err,sigma_fit,sigma_err



def PlotCerVSSciEnergy(array_Sci,array_Cer, energy, runNumber, particle):
    slope, intercept = np.polyfit(array_Sci, array_Cer, 1)
    array_Cer_fit = slope * array_Sci + intercept
    ss_res = np.sum((array_Cer - array_Cer_fit) ** 2)
    ss_tot = np.sum((array_Cer - np.mean(array_Cer)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    array_Sci_sorted = np.sort(array_Sci)
    plt.figure(figsize=(8, 6))
    plt.scatter(array_Sci, array_Cer, alpha=0.7, label='Events',marker='.',edgecolors = 'none', s = 2)
    plt.plot(array_Sci_sorted, slope * array_Sci_sorted + intercept, 'r-', 
         label=f'Fit: Cer. = {slope:.3f}Sci. + {intercept:.3f}\n$R^2$ = {r_squared:.3f}')
    plt.plot(array_Sci_sorted, array_Sci_sorted, 'b--',alpha=0.5, label='Cer. = Sci.')
    plt.plot(array_Sci_sorted, 0.5 * array_Sci_sorted, 'b:',alpha=0.5, label='Cer. = 0.5 Sci.')
    plt.title(f"Measured energy of {particle} beam, Run {runNumber}, E = {energy} GeV")
    plt.xlabel("Energy measured by Sci. channels [GeV]")
    plt.ylabel("Energy measured by Cer. channels [GeV]")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.system(f"mkdir -p results/plots/Run{runNumber}")
    plt.savefig(f"results/plots/Run{runNumber}/C_S_run{runNumber}_{var}_{textgain}_energy{energy}{filter_suffix}.png", dpi=300)
    plt.close()

def PlotCerVSSciRaw(array_Sci,array_Cer, energy, runNumber, particle):
    slope, intercept = np.polyfit(array_Sci, array_Cer, 1)
    array_Cer_fit = slope * array_Sci + intercept
    ss_res = np.sum((array_Cer - array_Cer_fit) ** 2)
    ss_tot = np.sum((array_Cer - np.mean(array_Cer)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    array_Sci_sorted = np.sort(array_Sci)
    plt.figure(figsize=(8, 6))
    plt.scatter(array_Sci, array_Cer, alpha=0.7, label='Events',marker='.',edgecolors = 'none', s = 2)
    plt.plot(array_Sci_sorted, slope * array_Sci_sorted + intercept, 'r-', 
         label=f'Fit: Cer. = {slope:.3f}Sci. + {intercept:.3f}\n$R^2$ = {r_squared:.3f}')
    plt.title(f"sum #ADC of {particle} beam, Run {runNumber}, E = {energy} GeV")
    plt.xlabel("sum #ADC from Sci. channels")
    plt.ylabel("sum #ADC from Cer. channels")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.system(f"mkdir -p results/plots/Run{runNumber}")
    plt.savefig(f"results/plots/Run{runNumber}/C_S_raw_run{runNumber}_{var}_{textgain}_energy{energy}{filter_suffix}.png", dpi=300)
    plt.close()

def PlotBias(energies,bias_Sci, bias_Cer, particle):
    plt.figure(figsize=(8, 6))
    plt.plot(energies, bias_Sci, 'bs-', label="Sci.")
    plt.plot(energies, bias_Cer, 'ro-', label="Cer.")
    plt.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label="response=1")
    plt.title(f"Bias of the measured energy, {particle} beam")
    plt.xlabel("Beam energy [GeV]")
    plt.ylabel("Measured energy / beam energy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.system(f"mkdir -p results/plots/{particle}")
    plt.savefig(f"results/plots/{particle}/bias_{textgain}{filter_suffix}.png", dpi=300)
    plt.close()

def PlotResolution(energies,resolution_Sci, resolution_Cer, particle):
    x = 1 / np.sqrt(energies)
    slope_sci, intercept_sci = np.polyfit(x, resolution_Sci, 1)
    slope_cer, intercept_cer = np.polyfit(x, resolution_Cer, 1)

    plt.figure(figsize=(8, 6))
    plt.plot(x, resolution_Sci, 'bs', label="Sci.")
    plt.plot(x, resolution_Cer, 'ro', label="Cer.")
    x_fit = np.linspace(min(x), max(x), 200)
    plt.plot(x_fit, slope_sci * x_fit + intercept_sci, 'b--', label=fr"$\frac{{{slope_sci*100:.0f}\%}}{{\sqrt{{E}}}} + {intercept_sci*100:.0f}\%$")  # Blue dashed line
    plt.plot(x_fit, slope_cer * x_fit + intercept_cer, 'r:', label=fr"$\frac{{{slope_cer*100:.0f}\%}}{{\sqrt{{E}}}} + {intercept_cer*100:.0f}\%$")
    plt.title(f"Resolution of the energy measurement, {particle} beam")
    plt.xlabel(r"$1/\sqrt{E}$")
    plt.ylabel(r"$\sigma / E$")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.system(f"mkdir -p results/plots/{particle}")
    plt.savefig(f"results/plots/{particle}/resolution_{textgain}{filter_suffix}.png", dpi=300)
    plt.close()



FERS_channels_Cer, conversion_factors_Cer = ReadADC2EnergyConversion(calibration_file_Cer)
FERS_channels_Sci, conversion_factors_Sci = ReadADC2EnergyConversion(calibration_file_Sci)
bias = {"Sci": [],"Cer": []}
rel_unc = {"Sci": [],"Cer": []}

for runNumber, energy, firstEvent, lastEvent in zip(runNumbers, energies, firstEvents, lastEvents):
    energy_measured = {}
    energy_raw = {}
    selections = {}
    for var, calibration_file in zip(["Cer","Sci"],[calibration_file_Cer,calibration_file_Sci]):
        if isHG:
            infile = f"results/root/Run{runNumber}/FERS_filtered_{var}{filter_suffix}.npz"
        else:
            infile = f"results/root/Run{runNumber}/FERS_filtered_{var}_LG{filter_suffix}.npz"
        FERSData = np.load(infile,"r")
        if isHG:
            selection = FERSData[f"FERS_{var}EnergyHG_subtracted_saturationcorrected"][firstEvent:lastEvent]>20000
        else:
            selection = FERSData[f"FERS_{var}EnergyLG_subtracted"][firstEvent:lastEvent]>1000
        selections[var] = selection
        FERS_channels, conversion_factors = ReadADC2EnergyConversion(calibration_file)
        FERSArrays = np.column_stack([FERSData[c][firstEvent:lastEvent] for c in FERS_channels])
        FERSEnergies = FERSArrays @ conversion_factors
        FERSRaw = np.sum(FERSArrays, axis=-1)
        energy_measured[var] = FERSEnergies
        energy_raw[var] = FERSRaw
    #energy_measured["Cer"] = energy_measured["Cer"][selections["Cer"] & selections["Sci"]]
    #energy_measured["Sci"] = energy_measured["Sci"][selections["Cer"] & selections["Sci"]]
    #energy_raw["Cer"] = energy_raw["Cer"][selections["Cer"] & selections["Sci"]]
    #energy_raw["Sci"] = energy_raw["Sci"][selections["Cer"] & selections["Sci"]]
    PlotEnergyRaw(energy_raw["Cer"], energy, "Cer", runNumber,50)
    PlotEnergyRaw(energy_raw["Sci"], energy, "Sci", runNumber,50)
    mu_fit_measured_Cer,mu_err_measured_Cer,sigma_fit_measured_Cer,sigma_err_measured_Cer = PlotEnergyMeasured(energy_measured["Cer"], energy, "Cer", runNumber,50)
    mu_fit_measured_Sci,mu_err_measured_Sci,sigma_fit_measured_Sci,sigma_err_measured_Sci = PlotEnergyMeasured(energy_measured["Sci"], energy, "Sci", runNumber,50)
    PlotEnergyRatioMeasured(energy_measured["Cer"],energy_measured["Sci"], energy, runNumber,bins = 50)
    bias["Cer"].append(mu_fit_measured_Cer/energy)
    bias["Sci"].append(mu_fit_measured_Sci/energy)
    rel_unc["Cer"].append(sigma_fit_measured_Cer/energy)
    rel_unc["Sci"].append(sigma_fit_measured_Sci/energy)
    PlotCerVSSciEnergy(energy_measured["Sci"],energy_measured["Cer"], energy, runNumber, particle)
    PlotCerVSSciRaw(energy_raw["Sci"],energy_raw["Cer"], energy, runNumber, particle)

PlotResolution(energies,rel_unc["Sci"], rel_unc["Cer"], particle)
PlotBias(energies,bias["Sci"], bias["Cer"], particle)



