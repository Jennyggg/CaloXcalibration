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
addToy = True
addPhotoStat = True
cali_suffix = ""
#calibration_file_Cer = "FERS_response/FERS_response_runNumber_1355-1406_7500events1366_energy_80_board2_3_4_5_6_7_8_9_10_11_12_13_14_15_Cer_HG_muonveto_passPSD_passCC1_passCC2_passCC3.json"
#calibration_file_Cer_calialg_toys = "FERS_response/FERS_response_runNumber_1355-1406_7500events1366_addtoys_energy_80_board2_3_4_5_6_7_8_9_10_11_12_13_14_15_Cer_HG_muonveto_passPSD_passCC1_passCC2_passCC3.json"
#calibration_file_Cer = "FERS_response/FERS_response_runNumber_1355-1406_7500events1366_2steps_addtoys_energy_80_board2_3_4_5_6_7_8_9_10_11_12_13_14_15_Cer_HG_muonveto_passPSD_passCC1_passCC2_passCC3.json"
#calibration_file_Cer_calialg_toys = "FERS_response/FERS_response_runNumber_1355-1406_7500events1366_2steps_addtoys_energy_80_board2_3_4_5_6_7_8_9_10_11_12_13_14_15_Cer_HG_muonveto_passPSD_passCC1_passCC2_passCC3.json"
calibration_file_Cer = "FERS_response/FERS_response_runNumber_1355-1406_7500events1366_5steps_addtoys_energy_80_board2_3_4_5_6_7_8_9_10_11_12_13_14_15_Cer_HG_muonveto_passPSD_passCC1_passCC2_passCC3.json"
calibration_file_Cer_calialg_toys = "FERS_response/FERS_response_runNumber_1355-1406_7500events1366_5steps_addtoys_energy_80_board2_3_4_5_6_7_8_9_10_11_12_13_14_15_Cer_HG_muonveto_passPSD_passCC1_passCC2_passCC3.json"

#calibration_file_Cer = "FERS_response/FERS_response_runNumber_1234-1252_1267-1275_energy_80_40_board7_8_2_3_4_10_11_12_0_1_5_6_9_13_Cer_HG.json"
#calibration_file_Cer = "FERS_response/FERS_response_runNumber1234_1231_1235_1232_1236_1237_1238_1239_1240_1241_1243_1245_1246_1248_1249_1250_1252_energy80_80_80_80_80_80_80_80_80_80_80_80_80_80_80_80_80_board7_8_2_3_4_10_11_12_0_1_5_6_9_13_Cer_HG.json"
#calibration_file_Sci = "FERS_response/FERS_response_runNumber1267_1268_1270_1271_1272_1273_1274_1275_energy40_40_40_40_40_40_40_40_board7_8_2_3_4_10_11_12_0_1_5_6_9_13_Sci_HG.json"
#calibration_file_Sci = "FERS_response/FERS_response_runNumber_1355-1406_7500events1366_energy_80_board2_3_4_5_6_7_8_9_10_11_12_13_14_15_Sci_HG_muonveto_passPSD_passCC1_passCC2_passCC3.json"
#calibration_file_Sci_calialg_toys = "FERS_response/FERS_response_runNumber_1355-1406_7500events1366_addtoys_energy_80_board2_3_4_5_6_7_8_9_10_11_12_13_14_15_Sci_HG_muonveto_passPSD_passCC1_passCC2_passCC3.json"
#calibration_file_Sci = "FERS_response/FERS_response_runNumber_1355-1406_7500events1366_2steps_addtoys_energy_80_board2_3_4_5_6_7_8_9_10_11_12_13_14_15_Sci_HG_muonveto_passPSD_passCC1_passCC2_passCC3.json"
#calibration_file_Sci_calialg_toys = "FERS_response/FERS_response_runNumber_1355-1406_7500events1366_2steps_addtoys_energy_80_board2_3_4_5_6_7_8_9_10_11_12_13_14_15_Sci_HG_muonveto_passPSD_passCC1_passCC2_passCC3.json"
calibration_file_Sci = "FERS_response/FERS_response_runNumber_1355-1406_7500events1366_5steps_addtoys_energy_80_board2_3_4_5_6_7_8_9_10_11_12_13_14_15_Sci_HG_muonveto_passPSD_passCC1_passCC2_passCC3.json"
calibration_file_Sci_calialg_toys = "FERS_response/FERS_response_runNumber_1355-1406_7500events1366_5steps_addtoys_energy_80_board2_3_4_5_6_7_8_9_10_11_12_13_14_15_Sci_HG_muonveto_passPSD_passCC1_passCC2_passCC3.json"
#runNumbers = [1292,1262,1266,1264,1290,1288]
#energies = [60,80,100,120,140,160]
#firstEvents = [0]*6
#lastEvents = [-1]*6
#runNumbers = [1266]
#energies = [100]
#irstEvents = [0]
#astEvents = [-1]
#particle = "pion"
#isFilter = True
#filter_suffix = "_addCC1"
#if not isFilter: filter_suffix = "_wofilter"

runNumbers_positrons = [1424,1423,1416,1514,1526,1527]
energies_positrons = [10,20,30,40,60,100]
filter_suffix_positrons = "_muonveto_holeveto_passPSD_passCC1_passCC2_passCC3"
runNumbers = [1442,1441,1439,1438,1437,1429,1431,1433,1434]
energies = [10,20,30,40,60,80,100,120,160]
firstEvents = [0]*9
lastEvents = [-1]*9
particle = "pion"
filter_suffix = "_muonveto_holeveto_failPSD_failCC1_failCC2_failCC3"

#runNumbers = [1261,1267,1260,1234]
#energies = [30,40,60,80]
#firstEvents = [0]*4
#lastEvents = [-1]*4
##particle = "positron"

#runNumbers = [1424,1423,1416,1514,1526,1527]
#energies = [10,20,30,40,60,100]
#firstEvents = [0]*6
#lastEvents = [-1]*6
#particle = "positron"
#filter_suffix = "_muonveto_holeveto_passPSD_passCC1_passCC2_passCC3"
xmax = 14
xmin = -14
ymax = 10
ymin = -10
W_ref = 1400
H_ref = 1600

runNumber_calibrate = 1355
energy_calibrate = 80
filter_suffix_calibrate = "_muonveto_holeveto_passPSD_passCC1_passCC2_passCC3"
def GetChannelGain(channel):
    if "Board9" in channel or "Board10" in channel:
        return 75
    else:
        return 50
def GetEMFraction(energy):
    #EM fraction of hadron showers in copper
    return 1-(energy/0.7)**(0.8-1)

def GetEHRatio(energy,pi_e_ratio):
    em_fraction = GetEMFraction(energy)
    print("energy ",energy, "em_fraction ",em_fraction, "pi_e_ratio ", pi_e_ratio,  "eh ratio ",(1-em_fraction) / (pi_e_ratio-em_fraction))
    return (1-em_fraction) / (pi_e_ratio-em_fraction)

def GetPositronResponse(energy, responses_e,energies_e):
    if energy in energies_e:
        return responses_e[energies_e.index(energy)]
    else:
        if energy > energies_e[0] and energy < energies_e[-1]:
            ifind = 0
            while(energies_e[ifind] < energy):
                ifind += 1
            response_e = np.interp(energy,[energies_e[ifind-1],energies_e[ifind]],[responses_e[ifind-1],responses_e[ifind]])
            return response_e
        else:
            #assume response_e is 1.0
            return 1.0

def ReadADC2EnergyConversion(conversion_file):
    branch_names = []
    conversion_factors = []
    with open(conversion_file, "r") as f: 
        response_dic = json.load(f)
    for key, value in response_dic.items():
        branch_names.append(key)
        conversion_factors.append(value["response"])
    return branch_names,np.array(conversion_factors)

def ReadADC2EnergyConversionToys(conversion_file):
    branch_names = []
    conversion_factors = []
    with open(conversion_file, "r") as f: 
        response_dic = json.load(f)
    for key, value in response_dic.items():
        branch_names.append(key)
        conversion_factors.append(np.array(value["response_toys"]))
    return branch_names,np.vstack(conversion_factors)

def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def PlotEnergyRaw(array, energy, var, runNumber,bins = 50,suffix=""):
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
    plt.savefig(f"results/plots/Run{runNumber}/ADC_sum_fit_run{runNumber}_{var}_{textgain}_energy{energy}{filter_suffix}{suffix}.png", dpi=300)
    plt.close()
    return mu_fit,mu_err,sigma_fit,sigma_err



def PlotEnergyMeasured(array,array_mean_response, energy, var, runNumber,bins = 50,suffix=""):
    range_min, range_max = max(0,energy*0.5-40),energy*2.5
    hist_fit, bin_edges = np.histogram(array, bins=bins, range=(range_min, range_max))
    if array_mean_response is not None:
        hist_mean_response, bin_edges = np.histogram(array_mean_response, bins=bins, range=(range_min, range_max))
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
    plt.step(bin_centers, hist_fit, where='mid', label="measured energy")
    if array_mean_response is not None:
        plt.step(bin_centers, hist_mean_response, where='mid', label="energy from mean response")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if not np.isnan(mu_fit):
        x_fit = np.linspace(range_min, range_max, 1000)
        plt.plot(x_fit, gaussian(x_fit, *popt), 'r-', linewidth=2, label="Gaussian Fit")
        fit_text = (f"μ = {mu_fit:.2f} ± {mu_err:.2f}\n"
                f"σ = {sigma_fit:.2f} ± {sigma_err:.2f}")
        plt.gca().text(
            0.95, 0.75, fit_text,
            transform=plt.gca().transAxes,
            fontsize=14,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray")
        )
    # Labels and legend
    plt.xlabel("Energy [GeV]",fontsize=15)
    plt.ylabel("Counts",fontsize=15)
    plt.legend(fontsize=15)
    plt.title(f"measured energy {var} {textgain}, Run{runNumber}, E={energy} GeV",fontsize=15)

    # Save to PNG
    plt.tight_layout()
    os.system(f"mkdir -p results/plots/Run{runNumber}")
    plt.savefig(f"results/plots/Run{runNumber}/measured_energy_fit_run{runNumber}_{var}_{textgain}_energy{energy}{filter_suffix}{suffix}.png", dpi=300)
    plt.close()
    return mu_fit,mu_err,sigma_fit,sigma_err


def PlotEnergyRatioMeasured(array_Cer,array_Sci, energy, runNumber,bins = 50,suffix=""):
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
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.step(bin_centers, hist_fit, where='mid', label="Cer./Sci.")
    if not np.isnan(mu_fit):
        x_fit = np.linspace(range_min, range_max, 1000)
        plt.plot(x_fit, gaussian(x_fit, *popt), 'r-', linewidth=2, label="Gaussian Fit")
        fit_text = (f"μ = {mu_fit:.2f} ± {mu_err:.2f}\n"
                f"σ = {sigma_fit:.2f} ± {sigma_err:.2f}")
        plt.gca().text(
            0.95, 0.8, fit_text,
            transform=plt.gca().transAxes,
            fontsize=14,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray")
        )
    # Labels and legend
    plt.xlabel("Cer./Sci. ratio",fontsize=14)
    plt.ylabel("Counts",fontsize=14)
    plt.legend(fontsize=14)
    plt.title(f"measured energy ratios between Cer. and Sci. {textgain}, Run{runNumber}, E={energy} GeV",fontsize=14)

    # Save to PNG
    plt.tight_layout()
    os.system(f"mkdir -p results/plots/Run{runNumber}")
    plt.savefig(f"results/plots/Run{runNumber}/CSratio_run{runNumber}_{textgain}_energy{energy}{filter_suffix}{suffix}.png", dpi=300)
    plt.close()
    return mu_fit,mu_err,sigma_fit,sigma_err



def constrained_fit(x, y, x0, y0):
    # Shift relative to the fixed point
    x_shift = x - x0
    y_shift = y - y0
    
    # Compute slope
    m = np.sum(x_shift * y_shift) / np.sum(x_shift**2)
    
    # Compute intercept from constraint
    b = y0 - m * x0
    
    return m, b

def PlotCerVSSciEnergy(array_Sci,array_Cer, energy, runNumber, particle, chi = None, suffix=""):
    #slope, intercept = np.polyfit(array_Sci, array_Cer, 1)
    array_Sci_fit = array_Sci[array_Sci>energy*0.5]
    array_Cer_fit = array_Cer[array_Sci>energy*0.5]
    slope, intercept = constrained_fit(array_Sci_fit, array_Cer_fit, energy, energy)
    array_Cer_fit = slope * array_Sci + intercept
    ss_res = np.sum((array_Cer - array_Cer_fit) ** 2)
    ss_tot = np.sum((array_Cer - np.mean(array_Cer)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    array_Sci_sorted = np.sort(array_Sci)
    plt.figure(figsize=(8, 6))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.scatter(array_Sci, array_Cer, alpha=0.7, label='Events',marker='.',edgecolors = 'none', s = 2)
    plt.plot(array_Sci_sorted, slope * array_Sci_sorted + intercept, 'r-', 
         label=f'Fit: Cer. = {slope:.3f}Sci. + {intercept:.3f}\n$R^2$ = {r_squared:.3f}')
    if chi is not None:
        slope_est = 1/chi
        intercept_est = (1-1/chi) * energy
        plt.plot(array_Sci_sorted, slope_est * array_Sci_sorted + intercept_est, 'g-',
             label=f'Fit: Cer. = {slope_est:.3f}Sci. + {intercept_est:.3f}')
    plt.plot(array_Sci_sorted, array_Sci_sorted, 'b--',alpha=0.5, label='Cer. = Sci.')
    plt.plot(array_Sci_sorted, 0.5 * array_Sci_sorted, 'b:',alpha=0.5, label='Cer. = 0.5 Sci.')
    plt.plot([0,energy], [energy,energy], 'k--',alpha=0.5)
    plt.plot([energy,energy], [0,energy], 'k--',alpha=0.5)
    plt.title(f"Measured energy of {particle} beam, Run {runNumber}, E = {energy} GeV",fontsize=14)
    plt.xlabel("Energy measured by Sci. channels [GeV]",fontsize=14)
    plt.ylabel("Energy measured by Cer. channels [GeV]",fontsize=14)
    plt.axis('equal')
    plt.ylim(-10,1.5*energy)
    plt.legend(fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.system(f"mkdir -p results/plots/Run{runNumber}")
    plt.savefig(f"results/plots/Run{runNumber}/C_S_run{runNumber}_{textgain}_energy{energy}{filter_suffix}{suffix}.png", dpi=300)
    plt.close()


def PlotCerVSSciResponse(array_Sci,array_Cer,  particle, chi = None, suffix=""):
    #slope, intercept = np.polyfit(array_Sci, array_Cer, 1)
    array_Sci_fit = array_Sci[array_Sci>0.5]
    array_Cer_fit = array_Cer[array_Sci>0.5]
    slope, intercept = constrained_fit(array_Sci_fit, array_Cer_fit, 1, 1)
    array_Cer_fit = slope * array_Sci + intercept
    ss_res = np.sum((array_Cer - array_Cer_fit) ** 2)
    ss_tot = np.sum((array_Cer - np.mean(array_Cer)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    array_Sci_sorted = np.sort(array_Sci)
    plt.figure(figsize=(8, 6))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.scatter(array_Sci, array_Cer, alpha=0.7, label='Events',marker='.',edgecolors = 'none', s = 2)
    plt.plot(array_Sci_sorted, slope * array_Sci_sorted + intercept, 'r-', 
         label=f'Fit: C/E = {slope:.3f}S/E + {intercept:.3f}\n$R^2$ = {r_squared:.3f}')
    if chi is not None:
        slope_est = 1/chi
        intercept_est = (1-1/chi)
        plt.plot(array_Sci_sorted, slope_est * array_Sci_sorted + intercept_est, 'g-',
             label=f'Fit: C/E = {slope_est:.3f}S/E + {intercept_est:.3f}')
    plt.plot(array_Sci_sorted, array_Sci_sorted, 'b--',alpha=0.5, label='Cer. = Sci.')
    plt.plot(array_Sci_sorted, 0.5 * array_Sci_sorted, 'b:',alpha=0.5, label='Cer. = 0.5 Sci.')
    plt.plot([0,1], [1,1], 'k--',alpha=0.5)
    plt.plot([1,1], [0,1], 'k--',alpha=0.5)
    plt.title(f"Measured energy response of {particle} beams",fontsize=14)
    plt.xlabel("S/E",fontsize=14)
    plt.ylabel("C/E",fontsize=14)
    plt.axis('equal')
    plt.ylim(-0.1,1.5)
    plt.legend(fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.system(f"mkdir -p results/plots/{particle}")
    plt.savefig(f"results/plots/{particle}/C_S_response_{textgain}_{filter_suffix}{suffix}.png", dpi=300)
    plt.close()



def PlotCerVSSciRaw(array_Sci,array_Cer, energy, runNumber, particle,suffix=""):
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
    plt.savefig(f"results/plots/Run{runNumber}/C_S_raw_run{runNumber}_{var}_{textgain}_energy{energy}{filter_suffix}{suffix}.png", dpi=300)
    plt.close()

def PlotBias(
    energies, 
    bias_Sci, bias_Cer, 
    bias_unc_Sci, bias_unc_Cer, 
    particle,suffix="",
    bias_dual=[],bias_unc_dual=[]
):
    plt.figure(figsize=(8, 6))
    plt.ylim(0.3,1.25)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # Plot Sci. with error bars
    plt.errorbar(
        energies, bias_Sci, yerr=bias_unc_Sci,
        fmt='bs-', capsize=3, label="Sci."
    )

    # Plot Cer. with error bars
    plt.errorbar(
        energies, bias_Cer, yerr=bias_unc_Cer,
        fmt='ro-', capsize=3, label="Cer."
    )
    if len(bias_dual) > 0:
        plt.errorbar(
        energies, bias_dual, yerr=bias_unc_dual,
        fmt='ko-', capsize=3, label="Combined"
    )
    # Reference line at 1.0
    plt.axhline(
        1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, 
        label="response=1"
    )

    # Titles and labels
    plt.title(f"Bias of the measured energy, {particle} beam",fontsize=15)
    plt.xlabel("Beam energy [GeV]",fontsize=15)
    plt.ylabel("Measured energy / beam energy",fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save
    os.system(f"mkdir -p results/plots/{particle}")
    plt.savefig(
        f"results/plots/{particle}/bias_{textgain}{filter_suffix}{suffix}.png", 
        dpi=300
    )
    plt.close()


def PlotFitK(
    energies, 
    bias_Sci, bias_Cer, 
    bias_unc_Sci, bias_unc_Cer, 
    particle,suffix=""
):
    plt.figure(figsize=(8, 6))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # Plot Sci. with error bars
    x = np.log(np.array(energies))
    y_Sci = np.log(1-np.array(bias_Sci))
    y_unc_Sci = np.array(bias_unc_Sci)/(1-np.array(bias_Sci))
    y_Cer = np.log(1-np.array(bias_Cer))
    y_unc_Cer = np.array(bias_unc_Cer)/(1-np.array(bias_Cer))
    plt.errorbar(
        x, y_Sci, yerr=y_unc_Sci,
        fmt='bs', capsize=3, label="Sci."
    )

    # Plot Cer. with error bars
    plt.errorbar(
        x, y_Cer, yerr=y_unc_Cer,
        fmt='ro', capsize=3, label="Cer."
    )
    slope_sci, intercept_sci = np.polyfit(
        x, y_Sci, 1, w=1/np.array(y_unc_Sci)
    )
    slope_cer, intercept_cer = np.polyfit(
        x, y_Cer, 1, w=1/np.array(y_unc_Cer)
    )

    x_fit = np.linspace(min(x), max(x), 200)
    plt.plot(
        x_fit, slope_sci * x_fit + intercept_sci, 'b-',
        label=f"{slope_sci:.2f}lnE  {intercept_sci:.2f}"
    )
    plt.plot(
        x_fit, slope_cer * x_fit + intercept_cer, 'r-',
        label=f"{slope_cer:.2f}lnE  {intercept_cer:.2f}"
    )
    # Titles and labels
    plt.title(f"Fit scaling factor, {particle} beam",fontsize=15)
    plt.xlabel("lnE",fontsize=15)
    plt.ylabel(rf"ln(1-$\pi/e$)",fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save
    os.system(f"mkdir -p results/plots/{particle}")
    plt.savefig(
        f"results/plots/{particle}/fitK_{textgain}{filter_suffix}{suffix}.png", 
        dpi=300
    )
    plt.close()









def PlotCSRatios(
    energies, 
    CSratios, 
    CSratios_unc,
    particle,suffix=""
):
    plt.figure(figsize=(8, 6))
    plt.ylim(0.5,1.05)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # Plot Sci. with error bars
    plt.errorbar(
        energies, CSratios, yerr=CSratios_unc,
        fmt='ks-', capsize=3, label="<C/S>"
    )

    # Reference line at 1.0
    plt.axhline(
        1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, 
        label="<C/S>=1"
    )

    # Titles and labels
    plt.title(f"Mean of the ratios Cer./Sci., {particle} beam",fontsize=15)
    plt.xlabel("Beam energy [GeV]",fontsize=15)
    plt.ylabel("Cer./Sci. ratios",fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save
    os.system(f"mkdir -p results/plots/{particle}")
    plt.savefig(
        f"results/plots/{particle}/CSratios_{textgain}{filter_suffix}{suffix}.png", 
        dpi=300
    )
    plt.close()

def PlotPiERatios(
    energies, 
    pi_e_ratios_Sci,  pi_e_ratios_Cer, 
    pi_e_ratios_unc_Sci,  pi_e_ratios_unc_Cer,
    particle,suffix=""
):
    plt.figure(figsize=(8, 6))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.errorbar(
        energies, pi_e_ratios_Sci, yerr=pi_e_ratios_unc_Sci,
        fmt='bs-', capsize=3, label="Sci."
    )
    plt.errorbar(
        energies, pi_e_ratios_Cer, yerr=pi_e_ratios_unc_Cer,
        fmt='ro-', capsize=3, label="Cer."
    )
    plt.axhline(
        1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, 
        label="π/e = 1"
    )
    # Titles and labels
    plt.title(f"Ratios of pion to positron responses",fontsize=15)
    plt.xlabel("Beam energy [GeV]",fontsize=15)
    plt.ylabel("π/e",fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save
    os.system(f"mkdir -p results/plots/{particle}")
    plt.savefig(
        f"results/plots/{particle}/PiEratios_{textgain}{filter_suffix}{suffix}.png", 
        dpi=300
    )
    plt.close()


def PlotEHRatios(
    energies, 
    e_h_ratios_Sci,  e_h_ratios_Cer,
    particle,suffix=""
):
    plt.figure(figsize=(8, 6))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    print(energies)
    print("e_h_ratios_Sci",e_h_ratios_Sci)
    print("e_h_ratios_Cer",e_h_ratios_Cer)
    plt.plot(
        energies, e_h_ratios_Sci,
        'bs-', label="Sci."
    )
    plt.plot(
        energies, e_h_ratios_Cer,
        'ro-', label="Cer."
    )
    plt.axhline(
        1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, 
        label="e/h = 1"
    )
    # Titles and labels
    plt.title(f"Ratios of e.m. to hadronic shower responses",fontsize=15)
    plt.xlabel("Beam energy [GeV]",fontsize=15)
    plt.ylabel("e/h",fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save
    os.system(f"mkdir -p results/plots/{particle}")
    plt.savefig(
        f"results/plots/{particle}/EHratios_{textgain}{filter_suffix}{suffix}.png", 
        dpi=300
    )
    plt.close()

def PlotResolution(
    energies, 
    resolution_Sci, resolution_Cer, 
    resolution_unc_Sci, resolution_unc_Cer, 
    particle,suffix="",
    resolution_dual = [], resolution_unc_dual = [],
    resolution_calialg_Sci = None, resolution_calialg_Cer = None,
    resolution_calialg_unc_Sci = None, resolution_calialg_unc_Cer = None,
    resolution_photostat_Sci = None, resolution_photostat_Cer = None,
    resolution_photostat_unc_Sci = None, resolution_photostat_unc_Cer = None,
):
    # x axis: 1/sqrt(E)
    x = 1 / np.sqrt(energies)

    # Weighted linear fits (use 1/uncertainty as weights)
    slope_sci, intercept_sci = np.polyfit(
        x, resolution_Sci, 1, w=1/np.array(resolution_unc_Sci)
    )
    slope_cer, intercept_cer = np.polyfit(
        x, resolution_Cer, 1, w=1/np.array(resolution_unc_Cer)
    )
    if len(resolution_dual)>0:
        slope_dual, intercept_dual = np.polyfit(
            x, resolution_dual, 1, w=1/np.array(resolution_unc_dual)
        )
    if resolution_calialg_Sci is not None:
        intercept_calialg_sci = np.polyfit(
            x, resolution_calialg_Sci, 0, w=1/np.array(resolution_calialg_unc_Sci)
        )
        intercept_calialg_sci = intercept_calialg_sci[0]
    if resolution_calialg_Cer is not None:
        intercept_calialg_cer = np.polyfit(
            x, resolution_calialg_Cer, 0, w=1/np.array(resolution_calialg_unc_Cer)
        )
        intercept_calialg_cer = intercept_calialg_cer[0]
    if resolution_photostat_Sci is not None:
        slope_photostat_sci, intercept_photostat_sci = np.polyfit(
            x, resolution_photostat_Sci, 1, w=1/np.array(resolution_photostat_unc_Sci)
        )
    if resolution_photostat_Cer is not None:
        slope_photostat_cer, intercept_photostat_cer = np.polyfit(
            x, resolution_photostat_Cer, 1, w=1/np.array(resolution_photostat_unc_Cer)
        )
    # Plot setup
    plt.figure(figsize=(8, 6))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # Plot with error bars
    plt.errorbar(
        x, resolution_Sci, yerr=resolution_unc_Sci, 
        fmt='bs', capsize=3, label="Sci."
    )
    plt.errorbar(
        x, resolution_Cer, yerr=resolution_unc_Cer, 
        fmt='ro', capsize=3, label="Cer."
    )
    if len(resolution_dual)>0:
        plt.errorbar(
            x, resolution_dual, yerr=resolution_unc_dual, 
            fmt='ko', capsize=3, label="Combined"
        )
    if resolution_calialg_Sci is not None:
        plt.errorbar(
            x, resolution_calialg_Sci, yerr=resolution_calialg_unc_Sci, 
            fmt='bd', capsize=3, label="Sci. cali. unc."
        )
    if resolution_calialg_Cer is not None:
        plt.errorbar(
            x, resolution_calialg_Cer, yerr=resolution_calialg_unc_Cer, 
            fmt='rh', capsize=3, label="Cer. cali. unc."
        )
    if resolution_photostat_Sci is not None:
        plt.errorbar(
            x, resolution_photostat_Sci, yerr=resolution_photostat_unc_Sci, 
            fmt='bv', capsize=3, label="Sci. stat. unc."
        )
    if resolution_photostat_Cer is not None:
        plt.errorbar(
            x, resolution_photostat_Cer, yerr=resolution_photostat_unc_Cer, 
            fmt='r^', capsize=3, label="Cer. stat. unc."
        )    
    # Line fits
    x_fit = np.linspace(min(x), max(x), 200)
    plt.plot(
        x_fit, slope_sci * x_fit + intercept_sci, 'b-',
        label=fr"$\frac{{{slope_sci*100:1f}\%}}{{\sqrt{{E}}}} + {intercept_sci*100:.1f}\%$"
    )
    plt.plot(
        x_fit, slope_cer * x_fit + intercept_cer, 'r-',
        label=fr"$\frac{{{slope_cer*100:.1f}\%}}{{\sqrt{{E}}}} + {intercept_cer*100:.1f}\%$"
    )
    if len(resolution_dual)>0:
        plt.plot(
            x_fit, slope_dual * x_fit + intercept_dual, 'k-',
            label=fr"$\frac{{{slope_dual*100:1f}\%}}{{\sqrt{{E}}}} + {intercept_dual*100:.1f}\%$"
        )
    if resolution_calialg_Sci is not None:
        plt.plot(
            x_fit, 0 * x_fit + intercept_calialg_sci, 'b--',
            label=f"{intercept_calialg_sci*100:.1f}%"
        )
    if resolution_calialg_Cer is not None:
        plt.plot(
            x_fit, 0 * x_fit + intercept_calialg_cer, 'r--',
            label=f"{intercept_calialg_cer*100:.1f}%"
        )
    if resolution_photostat_Sci is not None:
        plt.plot(
            x_fit, slope_photostat_sci * x_fit + intercept_photostat_sci, 'b:',
            label=fr"$\frac{{{slope_photostat_sci*100:.1f}\%}}{{\sqrt{{E}}}} + {intercept_photostat_sci*100:.1f}\%$"
        )
    if resolution_photostat_Cer is not None:
        plt.plot(
            x_fit, slope_photostat_cer * x_fit + intercept_photostat_cer, 'r:',
            label=fr"$\frac{{{slope_photostat_cer*100:.1f}\%}}{{\sqrt{{E}}}} + {intercept_photostat_cer*100:.1f}\%$"
        )
    # Labels, title, legend, etc.
    plt.title(f"Resolution of the energy measurement, {particle} beam",fontsize=15)
    plt.xlabel(r"$1/\sqrt{E}$",fontsize=15)
    plt.ylabel(r"$\sigma / <E>$",fontsize=15)
    plt.legend(fontsize=12,ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save
    os.system(f"mkdir -p results/plots/{particle}")
    if resolution_calialg_Sci is not None:
        plt.savefig(
            f"results/plots/{particle}/resolution_{textgain}{filter_suffix}{suffix}.png", 
            dpi=300
        )
    else:
        plt.savefig(
            f"results/plots/{particle}/resolution_{textgain}{filter_suffix}{suffix}.png", 
            dpi=300
        )
    plt.close()



bias = {"Sci": [],"Cer": [], "Combined": []}
bias_unc = {"Sci": [],"Cer": [], "Combined": []}
pi_e_ratio = {"Sci": [],"Cer": []}
pi_e_ratio_unc = {"Sci": [],"Cer": []}
e_h_ratio = {"Sci": [],"Cer": []}
resolution = {"Sci": [],"Cer": [],"Combined": []}
resolution_unc = {"Sci": [],"Cer": [],"Combined": []}
if addToy:
    resolution_calialg = {"Sci": [],"Cer": []}
    resolution_calialg_unc = {"Sci": [],"Cer": []}
else:
    resolution_calialg = {"Sci": None,"Cer": None}
    resolution_calialg_unc = {"Sci": None,"Cer": None}
if addPhotoStat:
    resolution_photostat = {"Sci": [],"Cer": []}
    resolution_photostat_unc = {"Sci": [],"Cer": []}
else:
    resolution_photostat = {"Sci": None,"Cer": None}
    resolution_photostat_unc = {"Sci": None,"Cer": None}
mean_conversion = {}
CSratios_mean = []
CSratios_mean_unc = []
CSratios_resolution = []
CSratios_resolution_unc = []


if isHG:
    infile_calibrate = f"results/root/Run{runNumber_calibrate}/FERS_filtered{filter_suffix_calibrate}.npz"
else:
    infile_calibrate = f"results/root/Run{runNumber_calibrate}/FERS_filtered_LG{filter_suffix_calibrate}.npz"
FERSData = np.load(infile_calibrate,"r")
for var, calibration_file in zip(["Cer","Sci"],[calibration_file_Cer,calibration_file_Sci]):
    if isHG:
        selection = FERSData[f"FERS_{var}EnergyHG_subtracted_saturationcorrected"][:]>20000
    else:
        selection = FERSData[f"FERS_{var}EnergyLG_subtracted"][:]>1000
    FERS_channels, conversion_factors = ReadADC2EnergyConversion(calibration_file)
    FERSArrays = np.column_stack([FERSData[c][:] for c in FERS_channels])
    FERSRaw = np.sum(FERSArrays, axis=-1)
    FERSRaw_mean = np.mean(FERSRaw)
    mean_conversion[var] = energy_calibrate/FERSRaw_mean
    print("mean_conversion",var,mean_conversion[var])


if particle == "pion":
    responses_positrons = {"Cer": [], "Sci": []}
    for runNumber,energy in zip(runNumbers_positrons, energies_positrons):
        if isHG:
            infile = f"results/root/Run{runNumber}/FERS_filtered{filter_suffix_positrons}.npz"
        else:
            infile = f"results/root/Run{runNumber}/FERS_filtered_LG{filter_suffix_positrons}.npz"
        FERSData = np.load(infile,"r")
        for var, calibration_file in zip(["Cer","Sci"],[calibration_file_Cer,calibration_file_Sci]):
            lastEvent = len(FERSData[list(FERSData.keys())[0]])
            FERS_channels, conversion_factors = ReadADC2EnergyConversion(calibration_file)
            FERSArrays = np.column_stack([FERSData[c][:] for c in FERS_channels])
            FERSEnergies = FERSArrays @ conversion_factors
            range_min, range_max = max(0,energy*0.8-40),energy*2.5
            hist_fit, bin_edges = np.histogram(FERSEnergies, bins=50, range=(range_min, range_max))
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            p0 = [hist_fit.max(), bin_centers[np.argmax(hist_fit)], 10.0]
            popt, pcov = curve_fit(gaussian, bin_centers, hist_fit, p0=p0)
            _, mu_fit, _  = popt
            responses_positrons[var].append(mu_fit / energy)

response_measured = {"Cer": None,"Sci": None}

for runNumber, energy, firstEvent, lastEvent in zip(runNumbers, energies, firstEvents, lastEvents):
    energy_measured = {}
    energy_raw = {}
    energy_unc_calialg = {}
    selections = {}
    if isHG:
        infile = f"results/root/Run{runNumber}/FERS_filtered{filter_suffix}.npz"
    else:
        infile = f"results/root/Run{runNumber}/FERS_filtered_LG{filter_suffix}.npz"
    FERSData = np.load(infile,"r")

    for var, calibration_file, calibration_file_calialg in zip(["Cer","Sci"],[calibration_file_Cer,calibration_file_Sci],[calibration_file_Cer_calialg_toys,calibration_file_Sci_calialg_toys]):
        if lastEvent == -1: lastEvent = len(FERSData[list(FERSData.keys())[0]])
        if isHG:
            selection = FERSData[f"FERS_{var}EnergyHG_subtracted_saturationcorrected"][firstEvent:lastEvent]>20000
        else:
            selection = FERSData[f"FERS_{var}EnergyLG_subtracted"][firstEvent:lastEvent]>1000
        selections[var] = selection
        FERS_channels, conversion_factors = ReadADC2EnergyConversion(calibration_file)
        FERS_gains = [GetChannelGain(ch) for ch in FERS_channels]
        if addToy:
            FERS_channels_calialg_toys, conversion_factors_calialg_toys = ReadADC2EnergyConversionToys(calibration_file_calialg)
            conversion_factors_calialg_toys = np.vstack([conversion_factors_calialg_toys[FERS_channels_calialg_toys.index(ch)] for ch in FERS_channels])
        FERSArrays = np.column_stack([FERSData[c][firstEvent:lastEvent] for c in FERS_channels])
        FERSEnergies = FERSArrays @ conversion_factors
        print("np.sqrt(FERSArrays @ (FERS_gains * conversion_factors * conversion_factors))",np.sqrt(FERSArrays @ (FERS_gains * conversion_factors * conversion_factors)))
        resolution_photostat[var].append(np.nanmean(np.sqrt(FERSArrays @ (FERS_gains * conversion_factors * conversion_factors)))/energy)
        resolution_photostat_unc[var].append(np.nanstd(np.sqrt(FERSArrays @ (FERS_gains * conversion_factors * conversion_factors)),ddof=1)/energy)
        if addToy:
            FERSEnergiesToys = FERSArrays @ conversion_factors_calialg_toys
            FERSEnergiesStd = np.std(FERSEnergiesToys,axis=-1,ddof=1)
            resolution_calialg[var].append(np.mean(FERSEnergiesStd/energy))
            resolution_calialg_unc[var].append(np.std(FERSEnergiesStd/energy,ddof=1))
        FERSRaw = np.sum(FERSArrays, axis=-1)
        energy_measured[var] = FERSEnergies
        energy_raw[var] = FERSRaw
        if response_measured[var] is None:
            response_measured[var] =  FERSEnergies/energy
        else:
            response_measured[var] = np.concatenate((response_measured[var],FERSEnergies/energy))
        print(f"{var} Mix: ",FERSRaw)
        print(f"{var} Mix mean: ",np.mean(FERSRaw))
        print(f"{var} Mix calibrated: ",FERSEnergies)
        print(f"{var} Mix calibrated mean: ",np.mean(FERSEnergies))

    PlotEnergyRaw(energy_raw["Cer"], energy, "Cer", runNumber,50,suffix=cali_suffix)
    PlotEnergyRaw(energy_raw["Sci"], energy, "Sci", runNumber,50,suffix=cali_suffix)
    mean_measured_Cer = np.mean(energy_measured["Cer"][energy_measured["Sci"]>energy/4])
    std_measured_Cer = np.std(energy_measured["Cer"][energy_measured["Sci"]>energy/4])
    mean_measured_unc_Cer = std_measured_Cer/np.sqrt(np.sum(energy_measured["Sci"]>energy/4))
    std_measured_unc_Cer = std_measured_Cer/np.sqrt(2*np.sum(energy_measured["Sci"]>energy/4)-2)
    mean_measured_Sci = np.mean(energy_measured["Sci"][energy_measured["Sci"]>energy/4])
    std_measured_Sci = np.std(energy_measured["Sci"][energy_measured["Sci"]>energy/4])
    mean_measured_unc_Sci = std_measured_Sci/np.sqrt(np.sum(energy_measured["Sci"]>energy/4))
    std_measured_unc_Sci = std_measured_Sci/np.sqrt(2*np.sum(energy_measured["Sci"]>energy/4)-2)
    mu_fit_measured_Cer,mu_err_measured_Cer,sigma_fit_measured_Cer,sigma_err_measured_Cer = PlotEnergyMeasured(energy_measured["Cer"],energy_raw["Cer"]*mean_conversion["Cer"], energy, "Cer", runNumber,50)
    mu_fit_measured_Sci,mu_err_measured_Sci,sigma_fit_measured_Sci,sigma_err_measured_Sci = PlotEnergyMeasured(energy_measured["Sci"], energy_raw["Sci"]*mean_conversion["Sci"],energy, "Sci", runNumber,50)
    mu_fit_CSratio,mu_err_CSratio,sigma_fit_CSratio,sigma_err_CSratio = PlotEnergyRatioMeasured(energy_measured["Cer"],energy_measured["Sci"], energy, runNumber,bins = 50)
    CSratios_mean.append(mu_fit_CSratio)
    CSratios_mean_unc.append(mu_err_CSratio)
    CSratios_resolution.append(sigma_fit_CSratio)
    CSratios_resolution_unc.append(sigma_err_CSratio)
    #bias["Cer"].append(mu_fit_measured_Cer/energy)
    #bias["Sci"].append(mu_fit_measured_Sci/energy)
    #bias_unc["Cer"].append(mu_err_measured_Cer/energy)
    #bias_unc["Sci"].append(mu_err_measured_Sci/energy)
    bias["Cer"].append(mean_measured_Cer/energy)
    bias["Sci"].append(mean_measured_Sci/energy)
    bias_unc["Cer"].append(mean_measured_unc_Cer/energy)
    bias_unc["Sci"].append(mean_measured_unc_Sci/energy)
    if particle == "pion":
        pi_e_ratio_Cer = mu_fit_measured_Cer/energy / GetPositronResponse(energy, responses_positrons["Cer"],energies_positrons)
        pi_e_ratio_Sci = mu_fit_measured_Sci/energy / GetPositronResponse(energy, responses_positrons["Sci"],energies_positrons)
        e_h_ratio_Cer = GetEHRatio(energy,pi_e_ratio_Cer)
        e_h_ratio_Sci = GetEHRatio(energy,pi_e_ratio_Sci)
        #chi = (1-1/e_h_ratio_Sci) / (1-1/e_h_ratio_Cer)
        chi=0.3571428571428571
        #chi = 1/1.8
        #chi=0.2
        #chi=0.59
        energy_combined = (energy_measured["Sci"]-chi*energy_measured["Cer"])/(1-chi)
        mu_fit_measured_dual,mu_err_measured_dual,sigma_fit_measured_dual,sigma_err_measured_dual = PlotEnergyMeasured(energy_combined, None,energy, "combined", runNumber,50)
        mean_measured_combined = np.mean(energy_combined[energy_measured["Sci"]>energy/4])
        std_measured_combined = np.std(energy_combined[energy_measured["Sci"]>energy/4])
        mean_measured_unc_combined = std_measured_combined/np.sqrt(np.sum(energy_measured["Sci"]>energy/4))
        std_measured_unc_combined = std_measured_combined/np.sqrt(2*np.sum(energy_measured["Sci"]>energy/4)-2)
        #resolution["Combined"].append(sigma_fit_measured_dual/mu_fit_measured_dual)
        #resolution_unc["Combined"].append(sigma_err_measured_dual/mu_fit_measured_dual)
        resolution["Combined"].append(std_measured_combined/mean_measured_combined)
        resolution_unc["Combined"].append(mean_measured_unc_combined/mean_measured_combined)
        print("energy ",energy)
        print("center energy_measured Sci ", mu_fit_measured_Sci)
        print("center energy_measured Cer ", mu_fit_measured_Cer)
        print("(S-chi*C)/(1-chi) ",(mu_fit_measured_Sci-chi*mu_fit_measured_Cer)/(1-chi))
        bias["Combined"].append(mu_fit_measured_dual/energy)
        bias_unc["Combined"].append(mu_err_measured_dual/energy)
        pi_e_ratio["Cer"].append(pi_e_ratio_Cer)
        pi_e_ratio["Sci"].append(pi_e_ratio_Sci)
        e_h_ratio["Cer"].append(e_h_ratio_Cer)
        e_h_ratio["Sci"].append(e_h_ratio_Sci)
        pi_e_ratio_unc["Cer"].append(mu_err_measured_Cer/energy / GetPositronResponse(energy, responses_positrons["Cer"],energies_positrons))
        pi_e_ratio_unc["Sci"].append(mu_err_measured_Sci/energy / GetPositronResponse(energy, responses_positrons["Sci"],energies_positrons))
    #resolution["Cer"].append(sigma_fit_measured_Cer/mu_fit_measured_Cer)
    #resolution["Sci"].append(sigma_fit_measured_Sci/mu_fit_measured_Sci)
    #resolution_unc["Cer"].append(sigma_err_measured_Cer/mu_fit_measured_Cer)
    #resolution_unc["Sci"].append(sigma_err_measured_Sci/mu_fit_measured_Sci)
    resolution["Cer"].append(std_measured_Cer/mean_measured_Cer)
    resolution["Sci"].append(std_measured_Sci/mean_measured_Sci)
    resolution_unc["Cer"].append(std_measured_unc_Cer/mean_measured_Cer)
    resolution_unc["Sci"].append(std_measured_unc_Sci/mean_measured_Sci)
    PlotCerVSSciEnergy(energy_measured["Sci"],energy_measured["Cer"], energy, runNumber, particle,chi = chi if particle == "pion" else None,suffix=cali_suffix)
    PlotCerVSSciRaw(energy_raw["Sci"],energy_raw["Cer"], energy, runNumber, particle,suffix=cali_suffix)

PlotCerVSSciResponse(response_measured["Sci"],response_measured["Cer"],  particle, chi = chi if particle == "pion" else None, suffix=cali_suffix)
PlotResolution(energies,resolution["Sci"], resolution["Cer"], resolution_unc["Sci"], resolution_unc["Cer"], particle,suffix=cali_suffix,resolution_dual = resolution["Combined"],resolution_unc_dual = resolution_unc["Combined"])
print("resolution_photostat['Sci']",resolution_photostat["Sci"])
print("resolution_photostat['Cer']",resolution_photostat["Cer"])
print("resolution_photostat_unc['Sci']",resolution_photostat_unc["Sci"])
print("resolution_photostat_unc['Cer']",resolution_photostat_unc["Cer"])
PlotResolution(energies,resolution["Sci"], resolution["Cer"], resolution_unc["Sci"], resolution_unc["Cer"], particle,suffix=cali_suffix+"_uncdecomp",\
resolution_calialg_Sci = resolution_calialg["Sci"], resolution_calialg_Cer = resolution_calialg["Cer"], resolution_calialg_unc_Sci = resolution_calialg_unc["Sci"], resolution_calialg_unc_Cer = resolution_calialg_unc["Cer"],\
resolution_photostat_Sci = resolution_photostat["Sci"], resolution_photostat_Cer = resolution_photostat["Cer"], resolution_photostat_unc_Sci = resolution_photostat_unc["Sci"], resolution_photostat_unc_Cer = resolution_photostat_unc["Cer"])

PlotBias(energies,bias["Sci"], bias["Cer"], bias_unc["Sci"], bias_unc["Cer"], particle,suffix=cali_suffix,bias_dual = bias["Combined"],bias_unc_dual = bias_unc["Combined"])
PlotCSRatios(energies, CSratios_mean, CSratios_mean_unc,particle,suffix=cali_suffix)
if particle == "pion":
    PlotFitK(energies, pi_e_ratio["Sci"],  pi_e_ratio["Cer"], pi_e_ratio_unc["Sci"],  pi_e_ratio_unc["Cer"], particle,suffix=cali_suffix)
    PlotPiERatios(energies, pi_e_ratio["Sci"],  pi_e_ratio["Cer"], pi_e_ratio_unc["Sci"],  pi_e_ratio_unc["Cer"],particle,suffix="")
    PlotEHRatios(energies, e_h_ratio["Sci"], e_h_ratio["Cer"], particle, suffix="")



