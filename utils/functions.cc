#include "ROOT/RVec.hxx"
#include <algorithm>
#include <iostream>
#include "TVectorF.h"
#include "TMatrixF.h"
#include "TF1.h"
#include "TDecompChol.h"
#include "TString.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TLine.h"
#include "TAxis.h"
#include <experimental/filesystem>
#include <iterator>
using namespace std;
namespace fs = std::experimental::filesystem;
ROOT::RVec<int> FillIndices(size_t n)
{
    ROOT::RVec<int> out(n);
    for (size_t i = 0; i < n; ++i)
        out[i] = i;
    return out;
}

float compute_median(ROOT::RVec<float> vec)
{
    if (vec.empty())
        return -9999;
    std::sort(vec.begin(), vec.end());
    size_t n = vec.size();
    if (n % 2 == 0)
        return 0.5 * (vec[n / 2 - 1] + vec[n / 2]);
    else
        return vec[n / 2];
}

size_t findTrigFireTime(const ROOT::VecOps::RVec<float> &vec, float val_min)
{
    for (size_t i = 0; i < vec.size(); ++i)
    {
        if (vec[i] < val_min / 2.0)
        {
            return i; // return the index of the first value below the threshold
        }
    }
    return -1; // return -1 if no value is below the threshold
}

ROOT::VecOps::RVec<float> clipToZero(const ROOT::VecOps::RVec<float> &vec)
{
    ROOT::VecOps::RVec<float> out;
    for (float v : vec)
    {
        if (fabs(v) < 3.0f)
            v = 0.0f; // clip to zero if below threshold
        out.push_back(v);
    }
    return out;
}

float SumRange(const ROOT::VecOps::RVec<float> &v, size_t i, size_t j)
{
    if (i >= v.size() || j > v.size() || i >= j)
        return 0.0;
    return std::accumulate(v.begin() + i, v.begin() + j, 0.0f);
}

float MaxRange(const ROOT::VecOps::RVec<float> &v, size_t i, size_t j)
{
    if (i >= v.size() || j > v.size() || i >= j)
        return 0.0;
    float maxVal = v[i];
    for (size_t k = i + 1; k < j; ++k)
        if (v[k] > maxVal)
            maxVal = v[k];
    return maxVal;
}
float MinRange(const ROOT::VecOps::RVec<float> &v, size_t i, size_t j)
{
    if (i >= v.size() || j > v.size() || i >= j)
        return 0.0;
    float minVal = v[i];
    for (size_t k = i + 1; k < j; ++k)
        if (v[k] < minVal)
            minVal = v[k];
    return minVal;
}
size_t ArgMinRange(const ROOT::VecOps::RVec<float> &v, size_t i, size_t j)
{
    if (i >= v.size() || j > v.size() || i >= j)
        return 0; // return 0 if the range is invalid
    auto minIt = std::min_element(v.begin() + i, v.begin() + j);
    return std::distance(v.begin(), minIt);
}

size_t ArgMaxRange(const ROOT::VecOps::RVec<float> &v, size_t i, size_t j, float threshold = 0.0f)
{
    if (i >= v.size() || j > v.size() || i >= j)
        return -1; // return -1 if the range is invalid
    auto maxIt = std::max_element(v.begin() + i, v.begin() + j);
    if (*maxIt < threshold)
        return -1; // return -1 if the maximum value is below the threshold
    return std::distance(v.begin(), maxIt);
}

unsigned int GetIdxFirstCross(float value, ROOT::RVec<float> v, unsigned int i_st, int direction) 
{
  unsigned int idx_end = direction>0 ? v.size() - 1 : 0;
  bool rising = value > v.at(i_st)? true : false; // Check if the given max value is greater than the given waveform amplitude at a given start time (i_st).

  // if it is: the value of the waveform needs to rise and otherwise it doesn't need so the function will return the i_st.
  unsigned int i = i_st;
  while( (i != idx_end)) 
  { // loop over the time bins
    if(rising && v.at(i) > value) break; // check if the rising variable is true and if the waveform value is going higher than the given amplitude value. And stops the loop.
    else if( !rising && v.at(i) < value) break; // check if the rising variable is false and if the waveform value is still lower than the given amplitude value. And stops the loop.
    i += direction; // Otherwise it moves to the next time bin.
  }

  return i;
}

void AnalyticalPolinomialSolver(unsigned int Np, float* in_x, float* in_y, unsigned int deg, float*& out_coeff)
{
    if (deg <= 0 || deg > 3) {
        std::cerr << "[ERROR]: You don't need AnalyticalPolinomialSolver for this\n";
        out_coeff = nullptr;
        return;
    }
    if (Np < deg + 1) {
        std::cerr << "[WARNING]: Not enough points for requested polynomial degree\n";
        out_coeff = nullptr;
        return;
    }

    TMatrixD A(Np, deg + 1);
    for (unsigned int i = 0; i < Np; ++i) {
        double x = in_x[i];
        A(i, 0) = 1.0;
        if (deg >= 1) A(i, 1) = x;
        if (deg >= 2) A(i, 2) = x * x;
        if (deg >= 3) A(i, 3) = x * x * x;
    }

    TVectorD y(Np);
    for (unsigned int i = 0; i < Np; ++i)
        y[i] = in_y[i];

    TMatrixD At = TMatrixD(TMatrixD::kTransposed, A);
    TMatrixD AtA = At * A;
    TVectorD Aty = At * y;

    // Suppress ROOT error messages
    int oldErrorLevel = gErrorIgnoreLevel;
    gErrorIgnoreLevel = kBreak;  // suppress all ROOT errors

    TDecompChol chol(AtA);
    Bool_t ok;
    TVectorD coeffs = chol.Solve(Aty, ok);

    // Restore original error level
    gErrorIgnoreLevel = oldErrorLevel;

    out_coeff = new float[deg + 1];

    if (!ok) {
        // std::cerr << "[WARNING]: Matrix not positive definite. Filling zeros.\n";
        for (unsigned int i = 0; i <= deg; ++i)
            out_coeff[i] = 0.0;
    } else {
        for (unsigned int i = 0; i <= deg; ++i)
            out_coeff[i] = coeffs[i];
    }
}

float PolyEval(float x, float* coeff, unsigned int deg) 
{
  float out = coeff[0] + x*coeff[1];
  for(unsigned int i=2; i<=deg; i++) 
  {
    out += coeff[i]*pow(x, i);
  }
  return out;
}

float fit_riseT(ROOT::RVec<float> vec, float fraction_amplitude, unsigned int PL_deg = 2, int polarity = 1, unsigned int bl_start = 600, unsigned int bl_end = 700, unsigned int window_start = 0, unsigned int window_end = 400,bool verbose = false, bool plot_fit = false, std::string plot_name = "debug", bool is_amplified = false)  {
    if (verbose) std::cout<<"start processing"<<std::endl;
    if (vec.empty()) return -9999;
    unsigned int j_90_pre = 0, j_10_pre = 0;
    unsigned int j_90_post = 0, j_10_post = 0;
    float baseline_RMS = 0;
    float baseline = 0;
    for(unsigned int j=bl_start; j<bl_end; j++) 
    {
      baseline += vec[j];
      baseline_RMS += vec[j] * vec[j];
    }
    if (verbose) std::cout<<std::endl;
    ROOT::RVec<float> time;
    for(float t = 0; t < vec.size(); t++) time.push_back(t);
    
    baseline = baseline / (bl_end - bl_start);
    baseline_RMS = sqrt(baseline_RMS/(bl_end - bl_start) - baseline*baseline);
    if (verbose) std::cout<<"baseline "<<baseline<<std::endl;
    if (verbose) std::cout<<"baseline_RMS "<<baseline_RMS<<std::endl;
    ROOT::RVec<float> vec_abs = vec;
    for (auto it = std::begin (vec_abs); it != std::end (vec_abs); ++it) *it = (*it - baseline) * polarity;
    float amp = *std::max_element(vec_abs.begin()+window_start, vec_abs.begin()+window_end);
    if (verbose) std::cout<<"amp "<<amp<<std::endl;
    unsigned int idx_peak = std::distance(vec_abs.begin(),std::max_element(vec_abs.begin()+window_start, vec_abs.begin()+window_end));
    if (verbose) std::cout<<"idx_peak "<<idx_peak<<std::endl;
    j_10_pre = GetIdxFirstCross(amp*0.1, vec_abs, idx_peak, -1);
    j_10_post = GetIdxFirstCross(amp*0.1, vec_abs, idx_peak, +1);
    j_90_pre = GetIdxFirstCross(amp*0.9, vec_abs, idx_peak, -1);
    j_90_post = GetIdxFirstCross(amp*0.9, vec_abs, idx_peak, +1);
    bool fittable = true;
    if (verbose) std::cout<<"window_start "<<window_start<<", window_end "<<window_end<<std::endl;
    fittable *= (idx_peak > window_start+1) && (idx_peak < window_end-1);
    if (verbose){
        if((idx_peak > window_start+1) && (idx_peak < window_end-1))
            std::cout<<"peak in window, continue processing"<<std::endl;
        else
            std::cout<<"peak out of window, not fittable"<<std::endl;
    }
    if(!fittable) return -9999;
    float thre_factor = 1;
    if(is_amplified) thre_factor = 2;
    fittable *= amp > 5 * baseline_RMS * thre_factor;
    fittable *= vec_abs[idx_peak+1] > 3 * baseline_RMS * thre_factor;
    fittable *= vec_abs[idx_peak-1] > 3 * baseline_RMS * thre_factor;
    fittable *= vec_abs[idx_peak+2] > 1.5 * baseline_RMS * thre_factor;
    fittable *= vec_abs[idx_peak-2] > 1.5 * baseline_RMS * thre_factor;
    if (verbose) {
        if((amp > 5 * baseline_RMS * thre_factor) && (vec_abs[idx_peak+1] > 3 * baseline_RMS * thre_factor) && (vec_abs[idx_peak-1] > 3*baseline_RMS * thre_factor) && (vec_abs[idx_peak+2] > 1.5 * baseline_RMS * thre_factor) && (vec_abs[idx_peak-2] > 1.5 * baseline_RMS * thre_factor))
            std::cout<<"peak fittable, continue processing"<<std::endl;
        else
            std::cout<<"peak is too low, not fittable"<<std::endl;
    }
    if(!fittable) return -9999;
    /*
    float start_level =   3 * baseline_RMS;

    unsigned int j_start =  GetIdxFirstCross( start_level, vec_abs, idx_peak, -1);
    unsigned int j_st = j_start;
    if (verbose) std::cout<<"j_start "<<j_start<<std::endl;
    if (amp*fraction_amplitude < start_level) j_st =  GetIdxFirstCross( amp*fraction_amplitude, vec_abs, idx_peak, -1);
    if (verbose) std::cout<<"j_st "<<j_st<<std::endl;

    unsigned int j_close = GetIdxFirstCross(amp*fraction_amplitude, vec_abs, j_st, +1);
    if (verbose) std::cout<<"finding j_close "<<j_close<<std::endl;
    if ( fabs(vec_abs[j_close-1] - fraction_amplitude*amp) < fabs(vec_abs[j_close] - fraction_amplitude*amp) ) j_close--;

    unsigned int span_j = (int) (min( j_90_pre-j_close , j_close-j_st)/1.5);

    if (j_90_pre - j_10_pre <= 3 * PL_deg) 
    {
        span_j = max((unsigned int)(PL_deg*0.5), span_j);
        span_j = max((unsigned int)1, span_j);
    }
    else 
    {
        span_j = max((unsigned int) PL_deg, span_j);
    }
    if (verbose) cout<<"j_close "<<j_close<<", span_j "<<span_j<<std::endl;
    fittable *= !((int)j_close + (int)span_j > (int)window_end || (int)j_close - (int)span_j < (int)window_start);

    if(!fittable) return -9999;
    float* coeff;
    int N_add = 1;
    if (span_j + N_add + j_close < j_90_pre) 
    {
        N_add++;
    }
    if (verbose) cout<<"running fit"<<endl;
    AnalyticalPolinomialSolver( 2*span_j + N_add , &(vec_abs[j_close - span_j]),&(time[j_close - span_j]), PL_deg, coeff);
    */
    float start_level = 1 * baseline_RMS * thre_factor;
    unsigned int j_start_level =  GetIdxFirstCross( start_level, vec_abs, idx_peak, -1);
    fittable *= (idx_peak - j_start_level > PL_deg);
    if (verbose) {
        if (idx_peak - j_start_level <= PL_deg) std::cout<<"peak is too narrow"<<std::endl;
    }
    if(!fittable) return -9999;
    unsigned int j_center = GetIdxFirstCross( fraction_amplitude*amp, vec_abs, idx_peak, -1);
    unsigned int j_end_level = idx_peak;
    if (amp > 20 * baseline_RMS) j_end_level = j_90_pre;
    unsigned int span_j = min(j_end_level-j_center, j_center-j_start_level);
    unsigned int j_st = j_center - span_j;
    unsigned int j_end = j_center + span_j;
    if (2*span_j <= PL_deg){
        if (j_end_level - j_center > j_center-j_start_level)
            j_end += PL_deg - 2*span_j + 1;
        else
            j_st -= PL_deg - 2*span_j + 1;
    }
    if (verbose) std::cout<<"running fit"<<endl;
    float* coeff;
    AnalyticalPolinomialSolver( j_end - j_st + 1 , &(vec_abs[j_st]),&(time[j_st]), PL_deg, coeff);
    if (verbose) std::cout<<"fitting result "<<PolyEval(fraction_amplitude*amp, coeff, PL_deg)<<endl;
    if (plot_fit){
        std::cout<<"j_end_level "<<j_end_level<<", j_end "<<j_end<<", j_st "<<j_st<<std::endl;
        for (unsigned int j = j_st; j<= j_end; j++)
          std::cout<<"time "<<j<<", amp "<<vec_abs[j]<<"; ";
        std::cout<<std::endl;
        TCanvas *c = new TCanvas("c", "Pulse shape fit", 800, 600);
        TGraph *g = new TGraph(window_end - window_start, &time[window_start], &vec_abs[window_start]);
        g->SetLineColor(kBlue);
        g->SetMarkerColor(kBlue);
        g->Draw("ALP");
        g->GetXaxis()->SetTitle("TS");
        g->GetYaxis()->SetTitle("Absolute amplitude");
        g->SetTitle("Pulse shape fit");
        std::string poly = "";
        for(unsigned int deg = 0; deg <=  PL_deg; deg++){
            poly.append(std::to_string(coeff[deg]));
            unsigned int x_count = deg;
            while(x_count > 0) {
                poly.append("*x");
                x_count--;
            }
            if(deg != PL_deg) poly.append(" + ");
        }
        std::cout<<"function "<<poly<<std::endl;
        TF1 * poly_fit = new TF1("fit",poly.c_str(),vec_abs[j_st],vec_abs[j_end]);
        std::vector<float> fit_time;
        std::vector<float> fit_amp;
        for(unsigned int it = 0; it <= 100; it++){
            float poly_amp = vec_abs[j_st] + (vec_abs[j_end]-vec_abs[j_st])/100*it;
            fit_amp.push_back(poly_amp);
            fit_time.push_back(poly_fit->Eval(poly_amp));
        }
        TGraph *g2 = new TGraph(101, &fit_time[0], &fit_amp[0]);
        g2->SetLineColor(kRed);
        g2->SetLineStyle(kSolid);
        g2->Draw("Csame");
        TLine* line_frac = new TLine(time[window_start], fraction_amplitude*amp, time[window_end], fraction_amplitude*amp);
        
        line_frac->SetLineColor(kRed);
        line_frac->SetLineStyle(kDashed);
        line_frac->Draw("same");
        c->SetGrid();
        unsigned int ifile = 0;
        std::string plot_name_modify = plot_name + "_" + std::to_string(ifile) + ".png";
        fs::path plot_path = plot_name_modify; 
        while(fs::exists(plot_path)){
            ifile++;
            plot_name_modify = plot_name + "_" + std::to_string(ifile) + ".png";
            plot_path = plot_name_modify;
        }
        c->SaveAs((plot_name_modify).c_str());
        //delete g;
        //delete g2;
        //delete poly_fit;
        //delete line_frac;
        //delete c;

    }
    return PolyEval(fraction_amplitude*amp, coeff, PL_deg);


}
size_t compute_triggerT(ROOT::RVec<float> vec) {
    if (vec.empty()) return -9999;
    size_t minT = std::distance(vec.begin(),std::min_element(vec.begin(), vec.end()));
    size_t maxT = std::distance(vec.begin(),std::max_element(vec.begin(), std::min_element(vec.begin(), vec.end())));
                         
    double half = (*std::min_element(vec.begin(), vec.end()) + *std::max_element(vec.begin(), std::min_element(vec.begin(), vec.end())))/2;
                      
    for(size_t iT = maxT; iT < minT; iT++){
        if( (vec[iT] > half) && (vec[iT+1] <= half) )         
            return (vec[iT] - half) > (half - vec[iT+1]) ? (iT+1) : iT ;      
    }
    return 0;          
}

float compute_peakAmp(ROOT::RVec<float> vec,unsigned int bl_start = 600, unsigned int bl_end = 700,int polarity = 1, unsigned int window_start = 0, unsigned int window_end = 400) {
    if (vec.empty()) return -9999;
    float peak_amp;
    if (polarity > 0)
        peak_amp = *std::max_element(vec.begin()+window_start, vec.begin()+window_end);
    else
        peak_amp = *std::min_element(vec.begin()+window_start, vec.begin()+window_end);
    float baseline = 0;
    for(unsigned int j=bl_start; j<bl_end; j++) 
    {
      baseline += vec[j];
    }
    baseline = baseline / (bl_end - bl_start);
    return (peak_amp-baseline)*polarity;
}
size_t compute_peakT(ROOT::RVec<float> vec, int polarity = 1) {
    if (vec.empty()) return -9999;
    if(polarity > 0)
        return std::distance(vec.begin(),std::max_element(vec.begin(), vec.end()));
    else
        return std::distance(vec.begin(),std::min_element(vec.begin(), vec.end()));
}