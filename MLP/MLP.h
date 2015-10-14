#ifndef MLP_H
#define MLP_H
#include <opencv2/opencv.hpp>
#include <wx/string.h>
#include <wx/thread.h>
#include <wx/event.h> 
#include <wx/log.h>


wxDECLARE_EVENT(wxEVT_COMMAND_MLP_START,        wxThreadEvent);
wxDECLARE_EVENT(wxEVT_COMMAND_MLP_UPDATE,       wxThreadEvent);
wxDECLARE_EVENT(wxEVT_COMMAND_MLP_UPDATE_PG,       wxThreadEvent);
wxDECLARE_EVENT(wxEVT_COMMAND_MLP_COMPLETE,     wxThreadEvent);



class MLP : public wxThread
{
public:
    MLP(wxEvtHandler* pParent);
    ~MLP();
    bool train();
    bool writeModule();
    bool readModule();
    int m_nNeuronsL1;
    int m_nNeuronsL2;
    int m_nInputs;
    int m_nClasses;
    int m_nLearningRateShift;
    int m_nTotalIteration;
    int m_nKfold;
    
    double m_dMomentumAlpha;
    bool m_bMomentum;
    
    double m_dDesiredOutput_rescale;
    double m_dInitalLearningRate;
    double m_dMinLearningRate;
    
    double getAccuracy();
    double getSigmod_tan(double x, double slope)
        { return (1-2*exp(-2*x*slope))/(1+2*exp(-2*x*slope));}
    double getSigmodDerivative_tan(double x, double slope)
        { return slope*(1+getSigmod_tan(x, slope))*(1-getSigmod_tan(x, slope));}
    double getLearningRate(int i_iteration, double slope = 0.5)
        { return (m_dInitalLearningRate - m_dMinLearningRate) / (1+exp(slope*(i_iteration - m_nLearningRateShift)) ) + m_dMinLearningRate;}
    void Sigmod_tan(cv::Mat* x, double slope = 0.5)
    {
        cv::exp(*x*-2*slope, *x);
        *x = (-2**x+cv::Scalar(1) )/(2**x+cv::Scalar(1));
        
    }
    void Sigmod_tanDerivative(cv::Mat* x, double slope = 0.5)
    {
        Sigmod_tan(x, slope);
        *x = slope *( (cv::Scalar(1)+*x).mul(cv::Scalar(1)-*x));
    }
    
    
    
    void openSampleFile(wxString fileName);
    void SetParameter(  int n_nuronL1, 
                        int n_nuronL2, 
                        double d_InitalLearningRate, 
                        double d_MinLearningRate,
                        int n_LearningRateShift,
                        int n_TotalIteration,
                        bool b_Momentum,
                        double d_MomentumAlpha,
                        int n_kFold);
    cv::Mat getNetworkResponse();
    virtual ExitCode Entry();
private:
    cv::Mat     m_data_input;
    cv::Mat     m_data_scaled2train;
    cv::Mat     m_weight_l1;
    cv::Mat     m_weight_l2;
    cv::Mat     m_weight_l3;
    wxEvtHandler* m_pHandler;
    void dataScale();
    void readMat(wxString inputName, cv::Mat* data);
    void writeMat(wxString outputName, cv::Mat* data);
    
    void readDataLine(cv::Mat* data, wxString line);

};
#endif // MLP_H
