#ifndef MLP_H
#define MLP_H
#include <opencv2/opencv.hpp>
#include <wx/string.h>
#include <wx/thread.h>
#include <wx/event.h> 

#define MLP_TOTAL_ITERATION 5

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
    
    bool m_bMomentum;
    double m_dDesiredOutput_rescale;
    double m_dInitalLearningRate;
    double m_dMinLearningRate;
    
    double getAccuracy();
    double getSigmod_tan(double x, double slope)
        { return (1-2*exp(-2*x*slope))/(1+2*exp(-2*x*slope));}
    double getSigmodDerivative_tan(double x, double slope)
        { return slope*(1+getSigmod_tan(x, slope))*(1-getSigmod_tan(x, slope));}
    double getLearningRate(int i_iteration, double slope)
        { return (m_dInitalLearningRate - m_dMinLearningRate) / (1+exp(slope*(i_iteration - m_nLearningRateShift)) ) + m_dMinLearningRate;}
    void openSampleFile(wxString fileName);
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
