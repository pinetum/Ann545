#ifndef MLP_H
#define MLP_H
#include <opencv2/opencv.hpp>
#include <wx/string.h>
#include <wx/thread.h>
#include <wx/event.h> 
#include <thread>
class MLP
{
public:
    MLP();
    ~MLP();
    bool train();
    bool writeModule();
    bool readModule();
    wxString getErrorMessage();
    double getAccuracy();
    void openSampleFile(wxString fileName);
    void threadEntry();
    void dumpTrainData(wxString fileName){writeMat(fileName, &m_data_Train);}
    cv::Mat getNetworkResponse(cv::Mat input, cv::Mat weight_L1, cv::Mat weight_L2, cv::Mat weight_L3);
    
private:
    wxString    str_errorMsg;
    cv::Mat     m_data_Train;
    cv::Mat     m_weight;
    std::thread* m_pthread;
    void dataScale();
    void readMat(wxString inputName, cv::Mat* data);
    void writeMat(wxString outputName, cv::Mat* data);

    void readDataLine(cv::Mat* data, wxString line);

};
#endif // MLP_H
