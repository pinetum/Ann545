#ifndef SOM_H
#define SOM_H
#include <wx/thread.h>
#include <wx/event.h> 
#include "opencv2/opencv.hpp"
#define CONTENT_SIZE 800




wxDECLARE_EVENT(wxEVT_COMMAND_SOM_START,        wxThreadEvent);
wxDECLARE_EVENT(wxEVT_COMMAND_SOM_UPDATE,       wxThreadEvent);
wxDECLARE_EVENT(wxEVT_COMMAND_SOM_COMPLETE,     wxThreadEvent);



class SOM : public wxThread
{
public:
    SOM(wxEvtHandler* pParent, int weightSize, int iterationTimes, cv::Mat input);
    ~SOM();
    virtual ExitCode Entry();
    wxEvtHandler* handler;
    double d_learningRate;
    double d_bandwith;
    int n_weightSize;
    int n_iterationTimes;
    cv::Mat m_input;
    void updateParameter(int epoch);
};

#endif // SOM_H
