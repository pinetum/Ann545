#ifndef SOM_H
#define SOM_H
#include <wx/thread.h>
#include <wx/event.h> 
#include "opencv2/opencv.hpp"
#include <wx/textfile.h>
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
    cv::Mat weights;
    void updateParameter(int epoch);
    void writeMat(wxString outputName, cv::Mat* data, wxString info = "");
    void shuffelRow(cv::Mat* x)
    {
        std::vector<int > s;

        for(int i = 0; i< x->rows; i++)
            s.push_back(i);
        cv::randShuffle(s);
        
        cv::Mat o(x->rows, x->cols, x->type());

        for (int i = 0; i <x->rows; i++)
            o.row(i) = x->row(s[i]) + 0 ;
        *x = o;
    }
};

#endif // SOM_H
