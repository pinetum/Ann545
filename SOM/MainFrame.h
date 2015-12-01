#ifndef MAINFRAME_H
#define MAINFRAME_H
#include "wxcrafter.h"
#include "SOM.h"


#define SQUARE_DATA 0
#define TRIANGLE_DATA 1

class MainFrame : public MainFrameBaseClass
{
public:
    MainFrame(wxWindow* parent);
    virtual ~MainFrame();
    bool b_threadRunning;
    SOM* p_Tsom;
    cv::Mat img ;
    cv::Mat readData(int data);
    void drawResults(cv::Mat weights, cv::Mat &img, wxString msg);
    void drawPts(cv::Mat &ptsData, cv::Mat &img);
    void OnExit(wxCommandEvent& event);
    void OnAbout(wxCommandEvent& event);
protected:
    virtual void OnBtnSquare(wxCommandEvent& event);
    virtual void OnBtnTriangle(wxCommandEvent& event);
    virtual void OnUpdateSquare(wxUpdateUIEvent& event);
    virtual void OnUpdateTriangle(wxUpdateUIEvent& event);
    
    
    void OnSOMStart(wxThreadEvent& evt);
    void OnSOMUpdate(wxThreadEvent& evt);
    void OnSOMComplete(wxThreadEvent& evt);
    
};
#endif // MAINFRAME_H
