#ifndef MAINFRAME_H
#define MAINFRAME_H
#include "wxcrafter.h"
#include "MLP.h"


class MainFrame : public MainFrameBaseClass
{
public:
    MainFrame(wxWindow* parent);
    static void showMessage(wxString msg);
    virtual ~MainFrame();
    time_t m_time_start;
    bool m_bMLPrunning;
    int m_nCPUs;
    MLP* m_MLP;
    wxString pathName;
    void OnExit(wxCommandEvent& event);
    void OnAbout(wxCommandEvent& event);
    wxString getTimer();
    void startTimer();
    void stopTimer(wxString TimerName);
    void OnMlpStart(wxThreadEvent& evt);
    void OnMlpUpdate(wxThreadEvent& evt);
    void OnMlpComplete(wxThreadEvent& evt);
    void OnMlpUpdatePg(wxThreadEvent& evt);
protected:
    virtual void OnUpdateParameterUI(wxUpdateUIEvent& event);
    virtual void OnUpdateUI(wxUpdateUIEvent& event);
    static MainFrame * m_pThis;
    
    virtual void OnLoadData(wxCommandEvent& event);
    virtual void OnTrainModel(wxCommandEvent& event);
};
#endif // MAINFRAME_H
