#ifndef MAINFRAME_H
#define MAINFRAME_H
#include "wxcrafter.h"
#include "RBF.h"









class MainFrame : public MainFrameBaseClass
{
public:
    
    MainFrame(wxWindow* parent);
    static void showMessage(wxString msg);
    virtual ~MainFrame();
    time_t m_time_start;
    bool m_bRBFrunning;
    RBF* m_RBF;
    wxString pathName;
    double d_nL1, d_nL2, d_rateIntial, d_rateMin, d_rateShift, d_iteration, d_nkFold, d_momentumAlpha, d_testDataRatio, d_terminalRatio;
    void getParameter();
    void OnExit(wxCommandEvent& event);
    void OnAbout(wxCommandEvent& event);
    wxString getTimer();
    void startTimer();
    void stopTimer(wxString TimerName);
    void OnRBFStart(wxThreadEvent& evt);
    void OnRBFUpdate(wxThreadEvent& evt);
    void OnRBFComplete(wxThreadEvent& evt);
    void OnRBFUpdatePg(wxThreadEvent& evt);
protected:
    virtual void OnUpdateParameterUI(wxUpdateUIEvent& event);
    virtual void OnUpdateUI(wxUpdateUIEvent& event);
    static MainFrame * m_pThis;
    
    virtual void OnLoadData(wxCommandEvent& event);
    virtual void OnTrainModel(wxCommandEvent& event);
};
#endif // MAINFRAME_H
