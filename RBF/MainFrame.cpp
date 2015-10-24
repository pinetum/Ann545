#include "MainFrame.h"
#include <wx/aboutdlg.h>
#include <wx/filedlg.h>
#include <wx/thread.h>

wxDEFINE_EVENT(wxEVT_COMMAND_RBF_START,        wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_RBF_UPDATE,       wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_RBF_UPDATE_PG,    wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_RBF_COMPLETE,     wxThreadEvent);



MainFrame *MainFrame::m_pThis = NULL;

MainFrame::MainFrame(wxWindow* parent)
    : MainFrameBaseClass(parent)
{
    m_pThis             = this; 
    m_RBF               = NULL;
    m_bRBFrunning       = false;
    SetSize(-1, -1);
    Center();
    Bind(wxEVT_COMMAND_RBF_START, &MainFrame::OnRBFStart, this);
    Bind(wxEVT_COMMAND_RBF_UPDATE, &MainFrame::OnRBFUpdate, this);
    Bind(wxEVT_COMMAND_RBF_UPDATE_PG, &MainFrame::OnRBFUpdatePg, this);
    Bind(wxEVT_COMMAND_RBF_COMPLETE, &MainFrame::OnRBFComplete, this);
    
    
    
    m_textCtrl_IterationTimes->SetLabel("1500");
    m_textCtrl_L1neurons->SetLabel("6");
    //m_textCtrl_L2neurons->SetLabel("8");
    m_textCtrl_LearnRateInital->SetLabel("0.5");
    m_textCtrl_LearnRateMin->SetLabel("0.001");
    m_textCtrl_LearnRateShift->SetLabel("500");
    m_textCtrl_KFold->SetLabel("10");

    m_textCtrl_TestDataRatio->SetLabel("0.5");
    m_textCtrl_TerminalRatio->SetLabel("0.1");
    pathName.Empty();
    m_choice_LearnAdjust->Select(0);

    
    
}

MainFrame::~MainFrame()
{
    
}
void MainFrame::OnRBFStart(wxThreadEvent& evt)
{
    startTimer();
    showMessage("[RBF]start running..");
    m_bRBFrunning = true;
}
void MainFrame::OnRBFUpdate(wxThreadEvent& evt)
{
    showMessage(evt.GetString());
}
void MainFrame::OnRBFComplete(wxThreadEvent& evt)
{
    //stopTimer("RBF Stop");
    m_RBF = NULL;
    m_bRBFrunning = false;
    showMessage(evt.GetString());
}
void MainFrame::OnRBFUpdatePg(wxThreadEvent& evt)
{
    int iteration = evt.GetInt() + 1 ;
    int total = m_RBF->m_nTotalIteration;
    
    m_gaugePg->SetValue(100*iteration/total);
    m_staticTextPg->SetLabel(wxString::Format("%d/%d", iteration, total));
    m_staticTextTimer->SetLabel(getTimer());

    
    
}


void MainFrame::showMessage(wxString msg){
		m_pThis->m_richTextCtrl->AppendText(msg<<"\n");
		int last_pos = m_pThis->m_richTextCtrl->GetLastPosition();
		m_pThis->m_richTextCtrl->ShowPosition(last_pos);
}
void MainFrame::startTimer()
{
    m_time_start = clock();
}
void MainFrame::stopTimer(wxString TimerName)
{
    showMessage(wxString::Format(_("[Time][%s] %s"),TimerName, getTimer()));
}
wxString MainFrame::getTimer()
{
    wxString str_time = "";
    time_t now = clock();
    unsigned long seconds       = 0;
    unsigned long milliseconds  = 0;
    unsigned long minutes       = 0;
    unsigned long hours         = 0;
    seconds         = (now-m_time_start)/CLOCKS_PER_SEC;
    milliseconds    = ((1000*(now-m_time_start))/CLOCKS_PER_SEC) - 1000*seconds;
    minutes         = seconds/60;
    seconds         = seconds - minutes*60;
    hours           = minutes/60;
    minutes         = minutes - hours*60;
    
    //if(hours > 0)
        str_time.Append(wxString::Format("%02d:",hours));
    //if(minutes > 0)
        str_time.Append(wxString::Format("%02d:",minutes));
    //if(seconds > 0)
        str_time.Append(wxString::Format("%02d:",seconds));
    //if(milliseconds > 0)
        str_time.Append(wxString::Format("%03d",milliseconds));
    return str_time;
    
    
}
void MainFrame::OnExit(wxCommandEvent& event)
{
    
    
    if(m_RBF !=NULL)
        m_RBF->Delete();
    wxUnusedVar(event);
    Close();
}

void MainFrame::OnAbout(wxCommandEvent& event)
{
    wxUnusedVar(event);
    wxAboutDialogInfo info;
    info.SetCopyright(_("My MainFrame"));
    info.SetLicence(_("GPL v2 or later"));
    info.SetDescription(_("Short description goes here"));
    ::wxAboutBox(info);
}
void MainFrame::OnLoadData(wxCommandEvent& event)
{
    
    wxString fileType = _("All suported formats(*.*)|*.*");
    wxFileDialog* openDialog = new wxFileDialog(this,_("openFile"),wxEmptyString,wxEmptyString,fileType,wxFD_OPEN,wxDefaultPosition);
	if(openDialog->ShowModal() == wxID_OK){
        pathName = openDialog->GetPath();
    }
    openDialog->Destroy();
    if(pathName.length() == 0)
        MainFrame::showMessage("[LoadData]File error");
    
        
    
}
void MainFrame::getParameter(){
    m_textCtrl_L1neurons->GetValue().ToDouble(&d_nL1);
    m_textCtrl_LearnRateInital->GetValue().ToDouble(&d_rateIntial);
    m_textCtrl_LearnRateMin->GetValue().ToDouble(&d_rateMin);
    m_textCtrl_IterationTimes->GetValue().ToDouble(&d_iteration);
    m_textCtrl_LearnRateShift->GetValue().ToDouble(&d_rateShift);
    m_textCtrl_KFold->GetValue().ToDouble(&d_nkFold);
    
    m_textCtrl_TestDataRatio->GetValue().ToDouble(&d_testDataRatio);
    m_textCtrl_TerminalRatio->GetValue().ToDouble(&d_terminalRatio);
}
void MainFrame::OnTrainModel(wxCommandEvent& event)
{
    m_RBF = new RBF(this);
    m_RBF->openSampleFile(pathName);
    
    getParameter();
    m_RBF->SetParameter(m_checkBox_DataRescale->IsChecked(),
                        (int)d_nL1, 
                        d_rateIntial, 
                        d_rateMin, 
                        d_rateShift, 
                        d_iteration,
                        (int)d_nkFold,
                        d_testDataRatio,
                        d_terminalRatio,
                        m_choice_LearnAdjust->GetSelection());
    
    m_RBF->Run();
        
}
void MainFrame::OnUpdateUI(wxUpdateUIEvent& event)
{
    
    event.Enable(!pathName.empty());

        
}
void MainFrame::OnUpdateParameterUI(wxUpdateUIEvent& event)
{
    
    event.Enable(!m_bRBFrunning);
}

