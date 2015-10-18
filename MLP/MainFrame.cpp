#include "MainFrame.h"
#include <wx/aboutdlg.h>
#include <wx/filedlg.h>
#include <wx/thread.h>

wxDEFINE_EVENT(wxEVT_COMMAND_MLP_START,        wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MLP_UPDATE,       wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MLP_UPDATE_PG,    wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MLP_COMPLETE,     wxThreadEvent);



MainFrame *MainFrame::m_pThis = NULL;

MainFrame::MainFrame(wxWindow* parent)
    : MainFrameBaseClass(parent)
{
    m_pThis             = this; 
    m_MLP               = NULL;
    m_bMLPrunning       = false;
    SetSize(-1, -1);
    Center();
    Bind(wxEVT_COMMAND_MLP_START, &MainFrame::OnMlpStart, this);
    Bind(wxEVT_COMMAND_MLP_UPDATE, &MainFrame::OnMlpUpdate, this);
    Bind(wxEVT_COMMAND_MLP_UPDATE_PG, &MainFrame::OnMlpUpdatePg, this);
    Bind(wxEVT_COMMAND_MLP_COMPLETE, &MainFrame::OnMlpComplete, this);
    
    
    
    m_textCtrl_IterationTimes->SetLabel("1500");
    m_textCtrl_L1neurons->SetLabel("6");
    m_textCtrl_L2neurons->SetLabel("8");
    m_textCtrl_LearnRateInital->SetLabel("0.5");
    m_textCtrl_LearnRateMin->SetLabel("0.01");
    m_textCtrl_LearnRateShift->SetLabel("500");
    m_textCtrl_KFold->SetLabel("10");
    m_textCtrl_MomentumAlpha->SetLabel("0.4");
    m_textCtrl_TestDataRatio->SetLabel("0.5");
    m_textCtrl_TerminalRatio->SetLabel("0.1");
    pathName.Empty();
    m_choice_LearnAdjust->Select(0);
    m_choice_TransferFunc->Select(0);
    
    
}

MainFrame::~MainFrame()
{
    
}
void MainFrame::OnMlpStart(wxThreadEvent& evt)
{
    startTimer();
    showMessage("[MLP]start running..");
    m_bMLPrunning = true;
}
void MainFrame::OnMlpUpdate(wxThreadEvent& evt)
{
    showMessage(evt.GetString());
}
void MainFrame::OnMlpComplete(wxThreadEvent& evt)
{
    //stopTimer("MLP Stop");
    m_MLP = NULL;
    m_bMLPrunning = false;
    showMessage(evt.GetString());
}
void MainFrame::OnMlpUpdatePg(wxThreadEvent& evt)
{
    int iteration = evt.GetInt() + 1 ;
    int total = m_MLP->m_nTotalIteration;
    
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
    
    
    if(m_MLP !=NULL)
        m_MLP->Delete();
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
    m_textCtrl_L2neurons->GetValue().ToDouble(&d_nL2);
    m_textCtrl_LearnRateInital->GetValue().ToDouble(&d_rateIntial);
    m_textCtrl_LearnRateMin->GetValue().ToDouble(&d_rateMin);
    m_textCtrl_IterationTimes->GetValue().ToDouble(&d_iteration);
    m_textCtrl_LearnRateShift->GetValue().ToDouble(&d_rateShift);
    m_textCtrl_KFold->GetValue().ToDouble(&d_nkFold);
    m_textCtrl_MomentumAlpha->GetValue().ToDouble(&d_momentumAlpha);
    m_textCtrl_TestDataRatio->GetValue().ToDouble(&d_testDataRatio);
    m_textCtrl_TerminalRatio->GetValue().ToDouble(&d_terminalRatio);
}
void MainFrame::OnTrainModel(wxCommandEvent& event)
{
    m_MLP = new MLP(this);
    m_MLP->openSampleFile(pathName);
    
    getParameter();
    m_MLP->SetParameter(m_checkBox_DataRescale->IsChecked(),
                        (int)d_nL1, 
                        (int)d_nL2, 
                        d_rateIntial, 
                        d_rateMin, 
                        d_rateShift, 
                        d_iteration, 
                        d_momentumAlpha,
                        (int)d_nkFold,
                        d_testDataRatio,
                        d_terminalRatio,
                        m_choice_LearnAdjust->GetSelection(),
                        m_choice_TransferFunc->GetSelection());
    
    m_MLP->Run();
        
}
void MainFrame::OnUpdateUI(wxUpdateUIEvent& event)
{
    
    event.Enable(!pathName.empty());

        
}
void MainFrame::OnUpdateParameterUI(wxUpdateUIEvent& event)
{
    
    event.Enable(!m_bMLPrunning);
}

