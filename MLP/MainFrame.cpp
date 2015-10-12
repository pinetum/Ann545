#include "MainFrame.h"
#include <wx/aboutdlg.h>
#include <wx/filedlg.h>


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
    SetSize(600, 400);
    Center();
    Bind(wxEVT_COMMAND_MLP_START, &MainFrame::OnMlpStart, this);
    Bind(wxEVT_COMMAND_MLP_UPDATE, &MainFrame::OnMlpUpdate, this);
    Bind(wxEVT_COMMAND_MLP_UPDATE_PG, &MainFrame::OnMlpUpdatePg, this);
    Bind(wxEVT_COMMAND_MLP_COMPLETE, &MainFrame::OnMlpComplete, this);
    
    
    m_textCtrl_IterationTimes->SetLabel("5000");
    m_textCtrl_L1neurons->SetLabel("6");
    m_textCtrl_L2neurons->SetLabel("8");
    m_textCtrl_LearnRateInital->SetLabel("0.3");
    m_textCtrl_LearnRateMin->SetLabel("0.05");
    m_textCtrl_LearnRateShift->SetLabel("4500");
    
}

MainFrame::~MainFrame()
{
    if(m_MLP !=NULL)
        delete m_MLP;
}
void MainFrame::OnMlpStart(wxThreadEvent& evt)
{
    startTimer();
    showMessage("MLP start Running..");
    m_bMLPrunning = true;
}
void MainFrame::OnMlpUpdate(wxThreadEvent& evt)
{
    showMessage(evt.GetString());
}
void MainFrame::OnMlpComplete(wxThreadEvent& evt)
{
    stopTimer("MLP Stop");
    m_MLP = NULL;
    m_bMLPrunning = false;
}
void MainFrame::OnMlpUpdatePg(wxThreadEvent& evt)
{
    int iteration = evt.GetInt() + 1 ;
    int total = m_MLP->m_nTotalIteration;
    m_gaugePg->SetValue(100*iteration/total);
    m_staticTextPg->SetLabel(wxString::Format("%d/%d", iteration, total));
    
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
    
    if(hours > 0)
        str_time.Append(wxString::Format("%dh ",hours));
    if(minutes > 0)
        str_time.Append(wxString::Format("%dm ",minutes));
    if(seconds > 0)
        str_time.Append(wxString::Format("%ds ",seconds));
    if(milliseconds > 0)
        str_time.Append(wxString::Format("%dms ",milliseconds));
    return str_time;
    
    
}
void MainFrame::OnExit(wxCommandEvent& event)
{
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
    wxString pathName = "";
    wxString fileType = _("All suported formats(*.*)|*.*");
    wxFileDialog* openDialog = new wxFileDialog(this,_("openFile"),wxEmptyString,wxEmptyString,fileType,wxFD_OPEN,wxDefaultPosition);
	if(openDialog->ShowModal() == wxID_OK){
        pathName = openDialog->GetPath();
    }
    openDialog->Destroy();
    if(pathName.length() == 0)
        MainFrame::showMessage("[LoadData]File error");
    else
    {
        m_MLP = new MLP(this);
        m_MLP->openSampleFile(pathName);
    }
}
void MainFrame::OnTrainModel(wxCommandEvent& event)
{
    if(!m_MLP->IsRunning())
    {
        double d_nL1, d_nL2, d_rateIntial, d_rateMin, d_rateShift, d_iteration;
        m_textCtrl_L1neurons->GetValue().ToDouble(&d_nL1);
        m_textCtrl_L2neurons->GetValue().ToDouble(&d_nL2);
        m_textCtrl_LearnRateInital->GetValue().ToDouble(&d_rateIntial);
        m_textCtrl_LearnRateMin->GetValue().ToDouble(&d_rateMin);
        m_textCtrl_IterationTimes->GetValue().ToDouble(&d_iteration);
        m_textCtrl_LearnRateShift->GetValue().ToDouble(&d_rateShift);
        
        m_MLP->SetParameter((int)d_nL1, 
                            (int)d_nL2, 
                            d_rateIntial, 
                            d_rateMin, 
                            d_rateShift, 
                            d_iteration, 
                            m_checkBox_Momentum->GetValue());
        m_MLP->Run();
    }
        
}
void MainFrame::OnValidate(wxCommandEvent& event)
{
    wxString pathName = "";
    wxString fileType = _("All suported formats(*.*)|*.*");
    wxFileDialog* openDialog = new wxFileDialog(this,_("openFile"),wxEmptyString,wxEmptyString,fileType,wxFD_OPEN,wxDefaultPosition);
	if(openDialog->ShowModal() == wxID_OK){
        pathName = openDialog->GetPath();
    }
    openDialog->Destroy();
    if(pathName.length() == 0)
        MainFrame::showMessage("[Validate]File error");
    else
    {
        startTimer();
        stopTimer("MLP Validating");
    }
}
void MainFrame::OnUpdateUI(wxUpdateUIEvent& event)
{
    if(m_MLP == NULL)
        event.Enable(false);
    else
    {
        event.Enable(true);
//        if(m_MLP->IsRunning())
//            event.Enable(false);
//        else
//            event.Enable(true);
            
    }
        
}
void MainFrame::OnLoadModel(wxCommandEvent& event)
{
}
void MainFrame::OnUpdateParameterUI(wxUpdateUIEvent& event)
{
    
    event.Enable(!m_bMLPrunning);
}
