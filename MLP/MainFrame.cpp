#include "MainFrame.h"
#include <wx/aboutdlg.h>
#include <wx/filedlg.h>

MainFrame *MainFrame::m_pThis = NULL;

MainFrame::MainFrame(wxWindow* parent)
    : MainFrameBaseClass(parent)
{
    m_pThis             = this; 
    m_MLP               = NULL;
    SetSize(500, 300);
    Center();
}

MainFrame::~MainFrame()
{
    if(m_MLP !=NULL)
        delete m_MLP;
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
        m_MLP = new MLP();
        m_MLP->openSampleFile(pathName);
    }
}
void MainFrame::OnTrainModel(wxCommandEvent& event)
{
    startTimer();
    if(!m_MLP->train())
        showMessage(m_MLP->getErrorMessage());
    stopTimer("MLP Training");
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
        event.Enable(true);
}
void MainFrame::OnLoadModel(wxCommandEvent& event)
{
}
