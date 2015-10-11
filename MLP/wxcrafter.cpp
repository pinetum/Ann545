//////////////////////////////////////////////////////////////////////
// This file was auto-generated by codelite's wxCrafter Plugin
// wxCrafter project file: wxcrafter.wxcp
// Do not modify this file by hand!
//////////////////////////////////////////////////////////////////////

#include "wxcrafter.h"


// Declare the bitmap loading function
extern void wxC9ED9InitBitmapResources();

static bool bBitmapLoaded = false;


MainFrameBaseClass::MainFrameBaseClass(wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style)
    : wxFrame(parent, id, title, pos, size, style)
{
    if ( !bBitmapLoaded ) {
        // We need to initialise the default bitmap handler
        wxXmlResource::Get()->AddHandler(new wxBitmapXmlHandler);
        wxC9ED9InitBitmapResources();
        bBitmapLoaded = true;
    }
    
    wxBoxSizer* boxSizer1 = new wxBoxSizer(wxVERTICAL);
    this->SetSizer(boxSizer1);
    
    m_mainPanel = new wxPanel(this, wxID_ANY, wxDefaultPosition, wxSize(-1,-1), wxTAB_TRAVERSAL);
    
    boxSizer1->Add(m_mainPanel, 1, wxEXPAND, 5);
    
    wxBoxSizer* boxSizer11 = new wxBoxSizer(wxVERTICAL);
    m_mainPanel->SetSizer(boxSizer11);
    
    wxBoxSizer* boxSizer35 = new wxBoxSizer(wxVERTICAL);
    
    boxSizer11->Add(boxSizer35, 0, wxALL|wxEXPAND, 5);
    
    wxBoxSizer* boxSizer37 = new wxBoxSizer(wxHORIZONTAL);
    
    boxSizer35->Add(boxSizer37, 1, wxALL|wxEXPAND, 5);
    
    m_buttonLoadModel = new wxButton(m_mainPanel, wxID_ANY, _("Load Model"), wxDefaultPosition, wxSize(-1,-1), 0);
    
    boxSizer37->Add(m_buttonLoadModel, 0, wxALL, 5);
    
    m_buttonLoadData = new wxButton(m_mainPanel, wxID_ANY, _("Load Data"), wxDefaultPosition, wxSize(-1,-1), 0);
    m_buttonLoadData->SetFocus();
    
    boxSizer37->Add(m_buttonLoadData, 0, wxALL, 5);
    
    m_buttonTrain = new wxButton(m_mainPanel, wxID_ANY, _("Train model"), wxDefaultPosition, wxSize(-1,-1), 0);
    
    boxSizer37->Add(m_buttonTrain, 0, wxALL, 5);
    
    m_buttonValidate = new wxButton(m_mainPanel, wxID_ANY, _("Validate"), wxDefaultPosition, wxSize(-1,-1), 0);
    
    boxSizer37->Add(m_buttonValidate, 0, wxALL, 5);
    
    m_staticText = new wxStaticText(m_mainPanel, wxID_ANY, wxT(""), wxDefaultPosition, wxSize(-1,-1), 0);
    
    boxSizer37->Add(m_staticText, 0, wxALL, 5);
    
    m_richTextCtrl = new wxRichTextCtrl(m_mainPanel, wxID_ANY, wxT(""), wxDefaultPosition, wxSize(-1,-1), wxTE_MULTILINE|wxTE_PROCESS_TAB|wxTE_PROCESS_ENTER|wxWANTS_CHARS);
    
    boxSizer11->Add(m_richTextCtrl, 1, wxALL|wxEXPAND, 5);
    
    wxBoxSizer* boxSizer45 = new wxBoxSizer(wxHORIZONTAL);
    
    boxSizer11->Add(boxSizer45, 0, wxALL|wxEXPAND, 5);
    
    m_staticTextPg = new wxStaticText(m_mainPanel, wxID_ANY, _("9999/9999"), wxDefaultPosition, wxSize(-1,-1), 0);
    
    boxSizer45->Add(m_staticTextPg, 0, wxALL, 5);
    
    m_gaugePg = new wxGauge(m_mainPanel, wxID_ANY, 100, wxDefaultPosition, wxSize(-1,-1), wxGA_HORIZONTAL);
    m_gaugePg->SetValue(0);
    
    boxSizer45->Add(m_gaugePg, 1, wxALL|wxEXPAND, 5);
    
    m_menuBar = new wxMenuBar(0);
    this->SetMenuBar(m_menuBar);
    
    m_name6 = new wxMenu();
    m_menuBar->Append(m_name6, _("File"));
    
    m_menuItem7 = new wxMenuItem(m_name6, wxID_EXIT, _("Exit\tAlt-X"), _("Quit"), wxITEM_NORMAL);
    m_name6->Append(m_menuItem7);
    
    m_name8 = new wxMenu();
    m_menuBar->Append(m_name8, _("Help"));
    
    m_menuItem9 = new wxMenuItem(m_name8, wxID_ABOUT, _("About..."), wxT(""), wxITEM_NORMAL);
    m_name8->Append(m_menuItem9);
    
    m_mainToolbar = this->CreateToolBar(wxTB_FLAT, wxID_ANY);
    m_mainToolbar->SetToolBitmapSize(wxSize(16,16));
    
    SetName(wxT("MainFrameBaseClass"));
    SetSizeHints(500,300);
    if ( GetSizer() ) {
         GetSizer()->Fit(this);
    }
    CentreOnParent(wxBOTH);
#if wxVERSION_NUMBER >= 2900
    if(!wxPersistenceManager::Get().Find(this)) {
        wxPersistenceManager::Get().RegisterAndRestore(this);
    } else {
        wxPersistenceManager::Get().Restore(this);
    }
#endif
    // Connect events
    m_buttonLoadModel->Connect(wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler(MainFrameBaseClass::OnLoadModel), NULL, this);
    m_buttonLoadData->Connect(wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler(MainFrameBaseClass::OnLoadData), NULL, this);
    m_buttonTrain->Connect(wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler(MainFrameBaseClass::OnTrainModel), NULL, this);
    m_buttonTrain->Connect(wxEVT_UPDATE_UI, wxUpdateUIEventHandler(MainFrameBaseClass::OnUpdateUI), NULL, this);
    m_buttonValidate->Connect(wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler(MainFrameBaseClass::OnValidate), NULL, this);
    m_buttonValidate->Connect(wxEVT_UPDATE_UI, wxUpdateUIEventHandler(MainFrameBaseClass::OnUpdateUI), NULL, this);
    this->Connect(m_menuItem7->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler(MainFrameBaseClass::OnExit), NULL, this);
    this->Connect(m_menuItem9->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler(MainFrameBaseClass::OnAbout), NULL, this);
    
}

MainFrameBaseClass::~MainFrameBaseClass()
{
    m_buttonLoadModel->Disconnect(wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler(MainFrameBaseClass::OnLoadModel), NULL, this);
    m_buttonLoadData->Disconnect(wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler(MainFrameBaseClass::OnLoadData), NULL, this);
    m_buttonTrain->Disconnect(wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler(MainFrameBaseClass::OnTrainModel), NULL, this);
    m_buttonTrain->Disconnect(wxEVT_UPDATE_UI, wxUpdateUIEventHandler(MainFrameBaseClass::OnUpdateUI), NULL, this);
    m_buttonValidate->Disconnect(wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler(MainFrameBaseClass::OnValidate), NULL, this);
    m_buttonValidate->Disconnect(wxEVT_UPDATE_UI, wxUpdateUIEventHandler(MainFrameBaseClass::OnUpdateUI), NULL, this);
    this->Disconnect(m_menuItem7->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler(MainFrameBaseClass::OnExit), NULL, this);
    this->Disconnect(m_menuItem9->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler(MainFrameBaseClass::OnAbout), NULL, this);
    
}
