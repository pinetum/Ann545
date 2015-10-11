//////////////////////////////////////////////////////////////////////
// This file was auto-generated by codelite's wxCrafter Plugin
// wxCrafter project file: wxcrafter.wxcp
// Do not modify this file by hand!
//////////////////////////////////////////////////////////////////////

#ifndef S1010329_MLP_WXCRAFTER_BASE_CLASSES_H
#define S1010329_MLP_WXCRAFTER_BASE_CLASSES_H

#include <wx/settings.h>
#include <wx/xrc/xmlres.h>
#include <wx/xrc/xh_bmp.h>
#include <wx/frame.h>
#include <wx/iconbndl.h>
#include <wx/artprov.h>
#include <wx/sizer.h>
#include <wx/panel.h>
#include <wx/button.h>
#include <wx/stattext.h>
#include <wx/richtext/richtextctrl.h>
#include <wx/menu.h>
#include <wx/toolbar.h>
#if wxVERSION_NUMBER >= 2900
#include <wx/persist.h>
#include <wx/persist/toplevel.h>
#include <wx/persist/bookctrl.h>
#include <wx/persist/treebook.h>
#endif

class MainFrameBaseClass : public wxFrame
{
protected:
    wxPanel* m_mainPanel;
    wxButton* m_buttonLoadModel;
    wxButton* m_buttonLoadData;
    wxButton* m_buttonTrain;
    wxButton* m_buttonValidate;
    wxStaticText* m_staticText;
    wxRichTextCtrl* m_richTextCtrl;
    wxMenuBar* m_menuBar;
    wxMenu* m_name6;
    wxMenuItem* m_menuItem7;
    wxMenu* m_name8;
    wxMenuItem* m_menuItem9;
    wxToolBar* m_mainToolbar;

protected:
    virtual void OnLoadModel(wxCommandEvent& event) { event.Skip(); }
    virtual void OnLoadData(wxCommandEvent& event) { event.Skip(); }
    virtual void OnTrainModel(wxCommandEvent& event) { event.Skip(); }
    virtual void OnUpdateUI(wxUpdateUIEvent& event) { event.Skip(); }
    virtual void OnValidate(wxCommandEvent& event) { event.Skip(); }
    virtual void OnExit(wxCommandEvent& event) { event.Skip(); }
    virtual void OnAbout(wxCommandEvent& event) { event.Skip(); }

public:
    wxButton* GetButtonLoadModel() { return m_buttonLoadModel; }
    wxButton* GetButtonLoadData() { return m_buttonLoadData; }
    wxButton* GetButtonTrain() { return m_buttonTrain; }
    wxButton* GetButtonValidate() { return m_buttonValidate; }
    wxStaticText* GetStaticText() { return m_staticText; }
    wxRichTextCtrl* GetRichTextCtrl() { return m_richTextCtrl; }
    wxPanel* GetMainPanel() { return m_mainPanel; }
    wxMenuBar* GetMenuBar() { return m_menuBar; }
    wxToolBar* GetMainToolbar() { return m_mainToolbar; }
    MainFrameBaseClass(wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("S1010329-MLP"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize(500,300), long style = wxCAPTION|wxRESIZE_BORDER|wxMAXIMIZE_BOX|wxMINIMIZE_BOX|wxSYSTEM_MENU|wxCLOSE_BOX);
    virtual ~MainFrameBaseClass();
};

#endif
