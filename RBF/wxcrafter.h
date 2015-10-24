//////////////////////////////////////////////////////////////////////
// This file was auto-generated by codelite's wxCrafter Plugin
// wxCrafter project file: wxcrafter.wxcp
// Do not modify this file by hand!
//////////////////////////////////////////////////////////////////////

#ifndef S1010329_RBF_WXCRAFTER_BASE_CLASSES_H
#define S1010329_RBF_WXCRAFTER_BASE_CLASSES_H

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
#include <wx/textctrl.h>
#include <wx/checkbox.h>
#include <wx/choice.h>
#include <wx/arrstr.h>
#include <wx/richtext/richtextctrl.h>
#include <wx/gauge.h>
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
    wxButton* m_buttonLoadData;
    wxButton* m_buttonTrain;
    wxStaticText* m_staticText;
    wxStaticText* m_staticText51;
    wxTextCtrl* m_textCtrl_L1neurons;
    wxStaticText* m_staticText_LearnRateInital;
    wxTextCtrl* m_textCtrl_LearnRateInital;
    wxStaticText* m_staticText61;
    wxTextCtrl* m_textCtrl_LearnRateShift;
    wxStaticText* m_staticText91;
    wxTextCtrl* m_textCtrl_KFold;
    wxStaticText* m_staticText_LearnRateMin;
    wxTextCtrl* m_textCtrl_LearnRateMin;
    wxStaticText* m_staticText77;
    wxTextCtrl* m_textCtrl_IterationTimes;
    wxStaticText* m_staticText129;
    wxTextCtrl* m_textCtrl_TestDataRatio;
    wxStaticText* m_staticText141;
    wxTextCtrl* m_textCtrl_TerminalRatio;
    wxCheckBox* m_checkBox_DataRescale;
    wxStaticText* m_staticText111;
    wxChoice* m_choice_LearnAdjust;
    wxRichTextCtrl* m_richTextCtrl;
    wxStaticText* m_staticTextPg;
    wxGauge* m_gaugePg;
    wxStaticText* m_staticTextTimer;
    wxMenuBar* m_menuBar;
    wxMenu* m_name6;
    wxMenuItem* m_menuItem7;
    wxMenu* m_name8;
    wxMenuItem* m_menuItem9;
    wxToolBar* m_mainToolbar;

protected:
    virtual void OnLoadData(wxCommandEvent& event) { event.Skip(); }
    virtual void OnTrainModel(wxCommandEvent& event) { event.Skip(); }
    virtual void OnUpdateUI(wxUpdateUIEvent& event) { event.Skip(); }
    virtual void OnUpdateParameterUI(wxUpdateUIEvent& event) { event.Skip(); }
    virtual void OnExit(wxCommandEvent& event) { event.Skip(); }
    virtual void OnAbout(wxCommandEvent& event) { event.Skip(); }

public:
    wxButton* GetButtonLoadData() { return m_buttonLoadData; }
    wxButton* GetButtonTrain() { return m_buttonTrain; }
    wxStaticText* GetStaticText() { return m_staticText; }
    wxStaticText* GetStaticText51() { return m_staticText51; }
    wxTextCtrl* GetTextCtrl_L1neurons() { return m_textCtrl_L1neurons; }
    wxStaticText* GetStaticText_LearnRateInital() { return m_staticText_LearnRateInital; }
    wxTextCtrl* GetTextCtrl_LearnRateInital() { return m_textCtrl_LearnRateInital; }
    wxStaticText* GetStaticText61() { return m_staticText61; }
    wxTextCtrl* GetTextCtrl_LearnRateShift() { return m_textCtrl_LearnRateShift; }
    wxStaticText* GetStaticText91() { return m_staticText91; }
    wxTextCtrl* GetTextCtrl_KFold() { return m_textCtrl_KFold; }
    wxStaticText* GetStaticText_LearnRateMin() { return m_staticText_LearnRateMin; }
    wxTextCtrl* GetTextCtrl_LearnRateMin() { return m_textCtrl_LearnRateMin; }
    wxStaticText* GetStaticText77() { return m_staticText77; }
    wxTextCtrl* GetTextCtrl_IterationTimes() { return m_textCtrl_IterationTimes; }
    wxStaticText* GetStaticText129() { return m_staticText129; }
    wxTextCtrl* GetTextCtrl_TestDataRatio() { return m_textCtrl_TestDataRatio; }
    wxStaticText* GetStaticText141() { return m_staticText141; }
    wxTextCtrl* GetTextCtrl_TerminalRatio() { return m_textCtrl_TerminalRatio; }
    wxCheckBox* GetCheckBox_DataRescale() { return m_checkBox_DataRescale; }
    wxStaticText* GetStaticText111() { return m_staticText111; }
    wxChoice* GetChoice_LearnAdjust() { return m_choice_LearnAdjust; }
    wxRichTextCtrl* GetRichTextCtrl() { return m_richTextCtrl; }
    wxStaticText* GetStaticTextPg() { return m_staticTextPg; }
    wxGauge* GetGaugePg() { return m_gaugePg; }
    wxStaticText* GetStaticTextTimer() { return m_staticTextTimer; }
    wxPanel* GetMainPanel() { return m_mainPanel; }
    wxMenuBar* GetMenuBar() { return m_menuBar; }
    wxToolBar* GetMainToolbar() { return m_mainToolbar; }
    MainFrameBaseClass(wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("S1010329-RBF"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize(500,300), long style = wxCAPTION|wxRESIZE_BORDER|wxMAXIMIZE_BOX|wxMINIMIZE_BOX|wxSYSTEM_MENU|wxCLOSE_BOX);
    virtual ~MainFrameBaseClass();
};

#endif
