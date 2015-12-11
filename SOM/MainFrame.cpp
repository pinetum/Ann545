#include "MainFrame.h"
#include <wx/aboutdlg.h>
#include <iostream>
#include <vector>
#include <wx/log.h>
wxDEFINE_EVENT(wxEVT_COMMAND_SOM_START,        wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_SOM_UPDATE,       wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_SOM_COMPLETE,     wxThreadEvent);



MainFrame::MainFrame(wxWindow* parent)
    : MainFrameBaseClass(parent)
{
    Bind(wxEVT_COMMAND_SOM_START, &MainFrame::OnSOMStart, this);
    Bind(wxEVT_COMMAND_SOM_UPDATE, &MainFrame::OnSOMUpdate, this);
    Bind(wxEVT_COMMAND_SOM_COMPLETE, &MainFrame::OnSOMComplete, this);
    b_threadRunning = true;
    p_Tsom = NULL;
}

MainFrame::~MainFrame()
{
}
void MainFrame::drawPts(cv::Mat &ptsData, cv::Mat &img)
{
    for(int i =0; i < ptsData.rows; i ++)
    {
        cv::Vec2d ptV = ptsData.at<cv::Vec2d>(i, 0);
        cv::Point pt(ptV[0], img.rows - ptV[1]);
        cv::circle(img, pt, 3, cv::Scalar(255), 1);
    }
}
void MainFrame::drawResults(cv::Mat& weights, cv::Mat &img, wxString msg)
{
    //wxLogMessage("weight size %d,%d", weights.rows, weights.cols);
    for(int r =0; r< weights.rows; r++)
    {
        for(int c = 0; c< weights.cols; c++)
        {
            cv::Point pt1, pt2, pt3;
            pt1.x=(int)weights.at<cv::Vec2d>(r, c)[0];
            pt1.y=(int)img.rows - weights.at<cv::Vec2d>(r, c)[1];
            
            if(r!=weights.rows-1)
            {
                pt2.x=(int)weights.at<cv::Vec2d>(r+1, c)[0];
                pt2.y=(int)img.rows - weights.at<cv::Vec2d>(r+1, c)[1];
                cv::line(img, pt1, pt2, cv::Scalar(255));
                
            }
            if(c!=weights.cols-1)
            {
                pt3.x=(int)weights.at<cv::Vec2d>(r, c+1)[0];
                pt3.y=(int)img.rows - weights.at<cv::Vec2d>(r, c+1)[1];
                cv::line(img, pt1, pt3, cv::Scalar(255));
            }
                
            cv::circle(img, pt1, 3, cv::Scalar(255), -1);
            //wxLogMessage("%d,%d", pt1.x, pt1.y);
            //wxLogMessage("r=%d,c=%d:pt1(%d,%d) pt2(%d,%d) pt3(%d,%d)", r, c, pt1.x, pt1.y, pt2.x, pt2.y, pt3.x, pt3.y);
//            char buff[100];
//            snprintf(buff, sizeof(buff), "%d, %d", r, c);
//            cv::putText(img,  buff, pt1, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
//            
        }
    }
    cv::putText(img, std::string(msg.mb_str()),cv::Point(20,20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255));
    
    
}
void MainFrame::OnExit(wxCommandEvent& event)
{
    if(p_Tsom!=NULL)
        p_Tsom->Delete();
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
void MainFrame::OnBtnSquare(wxCommandEvent& event)
{
    
    wxString str_iteration = m_textCtrl_IterationTimes->GetValue();
    wxString str_weightSize = m_textCtrl_weights->GetValue();
    double iterationTimes, weightSize;
    str_iteration.ToDouble(&iterationTimes);
    str_weightSize.ToDouble(&weightSize);
    cv::Mat input = readData(SQUARE_DATA);
    //input = input*CONTENT_SIZE;
    p_Tsom = new SOM(this, (int)weightSize, (int)iterationTimes, input);
    p_Tsom->Run();
    
}
void MainFrame::OnBtnTriangle(wxCommandEvent& event)
{
    wxString str_iteration = m_textCtrl_IterationTimes->GetValue();
    wxString str_weightSize = m_textCtrl_weights->GetValue();
    double iterationTimes, weightSize;
    str_iteration.ToDouble(&iterationTimes);
    str_weightSize.ToDouble(&weightSize);
    cv::Mat input = readData(TRIANGLE_DATA);
    //input = input*CONTENT_SIZE;
    //cv::Mat img2;
    //img2 = cv::Mat::zeros(CONTENT_SIZE, CONTENT_SIZE, CV_8UC1);
    //drawPts(input, img2);
    //imshow("input", img2);
    p_Tsom = new SOM(this, (int)weightSize, (int)iterationTimes, input);
    p_Tsom->Run();
}
void MainFrame::OnUpdateSquare(wxUpdateUIEvent& event)
{
    event.Enable(b_threadRunning);
}
void MainFrame::OnUpdateTriangle(wxUpdateUIEvent& event)
{
    event.Enable(b_threadRunning);
}
void MainFrame::OnSOMStart(wxThreadEvent& evt)
{
    b_threadRunning = false;
    
}
void MainFrame::OnSOMUpdate(wxThreadEvent& evt)
{
    cv::Mat weight = evt.GetPayload<cv::Mat>();
    int iteration = evt.GetInt();
    //SetTitle(wxString::Format("%d", iteration));
    img = cv::Mat::zeros(CONTENT_SIZE, CONTENT_SIZE, CV_8UC1);
    weight = weight * CONTENT_SIZE;
    drawResults(weight, img, evt.GetString());
    
    
    cv::imwrite(std::string(wxString::Format("/Users/QT/Downloads/SOM_IMAGE/%d.png", iteration).mb_str()), img);
    cv::imshow("result", img);
}
void MainFrame::OnSOMComplete(wxThreadEvent& evt)
{
    b_threadRunning = true;
    p_Tsom = NULL;
}
cv::Mat MainFrame::readData(int dataType)
{
    std::vector<double> data;
    FILE* fp = NULL;
    if(dataType == SQUARE_DATA)
        fp = fopen("/Users/QT/Dropbox/元智課程資料/研究所課程/類神經網路/hw4/HW_Kohonen_Data/_square.txt", "r");
    else if(dataType== TRIANGLE_DATA)
        fp = fopen("/Users/QT/Dropbox/元智課程資料/研究所課程/類神經網路/hw4/HW_Kohonen_Data/_triangle.txt", "r");
    if(fp==NULL)
    {
        cv::Mat empty;
        return empty;
    }
    char buffer[200];
    
    while(!feof(fp))
    {
        double x,y;
        fgets(buffer, 200, fp);
        sscanf(buffer, "%lf%lf", &x, &y);
        data.push_back(x);
        data.push_back(y);
        //printf("%f,%f\n", x, y);

    }
    fclose(fp);
    // read file end
    cv::Mat inputData(data.size()/2, 1, CV_64FC2, &data.front());
    return inputData.clone();
}
void MainFrame::OnBtnStop(wxCommandEvent& event)
{
    if(p_Tsom!=NULL)
        p_Tsom->Delete();
    
}
