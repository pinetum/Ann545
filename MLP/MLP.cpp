#include "MLP.h"
#include <stdio.h>
#include <wx/textfile.h>
#include <wx/tokenzr.h>
#include <wx/log.h> 
#include <vector>
#include <wx/progdlg.h>
#include "MainFrame.h"


MLP::MLP()
{
    str_errorMsg = "no error";
    m_pthread = NULL;
}

MLP::~MLP()
{
//    if(m_pthread != NULL)
//        if(m_pthread->joinable())
//            m_pthread->join();
}
void MLP::threadEntry()
{
    //wxProgressDialog* m_p_pgDlg = new wxProgressDialog("MLP Training", "training", 100, NULL);
    
    int n_k_totalIteration = 20;
    
    
    cv::Mat weight_1(1, 3 , CV_64FC1);
    cv::randu(weight_1, 0, 100);
    
    for(int i_iteration = 0; i_iteration < n_k_totalIteration; i_iteration++)
    {
        
        MainFrame::showMessage(wxString::Format("%d/%d",i_iteration, n_k_totalIteration));
        //MainFrame::m_staticText->SetLabel(wxString::Format("%d/%d",i_iteration, n_k_totalIteration));
        std::this_thread::sleep_for(std::chrono::seconds(1));
        //m_p_pgDlg->Update((i_iteration+1)/n_k_totalIteration*100);
    }
    //m_p_pgDlg->Destroy();
    
    return ;
}
wxString MLP::getErrorMessage()
{
    return str_errorMsg;
}
bool MLP::train()
{
    bool b_ret = true;
    if(m_data_Train.cols < 1 || m_data_Train.rows < 1)
    {
        str_errorMsg = wxString::Format("data matrix is empty....%d,%d", m_data_Train.cols, m_data_Train.rows);
        return false;
    }
    else
    {
        threadEntry();
        str_errorMsg = "trined";
    }
    
//    cv::Mat m = cv::Mat::ones(4, 3, CV_64F);    // 3 cols, 4 rows
//      cv::Mat row = cv::Mat::ones(1, 3, CV_64F);  // 3 cols
//  m.push_back(row);                       
    
    return b_ret;
}
void MLP::dataScale()
{
    
}
bool MLP::writeModule()
{
    
}
bool MLP::readModule()
{
    
}
double MLP::getAccuracy()
{
    double d_acc = 0;
    return d_acc;
}
void MLP::readMat(wxString inputName, cv::Mat* data)
{
    wxString    str_buffer = "";
    wxTextFile  tfile;
    tfile.Open(inputName);
    str_buffer = tfile.GetFirstLine();
    readDataLine(data, str_buffer);
    while(!tfile.Eof())
    {
        str_buffer = tfile.GetNextLine();
        readDataLine(data, str_buffer);
    }
}
void MLP::writeMat(wxString outputName, cv::Mat* data)
{
    wxTextFile  tfile;
    tfile.Open(outputName);
    for(int j = 0; j < data->rows; j++)
    {
        wxString str_line = "";
        for(int i = 0; i < data->cols; i++)
        {
            if(i != 0) //結尾逗號問題
                str_line.append(",");
            str_line.append(wxString::Format("%f",data->at<double>(cv::Point(i, j))));
        }
        tfile.AddLine(str_line);
    }
    tfile.Write();
}
void MLP::openSampleFile(wxString fileName)
{
    readMat(fileName, &m_data_Train);
}
/**
 * @brief push back element to data
 * @param data mat(target)
 * @param line (string)
 */
void MLP::readDataLine(cv::Mat* data, wxString line)
{
    if(!line.Contains(","))
        return;
    std::vector<double> ary_line;
    wxStringTokenizer tokenizer(line, ",");
    while (tokenizer.HasMoreTokens())
    {
        wxString    token = tokenizer.GetNextToken();
        double      v;
        if(token.ToDouble(&v))
        {
            ary_line.push_back(v);
        }
        else
        {
            str_errorMsg = "readDataLine error : wxString to double...";
        }
       // wxLogMessage(wxString::Format("%f\n",v));
    }
    cv::Mat row(1, ary_line.size(), CV_64F, &ary_line.front());
    if(data->rows < 1)
        *data = row.clone();
    else        
        data->push_back(row);
    //wxLogMessage(wxString::Format("%d,%d\n", data->cols, data->rows));
}