#include "MLP.h"
#include <wx/tokenzr.h>
#include <wx/log.h>
#include <wx/textfile.h>
#include <vector>




MLP::MLP(wxEvtHandler* pParent)
{
    m_pHandler   = pParent;
    m_nNeuronsL1 = 5;
    m_nNeuronsL2 = 5;
    m_nInputs    = 9;
    m_nClasses   = 2;
    m_dInitalLearningRate = 0.3;
    m_dMinLearningRate = 0.05;
    m_nLearningRateShift = 4000;
    m_nTotalIteration = 3000;
    m_bMomentum  = false;
    m_dDesiredOutput_rescale = 0.1;
}

MLP::~MLP()
{

}
wxThread::ExitCode MLP::Entry(){
    wxThreadEvent* evt_end;
    wxThreadEvent* evt_update;
    wxThreadEvent* evt_start;
    evt_start = new wxThreadEvent(wxEVT_COMMAND_MLP_START);
    wxQueueEvent(m_pHandler, evt_start);
    if(!m_data_input.data)
    {
        
        evt_end = new wxThreadEvent(wxEVT_COMMAND_MLP_COMPLETE);
        evt_end->SetString("[MLP]Error: input Matrix data empty");
        wxQueueEvent(m_pHandler, evt_end);
        return (wxThread::ExitCode)-1;
    }
    
    // data scaling (perpare m_data_scaled2train)
    dataScale();
    // inital weight...
    m_weight_l1.create(m_nInputs,       m_nNeuronsL1,   CV_64FC1);
    m_weight_l2.create(m_nNeuronsL1,    m_nNeuronsL2,   CV_64FC1);
    m_weight_l3.create(m_nNeuronsL2,    m_nClasses,     CV_64FC1);
    cv::randu(m_weight_l1, cv::Scalar(0), cv::Scalar(0.3));
    cv::randu(m_weight_l2, cv::Scalar(0), cv::Scalar(0.3));
    cv::randu(m_weight_l3, cv::Scalar(0), cv::Scalar(0.3));
    
    writeMat("./inital_W1.txt", &m_weight_l1);
    writeMat("./inital_W2.txt", &m_weight_l2);
    writeMat("./inital_W3.txt", &m_weight_l3);
    
    for(int i_iteration = 0; i_iteration < m_nTotalIteration; i_iteration++)
    {
        
        for(int i_dataRows = 0; i_dataRows < m_data_scaled2train.rows; i_dataRows++)
        {
            cv::Mat input = m_data_scaled2train(cv::Range(i_dataRows, i_dataRows+1), cv::Range(0, m_nInputs));
            
            cv::Mat response = input*m_weight_l1*m_weight_l2*m_weight_l3;
            //evt_update = new wxThreadEvent(wxEVT_COMMAND_MLP_UPDATE);
            //evt_update->SetString(wxString::Format("%.2f, %.2f", response.at<double>(0,0), response.at<double>(0,1)));
            //wxQueueEvent(m_pHandler, evt_update);
        }
        evt_update = new wxThreadEvent(wxEVT_COMMAND_MLP_UPDATE_PG);
        evt_update->SetInt(i_iteration);
        wxQueueEvent(m_pHandler, evt_update);
    }
    
    
    

    
    
    
    
    evt_end = new wxThreadEvent(wxEVT_COMMAND_MLP_COMPLETE);
    evt_end->SetString("[MLP]Complete!");
    wxQueueEvent(m_pHandler, evt_end);
    return (wxThread::ExitCode)0;
}



bool MLP::train()
{
    bool b_ret = true;

    return b_ret;
}
void MLP::dataScale()
{
    m_data_scaled2train.create(m_data_input.rows, m_nInputs + m_nClasses, CV_64FC1);
    for(int i =0; i < m_nInputs; i++)
    {
        // solution 1
        cv::normalize(m_data_input.col(i), m_data_scaled2train.col(i), 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
        // solution 2 (save min and max)
        //double min, max;
        //cv::minMaxLoc(m_data_input.col(i), &min, &max);
    }
                   ///////-----desired oupput-------///////
    
    // begin
    m_data_scaled2train.col(m_nInputs) = (m_data_input.col(m_nInputs)-cv::Scalar(4))/-2; // 2>>1, 4>>0
    cv::normalize(m_data_scaled2train.col(m_nInputs), 
                    m_data_scaled2train.col(m_nInputs), 
                    m_dDesiredOutput_rescale, 1 - m_dDesiredOutput_rescale,  // min, max
                    cv::NORM_MINMAX, -1, cv::Mat() );
    // malignant
    m_data_scaled2train.col(m_nInputs+1) = m_data_input.col(m_nInputs)*0.5-cv::Scalar(1); // if 4>>1, 2>>0
    cv::normalize(m_data_scaled2train.col(m_nInputs+1), 
                    m_data_scaled2train.col(m_nInputs+1), 
                    m_dDesiredOutput_rescale, 1 - m_dDesiredOutput_rescale,  // min, max
                    cv::NORM_MINMAX, -1, cv::Mat() );
    writeMat("./rescaled_output.txt", &m_data_scaled2train);
    
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
    tfile.Create(outputName);
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
void MLP::openSampleFile(wxString fileName){readMat(fileName, &m_data_input);}
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
            wxLogMessage(wxString::Format("readDataLine error : wxString to double..."));
        }
        
    }
    cv::Mat row(1, ary_line.size(), CV_64F, &ary_line.front());
    if(data->rows < 1)
        *data = row.clone();
    else        
        data->push_back(row);
    //wxLogMessage(wxString::Format("%d,%d\n", data->cols, data->rows));
}






