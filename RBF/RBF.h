#ifndef RBF_H
#define RBF_H
#include <opencv2/opencv.hpp>
#include <wx/string.h>
#include <wx/thread.h>
#include <wx/event.h> 
#include <wx/log.h>
#include <vector>
#include <omp.h>
#include <wx/tokenzr.h>
#include <wx/textfile.h>
#include <wx/progdlg.h>

#define RBF_ACTIVATION_BINARY       0
#define RBF_ACTIVATION_BIPOLOR      1

#define RBF_LEARNING_ADJ_STC        0
#define RBF_LEARNING_ADJ_EXPDEC     1
#define RBF_LEARNING_ADJ_BINSIG     2




wxDECLARE_EVENT(wxEVT_COMMAND_RBF_START,        wxThreadEvent);
wxDECLARE_EVENT(wxEVT_COMMAND_RBF_UPDATE,       wxThreadEvent);
wxDECLARE_EVENT(wxEVT_COMMAND_RBF_UPDATE_PG,       wxThreadEvent);
wxDECLARE_EVENT(wxEVT_COMMAND_RBF_COMPLETE,     wxThreadEvent);




class RBF : public wxThread
{
public:
    RBF(wxEvtHandler* pParent);
    ~RBF();
    cv::Mat     m_data_input;               // data input
    cv::Mat     m_data_scaled2train;        // scaled input training data
    cv::Mat     m_data_scaled2test;          // scaled input testing data
    cv::Mat     m_weight;                   //  weight
    cv::Mat     m_center;
    cv::Mat     m_sigma;

    
    
    
    int         m_LearnRateAdjMethod;
    int         m_ActivationType;
    int         m_nNeurons;
    int         m_nInputs;                  // # of network input vector dimension
    int         m_nClasses;                 // # of network output vector dimension
    int         m_nTotalIteration;          // epoch
    int         m_nKfold;                   // cross validation (k-fold)
    int         m_nLearningRateShift;       // epoch times: learning rate shift to minimum learning rate
    int         m_nTerminalThreshold;       //  m_nTerminalThreshold = m_dTerminalratio * m_nTotalIteration
    bool        m_bRescale;                 // data rescale?
    double      m_dMomentumAlpha;           // momentum Alpha
    double      m_dDesiredOutput_rescale;   // epsilon
    double      m_dInitalLearningRate;      // inital learning rate
    double      m_dMinLearningRate;         // minimum learning rate (only binary Sigmoid method use)
    double      m_dTerminalratio;           // m_nTerminalThreshold = m_dTerminalratio * m_nTotalIteration
    double      m_dRatioTestingDatas;       // ratio of datas use in Testing phase
    
    void SetParameter(  bool b_dataRescale,
                        int n_nuron,
                        double d_InitalLearningRate, 
                        double d_MinLearningRate,
                        int n_LearningRateShift,
                        int n_TotalIteration,
                        int n_kFold,
                        double d_testDataRatio,
                        double d_Terminalratio,
                        int LearnRateAdjMethod);
    
    // read csv file to cv::Mat data type contain all data with out any
    void openSampleFile(wxString fileName){readMat(fileName, &m_data_input);}
    
    // scale input data
    void dataScale();
    
    // training phase
    virtual ExitCode Entry();
    
    // test phase
    double getAccuracy();
    
    
    
    // activation functions
    void tanh(cv::Mat* x, cv::Mat* x_der = NULL, double slope = 0.5)
    {
        cv::Mat temp;
        cv::exp(*x*2*slope, temp);
        *x = (temp-cv::Scalar(1) )/(temp+cv::Scalar(1));
        if(x_der != NULL)
            *x_der = slope *( (cv::Scalar(1)+*x).mul(cv::Scalar(1)-*x));
    }
    void binSigmoid(cv::Mat* x, cv::Mat* x_der = NULL, double slope = 0.5)
    {
        cv::exp(*x*slope*-1, *x);
        *x = 1.0/(cv::Scalar(1)+*x);
        if(x_der != NULL)
            *x_der = (  cv::Scalar(1) - *x ).mul(*x)* slope;
    }    
    void transfer(cv::Mat* x, cv::Mat* x_der = NULL, double slope = 0.5)
    {
        switch(m_ActivationType)
        {
            case RBF_ACTIVATION_BINARY:
            {
                cv::exp(*x*slope*-1, *x);
                *x = 1.0/(cv::Scalar(1)+*x);
                if(x_der != NULL)
                    *x_der = (  cv::Scalar(1) - *x ).mul(*x)* slope;
                break;
            }
            case RBF_ACTIVATION_BIPOLOR:
            {
                cv::Mat temp;
                cv::exp(*x*2*slope, temp);
                *x = (temp-cv::Scalar(1) )/(temp+cv::Scalar(1));
                if(x_der != NULL)
                    *x_der = slope *( (cv::Scalar(1)+*x).mul(cv::Scalar(1)-*x));
                break;
            }
            default:
            {
                cv::exp(*x*slope*-1, *x);
                *x = 1.0/(cv::Scalar(1)+*x);
                if(x_der != NULL)
                    *x_der = (  cv::Scalar(1) - *x ).mul(*x)* slope;
                break;
            }
                
        }
    }
    // learning rate adjust...
    double getLearningRate(int i_iteration, double slope = 0.5)
    {
        switch(m_LearnRateAdjMethod)
        {
            case RBF_LEARNING_ADJ_EXPDEC:
                return m_dInitalLearningRate*exp(-1*i_iteration*slope/m_nTotalIteration);
            case RBF_LEARNING_ADJ_BINSIG:
                return (m_dInitalLearningRate - m_dMinLearningRate) / (1+exp(slope*(i_iteration - m_nLearningRateShift)) ) + m_dMinLearningRate;
            case RBF_LEARNING_ADJ_STC:
                return m_dInitalLearningRate/ (1+i_iteration/m_nLearningRateShift);
            default:
                return m_dInitalLearningRate/ ((1+i_iteration)/m_nLearningRateShift);
        }
    }
    
    
    
    
    
    
    
    
    
    void mNorm2(cv::Mat& y, cv::Mat& x, cv::Mat& c)
    {
        y.create(1, m_nNeurons, CV_64F);
                    // # of neurons
        for(int i =0; i < m_nNeurons; i++)
        {
            y.at<double>(0, i) = cv::norm(x, c.row(i), cv::NORM_L2);
        }
    }    
    
    
    
    
    
    void shuffelRow(cv::Mat* x)
    {
        std::vector<int > s;
        #pragma omp parallel for
        for(int i = 0; i< x->rows; i++)
            s.push_back(i);
        cv::randShuffle(s);

        cv::Mat o(x->rows, x->cols, x->type());
        #pragma omp parallel for
        for (int i = 0; i <x->rows; i++)
            o.row(i) = x->row(s[i]) + 0 ;
        *x = o;
    }
    
private:
    wxEvtHandler* m_pHandler;
    void readDataLine(cv::Mat* data, wxString line)
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
    }
    void readMat(wxString inputName, cv::Mat* data)
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
        tfile.Close();
    }
    void writeMat(wxString outputName, cv::Mat* data, wxString info = "")
    {
        wxTextFile  tfile;
        tfile.Create(outputName);
        
        if(info != "")
            tfile.AddLine(info);
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
        tfile.Close();
    }

    

};
#endif // RBF_H
