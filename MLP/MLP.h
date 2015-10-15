#ifndef MLP_H
#define MLP_H
#include <opencv2/opencv.hpp>
#include <wx/string.h>
#include <wx/thread.h>
#include <wx/event.h> 
#include <wx/log.h>


#define MLP_ACTIVATION_BINARY       0
#define MLP_ACTIVATION_BIPOLOR      1
#define MLP_LEARNING_ADJ_STC        0
#define MLP_LEARNING_ADJ_EXPDEC     1
#define MLP_LEARNING_ADJ_BINSIG     2




wxDECLARE_EVENT(wxEVT_COMMAND_MLP_START,        wxThreadEvent);
wxDECLARE_EVENT(wxEVT_COMMAND_MLP_UPDATE,       wxThreadEvent);
wxDECLARE_EVENT(wxEVT_COMMAND_MLP_UPDATE_PG,       wxThreadEvent);
wxDECLARE_EVENT(wxEVT_COMMAND_MLP_COMPLETE,     wxThreadEvent);



class MLP : public wxThread
{
public:
    MLP(wxEvtHandler* pParent);
    ~MLP();
    
    cv::Mat     m_data_input;               // data input
    cv::Mat     m_data_scaled2train;        // scaled input training data
    cv::Mat     m_data_scaled2test;          // scaled input testing data
    cv::Mat     m_weight_l1;                // layer 1 weight
    cv::Mat     m_weight_l2;                // layer 2 weight
    cv::Mat     m_weight_l3;                // layer 3 weight
    
    int         m_nNeuronsL1;               // number of neurons in hidden layer 1
    int         m_nNeuronsL2;               // number of neurons in hidden layer 2
    int         m_nInputs;                  // number of network input vector dimension
    int         m_nClasses;                 // number of network output vector dimension
    int         m_nTotalIteration;          // epoch
    int         m_nKfold;                   // cross validation (k-fold)
    int         m_nLearningRateShift;       // epoch times: learning rate shift to minimum learning rate
    
    int         m_LearnRateAdjMethod;
    int         m_ActivationType;
    
    bool        m_bMomentum;                // update weight with momentum ?
    double      m_dMomentumAlpha;           // momentum Alpha
    double      m_dDesiredOutput_rescale;
    double      m_dInitalLearningRate;      // inital learning rate
    double      m_dMinLearningRate;         // minimum learning rate (with )

    double      m_dRatioTestingDatas;       // ratio of datas use in Testing phase
    
    void SetParameter(  int n_nuronL1, 
                        int n_nuronL2, 
                        double d_InitalLearningRate, 
                        double d_MinLearningRate,
                        int n_LearningRateShift,
                        int n_TotalIteration,
                        bool b_Momentum,
                        double d_MomentumAlpha,
                        int n_kFold,
                        int LearnRateAdjMethod,
                        int ActivationType);
    
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
            case MLP_ACTIVATION_BINARY:
            {
                cv::exp(*x*slope*-1, *x);
                *x = 1.0/(cv::Scalar(1)+*x);
                if(x_der != NULL)
                    *x_der = (  cv::Scalar(1) - *x ).mul(*x)* slope;
                break;
            }
            case MLP_ACTIVATION_BIPOLOR:
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
            case MLP_LEARNING_ADJ_EXPDEC:
                return m_dInitalLearningRate*exp(-1*i_iteration*slope/m_nTotalIteration);
            case MLP_LEARNING_ADJ_BINSIG:
                return (m_dInitalLearningRate - m_dMinLearningRate) / (1+exp(slope*(i_iteration - m_nLearningRateShift)) ) + m_dMinLearningRate;
            case MLP_LEARNING_ADJ_STC:
                return m_dInitalLearningRate/ (1+i_iteration/m_nLearningRateShift);
            default:
                return m_dInitalLearningRate/ ((1+i_iteration)/m_nLearningRateShift);
        }
    }
    
    
private:
    wxEvtHandler* m_pHandler;
    void readMat(wxString inputName, cv::Mat* data);
    void writeMat(wxString outputName, cv::Mat* data);
    void readDataLine(cv::Mat* data, wxString line);

};
#endif // MLP_H
