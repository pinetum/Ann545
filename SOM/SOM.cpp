#include "SOM.h"
#include "wx/log.h"
#include "math.h"
#include "stdio.h"


SOM::SOM(wxEvtHandler* pParent, int weightSize, int iterationTimes, cv::Mat input)
{
    handler = pParent;
    n_weightSize = weightSize;
    n_iterationTimes = iterationTimes;
    m_input = input.clone();
}

SOM::~SOM()
{
}
void SOM::updateParameter(int epoch)
{
    //講義上的方法
//    d_learningRate = 1 /(1+(double)epoch/100);
//    double sigma = n_weightSize*0.5;
//    double lamda = 1.0*n_iterationTimes/log(sigma);
//    d_bandwith =  sigma*exp(-1*epoch/lamda)*0.9;
//    
    
  
    double initalSigma = sqrt(n_weightSize*n_weightSize)*0.5;
    
    
    
    d_learningRate = 1 /(1+(double)epoch/100);
    d_bandwith =  (initalSigma - sqrt(2)) / (1+exp(0.3*(epoch - 10)) ) + sqrt(2);

    
    
}
wxThread::ExitCode SOM::Entry(){
    wxThreadEvent* evt = new wxThreadEvent(wxEVT_COMMAND_SOM_START);
    wxQueueEvent(handler, evt);
    
    
    //record mu and sigma
    FILE* fp_mu = fopen("mu.csv", "w");
    FILE* fp_sigma = fopen("sigma.csv", "w");
    
    
    // inital weights
    weights = cv::Mat(n_weightSize, n_weightSize, CV_64FC2);
    
    cv::Mat initalKet = m_input.clone();
    shuffelRow(&initalKet);
    
    for(int i=0; i <n_weightSize; i++ )
        for(int j=0; j<n_weightSize; j++)
        {
            //隨機挑點
            //weights.at<cv::Vec2d>(i,j)= initalKet.at<cv::Vec2d>(n_weightSize*i+j,0);
            //隨機0~1
            cv::randu(weights.at<cv::Vec2d>(i,j), cv::Scalar(0.3), cv::Scalar(0.7));
        }
    //writeMat(wxString::Format("/Users/QT/Downloads/www/Weight0.txt"), &weights);
    // epoch loop
    for(int i_epoch=0; i_epoch < n_iterationTimes; i_epoch++)
    {
        //writeMat(wxString::Format("/Users/QT/Downloads/www/Weight%00d.txt", i_epoch), &weights);
        // update learnrate and neighbord width
        updateParameter(i_epoch+1);
        for(int i_input = 0; i_input<m_input.rows; i_input++)
        {
            //find winner
            cv::Vec2d x, w;
            // to save winner information
            int i_winner_r=0;
            int i_winner_c=0;
            double min_Distance=sqrt(n_weightSize*n_weightSize+n_weightSize*n_weightSize);
            x = m_input.at<cv::Vec2d>(i_input,0);
            for(int r=0; r < weights.rows; r++)
            {
                for(int c=0; c < weights.cols; c++)
                {
                    w = weights.at<cv::Vec2d>(r,c);
                    
                    //double d = sqrt( pow(x[0] - w[0], 2) + pow(x[1] - w[1], 2) );
                    double d = cv::norm(x, w, cv::NORM_L2);
                    
                    //wxLogMessage(wxString::Format("norm:%lf, x:%lf,%lf w::%lf,%lf", d, x[0], x[1], w[0], w[1]));
                    
                    if(d < min_Distance)
                    {
                        // find new winner....
                        min_Distance = d;
                        i_winner_r = r;
                        i_winner_c = c;
                    }
                }
            }
            //update weight
            for(int i_neuronsR =0; i_neuronsR < weights.rows; i_neuronsR++)
            {
                for(int i_neuronsC=0; i_neuronsC< weights.cols; i_neuronsC++)
                {
                    double dist = sqrt((i_neuronsR-i_winner_r)*(i_neuronsR-i_winner_r)+(i_neuronsC-i_winner_c)*(i_neuronsC-i_winner_c));
                    //wxLogMessage(wxString::Format("%f", dist));
                    if( dist > d_bandwith)
                        continue;
                    //double influence = exp(-(dist)/2/d_bandwith);
                    double influence = exp(-(dist)/2*d_bandwith);
                    //wxLogMessage(wxString::Format("influence:%lf", influence));
                    cv::Vec2d updateNeuron = weights.at<cv::Vec2d>(i_neuronsR, i_neuronsC);
                    updateNeuron += d_learningRate*influence*(x - updateNeuron);
                    weights.at<cv::Vec2d>(i_neuronsR, i_neuronsC) = updateNeuron;
                }
                
            }// all neuron for loop

            
        }//input data loop
        
        
        
        if(1)// (i_epoch+1) % 10 == 0 || i_epoch < 20)
        {
            evt = new wxThreadEvent(wxEVT_COMMAND_SOM_UPDATE);
            evt->SetPayload(weights.clone());
            evt->SetInt(i_epoch);
            evt->SetString(wxString::Format("epoch:%d mu:%lf, sigma:%lf",i_epoch, d_learningRate, d_bandwith));
            wxQueueEvent(handler, evt);
            
        }
        
        if(TestDestroy())
            break;
        
        fprintf(fp_mu, "%lf\n", d_learningRate);
        fprintf(fp_sigma, "%lf\n", d_bandwith);
        
    }//epoch loop
    writeMat(wxString::Format("/Users/QT/Downloads/www/Weight.txt"), &weights);
    
    
    fclose(fp_mu);
    fclose(fp_sigma);
    
    
    evt = new wxThreadEvent(wxEVT_COMMAND_SOM_COMPLETE);
    wxQueueEvent(handler, evt);
    
    return (wxThread::ExitCode)0;
}
void SOM::writeMat(wxString outputName, cv::Mat* data, wxString info )
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
                cv::Vec2d pt;
                pt = data->at<cv::Vec2d>(i, j);
                str_line.append(wxString::Format("%lf %lf",pt[0], pt[1]));
            }
            tfile.AddLine(str_line);
        }
        
        
        
        
        tfile.Write();
        tfile.Close();
    }
