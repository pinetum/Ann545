#include "SOM.h"
#include "wx/log.h"
#include "math.h"
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
    
    d_learningRate = 1 /(1+(double)epoch/200.);
    
    
    double sigma = n_weightSize/2.0;
    double lamda = 1.0*n_iterationTimes/log(sigma);
    d_bandwith =  sigma*exp(-epoch/lamda);

}
wxThread::ExitCode SOM::Entry(){
    wxThreadEvent* evt = new wxThreadEvent(wxEVT_COMMAND_SOM_START);
    wxQueueEvent(handler, evt);
    cv::Mat weights(n_weightSize, n_weightSize, CV_64FC2, cv::Scalar(40,40));
    cv::randu(weights, cv::Scalar(0.4*CONTENT_SIZE,0.4*CONTENT_SIZE), cv::Scalar(0.6*CONTENT_SIZE,0.6*CONTENT_SIZE));
    for(int i_epoch=0; i_epoch < n_iterationTimes; i_epoch++)
    {
         
        updateParameter(i_epoch+1);
        for(int i_input = 0; i_input<m_input.rows; i_input++)
        {
            //find winner
            cv::Vec2d x, w;
            int i_winner_r=0;
            int i_winner_c=0;
            double min_Distance=n_weightSize*n_weightSize;
            
            x = m_input.at<cv::Vec2d>(i_input,0);
            for(int r=0; r< weights.rows; r++)
            {
                for(int c=0; c<weights.cols; c++)
                {
                    w = m_input.at<cv::Vec2d>(r,c);
                    double d = cv::norm(x, w, cv::NORM_L2);
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
                    if( dist > d_bandwith)
                        continue;
                    double influence = exp(-dist/2*d_bandwith);
                    
                    cv::Vec2d updateNeuron = weights.at<cv::Vec2d>(i_neuronsR, i_neuronsC);
                    
                    weights.at<cv::Vec2d>(i_neuronsR, i_neuronsC) += d_learningRate*influence*(x - updateNeuron);
                    
                    
                }
                
            }// all neuron for loop

            
        }
        
        
        
        
        evt = new wxThreadEvent(wxEVT_COMMAND_SOM_UPDATE);
        evt->SetPayload(weights);
        evt->SetInt(i_epoch);
        
        evt->SetString(wxString::Format("%d LR:%lf, BW:%lf",i_epoch, d_learningRate, d_bandwith));
        wxQueueEvent(handler, evt);
    }
    
    
    
    
    
    
    evt = new wxThreadEvent(wxEVT_COMMAND_SOM_COMPLETE);
    wxQueueEvent(handler, evt);
    
    return (wxThread::ExitCode)0;
}