#include "MLP.h"
#include <wx/tokenzr.h>

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
    m_nKfold = 10;
    m_dDesiredOutput_rescale = 0.15;
}
void MLP::SetParameter(  int n_nuronL1, 
                    int n_nuronL2, 
                    double d_InitalLearningRate, 
                    double d_MinLearningRate,
                    int n_LearningRateShift,
                    int n_TotalIteration,
                    bool b_Momentum,
                    double d_MomentumAlpha,
                    int n_kFold)
{
    m_nNeuronsL1 = n_nuronL1;
    m_nNeuronsL2 = n_nuronL2;
    m_dInitalLearningRate = d_InitalLearningRate;
    m_dMinLearningRate = d_MinLearningRate;
    m_nLearningRateShift = n_LearningRateShift;
    m_nTotalIteration = n_TotalIteration;
    m_bMomentum  = b_Momentum;
    m_nKfold = n_kFold;
    m_dMomentumAlpha = d_MomentumAlpha;
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
    if(!m_data_input.data)  // check data is ok...
    {
        
        evt_end = new wxThreadEvent(wxEVT_COMMAND_MLP_COMPLETE);
        evt_end->SetString("[MLP]Error: input Matrix data empty");
        wxQueueEvent(m_pHandler, evt_end);
        return (wxThread::ExitCode)-1;
    }
    
    // data scaling (perpare m_data_scaled2train)
    dataScale();
    // inital weight... by cv::randu-(uniformly-distributed random number)
    m_weight_l1.create(m_nInputs,       m_nNeuronsL1,   CV_64F);
    m_weight_l2.create(m_nNeuronsL1,    m_nNeuronsL2,   CV_64F);
    m_weight_l3.create(m_nNeuronsL2,    m_nClasses,     CV_64F);
    cv::randu(m_weight_l1, cv::Scalar(double(0.1)), cv::Scalar(double(0.4)));
    cv::randu(m_weight_l2, cv::Scalar(double(0.1)), cv::Scalar(double(0.4)));
    cv::randu(m_weight_l3, cv::Scalar(double(0.1)), cv::Scalar(double(0.4)));
    // momentum weight updating...
    cv::Mat m_weight_momentum_l1, m_weight_momentum_l2, m_weight_momentum_l3;
    m_weight_momentum_l1 = cv::Mat::zeros(m_nInputs,       m_nNeuronsL1,   CV_64F);
    m_weight_momentum_l2 = cv::Mat::zeros(m_nNeuronsL1,    m_nNeuronsL2,   CV_64F);
    m_weight_momentum_l3 = cv::Mat::zeros(m_nNeuronsL2,    m_nClasses,     CV_64F);
    
    
    writeMat("./inital_W1.txt", &m_weight_l1);
    writeMat("./inital_W2.txt", &m_weight_l2);
    writeMat("./inital_W3.txt", &m_weight_l3);
    writeMat("./trainingScaledData.txt", &m_data_scaled2train);
    
    //MSE
    std::vector<double > vRMSE_training;
    std::vector<double > vRMSE_validation;
    
    
    //epoch for loop
    for(int i_iteration = 0; i_iteration < m_nTotalIteration; i_iteration++)
    {
        double dMSE_training_epoch    = 0;
        double dMSE_validation_epoch  = 0;
        int n_preFoldItems = m_data_scaled2train.rows/m_nKfold;
        
        
        // TO-DO shuffel
        
        // kfold for loop
        for(int i_kFold = 0; i_kFold < m_nKfold; i_kFold++)
        {
            cv::Mat MSE_training_Fold = cv::Mat::zeros(1, m_nClasses, CV_64FC1);
            cv::Mat MSE_valudation_Fold = cv::Mat::zeros(1, m_nClasses, CV_64FC1);
            
            // 切割Fold:index of validation
            int i_strt  = n_preFoldItems*i_kFold;
            int i_end   = i_strt + n_preFoldItems;
            if(i_kFold == m_nKfold -1)
            {
                i_end = m_data_scaled2train.rows;
            }
            cv::Mat validateData = m_data_scaled2train(cv::Range(i_strt, i_end), cv::Range::all()).clone();
            
            bool history = true;
            
            // training for loop
            
            for(int i_dataRows = 0; i_dataRows < m_data_scaled2train.rows; i_dataRows++)
            {
                
                // if row is in validation data range: continue loop..
                if(i_dataRows >= i_strt && i_dataRows < i_end)
                    continue;
                cv::Mat input = m_data_scaled2train(cv::Range(i_dataRows, i_dataRows+1), cv::Range(0, m_nInputs));                
                cv::Mat output_desired = m_data_scaled2train(cv::Range(i_dataRows, i_dataRows+1), cv::Range(m_nInputs, m_data_scaled2train.cols));
                
                cv::Mat summation_L1, summation_L2, summation_L3;
                cv::Mat output_L1, output_L2, output_L3;
                cv::Mat derivate_L1;
                cv::Mat derivate_L2;
                cv::Mat derivate_L3;
                
                // get network work response
                
                //Layer 1
                summation_L1 = input*m_weight_l1;
                output_L1 = summation_L1.clone();
                binSigmoid(&output_L1, &derivate_L1);
                
                //Layer 2
                summation_L2 = output_L1*m_weight_l2;
                output_L2 = summation_L2.clone();
                binSigmoid(&output_L2, &derivate_L2);
                
                //Layer 3 
                summation_L3 = output_L2*m_weight_l3;
                output_L3 = summation_L3.clone();
                binSigmoid(&output_L3, &derivate_L3);
                
                
                //SE(train)
                cv::Mat error = output_desired - output_L3 ;
                cv::Mat errorSquare;
                cv::pow(error.clone(), 2, errorSquare);
                MSE_training_Fold += errorSquare;
  
                // update weight (sequential update)
                
            
                double learnRate = getLearningRate(i_iteration);
                // update Layer 3 weight
                //cv::Mat derivate_L3 = summation_L3.clone();
                //binSigmoidDerivative(&derivate_L3);
                if(!cv::checkRange(derivate_L3))
                {
                    wxLogMessage("---nan---");
                    writeMat(wxString::Format("NanL3X%d.txt", i_dataRows), &summation_L3);
                    writeMat(wxString::Format("NanL3Y%d.txt", i_dataRows), &derivate_L3);
                    writeMat("./end_W1.txt", &m_weight_l1);
                    writeMat("./end_W2.txt", &m_weight_l2);
                    writeMat("./end_W3.txt", &m_weight_l3);
                    break;
                }
                
                cv::Mat delta_L3 = error.mul(derivate_L3); 
                
                m_weight_l3 += (delta_L3.t()*output_L2).t()*learnRate;
                
                

                // update Layer 2 weight 
                //cv::Mat derivate_L2 = summation_L2.clone();
                //binSigmoidDerivative(&derivate_L2);
                if(!cv::checkRange(derivate_L2))
                {
                    wxLogMessage("---nan---");
                    writeMat(wxString::Format("NanL2X%d.txt", i_dataRows), &summation_L2);
                    writeMat(wxString::Format("NanL2Y%d.txt", i_dataRows), &derivate_L2);
                    break;
                }
                cv::Mat delta_L2 = (delta_L3*m_weight_l3.t() ).mul(derivate_L2);
                m_weight_l2 += (delta_L2.t()*output_L1).t()*learnRate;
                
                
                
                
                
               
                
           
                // update Layer 1 weight
                //cv::Mat derivate_L1 = summation_L1.clone();
                //binSigmoidDerivative(&derivate_L1);
                if(!cv::checkRange(derivate_L1))
                {
                    wxLogMessage("---nan---");
                    writeMat(wxString::Format("NanL1X%d.txt", i_dataRows), &summation_L2);
                    writeMat(wxString::Format("NanL1Y%d.txt", i_dataRows), &derivate_L2);
                    break;
                }
                cv::Mat delta_L1 = (delta_L2*m_weight_l2.t()).mul(derivate_L1);
                m_weight_l1 += (delta_L1.t()*input).t()*learnRate;
                
                
                
                
                
                // update weight end
                
            } // training for loop end
            
            // validation for loop
            for(int i_validRows= i_strt; i_validRows < i_end; i_validRows++)
            {
                cv::Mat input = m_data_scaled2train(cv::Range(i_validRows, i_validRows+1), cv::Range(0, m_nInputs));                
                cv::Mat output_desired = m_data_scaled2train(cv::Range(i_validRows, i_validRows+1), cv::Range(m_nInputs, m_data_scaled2train.cols));
                cv::Mat response_L1, response_L2, response_L3;
                response_L1 = input*m_weight_l1;
                binSigmoid(&response_L1);
                response_L2 = response_L1*m_weight_l2;
                binSigmoid(&response_L2);
                response_L3 = response_L2*m_weight_l3;
                binSigmoid(&response_L3);
                
                
                //SE(train)
                cv::Mat error = output_desired - response_L3 ;
                cv::pow(error, 2, error);
                MSE_valudation_Fold += error;
            }// validation for loop end
            
            //writeMat("MSE_valudation_Fold.csv", &MSE_valudation_Fold);
            //writeMat("MSE_training_Fold.csv", &MSE_training_Fold);
            
            
            //MSE(fold)
            dMSE_training_epoch += sqrt(cv::sum(MSE_training_Fold)[0]/(m_data_scaled2train.rows-i_end+i_strt));
            dMSE_validation_epoch += sqrt(cv::sum(MSE_valudation_Fold)[0]/(i_end-i_strt));
            
            
        }// kfold for loop end
        
        //MSE(epoch)
        vRMSE_training.push_back(dMSE_training_epoch/(double)m_nKfold);
        vRMSE_validation.push_back(dMSE_validation_epoch/(double)m_nKfold);
        
        
        evt_update = new wxThreadEvent(wxEVT_COMMAND_MLP_UPDATE_PG);
        evt_update->SetInt(i_iteration);
        wxQueueEvent(m_pHandler, evt_update);
    }//epoch for loop end
    
    
    //------------------------save files---------------------//
    
    //save MSE history
    cv::Mat MSE_trainigData(1, vRMSE_training.size(), CV_64F, &vRMSE_training.front());
    cv::Mat MSE_validationData(1, vRMSE_validation.size(), CV_64F, &vRMSE_validation.front());
    writeMat("MSE_trainigData.csv", &MSE_trainigData);
    writeMat("MSE_validationData.csv", &MSE_validationData);
    
    
    //save weight result
    writeMat("./end_W1.txt", &m_weight_l1);
    writeMat("./end_W2.txt", &m_weight_l2);
    writeMat("./end_W3.txt", &m_weight_l3);
    
    //------------------------save files end-----------------//
    
    
    // post event2handler4stop
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
        double min, max;
        cv::minMaxLoc(m_data_input.col(i), &min, &max);
        cv::normalize(m_data_input.col(i), m_data_scaled2train.col(i), 0.001, 0.999, cv::NORM_MINMAX, -1, cv::Mat() );
           
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
    writeMat("./rescaled_input.txt", &m_data_scaled2train);
    
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
    tfile.Close();
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






