#include "RBF.h"





const char str_learnRateAdj[3][30] = {"Search then converge", "Exponential decay", "Binary Sigmoid"};
const char str_activation[2][30] = {"Binary Sigmoid", "bipolar sigmoid"};

RBF::RBF(wxEvtHandler* pParent)
{
    m_pHandler                  = pParent;
    m_nKfold                    = 10;
    m_nInputs                   = 9;//pima 8// cancer//9
    m_nClasses                  = 2;
    m_nNeurons                  = 5;
    m_nTotalIteration           = 3000;
    m_dMinLearningRate          = 0.05;
    m_nLearningRateShift        = 4000;
    m_LearnRateAdjMethod        = RBF_LEARNING_ADJ_BINSIG;
    m_dRatioTestingDatas        = 0.2;
    m_dInitalLearningRate       = 0.3;
    m_dDesiredOutput_rescale    = 0.15;
    m_nTerminalThreshold        = m_nTotalIteration*0.08;
}
void RBF::SetParameter(bool b_dataRescale,
                        int n_nuron,
                        double d_InitalLearningRate, 
                        double d_MinLearningRate,
                        int n_LearningRateShift,
                        int n_TotalIteration,
                        
                        int n_kFold,
                        double d_testDataRatio,
                        double d_Terminalratio,
                        int LearnRateAdjMethod)
{
    m_nKfold                = n_kFold;
    m_nNeurons              = n_nuron;
    m_bRescale              = b_dataRescale;
    m_dRatioTestingDatas    = d_testDataRatio;
    m_nTotalIteration       = n_TotalIteration;
    m_dMinLearningRate      = d_MinLearningRate;
    m_LearnRateAdjMethod    = LearnRateAdjMethod;
    m_nLearningRateShift    = n_LearningRateShift;
    m_dInitalLearningRate   = d_InitalLearningRate;
    m_dTerminalratio        = d_Terminalratio;
    
}
RBF::~RBF()
{

}
wxThread::ExitCode RBF::Entry(){
    wxThreadEvent* evt_end;
    wxThreadEvent* evt_update;
    wxThreadEvent* evt_start;
    evt_start = new wxThreadEvent(wxEVT_COMMAND_RBF_START);
    wxQueueEvent(m_pHandler, evt_start);
    if(!m_data_input.data)  // check data is ok...
    {
        evt_end = new wxThreadEvent(wxEVT_COMMAND_RBF_COMPLETE);
        evt_end->SetString("[RBF]Error: input Matrix data empty");
        wxQueueEvent(m_pHandler, evt_end);
        return (wxThread::ExitCode)-1;
    }
    // data scaling (perpare m_data_scaled2train and m_data_scaled2test)
    dataScale();
    // inital training parameters
    m_weight.create(m_nClasses, m_nNeurons, CV_64F);                   
    m_center.create(m_nNeurons, m_nInputs, CV_64F);
    m_sigma.create(1, m_nNeurons, CV_64F);
    
    shuffelRow(&m_data_scaled2train);
    
    cv::randu(m_weight, cv::Scalar(-0.4), cv::Scalar(0.4));
    m_center = m_data_scaled2train(cv::Range(0, m_nNeurons), cv::Range(0, m_nInputs)).clone();
    m_sigma  = cv::Mat(1, m_nNeurons, CV_64F, cv::Scalar((double)1/sqrt(m_nNeurons)));
    
    writeMat("weight.txt", &m_weight);
    writeMat("center.txt", &m_center);
    writeMat("sigma.txt", &m_sigma);
    
    
    
    //MSE
    std::vector<double > vMSE_training;
    std::vector<double > vMSE_validation;
    
    int early_terminateCounter = 0;
    //epoch for loop
    for(int i_iteration = 0; i_iteration < m_nTotalIteration; i_iteration++)
    {
        // shuffel
        //shuffelRow(&m_data_scaled2train);
        
        
        double dMSE_training_epoch    = 0;
        double dMSE_validation_epoch  = 0;
        
        int n_preFoldItems, n_loopFoldTimes;
        // leave one out 
        if(m_nKfold < 1)
        {
            n_preFoldItems = 1;
            n_loopFoldTimes = m_data_scaled2train.rows;
        }
        else
        {
            n_preFoldItems = m_data_scaled2train.rows/m_nKfold;
            n_loopFoldTimes = m_nKfold;
        }
        
            
        // kfold for loop
        for(int i_kFold = 0; i_kFold < n_loopFoldTimes; i_kFold++)
        {
            cv::Mat MSE_training_Fold = cv::Mat::zeros(1, m_nClasses, CV_64FC1);
            cv::Mat MSE_valudation_Fold = cv::Mat::zeros(1, m_nClasses, CV_64FC1);
            
            // 切割Fold:index of validation
            int i_strt  = n_preFoldItems*i_kFold;
            int i_end   = i_strt + n_preFoldItems;
            if(i_kFold == n_loopFoldTimes -1)
            {
                i_end = m_data_scaled2train.rows;
            }
            cv::Mat validateData = m_data_scaled2train.rowRange(i_strt, i_end);
            
            
            // training for loop
            //for(int i_dataRows = 0; i_dataRows < 1; i_dataRows++)
            for(int i_dataRows = 0; i_dataRows < m_data_scaled2train.rows; i_dataRows++)
            {
                
                // if row is in validation data range: continue loop..
                if(i_dataRows >= i_strt && i_dataRows < i_end)
                    continue;
                cv::Mat input = m_data_scaled2train(cv::Range(i_dataRows, i_dataRows+1), cv::Range(0, m_nInputs));                
                cv::Mat output_desired = m_data_scaled2train(cv::Range(i_dataRows, i_dataRows+1), cv::Range(m_nInputs, m_nInputs+m_nClasses));
                cv::Mat output_response, phi_result;
                cv::Mat delta_weight, delta_center, delta_sigma;
                cv::Mat norm, normSqure;
                cv::Mat sigmaThree, sigmaSqure;
                cv::pow(m_sigma, 3, sigmaThree);
                cv::pow(m_sigma, 2, sigmaSqure);
                //output_response;
                
                
                mNorm2(norm, input, m_center);
                cv::pow(norm, 2, normSqure);
                cv::exp(-1*normSqure/sigmaSqure, phi_result);
                output_response = phi_result*m_weight.t();
                
                
                //SE(train)
                cv::Mat error = output_desired - output_response;
                cv::Mat errorSquare;
                cv::pow(error, 2, errorSquare);
                // summation square error
                MSE_training_Fold += errorSquare/errorSquare.rows;
                
                
                
               
                
                
                
                // update parameter (sequential update)
                double learnRate = getLearningRate(i_iteration);
//                    // delta weight
                
               
                delta_weight = learnRate*error.t()*phi_result;
//                    // delta sigma

                
                delta_sigma = learnRate*(error*m_weight/sigmaThree).mul(phi_result).mul(normSqure);//  (1*# nurons)
                
                
                    // delta center
                //delta_center.create(m_center.rows, m_center.cols, CV_64F);
//                delta_center = (error*m_weight/sigmaSqure).mul(phi_result);
                
                

//                wxLogMessage(wxString::Format("%d %d, %d %d, %d, %d",   
//                                                m_center.rows,
//                                                m_center.cols,
//                                                delta_center.rows,
//                                                delta_center.cols,
//                                                phi_result.rows,
//                                                phi_result.cols));
//                
//                

                


                cv::Mat temp = (error*m_weight/sigmaSqure).mul(phi_result);
                delta_center.create(m_center.rows, m_center.cols, CV_64F);
                for(int i =0; i < delta_center.rows; i++)
                {
                    delta_center.row(i) = temp.at<double>(0, i)*(input-m_center.row(i));
                }
                delta_center = delta_center * learnRate;
                //writeMat("deltaCenter.txt", &delta_center);
                //writeMat("Center.txt", &m_center);
                
                m_center    += delta_center;
                m_weight    += delta_weight;
                m_sigma     += delta_sigma;
                
                
                
                // update parameter end
            } // training for loop end
            
            // validation for loop
            for(int i_validRows= i_strt; i_validRows < i_end; i_validRows++)
            {
                cv::Mat input = m_data_scaled2train(cv::Range(i_validRows, i_validRows+1), cv::Range(0, m_nInputs));                
                cv::Mat output_desired = m_data_scaled2train(cv::Range(i_validRows, i_validRows+1), cv::Range(m_nInputs, m_data_scaled2train.cols));
                cv::Mat output_response;
                cv::Mat phi_result;
                cv::Mat norm, normSqure;
                cv::Mat sigmaThree, sigmaSqure;
                cv::pow(m_sigma, 2, sigmaSqure);
                //output_response;
                
                
                mNorm2(norm, input, m_center);
                cv::pow(norm, 2, normSqure);
                cv::exp(-1*normSqure/sigmaSqure, phi_result);
                output_response = phi_result*m_weight.t();
                
                
                //SE(train)
                cv::Mat error = output_desired - output_response ;
                cv::pow(error, 2, error);
                MSE_valudation_Fold += error/error.rows;
            }// validation for loop end
            
            
            // root "mean" square error
            dMSE_training_epoch += sqrt(cv::sum(MSE_training_Fold)[0]/(m_data_scaled2train.rows-i_end+i_strt));
            dMSE_validation_epoch += sqrt(cv::sum(MSE_valudation_Fold)[0]/(i_end-i_strt));            
        }// kfold for loop end
        
        //mean k-fold MSE(epoch)
        vMSE_training.push_back(dMSE_training_epoch/(double)n_loopFoldTimes);
        vMSE_validation.push_back(dMSE_validation_epoch/(double)n_loopFoldTimes);
        
        // check can terminate?
//        if(i_iteration > 0)                 // check epoch > 0
//        {
//            double diff = vMSE_validation[i_iteration] - vMSE_validation[i_iteration-1];
//            if( diff >= 0) // validation MSE rise
//            {
//                if(early_terminateCounter == 0) // when MSE rise first epoch
//                {
//                    m_weight_terminal_l1 = m_weight_l1.clone();
//                    m_weight_terminal_l2 = m_weight_l2.clone();
//                    m_weight_terminal_l3 = m_weight_l3.clone();
//                }
//                early_terminateCounter++;
//                if(early_terminateCounter == m_nTerminalThreshold) // reach terminal condidtion 
//                {
//                    m_weight_l1 = m_weight_terminal_l1;
//                    m_weight_l2 = m_weight_terminal_l2;
//                    m_weight_l3 = m_weight_terminal_l3;
//                    early_terminateCounter = i_iteration;
//                    break;
//                }
//            }
//            else
//            {
//                early_terminateCounter = 0;
//            }
//        }
            
            
        
        
        // update progress bar and timer
        evt_update = new wxThreadEvent(wxEVT_COMMAND_RBF_UPDATE_PG);
        evt_update->SetInt(i_iteration);
        wxQueueEvent(m_pHandler, evt_update);
    }//epoch for loop end
    //------------------------save files---------------------//
    wxString str_result, str_resultFileName;
    str_result.Printf("Accuracy,%f,%f,%d,%d,%d,%d,%f,%f,%d,%f,%s", 
                                        getAccuracy(),
                                        m_dRatioTestingDatas,
                                        m_nNeurons, 
                                        early_terminateCounter,
                                        m_nTotalIteration,
                                        m_nKfold,
                                        m_dInitalLearningRate,
                                        m_dMinLearningRate,
                                        m_nLearningRateShift,
                                        m_dMomentumAlpha,
                                        str_learnRateAdj[m_LearnRateAdjMethod]);
                                        
     str_resultFileName = str_result.Clone();
     str_resultFileName.Replace(",", "-");
    //save MSE history col0:traing col1:validation
    cv::Mat mse_history(1, vMSE_training.size(), CV_64F, &vMSE_training.front());
    mse_history.push_back(cv::Mat(1, vMSE_validation.size(), CV_64F, &vMSE_validation.front()));
    mse_history = mse_history.t();
    writeMat(str_resultFileName.append(".csv"), &mse_history, str_result);
//    cv::Mat MSE_trainigData(vMSE_training.size(), 1, CV_64F, &vMSE_training.front());
//    cv::Mat MSE_validationData(vMSE_validation.size(), 1, CV_64F, &vMSE_validation.front());
//    writeMat("MSE_trainigData.csv", &MSE_trainigData);
//    writeMat("MSE_validationData.csv", &MSE_validationData);
    
    //save weight result
    writeMat(str_resultFileName << "_weight.txt", &m_weight);
    writeMat(str_resultFileName << "_center.txt", &m_center);
    writeMat(str_resultFileName << "_sigma.txt", &m_sigma);
    
    
    //------------------------save files end-----------------//
    // post event2handler4stop
    evt_end = new wxThreadEvent(wxEVT_COMMAND_RBF_COMPLETE);
    evt_end->SetString(wxString::Format("[RBF]Complete. %s",str_result));
    wxQueueEvent(m_pHandler, evt_end);
    return (wxThread::ExitCode)0;
}

void RBF::dataScale()
{
    cv::Mat rescaledResult(m_data_input.rows, m_nInputs + m_nClasses, CV_64FC1);
    
    
    // scale input colum
    if(m_bRescale)
    {
        for(int i =0; i < m_nInputs; i++)
            cv::normalize(m_data_input.col(i), rescaledResult.col(i), 0.001, 0.999, cv::NORM_MINMAX, -1, cv::Mat() );
    }
    else
        rescaledResult = m_data_input.clone();
    
    
//    cancer
    // scale oupput colum
        // begin  2>>1, 4>>0
    rescaledResult.col(m_nInputs) = (m_data_input.col(m_nInputs)-cv::Scalar(4))/-2; 
    cv::normalize(rescaledResult.col(m_nInputs), 
                    rescaledResult.col(m_nInputs), 
                    m_dDesiredOutput_rescale, 1 - m_dDesiredOutput_rescale,  // min, max
                    cv::NORM_MINMAX, -1, cv::Mat() );
        // malignant 4>>1, 2>>0
    rescaledResult.col(m_nInputs+1) = m_data_input.col(m_nInputs)*0.5-cv::Scalar(1); // if 
    cv::normalize(rescaledResult.col(m_nInputs+1), 
                    rescaledResult.col(m_nInputs+1), 
                    m_dDesiredOutput_rescale, 1 - m_dDesiredOutput_rescale,  // min, max
                    cv::NORM_MINMAX, -1, cv::Mat() );

// pima
    // scale oupput colum
//    cv::normalize(rescaledResult.col(m_nInputs), 
//                    rescaledResult.col(m_nInputs), 
//                    m_dDesiredOutput_rescale, 1 - m_dDesiredOutput_rescale,  // min, max
//                    cv::NORM_MINMAX, -1, cv::Mat() );
//        // malignant 4>>1, 2>>0
//    rescaledResult.col(m_nInputs+1) = (m_data_input.col(m_nInputs)-cv::Scalar(1))*-1; // if 
//    cv::normalize(rescaledResult.col(m_nInputs+1), 
//                    rescaledResult.col(m_nInputs+1), 
//                    m_dDesiredOutput_rescale, 1 - m_dDesiredOutput_rescale,  // min, max
//                    cv::NORM_MINMAX, -1, cv::Mat() );



    // seperate scaled data to two part (training and testing)
    int n_testDataRows  = m_dRatioTestingDatas*rescaledResult.rows;
    m_data_scaled2test  = rescaledResult.rowRange(0, n_testDataRows).clone();
    m_data_scaled2train = rescaledResult.rowRange(n_testDataRows, rescaledResult.rows).clone();
    writeMat("m_data_scaled2test.txt", &m_data_scaled2test);
    writeMat("m_data_scaled2train.txt", &m_data_scaled2train);
}

double RBF::getAccuracy()
{
    int n_timesCorrect  = 0;
    for(int i= 0; i < m_data_scaled2test.rows; i++)
    {
        cv::Mat input = m_data_scaled2test(cv::Range(i, i+1), cv::Range(0, m_nInputs));                
        cv::Mat output_desired = m_data_scaled2test(cv::Range(i, i+1), cv::Range(m_nInputs, m_data_scaled2test.cols));
        cv::Mat output_response;
        cv::Mat norm, normSqure;
        cv::Mat phi_result;
        cv::Mat sigmaThree, sigmaSqure;
        cv::pow(m_sigma, 3, sigmaThree);
        cv::pow(m_sigma, 2, sigmaSqure);
        //output_response;
        
        
        mNorm2(norm, input, m_center);
        cv::pow(norm, 2, normSqure);
        cv::exp(-1*normSqure/sigmaSqure, phi_result);
        output_response = phi_result*m_weight.t();
        
        
        
        cv::Point max_loc_response, max_loc_desert;
        cv::minMaxLoc(output_response, NULL, NULL, NULL, &max_loc_response);
        cv::minMaxLoc(output_desired, NULL, NULL, NULL, &max_loc_desert);
        
         if(max_loc_response == max_loc_desert)
             n_timesCorrect++;
        
    }
    return (double)n_timesCorrect/m_data_scaled2test.rows;
}







