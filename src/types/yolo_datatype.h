#ifndef RK3588_DEMO_NN_DATATYPE_H
#define RK3588_DEMO_NN_DATATYPE_H

#include <opencv2/opencv.hpp>
#include "process/preprocess.h"

typedef struct _nn_object_s {
    float x;        
    float y;        
    float w;        
    float h;        
    float score;    
    int class_id;   
} nn_object_s;

struct Detection
{
    int class_id{0};                
    std::string className{};         
    float confidence{0.0};           
    cv::Scalar color{};              
    cv::Rect box{};                  
    int id;                          
    bool has_letterbox = false;      
    LetterBoxInfo letterbox_info;    
    bool has_xyz = false;            
    float x = 0, y = 0, z = 0;      
};

#endif // RK3588_DEMO_NN_DATATYPE_H
