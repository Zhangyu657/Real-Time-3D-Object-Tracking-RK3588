
 #ifndef RK3588_DEMO_PREPROCESS_H
#define RK3588_DEMO_PREPROCESS_H

#include <opencv2/opencv.hpp>
#include "types/datatype.h"

struct LetterBoxInfo
{
    float scale = 1.0f;
    int pad_w = 0;
    int pad_h = 0;
    int input_w = 0;
    int input_h = 0;
    bool hor = false;
};

LetterBoxInfo letterbox(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio);
LetterBoxInfo letterbox_rga(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio);

void cvimg2tensor(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor);
void cvimg2tensor_rga(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor);

#endif // RK3588_DEMO_PREPROCESS_H
