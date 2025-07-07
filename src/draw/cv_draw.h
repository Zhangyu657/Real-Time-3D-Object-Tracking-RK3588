
#ifndef RK3588_DEMO_CV_DRAW_H
#define RK3588_DEMO_CV_DRAW_H

#include <opencv2/opencv.hpp>
#include "types/yolo_datatype.h"
#include "process/preprocess.h"

void scale_coords_back(const LetterBoxInfo& info, int net_input_w, int net_input_h,
                       int image_w, int image_h, cv::Rect& bbox);

void scale_coords_back(const LetterBoxInfo& info, int net_input_w, int net_input_h,
                       int origin_img_w, int origin_img_h, cv::Rect& box);

void DrawDetections(cv::Mat& img, const std::vector<Detection>& objects,
                    int net_input_w, int net_input_h);

void DrawDetections(cv::Mat& img, const std::vector<Detection>& objects);

#endif // RK3588_DEMO_CV_DRAW_H
