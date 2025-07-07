
/* 
#ifndef RK3588_DEMO_CV_DRAW_H
#define RK3588_DEMO_CV_DRAW_H

#include <opencv2/opencv.hpp>

#include "types/yolo_datatype.h"

// draw detections on img
void DrawDetections(cv::Mat& img, const std::vector<Detection>& objects);

#endif //RK3588_DEMO_CV_DRAW_H */


/* #ifndef RK3588_DEMO_CV_DRAW_H
#define RK3588_DEMO_CV_DRAW_H

#include <opencv2/opencv.hpp>
#include "types/yolo_datatype.h"
#include "process/preprocess.h"



// 坐标反映射：从网络输入图尺寸坐标还原为原图坐标
void scale_coords_back(const LetterBoxInfo& info, int net_input_w, int net_input_h,
                       int image_w, int image_h, cv::Rect& bbox);
void scale_coords_back(const LetterBoxInfo& info, int net_input_w, int net_input_h,
                        int origin_img_w, int origin_img_h, cv::Rect& box);
// 绘图函数：会自动进行坐标还原
//void DrawDetections(cv::Mat& img, const std::vector<Detection>& objects,
                   // int net_input_w = 640, int net_input_h = 640);
void DrawDetections(cv::Mat& img, const std::vector<Detection>& objects,
                        int net_input_w, int net_input_h);
void DrawDetections(cv::Mat& img, const std::vector<Detection>& objects);

#endif // RK3588_DEMO_CV_DRAW_H
 */

/* //3d
 #ifndef RK3588_DEMO_CV_DRAW_H
#define RK3588_DEMO_CV_DRAW_H

#include <opencv2/opencv.hpp>
#include "types/yolo_datatype.h"
#include "process/preprocess.h"

// 坐标反映射：从网络输入图尺寸坐标还原为原图坐标
void scale_coords_back(const LetterBoxInfo& info, int net_input_w, int net_input_h,
                       int image_w, int image_h, cv::Rect& bbox);

void scale_coords_back(const LetterBoxInfo& info, int net_input_w, int net_input_h,
                       int origin_img_w, int origin_img_h, cv::Rect& box);

// 绘图函数：自动进行坐标还原，可显示类名、置信度和XYZ坐标（若有）
void DrawDetections(cv::Mat& img, const std::vector<Detection>& objects,
                    int net_input_w, int net_input_h);

void DrawDetections(cv::Mat& img, const std::vector<Detection>& objects);

#endif // RK3588_DEMO_CV_DRAW_H */


//3d
// 用户添加的注释，可能用于标记该文件与3D功能相关

// 这是标准的 "Include Guard"（头文件保护宏），防止同一个头文件在一次编译中被多次包含
#ifndef RK3588_DEMO_CV_DRAW_H
// 定义宏 RK3588_DEMO_CV_DRAW_H，与上面的 #ifndef 配对使用，确保本文件只被包含一次
#define RK3588_DEMO_CV_DRAW_H

// 引入OpenCV主头文件，以便使用 cv::Mat, cv::Rect 等数据结构和函数
#include <opencv2/opencv.hpp>
// 引入自定义的YOLO数据类型头文件，可能定义了 Detection 结构体
#include "types/yolo_datatype.h"
// 引入自定义的预处理头文件，可能定义了 LetterBoxInfo 结构体
#include "process/preprocess.h"

// 这是一个已有的注释，描述下方函数的功能：将坐标从网络输入尺寸还原为原始图像尺寸
// 坐标反映射：从网络输入图尺寸坐标还原为原图坐标

// `scale_coords_back` 函数的声明（原型）。这是一个重载版本
void scale_coords_back(const LetterBoxInfo& info, int net_input_w, int net_input_h,
                       int image_w, int image_h, cv::Rect& bbox);

// `scale_coords_back` 函数的另一个重载声明，参数名略有不同，功能应保持一致
void scale_coords_back(const LetterBoxInfo& info, int net_input_w, int net_input_h,
                       int origin_img_w, int origin_img_h, cv::Rect& box);

// 这是一个已有的注释，描述下方绘图函数的功能
// 绘图函数：自动进行坐标还原，可显示类名、置信度和XYZ坐标（若有）

// `DrawDetections` 函数的声明。这个重载版本接受网络输入尺寸作为参数
void DrawDetections(cv::Mat& img, const std::vector<Detection>& objects,
                    int net_input_w, int net_input_h);

// `DrawDetections` 函数的另一个重载声明。这个版本不直接接受网络输入尺寸，可能在实现中使用了默认值
void DrawDetections(cv::Mat& img, const std::vector<Detection>& objects);

// 结束 #ifndef RK3588_DEMO_CV_DRAW_H 定义的条件编译块
#endif // RK3588_DEMO_CV_DRAW_H