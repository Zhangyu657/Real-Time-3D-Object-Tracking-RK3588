/* // 预处理

#ifndef RK3588_DEMO_PREPROCESS_H
#define RK3588_DEMO_PREPROCESS_H

#include <opencv2/opencv.hpp>
#include "types/datatype.h"

struct LetterBoxInfo
{
    bool hor;
    int pad;
};

LetterBoxInfo letterbox(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio);
LetterBoxInfo letterbox_rga(const cv::Mat& img, cv::Mat& img_letterbox, float wh_ratio);
void cvimg2tensor(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor);
void cvimg2tensor_rga(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor);

#endif // RK3588_DEMO_PREPROCESS_H
 */
 /* #ifndef RK3588_DEMO_PREPROCESS_H
#define RK3588_DEMO_PREPROCESS_H

#include <opencv2/opencv.hpp>
#include "types/datatype.h"

// ✅ 正确的结构体：包含所有后续用到的字段
struct LetterBoxInfo
{
    bool hor = false;       // 是否横向填充（true=左右填充，false=上下填充）
    int pad = 0;            // 填充的像素数（用于坐标还原）
    int input_w = 0;        // LetterBox处理后的宽度
    int input_h = 0;        // LetterBox处理后的高度
    float ratio = 1.0f;     // 放缩比例（原图尺寸和输入尺寸之间）
};

// ✅ 函数声明
LetterBoxInfo letterbox(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio);
LetterBoxInfo letterbox_rga(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio);
void cvimg2tensor(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor);
void cvimg2tensor_rga(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor);

#endif // RK3588_DEMO_PREPROCESS_H
 */


/*  #ifndef RK3588_DEMO_PREPROCESS_H
#define RK3588_DEMO_PREPROCESS_H

#include <opencv2/opencv.hpp>
#include "types/datatype.h"

// ✅ 正确的结构体，包含坐标还原所需信息
struct LetterBoxInfo
{
    float scale = 1.0f;  // 原图 -> 输入尺寸的缩放比例
    int pad_w = 0;       // 左右填充（horizontal）
    int pad_h = 0;       // 上下填充（vertical）
    int input_w = 0;     // 输入宽
    int input_h = 0;     // 输入高
    bool hor = false;    // 是否是水平填充（true 表示宽度方向填充）
};


// ✅ 修改后的函数声明，统一使用 (input_w, input_h)
//LetterBoxInfo letterbox(const cv::Mat &img, cv::Mat &img_letterbox, int input_w, int input_h);
//LetterBoxInfo letterbox_rga(const cv::Mat &img, cv::Mat &img_letterbox, int input_w, int input_h);
LetterBoxInfo letterbox(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio);
LetterBoxInfo letterbox_rga(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio);

void cvimg2tensor(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor);
void cvimg2tensor_rga(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor);

#endif // RK3588_DEMO_PREPROCESS_H
 */

 // 这是 "Include Guard"（头文件保护宏），用于防止本头文件在一次编译中被重复包含
#ifndef RK3588_DEMO_PREPROCESS_H
// 如果宏 RK3588_DEMO_PREPROCESS_H 未被定义，则定义它，与 #ifndef 配对使用
#define RK3588_DEMO_PREPROCESS_H

// 引入OpenCV主头文件，以便使用 cv::Mat 等数据结构和函数
#include <opencv2/opencv.hpp>
// 引入自定义的头文件 "datatype.h"，其中可能定义了 tensor_data_s 等与张量相关的数据类型
#include "types/datatype.h"

// 这是一个已有的注释，说明下方的结构体定义是正确的，并且包含了坐标还原所需的信息
// ✅ 正确的结构体，包含坐标还原所需信息
// 定义一个名为 LetterBoxInfo 的结构体，用于存储 letterbox 预处理过程中的关键信息
struct LetterBoxInfo
{
    // 成员变量 scale，用于记录从原图到模型输入尺寸的缩放比例，默认为 1.0
    float scale = 1.0f;
    // 成员变量 pad_w，用于记录在宽度（水平）方向上填充的像素数，默认为 0
    int pad_w = 0;
    // 成员变量 pad_h，用于记录在高度（垂直）方向上填充的像素数，默认为 0
    int pad_h = 0;
    // 成员变量 input_w，记录 letterbox 目标图像的宽度，默认为 0
    int input_w = 0;
    // 成员变量 input_h，记录 letterbox 目标图像的高度，默认为 0
    int input_h = 0;
    // 成员变量 hor，布尔型标志，用于表示是否为水平填充（此成员可能用于特定逻辑），默认为 false
    bool hor = false;
}; // 结构体定义结束


// 这是一个已有的注释，说明下方的函数声明经过了修改
// ✅ 修改后的函数声明，统一使用 (input_w, input_h)
// 这是一个被注释掉的函数声明，可能是旧版本的 letterbox 函数接口
//LetterBoxInfo letterbox(const cv::Mat &img, cv::Mat &img_letterbox, int input_w, int input_h);
// 这是一个被注释掉的函数声明，可能是旧版本的 letterbox_rga 函数接口
//LetterBoxInfo letterbox_rga(const cv::Mat &img, cv::Mat &img_letterbox, int input_w, int input_h);

// 声明一个使用 OpenCV 实现的 letterbox 函数
LetterBoxInfo letterbox(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio);
// 声明一个使用 RGA 硬件加速实现的 letterbox 函数
LetterBoxInfo letterbox_rga(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio);

// 声明一个使用 OpenCV 将 cv::Mat 图像转换为张量（tensor_data_s）的函数
void cvimg2tensor(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor);
// 声明一个使用 RGA 硬件加速将 cv::Mat 图像转换为张量（tensor_data_s）的函数
void cvimg2tensor_rga(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor);

// 结束 #ifndef RK3588_DEMO_PREPROCESS_H 定义的条件编译块
#endif // RK3588_DEMO_PREPROCESS_H