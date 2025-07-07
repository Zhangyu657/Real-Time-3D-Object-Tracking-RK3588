

 // 这是 "Include Guard"（头文件保护宏），用于防止本头文件在一次编译中被重复包含
#ifndef RK3588_DEMO_NN_DATATYPE_H
// 如果宏 RK3588_DEMO_NN_DATATYPE_H 未被定义，则定义它，与 #ifndef 配对使用
#define RK3588_DEMO_NN_DATATYPE_H

// 引入OpenCV主头文件，以使用 cv::Rect, cv::Scalar 等数据结构
#include <opencv2/opencv.hpp>
// 引入自定义的预处理头文件，以使用其中定义的 LetterBoxInfo 结构体
#include "process/preprocess.h" 

// 这是一个已有的注释，说明下方结构体的可能用途
// 原始结构体（可能用于量化输出或中间处理）
// 使用 C 风格的 typedef struct 定义一个名为 _nn_object_s 的结构体，并为其创建别名 nn_object_s
typedef struct _nn_object_s {
    float x;        // 目标的中心点 x 坐标
    float y;        // 目标的中心点 y 坐标
    float w;        // 目标的宽度
    float h;        // 目标的高度
    float score;    // 目标的置信度分数
    int class_id;   // 目标的类别ID
} nn_object_s;      // 结构体类型别名

// 这是一个已有的注释，说明下方的结构体是主要的检测结果表示形式
// 主检测结构体
// 定义一个名为 Detection 的 C++ 风格结构体，用于存储一个完整的检测目标信息
/* struct Detection
{
    // 目标的类别ID，使用C++11的花括号初始化语法为其设置默认值 0
    int class_id{0};
    // 目标的类别名称字符串，默认为空字符串
    std::string className{};
    // 目标的置信度分数，默认为 0.0
    float confidence{0.0};
    // 用于可视化绘制的颜色，默认为空（黑色）
    cv::Scalar color{};
    // 目标的边界框，使用OpenCV的Rect表示，默认为空
    cv::Rect box{};
    // 目标的追踪ID（如果使用了追踪算法）
    int id;
    // 这是一个已有的注释，说明下方成员变量的用途
    // ✅ letterbox 信息（用于坐标还原）
    // 布尔型标志，表示此检测结果的坐标是否基于letterbox处理后的图像，默认为 false
    bool has_letterbox = false;
    // 存储 letterbox 变换信息的结构体
    LetterBoxInfo letterbox_info;

    // 这是一个已有的注释，说明下方成员变量的用途
    // ✅ XYZ 深度坐标（可选）
    // 布尔型标志，表示此检测结果是否包含三维空间坐标，默认为 false
    bool has_xyz = false;
    // 目标在相机坐标系下的 x, y, z 坐标，默认为 0
    float x = 0, y = 0, z = 0;
}; // 结构体定义结束
 */

 struct Detection
{
    int class_id{0};                // 目标的类别ID
    std::string className{};         // 目标的类别名称字符串
    float confidence{0.0};           // 目标的置信度分数
    cv::Scalar color{};              // 用于可视化绘制的颜色
    cv::Rect box{};                  // 目标的边界框
    int id;                          // 目标的追踪ID
    bool has_letterbox = false;      // 是否基于letterbox处理后的图像
    LetterBoxInfo letterbox_info;    // letterbox变换信息
    bool has_xyz = false;            // 是否包含三维空间坐标
    float x = 0, y = 0, z = 0;      // 三维空间坐标
};
// 结束 #ifndef RK3588_DEMO_NN_DATATYPE_H 定义的条件编译块
#endif // RK3588_DEMO_NN_DATATYPE_H