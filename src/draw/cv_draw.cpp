/* 
#include "cv_draw.h"

#include "utils/logging.h"

// 在img上画出检测结果
void DrawDetections(cv::Mat &img, const std::vector<Detection> &objects)
{
    NN_LOG_DEBUG("draw %ld objects", objects.size());
    for (const auto &object : objects)
    {
        cv::rectangle(img, object.box, object.color, 2);
        // class name with confidence
        std::string draw_string = object.className + " " + std::to_string(object.confidence);

        cv::putText(img, draw_string, cv::Point(object.box.x, object.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 1,
                    object.color, 2);
    }
} */

/* #include "cv_draw.h"
#include "utils/logging.h"

void scale_coords_back(const LetterBoxInfo& info, int net_input_w, int net_input_h,
                       int image_w, int image_h, cv::Rect& bbox)
{
    float r = std::min(float(net_input_w) / image_w, float(net_input_h) / image_h);
    //int pad_x = info.hor ? info.pad : 0;
    //int pad_y = info.hor ? 0 : info.pad;
    int pad_x = info.pad_w;
    int pad_y = info.pad_h;


    bbox.x = std::max(int((bbox.x - pad_x) / r), 0);
    bbox.y = std::max(int((bbox.y - pad_y) / r), 0);
    bbox.width = int(bbox.width / r);
    bbox.height = int(bbox.height / r);
}

void DrawDetections(cv::Mat& img, const std::vector<Detection>& objects,
                    int net_input_w, int net_input_h)
{
    NN_LOG_DEBUG("draw %ld objects", objects.size());
    for (const auto& object : objects)
    {
        cv::Rect box = object.box;

        // 若存在letterbox信息则进行坐标反变换
        if (object.has_letterbox)
        {
            scale_coords_back(object.letterbox_info, net_input_w, net_input_h,
                              img.cols, img.rows, box);
        }

        cv::rectangle(img, box, object.color, 2);

        std::ostringstream oss;
        oss << object.className << " " << std::fixed << std::setprecision(2) << object.confidence;
        cv::putText(img, oss.str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 1,
                    object.color, 2);
    }
}
 */

 /* #include <opencv2/opencv.hpp>
#include <iomanip>
#include <sstream>
#include "types/yolo_datatype.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"
#include "process/preprocess.h"


// 将 letterbox 缩放后的框还原到原图坐标
void scale_coords_back(const LetterBoxInfo& info, int net_input_w, int net_input_h,
    int origin_img_w, int origin_img_h, cv::Rect& box)
{
float scale = info.scale;
int pad_w = info.pad_w;
int pad_h = info.pad_h;

// 转 float 计算更精确
float x = (box.x - pad_w) / scale;
float y = (box.y - pad_h) / scale;
float w = box.width / scale;
float h = box.height / scale;

// 限制边界，避免越界访问
int x0 = std::max(int(x), 0);
int y0 = std::max(int(y), 0);
int x1 = std::min(int(x + w), origin_img_w);
int y1 = std::min(int(y + h), origin_img_h);

box = cv::Rect(cv::Point(x0, y0), cv::Point(x1, y1));
}


// 主绘图函数
void DrawDetections(cv::Mat& img, const std::vector<Detection>& objects)
{
    for (const auto& object : objects)
    {
        cv::Rect box = object.box;
        if (object.has_letterbox)
        {
            scale_coords_back(object.letterbox_info, 640, 640, img.cols, img.rows, box);
        }

        cv::rectangle(img, box, object.color, 2);
        std::ostringstream oss;
        oss << object.className << " " << std::fixed << std::setprecision(2) << object.confidence;
        cv::putText(img, oss.str(), cv::Point(box.x, std::max(0, box.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, object.color, 2);
    }
}
 */

/*  //3d
 #include <opencv2/opencv.hpp>
#include <iomanip>
#include <sstream>
#include "types/yolo_datatype.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"
#include "process/preprocess.h"

// 将 letterbox 缩放后的框还原到原图坐标
void scale_coords_back(const LetterBoxInfo& info, int net_input_w, int net_input_h,
    int origin_img_w, int origin_img_h, cv::Rect& box)
{
    float scale = info.scale;
    int pad_w = info.pad_w;
    int pad_h = info.pad_h;

    // 转 float 计算更精确
    float x = (box.x - pad_w) / scale;
    float y = (box.y - pad_h) / scale;
    float w = box.width / scale;
    float h = box.height / scale;

    // 限制边界，避免越界访问
    int x0 = std::max(int(x), 0);
    int y0 = std::max(int(y), 0);
    int x1 = std::min(int(x + w), origin_img_w);
    int y1 = std::min(int(y + h), origin_img_h);

    box = cv::Rect(cv::Point(x0, y0), cv::Point(x1, y1));
}

// 主绘图函数（支持 XYZ 坐标显示）
void DrawDetections(cv::Mat& img, const std::vector<Detection>& objects)
{
    for (const auto& object : objects)
    {
        cv::Rect box = object.box;
        if (object.has_letterbox)
        {
            scale_coords_back(object.letterbox_info, 640, 640, img.cols, img.rows, box);
        }

        cv::rectangle(img, box, object.color, 2);

        std::ostringstream oss;
        oss << object.className << " " << std::fixed << std::setprecision(2) << object.confidence;
        if (object.has_xyz)
        {
            oss << " | X:" << object.x << " Y:" << object.y << " Z:" << object.z;
        }

        cv::putText(img, oss.str(), cv::Point(box.x, std::max(0, box.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, object.color, 2);
    }
}
 */

 //3d
 // 这是一个用户添加的注释，可能用来标记该文件与3D功能相关

 // 引入OpenCV主头文件，它包含了所有核心和常用功能
 #include <opencv2/opencv.hpp>
// 引入 C++ 标准库头文件 <iomanip>，用于格式化输入输出流（例如设置精度）
#include <iomanip>
// 引入 C++ 标准库头文件 <sstream>，用于字符串流操作，方便构建字符串
#include <sstream>
// 引入自定义头文件 "yolo_datatype.h"，该文件可能定义了YOLO检测结果相关的数据结构（如 Detection）
#include "types/yolo_datatype.h"
// 引入自定义头文件 "logging.h"，该文件可能提供了日志记录功能
#include "utils/logging.h"
// 引入自定义头文件 "cv_draw.h"，该文件可能包含了在此处实现的绘图函数的声明
#include "draw/cv_draw.h"
// 引入自定义头文件 "preprocess.h"，该文件可能定义了图像预处理相关的函数和数据结构（如 LetterBoxInfo）
#include "process/preprocess.h"

// 函数功能：将经过 letterbox 预处理后的边界框坐标还原到原始图像的坐标系中
// info: 包含缩放比例和填充信息的 LetterBoxInfo 结构体
// net_input_w: 神经网络的输入宽度
// net_input_h: 神经网络的输入高度
// origin_img_w: 原始图像的宽度
// origin_img_h: 原始图像的高度
// box: 需要被转换的边界框，函数将直接修改这个对象
void scale_coords_back(const LetterBoxInfo& info, int net_input_w, int net_input_h,
    int origin_img_w, int origin_img_h, cv::Rect& box)
{
    // 从 info 结构体中获取缩放比例
    float scale = info.scale;
    // 从 info 结构体中获取宽度方向上的填充值
    int pad_w = info.pad_w;
    // 从 info 结构体中获取高度方向上的填充值
    int pad_h = info.pad_h;

    // 转换为浮点数进行计算可以保证更高的精度
    // 计算还原后的 x 坐标：先减去填充，再除以缩放比例
    float x = (box.x - pad_w) / scale;
    // 计算还原后的 y 坐标：先减去填充，再除以缩放比例
    float y = (box.y - pad_h) / scale;
    // 计算还原后的宽度：直接除以缩放比例
    float w = box.width / scale;
    // 计算还原后的高度：直接除以缩放比例
    float h = box.height / scale;

    // 限制边界框的坐标，确保它不会超出原始图像的范围，防止越界访问
    // 计算框的左上角 x 坐标，并确保其不小于 0
    int x0 = std::max(int(x), 0);
    // 计算框的左上角 y 坐标，并确保其不小于 0
    int y0 = std::max(int(y), 0);
    // 计算框的右下角 x 坐标，并确保其不超过原始图像的宽度
    int x1 = std::min(int(x + w), origin_img_w);
    // 计算框的右下角 y 坐标，并确保其不超过原始图像的高度
    int y1 = std::min(int(y + h), origin_img_h);

    // 使用计算出的两个角点 (x0, y0) 和 (x1, y1) 创建一个新的 cv::Rect 对象，并更新传入的 box 引用
    box = cv::Rect(cv::Point(x0, y0), cv::Point(x1, y1));
}

// 函数功能：在图像上绘制检测到的目标框和信息（标签、置信度、可选的XYZ坐标）
// img: 需要进行绘制的目标图像
// objects: 一个包含所有检测结果的 std::vector 容器，每个结果是一个 Detection 结构体
void DrawDetections(cv::Mat& img, const std::vector<Detection>& objects)
{
    // 使用 for-each 循环遍历 `objects` 容器中的每一个检测到的目标 `object`
    for (const auto& object : objects)
    {
        // 复制当前目标的边界框信息到一个局部变量 box
        cv::Rect box = object.box;
        // 检查该目标的坐标是否是基于 letterbox 预处理后的图像
        if (object.has_letterbox)
        {
            // 如果是，则调用 scale_coords_back 函数将坐标从网络输入尺寸（这里硬编码为640x640）还原到原始图像尺寸
            scale_coords_back(object.letterbox_info, 640, 640, img.cols, img.rows, box);
        }

        // 使用OpenCV的 rectangle 函数在图像上绘制边界框
        // img: 目标图像
        // box: 边界框的位置和大小
        // object.color: 边界框的颜色
        // 2: 边界框线条的粗细
        cv::rectangle(img, box, object.color, 2);

        // 创建一个字符串流对象 oss，用于方便地构建要显示的文本标签
        std::ostringstream oss;
        // 将类别名称、空格、以及固定两位小数精度的置信度写入字符串流
        oss << object.className << " " << std::fixed << std::setprecision(2) << object.confidence;
        // 检查该目标是否包含三维坐标信息 (X, Y, Z)
        if (object.has_xyz)
        {
            // 如果有，则将XYZ坐标信息追加到字符串流中
            oss << " | X:" << object.x << " Y:" << object.y << " Z:" << object.z;
        }

        // 使用OpenCV的 putText 函数将构建好的文本标签绘制到图像上
        // img: 目标图像
        // oss.str(): 要绘制的完整字符串
        // cv::Point(box.x, std::max(0, box.y - 5)): 文本的起始绘制位置（左下角），通常在框的左上角上方5个像素处。std::max确保文本不会绘制到图像区域之外。
        // cv::FONT_HERSHEY_SIMPLEX: 字体类型
        // 0.6: 字体大小（缩放比例）
        // object.color: 字体颜色
        // 2: 字体线条的粗细
        cv::putText(img, oss.str(), cv::Point(box.x, std::max(0, box.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, object.color, 2);
    }
}