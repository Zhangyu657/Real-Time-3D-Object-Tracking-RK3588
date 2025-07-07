/* // 预处理

#include "preprocess.h"

#include "utils/logging.h"
#include "im2d.h"
#include "rga.h"

// opencv 版本的 letterbox
LetterBoxInfo letterbox(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio)
{

    // img has to be 3 channels
    if (img.channels() != 3)
    {
        NN_LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }
    float img_width = img.cols;
    float img_height = img.rows;

    int letterbox_width = 0;
    int letterbox_height = 0;

    LetterBoxInfo info;
    int padding_hor = 0;
    int padding_ver = 0;

    if (img_width / img_height > wh_ratio)
    {
        info.hor = false;
        letterbox_width = img_width;
        letterbox_height = img_width / wh_ratio;
        info.pad = (letterbox_height - img_height) / 2.f;
        padding_hor = 0;
        padding_ver = info.pad;
    }
    else
    {
        info.hor = true;
        letterbox_width = img_height * wh_ratio;
        letterbox_height = img_height;
        info.pad = (letterbox_width - img_width) / 2.f;
        padding_hor = info.pad;
        padding_ver = 0;
    }

    // 使用cv::copyMakeBorder函数进行填充边界
    cv::copyMakeBorder(img, img_letterbox, padding_ver, padding_ver, padding_hor, padding_hor, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return info;
}

// opencv resize
void cvimg2tensor(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor)
{
    // img has to be 3 channels
    if (img.channels() != 3)
    {
        NN_LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }
    // BGR to RGB
    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    // resize img
    cv::Mat img_resized;
    // resize img
    cv::resize(img_rgb, img_resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    // BGR to RGB
    memcpy(tensor.data, img_resized.data, tensor.attr.size);
}

// rga 版本的 resize
void cvimg2tensor_rga(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor)
{
    // img has to be 3 channels
    if (img.channels() != 3)
    {
        NN_LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }

    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));
    rga_buffer_t src = wrapbuffer_virtualaddr((void *)img_rgb.data, img.cols, img.rows, RK_FORMAT_RGB_888);
    rga_buffer_t dst = wrapbuffer_virtualaddr((void *)tensor.data, width, height, RK_FORMAT_RGB_888);
    int ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret)
    {
        NN_LOG_ERROR("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        exit(-1);
    }
    imresize(src, dst);
}

// rga 版本的 letterbox
LetterBoxInfo letterbox_rga(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio)
{
    // img has to be 3 channels
    if (img.channels() != 3)
    {
        NN_LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }
    float img_width = img.cols;
    float img_height = img.rows;

    int letterbox_width = 0;
    int letterbox_height = 0;

    LetterBoxInfo info;
    int padding_hor = 0;
    int padding_ver = 0;

    if (img_width / img_height > wh_ratio)
    {
        info.hor = false;
        letterbox_width = img_width;
        letterbox_height = img_width / wh_ratio;
        info.pad = (letterbox_height - img_height) / 2.f;
        padding_hor = 0;
        padding_ver = info.pad;
    }
    else
    {
        info.hor = true;
        letterbox_width = img_height * wh_ratio;
        letterbox_height = img_height;
        info.pad = (letterbox_width - img_width) / 2.f;
        padding_hor = info.pad;
        padding_ver = 0;
    }
    // rga add border
    img_letterbox = cv::Mat::zeros(letterbox_height, letterbox_width, CV_8UC3);

    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));

    // NN_LOG_INFO("img size: %d, %d", img.cols, img.rows);

    rga_buffer_t src = wrapbuffer_virtualaddr((void *)img.data, img.cols, img.rows, RK_FORMAT_RGB_888);
    rga_buffer_t dst = wrapbuffer_virtualaddr((void *)img_letterbox.data, img_letterbox.cols, img_letterbox.rows, RK_FORMAT_RGB_888);
    int ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret)
    {
        NN_LOG_ERROR("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        exit(-1);
    }

    immakeBorder(src, dst, padding_ver, padding_ver, padding_hor, padding_hor, 0, 0, 0);

    return info;
} */


/* #include "preprocess.h"
#include "utils/logging.h"
#include "im2d.h"
#include "rga.h"


LetterBoxInfo letterbox(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio)
{
    if (img.channels() != 3)
    {
        NN_LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }

    float img_width = img.cols;
    float img_height = img.rows;

    int pad_w = 0, pad_h = 0;
    float scale = std::min(640.0f / img_width, 640.0f / img_height);

    int resized_w = int(round(img_width * scale));
    int resized_h = int(round(img_height * scale));

    pad_w = (640 - resized_w) / 2;
    pad_h = (640 - resized_h) / 2;

    LetterBoxInfo info;
    info.scale = scale;
    info.pad_w = pad_w;
    info.pad_h = pad_h;
    info.input_w = 640;  // 真实的网络输入尺寸
    info.input_h = 640;

    // resize
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resized_w, resized_h));

    // padding
    cv::copyMakeBorder(resized, img_letterbox,
                       pad_h, 640 - resized_h - pad_h,
                       pad_w, 640 - resized_w - pad_w,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return info;
}

// OpenCV resize
void cvimg2tensor(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor)
{
    if (img.channels() != 3)
    {
        NN_LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }
    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

    cv::Mat img_resized;
    cv::resize(img_rgb, img_resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    memcpy(tensor.data, img_resized.data, tensor.attr.size);
}

// RGA resize
void cvimg2tensor_rga(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor)
{
    if (img.channels() != 3)
    {
        NN_LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }

    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));

    rga_buffer_t src = wrapbuffer_virtualaddr((void *)img_rgb.data, img.cols, img.rows, RK_FORMAT_RGB_888);
    rga_buffer_t dst = wrapbuffer_virtualaddr((void *)tensor.data, width, height, RK_FORMAT_RGB_888);

    int ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret)
    {
        NN_LOG_ERROR("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        exit(-1);
    }
    imresize(src, dst);
}

// RGA 版本的 letterbox
LetterBoxInfo letterbox_rga(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio)
{
    if (img.channels() != 3)
    {
        NN_LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }

    float img_width = img.cols;
    float img_height = img.rows;

    int letterbox_width = 0;
    int letterbox_height = 0;
    int padding_hor = 0;
    int padding_ver = 0;

    LetterBoxInfo info;

    if (img_width / img_height > wh_ratio)
    {
        info.hor = false;
        letterbox_width = img_width;
        letterbox_height = img_width / wh_ratio;
        padding_ver = (letterbox_height - img_height) / 2;
        padding_hor = 0;
    }
    else
    {
        info.hor = true;
        letterbox_width = img_height * wh_ratio;
        letterbox_height = img_height;
        padding_hor = (letterbox_width - img_width) / 2;
        padding_ver = 0;
    }

    info.input_w = letterbox_width;
    info.input_h = letterbox_height;
    info.pad_w = padding_hor;
    info.pad_h = padding_ver;
    info.scale = std::min((float)letterbox_width / img_width, (float)letterbox_height / img_height);

    img_letterbox = cv::Mat::zeros(letterbox_height, letterbox_width, CV_8UC3);

    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));

    rga_buffer_t src = wrapbuffer_virtualaddr((void *)img.data, img.cols, img.rows, RK_FORMAT_RGB_888);
    rga_buffer_t dst = wrapbuffer_virtualaddr((void *)img_letterbox.data, img_letterbox.cols, img_letterbox.rows, RK_FORMAT_RGB_888);

    int ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret)
    {
        NN_LOG_ERROR("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        exit(-1);
    }

    immakeBorder(src, dst, padding_ver, padding_ver, padding_hor, padding_hor, 0, 0, 0);
    return info;
}
 */

/*  // preprocess.cpp
#include "preprocess.h"
#include "utils/logging.h"
#include "im2d.h"
#include "rga.h"

// OpenCV 版本的 letterbox（标准化输入尺寸为 640x640）
LetterBoxInfo letterbox(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio)
{
    if (img.channels() != 3)
    {
        NN_LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }

    float img_width = img.cols;
    float img_height = img.rows;

    float scale = std::min(640.0f / img_width, 640.0f / img_height);
    int resized_w = static_cast<int>(round(img_width * scale));
    int resized_h = static_cast<int>(round(img_height * scale));

    int pad_w = (640 - resized_w) / 2;
    int pad_h = (640 - resized_h) / 2;

    LetterBoxInfo info;
    info.scale = scale;
    info.pad_w = pad_w;
    info.pad_h = pad_h;
    info.input_w = 640;
    info.input_h = 640;

    // resize
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resized_w, resized_h));

    // padding
    cv::copyMakeBorder(resized, img_letterbox,
                       pad_h, 640 - resized_h - pad_h,
                       pad_w, 640 - resized_w - pad_w,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return info;
}

// OpenCV resize
void cvimg2tensor(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor)
{
    if (img.channels() != 3)
    {
        NN_LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }
    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

    cv::Mat img_resized;
    cv::resize(img_rgb, img_resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    memcpy(tensor.data, img_resized.data, tensor.attr.size);
}

// RGA resize
void cvimg2tensor_rga(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor)
{
    if (img.channels() != 3)
    {
        NN_LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }

    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));

    rga_buffer_t src = wrapbuffer_virtualaddr((void *)img_rgb.data, img.cols, img.rows, RK_FORMAT_RGB_888);
    rga_buffer_t dst = wrapbuffer_virtualaddr((void *)tensor.data, width, height, RK_FORMAT_RGB_888);

    int ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret)
    {
        NN_LOG_ERROR("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        exit(-1);
    }
    imresize(src, dst);
}
// RGA 版本的 letterbox（标准化输入尺寸为 640x640）
LetterBoxInfo letterbox_rga(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio)
{
    if (img.channels() != 3)
    {
        NN_LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }

    float img_width = img.cols;
    float img_height = img.rows;

    float scale = std::min(640.0f / img_width, 640.0f / img_height);
    int resized_w = static_cast<int>(round(img_width * scale));
    int resized_h = static_cast<int>(round(img_height * scale));

    int pad_w = (640 - resized_w) / 2;
    int pad_h = (640 - resized_h) / 2;

    LetterBoxInfo info;
    info.scale = scale;
    info.pad_w = pad_w;
    info.pad_h = pad_h;
    info.input_w = 640;
    info.input_h = 640;

    img_letterbox = cv::Mat::zeros(640, 640, CV_8UC3);

    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(img_rgb, resized, cv::Size(resized_w, resized_h));

    im_rect src_rect = {0};
    im_rect dst_rect = {0};

    rga_buffer_t src = wrapbuffer_virtualaddr((void *)resized.data, resized.cols, resized.rows, RK_FORMAT_RGB_888);
    rga_buffer_t dst = wrapbuffer_virtualaddr((void *)img_letterbox.data, img_letterbox.cols, img_letterbox.rows, RK_FORMAT_RGB_888);

    int ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret)
    {
        NN_LOG_ERROR("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        exit(-1);
    }

    immakeBorder(src, dst,
                 pad_h, 640 - resized_h - pad_h,
                 pad_w, 640 - resized_w - pad_w,
                 0, 0, 0);

    return info;
}

 */


 // preprocess.cpp

// 引入 "preprocess.h" 头文件，其中应包含了本文件中所实现函数的声明
#include "preprocess.h"
// 引入自定义的日志工具头文件，用于打印日志信息
#include "utils/logging.h"
// 引入 im2d.h 头文件，这是使用 Rockchip RGA (2D图形加速单元) 功能所必需的
#include "im2d.h"
// 引入 rga.h 头文件，同样是 RGA 功能所需的核心头文件
#include "rga.h"

// 这是一个已有的注释，说明下方函数的功能
// OpenCV 版本的 letterbox（标准化输入尺寸为 640x640）
// img: 输入的原始OpenCV图像
// img_letterbox: 函数处理后输出的、经过letterbox操作的图像
// wh_ratio: 宽高比（此函数中未使用，但可能为兼容其他版本而保留）
// 返回值: 包含缩放和平移信息的 LetterBoxInfo 结构体
LetterBoxInfo letterbox(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio)
{
    // 检查输入图像是否为3通道（如 BGR 或 RGB）
    if (img.channels() != 3)
    {
        // 如果不是，则记录错误日志
        NN_LOG_ERROR("img has to be 3 channels");
        // 异常退出程序
        exit(-1);
    }

    // 获取输入图像的宽度
    float img_width = img.cols;
    // 获取输入图像的高度
    float img_height = img.rows;

    // 计算缩放比例，取宽、高方向缩放比例中的较小值，以保持原始宽高比
    float scale = std::min(640.0f / img_width, 640.0f / img_height);
    // 根据缩放比例计算缩放后的新宽度
    int resized_w = static_cast<int>(round(img_width * scale));
    // 根据缩放比例计算缩放后的新高度
    int resized_h = static_cast<int>(round(img_height * scale));

    // 计算宽度方向上需要填充的像素数（黑边），使其居中
    int pad_w = (640 - resized_w) / 2;
    // 计算高度方向上需要填充的像素数（黑边），使其居中
    int pad_h = (640 - resized_h) / 2;

    // 创建一个 LetterBoxInfo 结构体来存储变换信息
    LetterBoxInfo info;
    // 保存缩放比例
    info.scale = scale;
    // 保存宽度方向的填充值
    info.pad_w = pad_w;
    // 保存高度方向的填充值
    info.pad_h = pad_h;
    // 保存目标输入的宽度
    info.input_w = 640;
    // 保存目标输入的高度
    info.input_h = 640;

    // --- 执行图像变换 ---
    // 定义一个 cv::Mat 用于存放缩放后的图像
    cv::Mat resized;
    // 使用 OpenCV 的 resize 函数将原图缩放到计算出的新尺寸
    cv::resize(img, resized, cv::Size(resized_w, resized_h));

    // 使用 OpenCV 的 copyMakeBorder 函数在缩放后的图像周围添加黑边
    cv::copyMakeBorder(resized, img_letterbox,
                       pad_h, 640 - resized_h - pad_h, // 上下边框的高度
                       pad_w, 640 - resized_w - pad_w, // 左右边框的宽度
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0)); // 边框类型：常量；边框颜色：黑
    // 返回包含变换信息的结构体
    return info;
}

// 这是一个已有的注释，说明下方函数的功能
// OpenCV resize
// img: 输入的OpenCV图像
// width, height: 目标张量的宽度和高度
// tensor: 用于接收图像数据的输出张量结构体
void cvimg2tensor(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor)
{
    // 检查输入图像是否为3通道
    if (img.channels() != 3)
    {
        // 记录错误日志
        NN_LOG_ERROR("img has to be 3 channels");
        // 异常退出
        exit(-1);
    }
    // 定义一个 cv::Mat 用于存放颜色空间转换后的图像
    cv::Mat img_rgb;
    // 将 OpenCV 默认的 BGR 格式转换为神经网络常用的 RGB 格式
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

    // 定义一个 cv::Mat 用于存放缩放后的图像
    cv::Mat img_resized;
    // 使用 OpenCV 的 resize 函数将 RGB 图像缩放到指定的宽高
    cv::resize(img_rgb, img_resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    // 使用 memcpy 将缩放后图像的像素数据直接拷贝到张量的数据区
    memcpy(tensor.data, img_resized.data, tensor.attr.size);
}

// 这是一个已有的注释，说明下方函数的功能
// RGA resize
// (参数同上一个函数，但此函数使用RGA硬件加速)
void cvimg2tensor_rga(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor)
{
    // 检查输入图像是否为3通道
    if (img.channels() != 3)
    {
        // 记录错误日志
        NN_LOG_ERROR("img has to be 3 channels");
        // 异常退出
        exit(-1);
    }

    // 定义一个 cv::Mat 用于存放颜色空间转换后的图像
    cv::Mat img_rgb;
    // 将 OpenCV 默认的 BGR 格式转换为 RGB 格式
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

    // 定义 RGA 操作所需的源和目标矩形区域结构体
    im_rect src_rect;
    im_rect dst_rect;
    // 将结构体内存清零
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));

    // 将源图像(img_rgb)的内存地址和属性封装成 RGA 能识别的 rga_buffer_t 结构
    rga_buffer_t src = wrapbuffer_virtualaddr((void *)img_rgb.data, img.cols, img.rows, RK_FORMAT_RGB_888);
    // 将目标张量(tensor)的内存地址和属性封装成 RGA 能识别的 rga_buffer_t 结构
    rga_buffer_t dst = wrapbuffer_virtualaddr((void *)tensor.data, width, height, RK_FORMAT_RGB_888);

    // 检查 RGA 操作的参数是否有效
    int ret = imcheck(src, dst, src_rect, dst_rect);
    // 如果检查返回错误
    if (IM_STATUS_NOERROR != ret)
    {
        // 记录致命错误并退出
        NN_LOG_ERROR("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        exit(-1);
    }
    // 调用 RGA 的 imresize 函数执行硬件加速的图像缩放
    imresize(src, dst);
}

// 这是一个已有的注释，说明下方函数的功能
// RGA 版本的 letterbox（标准化输入尺寸为 640x640）
// (参数和返回值同第一个 letterbox 函数)
LetterBoxInfo letterbox_rga(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio)
{
    // 检查输入图像是否为3通道
    if (img.channels() != 3)
    {
        // 记录错误日志
        NN_LOG_ERROR("img has to be 3 channels");
        // 异常退出
        exit(-1);
    }

    // 获取输入图像的宽度
    float img_width = img.cols;
    // 获取输入图像的高度
    float img_height = img.rows;

    // 计算缩放比例，逻辑同OpenCV版本
    float scale = std::min(640.0f / img_width, 640.0f / img_height);
    // 计算缩放后的新宽度
    int resized_w = static_cast<int>(round(img_width * scale));
    // 计算缩放后的新高度
    int resized_h = static_cast<int>(round(img_height * scale));

    // 计算宽度方向的填充值
    int pad_w = (640 - resized_w) / 2;
    // 计算高度方向的填充值
    int pad_h = (640 - resized_h) / 2;

    // 创建并填充 LetterBoxInfo 结构体
    LetterBoxInfo info;
    info.scale = scale;
    info.pad_w = pad_w;
    info.pad_h = pad_h;
    info.input_w = 640;
    info.input_h = 640;

    // 创建一个 640x640 的全黑图像作为目标 letterbox 图像的背景
    img_letterbox = cv::Mat::zeros(640, 640, CV_8UC3);

    // 定义一个 cv::Mat 用于存放颜色空间转换后的图像
    cv::Mat img_rgb;
    // 将输入图像从 BGR 转换为 RGB
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

    // 定义一个 cv::Mat 用于存放缩放后的图像
    cv::Mat resized;
    // 注意：这里仍然使用 OpenCV 进行缩放，而非 RGA。这可能不是最高效的方式。
    cv::resize(img_rgb, resized, cv::Size(resized_w, resized_h));

    // 定义 RGA 操作所需的源和目标矩形区域，并初始化为0
    im_rect src_rect = {0};
    im_rect dst_rect = {0};

    // 将缩放后的小图(resized)封装为 RGA 的源缓冲
    rga_buffer_t src = wrapbuffer_virtualaddr((void *)resized.data, resized.cols, resized.rows, RK_FORMAT_RGB_888);
    // 将最终的 letterbox 大图封装为 RGA 的目标缓冲
    rga_buffer_t dst = wrapbuffer_virtualaddr((void *)img_letterbox.data, img_letterbox.cols, img_letterbox.rows, RK_FORMAT_RGB_888);

    // 检查 RGA 操作的参数是否有效
    int ret = imcheck(src, dst, src_rect, dst_rect);
    // 如果检查失败
    if (IM_STATUS_NOERROR != ret)
    {
        // 记录致命错误并退出
        NN_LOG_ERROR("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        exit(-1);
    }

    // 调用 RGA 的 immakeBorder 函数，将源图像（小图）拷贝到目标图像（大图）的指定位置，从而实现 padding
    immakeBorder(src, dst,
                 pad_h, 640 - resized_h - pad_h, // 上下边框
                 pad_w, 640 - resized_w - pad_w, // 左右边框
                 0, 0, 0);                       // 边框颜色（R, G, B）

    // 返回包含变换信息的结构体
    return info;
}