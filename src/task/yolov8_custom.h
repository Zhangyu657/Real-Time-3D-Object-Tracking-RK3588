/* #ifndef RK3588_DEMO_YOLOV8_CUSTOM_H
#define RK3588_DEMO_YOLOV8_CUSTOM_H

#include "engine/engine.h"

#include <memory>

#include <opencv2/opencv.hpp>
#include "process/preprocess.h"
#include "types/yolo_datatype.h"

class Yolov8Custom
{
public:
    Yolov8Custom();
    ~Yolov8Custom();

    nn_error_e LoadModel(const char *model_path);

    nn_error_e Run(const cv::Mat &img, std::vector<Detection> &objects);

private:
    nn_error_e Preprocess(const cv::Mat &img, const std::string process_type, cv::Mat &image_letterbox);
    nn_error_e Inference();
    nn_error_e Postprocess(const cv::Mat &img, std::vector<Detection> &objects);

    bool ready_;
    LetterBoxInfo letterbox_info_;
    tensor_data_s input_tensor_;
    std::vector<tensor_data_s> output_tensors_;
    bool want_float_;
    std::vector<int32_t> out_zps_;
    std::vector<float> out_scales_;
    std::shared_ptr<NNEngine> engine_;
};

#endif // RK3588_DEMO_YOLOV8_CUSTOM_H */


/* #ifndef RK3588_DEMO_YOLOV8_CUSTOM_H
#define RK3588_DEMO_YOLOV8_CUSTOM_H

#include "engine/engine.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include "process/preprocess.h"
#include "types/yolo_datatype.h"

class Yolov8Custom
{
public:
    Yolov8Custom();
    ~Yolov8Custom();

    nn_error_e LoadModel(const char *model_path);
    nn_error_e Run(const cv::Mat &img, std::vector<Detection> &objects);

private:
    nn_error_e Preprocess(const cv::Mat &img, const std::string process_type, cv::Mat &image_letterbox);
    nn_error_e Inference();
    nn_error_e Postprocess(const cv::Mat &img, std::vector<Detection> &objects);

    bool ready_;
    bool want_float_;
    LetterBoxInfo letterbox_info_;
    
    tensor_data_s input_tensor_;
    tensor_data_s output_tensor_;  // ✅ 单个输出 tensor
    int32_t out_zp_ = 0;           // ✅ 单个量化偏移
    float out_scale_ = 1.0f;       // ✅ 单个量化比例

    std::shared_ptr<NNEngine> engine_;
};

#endif // RK3588_DEMO_YOLOV8_CUSTOM_H
 */

 /* #ifndef RK3588_DEMO_YOLOV8_CUSTOM_H
 #define RK3588_DEMO_YOLOV8_CUSTOM_H
 
 #include "engine/engine.h"
 #include <memory>
 #include <opencv2/opencv.hpp>
 #include "process/preprocess.h"
 #include "types/yolo_datatype.h"
 #include <librealsense2/rs.hpp>  // 添加 RealSense 支持
 

 class Yolov8Custom
 {
 public:
     Yolov8Custom();
     ~Yolov8Custom();
 
     
     nn_error_e LoadModel(const char *model_path);
 
     nn_error_e Run(const cv::Mat &img, std::vector<Detection> &objects, LetterBoxInfo &info);
     nn_error_e Run(const cv::Mat &img, std::vector<Detection> &objects);
 
     // ✅ 新增：可传入深度帧与相机内参用于坐标计算
     void SetDepthContext(const rs2::depth_frame& depth, const rs2_intrinsics& intrin)
     {
         depth_frame_ = depth;
         intrinsics_ = intrin;
         has_depth_ = true;
     }
 
 private:
    
     nn_error_e Preprocess(const cv::Mat &img, const std::string process_type, cv::Mat &image_letterbox);
 
   
     nn_error_e Inference();
 
    
     nn_error_e Postprocess(const cv::Mat &img_letterbox, std::vector<Detection> &objects);
 
 private:
     bool ready_ = false;
     bool want_float_ = false;               // 是否为 float 模型输出
     LetterBoxInfo letterbox_info_;         // 用于后处理中的坐标还原
 
     tensor_data_s input_tensor_;
     tensor_data_s output_tensor_;
 
     int32_t out_zp_ = 0;                   // int8 zero-point
     float out_scale_ = 1.0f;               // int8 scale
 
     std::shared_ptr<NNEngine> engine_;     // 推理引擎
 
     // ✅ 深度帧上下文（用于计算 3D 坐标）
     rs2::depth_frame depth_frame_;
     rs2_intrinsics intrinsics_{};
     bool has_depth_ = false;
 };
 
 #endif // RK3588_DEMO_YOLOV8_CUSTOM_H
  */

 /*  #ifndef RK3588_DEMO_YOLOV8_CUSTOM_H
#define RK3588_DEMO_YOLOV8_CUSTOM_H

#include "engine/engine.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include "process/preprocess.h"
#include "types/yolo_datatype.h"
#include <librealsense2/rs.hpp>  // ✅ RealSense 支持


class Yolov8Custom
{
public:
    Yolov8Custom();
    ~Yolov8Custom();

  
    nn_error_e LoadModel(const char *model_path);

    nn_error_e Run(const cv::Mat &img, std::vector<Detection> &objects);

    nn_error_e Run(const cv::Mat &img, std::vector<Detection> &objects, LetterBoxInfo &info);

    void SetDepthContext(const rs2::depth_frame &depth, const rs2_intrinsics &intrin)
    {
        depth_frame_ = depth;
        intrinsics_ = intrin;
        has_depth_ = true;
    }

    const rs2::depth_frame &GetDepth() const { return depth_frame_; }

   
    const rs2_intrinsics &GetIntrinsics() const { return intrinsics_; }

private:
    // 模型执行前的图像预处理
    nn_error_e Preprocess(const cv::Mat &img, const std::string process_type, cv::Mat &image_letterbox);

    // 执行模型推理
    nn_error_e Inference();

    // 模型输出后处理（坐标映射等）
    nn_error_e Postprocess(const cv::Mat &img_letterbox, std::vector<Detection> &objects);

private:
    bool ready_ = false;
    bool want_float_ = false;

    LetterBoxInfo letterbox_info_;          // Letterbox 映射信息
    tensor_data_s input_tensor_;
    tensor_data_s output_tensor_;
    std::optional<rs2::depth_frame> depth_frame_;  // ✅ 正确放在类的 private 部分
    int32_t out_zp_ = 0;                    // INT8 模型 zero-point
    float out_scale_ = 1.0f;                // INT8 模型 scale

    std::shared_ptr<NNEngine> engine_;      // 底层 RKNN 推理引擎

    // ✅ RealSense 深度上下文
    rs2::depth_frame depth_frame_;
    rs2_intrinsics intrinsics_{};
    bool has_depth_ = false;
};

#endif // RK3588_DEMO_YOLOV8_CUSTOM_H
 */


 /* #ifndef RK3588_DEMO_YOLOV8_CUSTOM_H
#define RK3588_DEMO_YOLOV8_CUSTOM_H

#include "engine/engine.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include "process/preprocess.h"
#include "types/yolo_datatype.h"
#include <librealsense2/rs.hpp>
#include <optional>


class Yolov8Custom
{
public:
    Yolov8Custom();
    ~Yolov8Custom();

    nn_error_e LoadModel(const char *model_path);
    nn_error_e Run(const cv::Mat &img, std::vector<Detection> &objects, LetterBoxInfo &info);
    nn_error_e Run(const cv::Mat &img, std::vector<Detection> &objects);

    // ✅ 设置深度帧和相机内参
    void SetDepthContext(const rs2::depth_frame& depth, const rs2_intrinsics& intrin)
    {
        depth_frame_ = depth;
        intrinsics_ = intrin;
        has_depth_ = true;
    }

private:
    nn_error_e Preprocess(const cv::Mat &img, const std::string process_type, cv::Mat &image_letterbox);
    nn_error_e Inference();
    nn_error_e Postprocess(const cv::Mat &img_letterbox, std::vector<Detection> &objects);

private:
    bool ready_ = false;
    bool want_float_ = false;
    LetterBoxInfo letterbox_info_;
    tensor_data_s input_tensor_;
    tensor_data_s output_tensor_;
    int32_t out_zp_ = 0;
    float out_scale_ = 1.0f;
    std::shared_ptr<NNEngine> engine_;

    // ✅ 深度帧支持
    std::optional<rs2::depth_frame> depth_frame_;
    rs2_intrinsics intrinsics_{};
    bool has_depth_ = false;
};

#endif // RK3588_DEMO_YOLOV8_CUSTOM_H
 */

 // 这是 "Include Guard"（头文件保护宏），用于防止本头文件在一次编译中被重复包含
#ifndef RK3588_DEMO_YOLOV8_CUSTOM_H
// 如果宏 RK3588_DEMO_YOLOV8_CUSTOM_H 未被定义，则定义它，与 #ifndef 配对使用
#define RK3588_DEMO_YOLOV8_CUSTOM_H

// 引入 "engine.h" 头文件，其中定义了推理引擎的基类接口 NNEngine
#include "engine/engine.h"
// 引入 C++ 标准库头文件 <memory>，以使用 std::shared_ptr 等智能指针
#include <memory>
// 引入 OpenCV 主头文件，以使用 cv::Mat 等图像处理功能
#include <opencv2/opencv.hpp>
// 引入自定义的预处理头文件，其中声明了 letterbox 等函数和 LetterBoxInfo 结构体
#include "process/preprocess.h"
// 引入自定义的YOLO数据类型头文件，其中定义了 Detection 等结构体
#include "types/yolo_datatype.h"
// 引入 Intel RealSense SDK 的主头文件，以支持深度相机
#include <librealsense2/rs.hpp>
// 引入 C++ 标准库头文件 <optional>，用于处理可能存在也可能不存在的值（如此处的深度帧）
#include <optional>

/**
 * @brief 这是一个Doxygen风格的注释块，描述下方类的功能：
 * 自定义 YOLOv8 推理类，适配 RKNN float32 / float16 / int8 单输出模型结构
 */
// 定义一个名为 Yolov8Custom 的类，封装了YOLOv8模型的完整推理流程
class Yolov8Custom
{
// public 访问修饰符，表示接下来的成员（函数、变量）可以在类外部被访问
public:
    // 声明类的构造函数
    Yolov8Custom();
    // 声明类的析构函数
    ~Yolov8Custom();

    // 声明加载模型文件的方法
    nn_error_e LoadModel(const char *model_path);
    // 声明运行检测流程的方法，此重载版本会通过参数返回 LetterBox 信息
    nn_error_e Run(const cv::Mat &img, std::vector<Detection> &objects, LetterBoxInfo &info);
    // 声明运行检测流程的方法的另一个重载版本
    nn_error_e Run(const cv::Mat &img, std::vector<Detection> &objects);

    // 这是一个已有的注释，说明下方函数的功能
    // ✅ 设置深度帧和相机内参
    // 定义一个内联（inline）成员函数，用于设置深度相机的数据和参数
    void SetDepthContext(const rs2::depth_frame& depth, const rs2_intrinsics& intrin)
    {
        // 将传入的深度帧对象赋值给成员变量 depth_frame_
        depth_frame_ = depth;
        // 将传入的相机内参对象赋值给成员变量 intrinsics_
        intrinsics_ = intrin;
        // 设置标志位，表示当前已有深度信息
        has_depth_ = true;
    }

// private 访问修饰符，表示接下来的成员只能被本类的成员函数访问
private:
    // 声明私有的图像预处理方法
    nn_error_e Preprocess(const cv::Mat &img, const std::string process_type, cv::Mat &image_letterbox);
    // 声明私有的模型推理方法
    nn_error_e Inference();
    // 声明私有的后处理方法
    nn_error_e Postprocess(const cv::Mat &img_letterbox, std::vector<Detection> &objects);

// 第二个 private 访问修饰符（与前一个等效），下方是私有成员变量
private:
    // 布尔型标志，表示模型是否已成功加载并准备就绪
    bool ready_ = false;
    // 布尔型标志，表示是否需要将模型输出转换为 float32 类型进行处理
    bool want_float_ = false;
    // 成员变量，用于存储图像预处理（letterbox）的相关信息
    LetterBoxInfo letterbox_info_;
    // 成员变量，用于存储模型的输入张量数据
    tensor_data_s input_tensor_;
    // 成员变量，用于存储模型的输出张量数据
    tensor_data_s output_tensor_;
    // 成员变量，用于存储量化模型的零点（zero-point）参数
    int32_t out_zp_ = 0;
    // 成员变量，用于存储量化模型的缩放因子（scale）参数
    float out_scale_ = 1.0f;
    // 指向推理引擎基类的共享指针，用于管理引擎对象的生命周期
    std::shared_ptr<NNEngine> engine_;

    // 这是一个已有的注释，说明下方的成员变量用于支持深度帧
    // ✅ 深度帧支持
    // 使用 std::optional 包装深度帧对象，因为它可能不存在
    std::optional<rs2::depth_frame> depth_frame_;
    // 成员变量，用于存储深度相机的内参（焦距、主点等）
    rs2_intrinsics intrinsics_{};
    // 布尔型标志，表示当前是否有可用的深度信息
    bool has_depth_ = false;
}; // 类定义结束

// 结束 #ifndef RK3588_DEMO_YOLOV8_CUSTOM_H 定义的条件编译块
#endif // RK3588_DEMO_YOLOV8_CUSTOM_H