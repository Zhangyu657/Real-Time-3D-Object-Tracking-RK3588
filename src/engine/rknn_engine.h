// 继承自NNEngine，实现NNEngine的接口

/* #ifndef RK3588_DEMO_RKNN_ENGINE_H
#define RK3588_DEMO_RKNN_ENGINE_H

#include "engine.h"

#include <vector>

#include <rknn_api.h>

// 继承自NNEngine，实现NNEngine的接口
class RKEngine : public NNEngine
{
public:
    RKEngine() : rknn_ctx_(0), ctx_created_(false), input_num_(0), output_num_(0){}; // 构造函数，初始化
    ~RKEngine() override;                                                            // 析构函数

    nn_error_e LoadModelFile(const char *model_file) override;                                                         // 加载模型文件
    const std::vector<tensor_attr_s> &GetInputShapes() override;                                                       // 获取输入张量的形状
    const std::vector<tensor_attr_s> &GetOutputShapes() override;                                                      // 获取输出张量的形状
    nn_error_e Run(std::vector<tensor_data_s> &inputs, std::vector<tensor_data_s> &outputs, bool want_float) override; // 运行模型

private:
    // rknn context
    rknn_context rknn_ctx_; // rknn context
    bool ctx_created_;      // rknn context是否创建

    uint32_t input_num_;  // 输入的数量
    uint32_t output_num_; // 输出的数量

    std::vector<tensor_attr_s> in_shapes_;  // 输入张量的形状
    std::vector<tensor_attr_s> out_shapes_; // 输出张量的形状
};

#endif // RK3588_DEMO_RKNN_ENGINE_H
 */
 // 这是 "Include Guard"（头文件保护宏），防止本头文件在一次编译中被多次包含
#ifndef RK3588_DEMO_RKNN_ENGINE_H
// 定义宏 RK3588_DEMO_RKNN_ENGINE_H，与上面的 #ifndef 配对使用
#define RK3588_DEMO_RKNN_ENGINE_H

// 引入 "engine.h" 头文件，该文件定义了 NNEngine 抽象基类
#include "engine.h"

// 引入 C++ 标准库的 <vector> 头文件，以使用 std::vector
#include <vector>

// 引入 Rockchip 官方提供的 RKNN C API 头文件
#include <rknn_api.h>

// 这个类继承自 NNEngine，是 NNEngine 接口的一个具体实现，专门用于 RKNN
class RKEngine : public NNEngine
{
// public 访问修饰符，表示接下来的成员（函数、变量）可以在类外部被访问
public:
    // 构造函数：使用成员初始化列表对私有成员变量进行初始化
    RKEngine() : rknn_ctx_(0), ctx_created_(false), input_num_(0), output_num_(0){};
    // 析构函数：使用 override 关键字明确表示它覆盖了基类的虚析构函数
    ~RKEngine() override;

    // 覆盖基类的 LoadModelFile 纯虚函数，用于加载 RKNN 模型文件
    nn_error_e LoadModelFile(const char *model_file) override;
    // 覆盖基类的 GetInputShapes 纯虚函数，用于获取模型输入张量的属性
    const std::vector<tensor_attr_s> &GetInputShapes() override;
    // 覆盖基类的 GetOutputShapes 纯虚函数，用于获取模型输出张量的属性
    const std::vector<tensor_attr_s> &GetOutputShapes() override;
    // 覆盖基类的 Run 纯虚函数，用于执行模型推理
    nn_error_e Run(std::vector<tensor_data_s> &inputs, std::vector<tensor_data_s> &outputs, bool want_float) override;

// private 访问修饰符，表示接下来的成员只能被本类的成员函数访问
private:
    // rknn context 注释块
    // 定义一个 rknn_context 类型的成员变量，用于保存 RKNN 的上下文句柄
    rknn_context rknn_ctx_;
    // 定义一个布尔型标志，用于记录 rknn_ctx_ 是否已成功创建
    bool ctx_created_;

    // 定义一个无符号32位整型变量，用于存储模型输入张量的数量
    uint32_t input_num_;
    // 定义一个无符号32位整型变量，用于存储模型输出张量的数量
    uint32_t output_num_;

    // 定义一个 std::vector，用于存储所有输入张量的属性（如维度、类型等）
    std::vector<tensor_attr_s> in_shapes_;
    // 定义一个 std::vector，用于存储所有输出张量的属性
    std::vector<tensor_attr_s> out_shapes_;
// 类定义结束
};

// 结束 #ifndef RK3588_DEMO_RKNN_ENGINE_H 定义的条件编译块
#endif // RK3588_DEMO_RKNN_ENGINE_H