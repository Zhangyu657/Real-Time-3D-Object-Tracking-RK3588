// 接口定义

/* #ifndef RK3588_DEMO_ENGINE_H
#define RK3588_DEMO_ENGINE_H

#include "types/error.h"
#include "types/datatype.h"

#include <vector>
#include <memory>

class NNEngine
{
public:
    // 这里全部使用纯虚函数（=0），作用是将NNEngine定义为一个抽象类，不能实例化，只能作为基类使用
    // 具体实现需要在子类中实现，这里的实现只是为了定义接口
    // 用这种方式实现封装，可以使得不同的引擎的接口一致，方便使用；也可以隐藏不同引擎的实现细节，方便维护
    virtual ~NNEngine(){};                                                                                               // 析构函数
    virtual nn_error_e LoadModelFile(const char *model_file) = 0;                                                        // 加载模型文件，=0表示纯虚函数，必须在子类中实现
    virtual const std::vector<tensor_attr_s> &GetInputShapes() = 0;                                                      // 获取输入张量的形状
    virtual const std::vector<tensor_attr_s> &GetOutputShapes() = 0;                                                     // 获取输出张量的形状
    virtual nn_error_e Run(std::vector<tensor_data_s> &inputs, std::vector<tensor_data_s> &outpus, bool want_float) = 0; // 运行模型
    
};

std::shared_ptr<NNEngine> CreateRKNNEngine(); // 创建RKNN引擎

#endif // RK3588_DEMO_ENGINE_H
 */

 // 这一行是 "Include Guard"（头文件保护宏），用于防止此头文件在一次编译中被重复包含
#ifndef RK3588_DEMO_ENGINE_H
// 如果宏 RK3588_DEMO_ENGINE_H 未被定义，则定义它。这与 #ifndef 指令配对使用
#define RK3588_DEMO_ENGINE_H

// 引入自定义的错误码类型头文件，可能定义了 nn_error_e 枚举
#include "types/error.h"
// 引入自定义的数据类型头文件，可能定义了 tensor_attr_s, tensor_data_s 等结构体
#include "types/datatype.h"

// 引入 C++ 标准库中的 <vector> 头文件，以使用 std::vector 容器
#include <vector>
// 引入 C++ 标准库中的 <memory> 头文件，以使用 std::shared_ptr 等智能指针
#include <memory>

// 定义一个名为 NNEngine 的类，它将作为所有神经网络推理引擎的基类接口
class NNEngine
{
// public 访问修饰符，表示接下来的成员（函数、变量）可以在类外部被访问
public:
    // 这里全部使用纯虚函数（=0），作用是将 NNEngine 定义为一个抽象类（接口类），它不能被直接实例化，只能作为基类被继承
    // 具体的功能必须在派生类中实现，基类中只定义派生类需要遵循的接口规范
    // 通过这种方式实现封装，可以统一不同引擎的调用接口，方便切换和使用；同时也能隐藏不同引擎的内部实现细节，便于代码维护

    // 声明一个虚析构函数。当通过基类指针删除派生类对象时，这能确保派生类的析构函数被正确调用
    virtual ~NNEngine(){};

    // 声明一个纯虚函数 LoadModelFile，用于加载模型文件。`=0` 表示这个函数没有实现，必须在派生类中提供具体实现
    virtual nn_error_e LoadModelFile(const char *model_file) = 0;

    // 声明一个纯虚函数 GetInputShapes，用于获取模型输入张量的属性（如形状、数据类型等）。返回一个常量的引用，避免不必要的数据拷贝
    virtual const std::vector<tensor_attr_s> &GetInputShapes() = 0;

    // 声明一个纯虚函数 GetOutputShapes，用于获取模型输出张量的属性
    virtual const std::vector<tensor_attr_s> &GetOutputShapes() = 0;

    // 声明一个纯虚函数 Run，用于执行模型推理。它接收输入数据，并填充输出数据
    virtual nn_error_e Run(std::vector<tensor_data_s> &inputs, std::vector<tensor_data_s> &outpus, bool want_float) = 0;

// 类定义结束
};

// 声明一个工厂函数 CreateRKNNEngine。该函数用于创建一个具体的 RKNN 引擎实例，并返回一个指向 NNEngine 基类接口的共享指针 (std::shared_ptr)
std::shared_ptr<NNEngine> CreateRKNNEngine();

// 结束 #ifndef RK3588_DEMO_ENGINE_H 定义的条件编译块
#endif // RK3588_DEMO_ENGINE_H