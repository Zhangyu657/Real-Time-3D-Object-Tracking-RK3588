/* // 定义数据类型

#ifndef RK3588_DEMO_DATATYPE_H
#define RK3588_DEMO_DATATYPE_H

#include <stdint.h>
#include <stdlib.h>

#include "utils/logging.h"
#include "types/error.h"

typedef enum _tensor_layout
{
    NN_TENSORT_LAYOUT_UNKNOWN = 0,
    NN_TENSOR_NCHW = 1,
    NN_TENSOR_NHWC = 2,
    NN_TENSOR_OTHER = 3,
} tensor_layout_e;

typedef enum _tensor_datatype
{
    NN_TENSOR_INT8 = 1,
    NN_TENSOR_UINT8 = 2,
    NN_TENSOR_FLOAT = 3,
    NN_TENSOR_FLOAT16 = 4,
} tensor_datatype_e;

static const int g_max_num_dims = 4;


typedef struct
{
    uint32_t index;
    uint32_t n_dims;
    uint32_t dims[g_max_num_dims];
    uint32_t n_elems;
    uint32_t size;
    tensor_datatype_e type;
    tensor_layout_e layout;
    int32_t zp;
    float scale;
} tensor_attr_s;

typedef struct
{
    tensor_attr_s attr;
    void *data;
} tensor_data_s;



static size_t nn_tensor_type_to_size(tensor_datatype_e type)
{
    switch (type)
    {
    case NN_TENSOR_INT8:
        return sizeof(int8_t);
    case NN_TENSOR_UINT8:
        return sizeof(uint8_t);
    case NN_TENSOR_FLOAT:
        return sizeof(float);
    case NN_TENSOR_FLOAT16:
        return sizeof(uint16_t);
    default:
        NN_LOG_ERROR("unsupported tensor type");
        exit(-1);
    }
}

static void nn_tensor_attr_to_cvimg_input_data(const tensor_attr_s &attr, tensor_data_s &data)
{
    if (attr.n_dims != 4)
    {
        NN_LOG_ERROR("unsupported input dims");
        exit(-1);
    }
    data.attr.n_dims = attr.n_dims;
    data.attr.index = 0;
    data.attr.type = NN_TENSOR_UINT8;
    data.attr.layout = NN_TENSOR_NHWC;
    if (attr.layout == NN_TENSOR_NCHW)
    {
        data.attr.dims[0] = attr.dims[0];
        data.attr.dims[1] = attr.dims[2];
        data.attr.dims[2] = attr.dims[3];
        data.attr.dims[3] = attr.dims[1];
    }
    else if (attr.layout == NN_TENSOR_NHWC)
    {
        data.attr.dims[0] = attr.dims[0];
        data.attr.dims[1] = attr.dims[1];
        data.attr.dims[2] = attr.dims[2];
        data.attr.dims[3] = attr.dims[3];
    }
    else
    {
        NN_LOG_ERROR("unsupported input layout");
        exit(-1);
    }
    // multiply all dims
    data.attr.n_elems = data.attr.dims[0] * data.attr.dims[1] *
                        data.attr.dims[2] * data.attr.dims[3];
    data.attr.size = data.attr.n_elems * sizeof(uint8_t);
}

#endif // RK3588_DEMO_DATATYPE_H
 */


 // 这是一个文件顶部的注释，说明本文件的作用是定义数据类型
// 定义数据类型

// 这是 "Include Guard"（头文件保护宏），用于防止本头文件在一次编译中被重复包含
#ifndef RK3588_DEMO_DATATYPE_H
// 如果宏 RK3588_DEMO_DATATYPE_H 未被定义，则定义它，与 #ifndef 配对使用
#define RK3588_DEMO_DATATYPE_H

// 引入 C 标准库头文件 <stdint.h>，以使用 int8_t, uint8_t, uint32_t 等固定宽度的整数类型
#include <stdint.h>
// 引入 C 标准库头文件 <stdlib.h>，以使用 exit() 等函数
#include <stdlib.h>

// 引入自定义的日志工具头文件，用于打印日志信息
#include "utils/logging.h"
// 引入自定义的错误码头文件，其中可能定义了 nn_error_e 等
#include "types/error.h"

// 定义一个枚举类型，用于表示张量的数据布局（维度顺序）
typedef enum _tensor_layout
{
    NN_TENSORT_LAYOUT_UNKNOWN = 0, // 未知布局
    NN_TENSOR_NCHW = 1,            // NCHW 布局 (批次数, 通道数, 高度, 宽度)
    NN_TENSOR_NHWC = 2,            // NHWC 布局 (批次数, 高度, 宽度, 通道数)
    NN_TENSOR_OTHER = 3,           // 其他布局
} tensor_layout_e;                 // 枚举类型别名

// 定义一个枚举类型，用于表示张量元素的数据类型
typedef enum _tensor_datatype
{
    NN_TENSOR_INT8 = 1,    // 8位有符号整型
    NN_TENSOR_UINT8 = 2,   // 8位无符号整型
    NN_TENSOR_FLOAT = 3,   // 32位浮点型
    NN_TENSOR_FLOAT16 = 4, // 16位浮点型 (通常用 uint16_t 存储)
} tensor_datatype_e;       // 枚举类型别名

// 定义一个静态常量，表示张量的最大维度数，这里设为4 (N, C, H, W)
static const int g_max_num_dims = 4;


// 定义一个结构体，用于描述张量的所有属性（元数据）
typedef struct
{
    uint32_t index;             // 张量的索引号
    uint32_t n_dims;            // 张量的维度数量
    uint32_t dims[g_max_num_dims]; // 存储每个维度大小的数组
    uint32_t n_elems;           // 张量的总元素数量
    uint32_t size;              // 张量数据所占的总字节大小
    tensor_datatype_e type;     // 张量的数据类型
    tensor_layout_e layout;     // 张量的数据布局
    int32_t zp;                 // 量化参数：零点 (zero-point)
    float scale;                // 量化参数：缩放因子 (scale)
} tensor_attr_s;                // 结构体类型别名

// 定义一个结构体，用于将张量属性和张量数据指针结合在一起
typedef struct
{
    tensor_attr_s attr; // 张量的属性
    void *data;         // 指向张量实际数据内存的 void 指针
} tensor_data_s;        // 结构体类型别名


// 定义一个静态辅助函数，根据张量数据类型枚举返回其对应的字节大小
static size_t nn_tensor_type_to_size(tensor_datatype_e type)
{
    // 使用 switch 语句判断数据类型
    switch (type)
    {
    case NN_TENSOR_INT8:    // 如果是 int8
        return sizeof(int8_t); // 返回 int8_t 的大小
    case NN_TENSOR_UINT8:   // 如果是 uint8
        return sizeof(uint8_t); // 返回 uint8_t 的大小
    case NN_TENSOR_FLOAT:   // 如果是 float
        return sizeof(float);  // 返回 float 的大小
    case NN_TENSOR_FLOAT16: // 如果是 float16
        return sizeof(uint16_t); // 返回 uint16_t 的大小 (通常 float16 在内存中用16位整数表示)
    default:                // 如果是其他不支持的类型
        NN_LOG_ERROR("unsupported tensor type"); // 记录错误日志
        exit(-1);           // 异常退出程序
    }
}

// 定义一个静态辅助函数，将从模型获取的输入张量属性（attr）转换/适配为适合OpenCV图像输入的张量属性（填充到data中）
static void nn_tensor_attr_to_cvimg_input_data(const tensor_attr_s &attr, tensor_data_s &data)
{
    // 检查模型输入的维度数是否为4，如果不是则不支持
    if (attr.n_dims != 4)
    {
        // 记录错误日志
        NN_LOG_ERROR("unsupported input dims");
        // 异常退出
        exit(-1);
    }
    // 复制维度数
    data.attr.n_dims = attr.n_dims;
    // 设置输入索引为0
    data.attr.index = 0;
    // 将输入数据的类型固定为 UINT8，因为OpenCV图像数据通常是这个类型
    data.attr.type = NN_TENSOR_UINT8;
    // 将输入数据的布局固定为 NHWC，这是OpenCV图像在内存中的常见布局
    data.attr.layout = NN_TENSOR_NHWC;
    // 根据模型输入的原始布局（NCHW或NHWC）来调整维度顺序
    if (attr.layout == NN_TENSOR_NCHW) // 如果模型需要 NCHW
    {
        data.attr.dims[0] = attr.dims[0]; // N (批次数)
        data.attr.dims[1] = attr.dims[2]; // H (高度)
        data.attr.dims[2] = attr.dims[3]; // W (宽度)
        data.attr.dims[3] = attr.dims[1]; // C (通道数) -> 调整为 NHWC
    }
    else if (attr.layout == NN_TENSOR_NHWC) // 如果模型需要 NHWC
    {
        data.attr.dims[0] = attr.dims[0]; // N
        data.attr.dims[1] = attr.dims[1]; // H
        data.attr.dims[2] = attr.dims[2]; // W
        data.attr.dims[3] = attr.dims[3]; // C -> 顺序保持不变
    }
    else // 如果是其他不支持的布局
    {
        // 记录错误日志
        NN_LOG_ERROR("unsupported input layout");
        // 异常退出
        exit(-1);
    }
    // 根据调整后的维度重新计算总元素数量
    data.attr.n_elems = data.attr.dims[0] * data.attr.dims[1] *
                        data.attr.dims[2] * data.attr.dims[3];
    // 根据总元素数量和数据类型（UINT8）重新计算总字节大小
    data.attr.size = data.attr.n_elems * sizeof(uint8_t);
}

// 结束 #ifndef RK3588_DEMO_DATATYPE_H 定义的条件编译块
#endif // RK3588_DEMO_DATATYPE_H