// rknn_engine.h的实现

/* #include "rknn_engine.h"

#include <string.h>

#include "utils/engine_helper.h"
#include "utils/logging.h"

static const int g_max_io_num = 10; // 最大输入输出张量的数量

nn_error_e RKEngine::LoadModelFile(const char *model_file)
{
    int model_len = 0;                               // 模型文件大小
    auto model = load_model(model_file, &model_len); // 加载模型文件
    if (model == nullptr)
    {
        NN_LOG_ERROR("load model file %s fail!", model_file);
        return NN_LOAD_MODEL_FAIL; // 返回错误码：加载模型文件失败
    }
    int ret = rknn_init(&rknn_ctx_, model, model_len, 0, NULL); // 初始化rknn context
    if (ret < 0)
    {
        NN_LOG_ERROR("rknn_init fail! ret=%d", ret);
        return NN_RKNN_INIT_FAIL; // 返回错误码：初始化rknn context失败
    }
    // 打印初始化成功信息
    NN_LOG_INFO("rknn_init success!");
    ctx_created_ = true;

    // 获取rknn版本信息
    rknn_sdk_version version;
    ret = rknn_query(rknn_ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        NN_LOG_ERROR("rknn_query fail! ret=%d", ret);
        return NN_RKNN_QUERY_FAIL;
    }
    // 打印rknn版本信息
    NN_LOG_INFO("RKNN API version: %s", version.api_version);
    NN_LOG_INFO("RKNN Driver version: %s", version.drv_version);

    // 获取输入输出个数
    rknn_input_output_num io_num;
    ret = rknn_query(rknn_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        NN_LOG_ERROR("rknn_query fail! ret=%d", ret);
        return NN_RKNN_QUERY_FAIL;
    }
    NN_LOG_INFO("model input num: %d, output num: %d", io_num.n_input, io_num.n_output);

    // 保存输入输出个数
    input_num_ = io_num.n_input;
    output_num_ = io_num.n_output;

    // 输入属性
    NN_LOG_INFO("input tensors:");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(rknn_ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            NN_LOG_ERROR("rknn_query fail! ret=%d", ret);
            return NN_RKNN_QUERY_FAIL;
        }
        print_tensor_attr(&(input_attrs[i]));
        // set input_shapes_
        in_shapes_.push_back(rknn_tensor_attr_convert(input_attrs[i]));
    }

    // 输出属性
    NN_LOG_INFO("output tensors:");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(rknn_ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            NN_LOG_ERROR("rknn_query fail! ret=%d", ret);
            return NN_RKNN_QUERY_FAIL;
        }
        print_tensor_attr(&(output_attrs[i]));
        // set output_shapes_
        out_shapes_.push_back(rknn_tensor_attr_convert(output_attrs[i]));
    }

    return NN_SUCCESS;
}

// 获取输入张量的形状
const std::vector<tensor_attr_s> &RKEngine::GetInputShapes()
{
    return in_shapes_;
}

// 获取输出张量的形状
const std::vector<tensor_attr_s> &RKEngine::GetOutputShapes()
{
    return out_shapes_;
}


nn_error_e RKEngine::Run(std::vector<tensor_data_s> &inputs, std::vector<tensor_data_s> &outputs, bool want_float)
{
    // 检查输入输出张量的数量是否匹配
    if (inputs.size() != input_num_)
    {
        NN_LOG_ERROR("inputs num not match! inputs.size()=%ld, input_num_=%d", inputs.size(), input_num_);
        return NN_IO_NUM_NOT_MATCH;
    }
    if (outputs.size() != output_num_)
    {
        NN_LOG_ERROR("outputs num not match! outputs.size()=%ld, output_num_=%d", outputs.size(), output_num_);
        return NN_IO_NUM_NOT_MATCH;
    }

    // 设置rknn inputs
    rknn_input rknn_inputs[g_max_io_num];
    for (int i = 0; i < inputs.size(); i++)
    {
        // 将自定义的tensor_data_s转换为rknn_input
        rknn_inputs[i] = tensor_data_to_rknn_input(inputs[i]);
    }
    int ret = rknn_inputs_set(rknn_ctx_, (uint32_t)inputs.size(), rknn_inputs);
    if (ret < 0)
    {
        NN_LOG_ERROR("rknn_inputs_set fail! ret=%d", ret);
        return NN_RKNN_INPUT_SET_FAIL;
    }

    // 推理
    NN_LOG_DEBUG("rknn running...");
    ret = rknn_run(rknn_ctx_, nullptr);
    if (ret < 0)
    {
        NN_LOG_ERROR("rknn_run fail! ret=%d", ret);
        return NN_RKNN_RUNTIME_ERROR;
    }

    // 获得输出
    rknn_output rknn_outputs[g_max_io_num];
    memset(rknn_outputs, 0, sizeof(rknn_outputs));
    for (int i = 0; i < output_num_; ++i)
    {
        rknn_outputs[i].want_float = want_float ? 1 : 0;
    }
    ret = rknn_outputs_get(rknn_ctx_, output_num_, rknn_outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        NN_LOG_ERROR("rknn_outputs_get fail! ret=%d", ret);
        return NN_RKNN_OUTPUT_GET_FAIL;
    }

    NN_LOG_DEBUG("output num: %d", output_num_);
    // copy rknn outputs to tensor_data_s
    for (int i = 0; i < output_num_; ++i)
    {
        // 将rknn_output转换为自定义的tensor_data_s
        rknn_output_to_tensor_data(rknn_outputs[i], outputs[i]);
        NN_LOG_DEBUG("output[%d] size=%d", i, outputs[i].attr.size);
        free(rknn_outputs[i].buf); // 释放缓存
    }
    return NN_SUCCESS;
}

// 析构函数
RKEngine::~RKEngine()
{
    if (ctx_created_)
    {
        rknn_destroy(rknn_ctx_);
        NN_LOG_INFO("rknn context destroyed!");
    }
}

// 创建RKNN引擎
std::shared_ptr<NNEngine> CreateRKNNEngine()
{
    return std::make_shared<RKEngine>();
}
 */

 // 引入 rknn_engine.h 头文件，其中包含了 RKEngine 类的声明
#include "rknn_engine.h"

// 引入 C 标准库的 string.h，用于使用 memset 等内存操作函数
#include <string.h>

// 引入自定义的工具类头文件，可能包含 load_model, print_tensor_attr 等辅助函数
#include "utils/engine_helper.h"
// 引入自定义的日志工具头文件，用于打印日志信息
#include "utils/logging.h"

// 定义一个静态常量，表示模型最大支持的输入输出张量数量
static const int g_max_io_num = 10; // 最大输入输出张量的数量

/**
 * @brief 加载模型文件、初始化rknn context、获取rknn版本信息、获取输入输出张量的信息
 * @param model_file 模型文件路径
 * @return nn_error_e 错误码
 */
// 实现 RKEngine 类的 LoadModelFile 方法
nn_error_e RKEngine::LoadModelFile(const char *model_file)
{
    // 定义一个整型变量，用于存储模型文件的大小
    int model_len = 0;                               // 模型文件大小
    // 调用辅助函数 load_model 从指定路径加载模型文件到内存中，并获取其长度
    auto model = load_model(model_file, &model_len); // 加载模型文件
    // 检查模型是否加载成功
    if (model == nullptr)
    {
        // 如果加载失败，记录错误日志
        NN_LOG_ERROR("load model file %s fail!", model_file);
        // 返回“加载模型失败”的错误码
        return NN_LOAD_MODEL_FAIL; // 返回错误码：加载模型文件失败
    }
    // 调用 rknn_init 函数，使用加载的模型数据初始化 RKNN 上下文(context)
    int ret = rknn_init(&rknn_ctx_, model, model_len, 0, NULL); // 初始化rknn context
    // 检查 rknn_init 是否执行成功
    if (ret < 0)
    {
        // 如果初始化失败，记录错误日志
        NN_LOG_ERROR("rknn_init fail! ret=%d", ret);
        // 返回“RKNN初始化失败”的错误码
        return NN_RKNN_INIT_FAIL; // 返回错误码：初始化rknn context失败
    }
    // 记录初始化成功的日志信息
    NN_LOG_INFO("rknn_init success!");
    // 设置标志位，表示 RKNN 上下文已成功创建
    ctx_created_ = true;

    // 获取rknn版本信息
    // 定义一个 rknn_sdk_version 结构体变量，用于存储版本信息
    rknn_sdk_version version;
    // 调用 rknn_query 查询 SDK 版本信息
    ret = rknn_query(rknn_ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    // 检查查询操作是否成功
    if (ret < 0)
    {
        // 如果查询失败，记录错误日志
        NN_LOG_ERROR("rknn_query fail! ret=%d", ret);
        // 返回“RKNN查询失败”的错误码
        return NN_RKNN_QUERY_FAIL;
    }
    // 记录并打印 RKNN API 和驱动的版本信息
    NN_LOG_INFO("RKNN API version: %s", version.api_version);
    NN_LOG_INFO("RKNN Driver version: %s", version.drv_version);

    // 获取输入输出个数
    // 定义一个 rknn_input_output_num 结构体变量，用于存储输入输出张量的数量
    rknn_input_output_num io_num;
    // 调用 rknn_query 查询模型的输入输出数量
    ret = rknn_query(rknn_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    // 检查查询操作是否成功
    if (ret != RKNN_SUCC)
    {
        // 如果查询失败，记录错误日志
        NN_LOG_ERROR("rknn_query fail! ret=%d", ret);
        // 返回“RKNN查询失败”的错误码
        return NN_RKNN_QUERY_FAIL;
    }
    // 记录并打印模型的输入和输出张量数量
    NN_LOG_INFO("model input num: %d, output num: %d", io_num.n_input, io_num.n_output);

    // 将获取到的输入输出数量保存到类的成员变量中
    input_num_ = io_num.n_input;
    output_num_ = io_num.n_output;

    // 获取输入张量的属性
    NN_LOG_INFO("input tensors:");
    // 定义一个 rknn_tensor_attr 数组，用于存储所有输入张量的属性
    rknn_tensor_attr input_attrs[io_num.n_input];
    // 使用 memset 将数组内存清零，进行初始化
    memset(input_attrs, 0, sizeof(input_attrs));
    // 遍历所有输入张量
    for (int i = 0; i < io_num.n_input; i++)
    {
        // 设置当前要查询的张量的索引
        input_attrs[i].index = i;
        // 调用 rknn_query 查询指定索引的输入张量的属性
        ret = rknn_query(rknn_ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        // 检查查询操作是否成功
        if (ret != RKNN_SUCC)
        {
            // 如果查询失败，记录错误日志
            NN_LOG_ERROR("rknn_query fail! ret=%d", ret);
            // 返回“RKNN查询失败”的错误码
            return NN_RKNN_QUERY_FAIL;
        }
        // 调用辅助函数打印当前张量的属性信息
        print_tensor_attr(&(input_attrs[i]));
        // 将 rknn_tensor_attr 转换为自定义的 tensor_attr_s 类型，并存入成员变量 in_shapes_
        in_shapes_.push_back(rknn_tensor_attr_convert(input_attrs[i]));
    }

    // 获取输出张量的属性
    NN_LOG_INFO("output tensors:");
    // 定义一个 rknn_tensor_attr 数组，用于存储所有输出张量的属性
    rknn_tensor_attr output_attrs[io_num.n_output];
    // 使用 memset 将数组内存清零，进行初始化
    memset(output_attrs, 0, sizeof(output_attrs));
    // 遍历所有输出张量
    for (int i = 0; i < io_num.n_output; i++)
    {
        // 设置当前要查询的张量的索引
        output_attrs[i].index = i;
        // 调用 rknn_query 查询指定索引的输出张量的属性
        ret = rknn_query(rknn_ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        // 检查查询操作是否成功
        if (ret != RKNN_SUCC)
        {
            // 如果查询失败，记录错误日志
            NN_LOG_ERROR("rknn_query fail! ret=%d", ret);
            // 返回“RKNN查询失败”的错误码
            return NN_RKNN_QUERY_FAIL;
        }
        // 调用辅助函数打印当前张量的属性信息
        print_tensor_attr(&(output_attrs[i]));
        // 将 rknn_tensor_attr 转换为自定义的 tensor_attr_s 类型，并存入成员变量 out_shapes_
        out_shapes_.push_back(rknn_tensor_attr_convert(output_attrs[i]));
    }

    // 如果所有步骤都成功完成，返回成功状态码
    return NN_SUCCESS;
}

// 实现 GetInputShapes 方法，用于获取输入张量的形状信息
const std::vector<tensor_attr_s> &RKEngine::GetInputShapes()
{
    // 返回存储输入张量属性的成员变量 in_shapes_ 的常量引用
    return in_shapes_;
}

// 实现 GetOutputShapes 方法，用于获取输出张量的形状信息
const std::vector<tensor_attr_s> &RKEngine::GetOutputShapes()
{
    // 返回存储输出张量属性的成员变量 out_shapes_ 的常量引用
    return out_shapes_;
}

/**
 * @brief 运行模型，获得推理结果
 * @param inputs 输入张量
 * @param outputs 输出张量
 * @param want_float 是否需要float类型的输出
 * @return nn_error_e 错误码
 */
// 实现 Run 方法，用于执行模型推理
nn_error_e RKEngine::Run(std::vector<tensor_data_s> &inputs, std::vector<tensor_data_s> &outputs, bool want_float)
{
    // 检查外部传入的输入张量数量是否与模型定义的数量匹配
    if (inputs.size() != input_num_)
    {
        // 如果不匹配，记录错误日志
        NN_LOG_ERROR("inputs num not match! inputs.size()=%ld, input_num_=%d", inputs.size(), input_num_);
        // 返回“输入输出数量不匹配”的错误码
        return NN_IO_NUM_NOT_MATCH;
    }
    // 检查外部传入的用于接收结果的输出张量数量是否与模型定义的数量匹配
    if (outputs.size() != output_num_)
    {
        // 如果不匹配，记录错误日志
        NN_LOG_ERROR("outputs num not match! outputs.size()=%ld, output_num_=%d", outputs.size(), output_num_);
        // 返回“输入输出数量不匹配”的错误码
        return NN_IO_NUM_NOT_MATCH;
    }

    // 定义一个 rknn_input 数组，用于设置给 RKNN API
    rknn_input rknn_inputs[g_max_io_num];
    // 遍历所有输入数据
    for (int i = 0; i < inputs.size(); i++)
    {
        // 调用辅助函数将自定义的 tensor_data_s 结构转换为 rknn_input 结构
        rknn_inputs[i] = tensor_data_to_rknn_input(inputs[i]);
    }
    // 调用 rknn_inputs_set 将准备好的输入数据设置到 RKNN 上下文中
    int ret = rknn_inputs_set(rknn_ctx_, (uint32_t)inputs.size(), rknn_inputs);
    // 检查设置操作是否成功
    if (ret < 0)
    {
        // 如果失败，记录错误日志
        NN_LOG_ERROR("rknn_inputs_set fail! ret=%d", ret);
        // 返回“RKNN输入设置失败”的错误码
        return NN_RKNN_INPUT_SET_FAIL;
    }

    // 开始执行模型推理
    NN_LOG_DEBUG("rknn running...");
    // 调用 rknn_run 执行推理，第二个参数为 nullptr 表示使用默认的同步模式
    ret = rknn_run(rknn_ctx_, nullptr);
    // 检查推理是否成功
    if (ret < 0)
    {
        // 如果失败，记录错误日志
        NN_LOG_ERROR("rknn_run fail! ret=%d", ret);
        // 返回“RKNN运行时错误”的错误码
        return NN_RKNN_RUNTIME_ERROR;
    }

    // 获取模型的推理输出结果
    // 定义一个 rknn_output 数组，用于接收推理结果
    rknn_output rknn_outputs[g_max_io_num];
    // 使用 memset 将数组内存清零，进行初始化
    memset(rknn_outputs, 0, sizeof(rknn_outputs));
    // 遍历所有输出张量
    for (int i = 0; i < output_num_; ++i)
    {
        // 根据 want_float 参数设置是否需要浮点型输出
        rknn_outputs[i].want_float = want_float ? 1 : 0;
    }
    // 调用 rknn_outputs_get 获取推理结果
    ret = rknn_outputs_get(rknn_ctx_, output_num_, rknn_outputs, NULL);
    // 检查获取操作是否成功
    if (ret < 0)
    {
        // 如果失败，使用 printf 打印直接错误信息（可能日志系统此时不可用）
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        // 记录错误日志
        NN_LOG_ERROR("rknn_outputs_get fail! ret=%d", ret);
        // 返回“RKNN获取输出失败”的错误码
        return NN_RKNN_OUTPUT_GET_FAIL;
    }

    // 记录输出张量的数量
    NN_LOG_DEBUG("output num: %d", output_num_);
    // 将 rknn_output 的结果拷贝到用户提供的 outputs 向量中
    for (int i = 0; i < output_num_; ++i)
    {
        // 调用辅助函数将 rknn_output 结构转换为自定义的 tensor_data_s 结构
        rknn_output_to_tensor_data(rknn_outputs[i], outputs[i]);
        // 记录每个输出张量的大小
        NN_LOG_DEBUG("output[%d] size=%d", i, outputs[i].attr.size);
        // 释放由 rknn_outputs_get 内部为 buf 分配的内存
        free(rknn_outputs[i].buf);
    }
    // 如果所有步骤都成功完成，返回成功状态码
    return NN_SUCCESS;
}

// 实现 RKEngine 类的析构函数
RKEngine::~RKEngine()
{
    // 检查 RKNN 上下文是否已经被创建
    if (ctx_created_)
    {
        // 如果已创建，则调用 rknn_destroy 进行销毁，释放资源
        rknn_destroy(rknn_ctx_);
        // 记录销毁成功的日志信息
        NN_LOG_INFO("rknn context destroyed!");
    }
}

// 实现 CreateRKNNEngine 工厂函数
std::shared_ptr<NNEngine> CreateRKNNEngine()
{
    // 使用 std::make_shared 创建一个 RKEngine 类的实例，并返回一个指向基类 NNEngine 的共享指针
    return std::make_shared<RKEngine>();
}