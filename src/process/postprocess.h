/* 
#ifndef RK3588_DEMO_POSTPROCESS_H
#define RK3588_DEMO_POSTPROCESS_H

#include <stdint.h>
#include <vector>

int get_top(float *pfProb, float *pfMaxProb, uint32_t *pMaxClass, uint32_t outputCount, uint32_t topNum);

namespace yolo
{
    int GetConvDetectionResult(float **pBlob, std::vector<float> &DetectiontRects);                                                               // 浮点数版本
    int GetConvDetectionResultInt8(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects); // int8版本
}

#endif // RK3588_DEMO_POSTPROCESS_H */

/* #ifndef RK3588_DEMO_POSTPROCESS_H
#define RK3588_DEMO_POSTPROCESS_H

#include <stdint.h>
#include <vector>

namespace yolo
{
    /**
     * @brief 后处理函数：适用于 float32 / float16 模型
     * @param output 模型输出 float 指针，维度为 [5, 8400]
     * @param DetectiontRects 存储检测结果的数组（格式：classId, conf, x1, y1, x2, y2）
     * @return 0 表示成功
     
    int PostProcessSingleOutput(float* output, std::vector<float>& DetectiontRects);

    /**
     * @brief 后处理函数：适用于 int8 量化模型
     * @param output 模型输出 int8 指针，维度为 [5, 8400]
     * @param zp zero-point
     * @param scale 缩放因子
     * @param DetectiontRects 存储检测结果的数组（格式：classId, conf, x1, y1, x2, y2）
     * @return 0 表示成功
     
    int PostProcessSingleOutputInt8(int8_t* output, int zp, float scale, std::vector<float>& DetectiontRects);
}

#endif  // RK3588_DEMO_POSTPROCESS_H */


/* #ifndef RK3588_DEMO_POSTPROCESS_H
#define RK3588_DEMO_POSTPROCESS_H

#include <stdint.h>
#include <vector>
#include "types/yolo_datatype.h"   // 包含 Detection 与 LetterBoxInfo 结构体定义

#include "process/preprocess.h"   // ✅ LetterBoxInfo 结构体
namespace yolo
{
    int PostProcessSingleOutput(float* output, std::vector<Detection>& objects, const LetterBoxInfo& letterbox_info);
    int PostProcessWithLetterBox(float* output, std::vector<Detection>& results,
        int num_attrs, int input_w, int input_h,
        const LetterBoxInfo& info,
        int src_width, int src_height);

   
    int PostProcessSingleOutputInt8(int8_t* output, int zp, float scale, std::vector<Detection>& objects, const LetterBoxInfo& letterbox_info);
}

#endif  // RK3588_DEMO_POSTPROCESS_H


 */

 // 这是 "Include Guard"（头文件保护宏），用于防止本头文件在一次编译中被重复包含
#ifndef RK3588_DEMO_POSTPROCESS_H
// 如果宏 RK3588_DEMO_POSTPROCESS_H 未被定义，则定义它，与 #ifndef 配对使用
#define RK3588_DEMO_POSTPROCESS_H

// 引入 C 标准库头文件 <stdint.h>，以使用 int8_t 等固定宽度的整数类型
#include <stdint.h>
// 引入 C++ 标准库头文件 <vector>，以使用 std::vector 容器
#include <vector>
// 引入自定义的头文件 "yolo_datatype.h"，注释表明其中包含了 Detection 和 LetterBoxInfo 结构体的定义
#include "types/yolo_datatype.h"

// 引入自定义的头文件 "preprocess.h"，注释表明其中也包含了 LetterBoxInfo 结构体
#include "process/preprocess.h"
// 定义一个名为 yolo 的命名空间，用于封装与YOLO模型相关的代码，避免命名冲突
namespace yolo
{
    /**
     * @brief 这是一个Doxygen风格的注释块，描述下方函数的功能：对浮点型模型的输出进行后处理
     * @param output 模型的浮点数（float）输出数据指针，注释中假设其维度为 [5, 8400]
     * @param objects 用于存储最终检测结果的向量引用
     * @param letterbox_info 包含图像预处理（LetterBox）信息的结构体，用于将坐标还原到原图尺寸
     * @return 返回一个整数，通常 0 表示成功
     */
    // 声明一个后处理函数，适用于处理 float32 或 float16 模型的单输出情况
    int PostProcessSingleOutput(float* output, std::vector<Detection>& objects, const LetterBoxInfo& letterbox_info);
    
    // 声明另一个后处理函数，其参数更为详细，同样适用于浮点型模型
    int PostProcessWithLetterBox(float* output, std::vector<Detection>& results,
        int num_attrs, int input_w, int input_h,
        const LetterBoxInfo& info,
        int src_width, int src_height);

    /**
     * @brief Doxygen风格注释块，描述下方函数的功能：对int8量化模型的输出进行后处理
     * @param output 模型的整型（int8_t）输出数据指针，注释中假设其维度为 [5, 8400]
     * @param zp 量化参数 "zero-point"（零点）
     * @param scale 量化参数 "scale"（缩放因子）
     * @param objects 用于存储最终检测结果的向量引用
     * @param letterbox_info 包含 LetterBox 信息的结构体，用于坐标还原
     * @return 返回一个整数，通常 0 表示成功
     */
    // 声明一个后处理函数，专门用于处理 int8 量化模型的单输出情况
    int PostProcessSingleOutputInt8(int8_t* output, int zp, float scale, std::vector<Detection>& objects, const LetterBoxInfo& letterbox_info);
// yolo 命名空间结束
}

// 结束 #ifndef RK3588_DEMO_POSTPROCESS_H 定义的条件编译块
#endif  // RK3588_DEMO_POSTPROCESS_H