/* 
#include "postprocess.h"

#include <string.h>
#include <stdlib.h>

#include <algorithm>

#include "utils/logging.h"

int get_top(float *pfProb, float *pfMaxProb, uint32_t *pMaxClass, uint32_t outputCount, uint32_t topNum)
{
    uint32_t i, j;

#define MAX_TOP_NUM 20
    if (topNum > MAX_TOP_NUM)
        return 0;

    memset(pfMaxProb, 0, sizeof(float) * topNum);
    memset(pMaxClass, 0xff, sizeof(float) * topNum);

    for (j = 0; j < topNum; j++)
    {
        for (i = 0; i < outputCount; i++)
        {
            if ((i == *(pMaxClass + 0)) || (i == *(pMaxClass + 1)) || (i == *(pMaxClass + 2)) || (i == *(pMaxClass + 3)) ||
                (i == *(pMaxClass + 4)))
            {
                continue;
            }

            if (pfProb[i] > *(pfMaxProb + j))
            {
                *(pfMaxProb + j) = pfProb[i];
                *(pMaxClass + j) = i;
            }
        }
    }

    return 1;
}

namespace yolo
{
    typedef struct
    {
        float xmin;
        float ymin;
        float xmax;
        float ymax;
        float score;
        int classId;
    } DetectRect;
    static int input_w = 640;
    static int input_h = 640;
    static float objectThreshold = 0.2;
    static float nmsThreshold = 0.25;
    static int headNum = 3;
    static int class_num = 5; // 类别数
    static int strides[3] = {8, 16, 32};
    static int mapSize[3][2] = {{80, 80}, {40, 40}, {20, 20}};
#define ZQ_MAX(a, b) ((a) > (b) ? (a) : (b))
#define ZQ_MIN(a, b) ((a) < (b) ? (a) : (b))
    static inline float fast_exp(float x)
    {
        // return exp(x);
        union
        {
            uint32_t i;
            float f;
        } v;
        v.i = (12102203.1616540672 * x + 1064807160.56887296);
        return v.f;
    }

    float sigmoid(float x)
    {
        return 1 / (1 + fast_exp(-x));
    }

    static inline float IOU(float XMin1, float YMin1, float XMax1, float YMax1, float XMin2, float YMin2, float XMax2, float YMax2)
    {
        float Inter = 0;
        float Total = 0;
        float XMin = 0;
        float YMin = 0;
        float XMax = 0;
        float YMax = 0;
        float Area1 = 0;
        float Area2 = 0;
        float InterWidth = 0;
        float InterHeight = 0;

        XMin = ZQ_MAX(XMin1, XMin2);
        YMin = ZQ_MAX(YMin1, YMin2);
        XMax = ZQ_MIN(XMax1, XMax2);
        YMax = ZQ_MIN(YMax1, YMax2);

        InterWidth = XMax - XMin;
        InterHeight = YMax - YMin;

        InterWidth = (InterWidth >= 0) ? InterWidth : 0;
        InterHeight = (InterHeight >= 0) ? InterHeight : 0;

        Inter = InterWidth * InterHeight;

        Area1 = (XMax1 - XMin1) * (YMax1 - YMin1);
        Area2 = (XMax2 - XMin2) * (YMax2 - YMin2);

        Total = Area1 + Area2 - Inter;

        return float(Inter) / float(Total);
    }

    static float DeQnt2F32(int8_t qnt, int zp, float scale)
    {
        return ((float)qnt - (float)zp) * scale;
    }

    std::vector<float> GenerateMeshgrid()
    {
        std::vector<float> meshgrid;
        if (headNum == 0)
        {
            NN_LOG_ERROR("=== yolov8 Meshgrid  Generate failed! ");
            exit(-1);
        }

        for (int index = 0; index < headNum; index++)
        {
            for (int i = 0; i < mapSize[index][0]; i++)
            {
                for (int j = 0; j < mapSize[index][1]; j++)
                {
                    meshgrid.push_back(float(j + 0.5));
                    meshgrid.push_back(float(i + 0.5));
                }
            }
        }

        printf("=== yolov8 Meshgrid  Generate success! \n");
        return meshgrid;
    }
    // int8版本
    int GetConvDetectionResultInt8(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale,
                                   std::vector<float> &DetectiontRects)
    {
        static auto meshgrid = GenerateMeshgrid();
        int ret = 0;

        int gridIndex = -2;
        float xmin = 0, ymin = 0, xmax = 0, ymax = 0;
        float cls_val = 0;
        float cls_max = 0;
        int cls_index = 0;

        int quant_zp_cls = 0, quant_zp_reg = 0;
        float quant_scale_cls = 0, quant_scale_reg = 0;

        DetectRect temp;
        std::vector<DetectRect> detectRects;

        for (int index = 0; index < headNum; index++)
        {
            int8_t *reg = (int8_t *)pBlob[index * 2 + 0];
            int8_t *cls = (int8_t *)pBlob[index * 2 + 1];

            quant_zp_reg = qnt_zp[index * 2 + 0];
            quant_zp_cls = qnt_zp[index * 2 + 1];

            quant_scale_reg = qnt_scale[index * 2 + 0];
            quant_scale_cls = qnt_scale[index * 2 + 1];

            for (int h = 0; h < mapSize[index][0]; h++)
            {
                for (int w = 0; w < mapSize[index][1]; w++)
                {
                    gridIndex += 2;

                    for (int cl = 0; cl < class_num; cl++)
                    {
                        cls_val = sigmoid(
                            DeQnt2F32(cls[cl * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w],
                                      quant_zp_cls, quant_scale_cls));

                        if (0 == cl)
                        {
                            cls_max = cls_val;
                            cls_index = cl;
                        }
                        else
                        {
                            if (cls_val > cls_max)
                            {
                                cls_max = cls_val;
                                cls_index = cl;
                            }
                        }
                    }

                    if (cls_max > objectThreshold)
                    {
                        xmin = (meshgrid[gridIndex + 0] -
                                DeQnt2F32(reg[0 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w],
                                          quant_zp_reg, quant_scale_reg)) *
                               strides[index];
                        ymin = (meshgrid[gridIndex + 1] -
                                DeQnt2F32(reg[1 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w],
                                          quant_zp_reg, quant_scale_reg)) *
                               strides[index];
                        xmax = (meshgrid[gridIndex + 0] +
                                DeQnt2F32(reg[2 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w],
                                          quant_zp_reg, quant_scale_reg)) *
                               strides[index];
                        ymax = (meshgrid[gridIndex + 1] +
                                DeQnt2F32(reg[3 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w],
                                          quant_zp_reg, quant_scale_reg)) *
                               strides[index];

                        xmin = xmin > 0 ? xmin : 0;
                        ymin = ymin > 0 ? ymin : 0;
                        xmax = xmax < input_w ? xmax : input_w;
                        ymax = ymax < input_h ? ymax : input_h;

                        if (xmin >= 0 && ymin >= 0 && xmax <= input_w && ymax <= input_h)
                        {
                            temp.xmin = xmin / input_w;
                            temp.ymin = ymin / input_h;
                            temp.xmax = xmax / input_w;
                            temp.ymax = ymax / input_h;
                            temp.classId = cls_index;
                            temp.score = cls_max;
                            detectRects.push_back(temp);
                        }
                    }
                }
            }
        }

        std::sort(detectRects.begin(), detectRects.end(),
                  [](DetectRect &Rect1, DetectRect &Rect2) -> bool
                  { return (Rect1.score > Rect2.score); });

        NN_LOG_DEBUG("NMS Before num :%ld", detectRects.size());
        for (int i = 0; i < detectRects.size(); ++i)
        {
            float xmin1 = detectRects[i].xmin;
            float ymin1 = detectRects[i].ymin;
            float xmax1 = detectRects[i].xmax;
            float ymax1 = detectRects[i].ymax;
            int classId = detectRects[i].classId;
            float score = detectRects[i].score;

            if (classId != -1)
            {
                // 将检测结果按照classId、score、xmin1、ymin1、xmax1、ymax1 的格式存放在vector<float>中
                DetectiontRects.push_back(float(classId));
                DetectiontRects.push_back(float(score));
                DetectiontRects.push_back(float(xmin1));
                DetectiontRects.push_back(float(ymin1));
                DetectiontRects.push_back(float(xmax1));
                DetectiontRects.push_back(float(ymax1));

                for (int j = i + 1; j < detectRects.size(); ++j)
                {
                    float xmin2 = detectRects[j].xmin;
                    float ymin2 = detectRects[j].ymin;
                    float xmax2 = detectRects[j].xmax;
                    float ymax2 = detectRects[j].ymax;
                    float iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);
                    if (iou > nmsThreshold)
                    {
                        detectRects[j].classId = -1;
                    }
                }
            }
        }

        return ret;
    }
    // 浮点数版本
    int GetConvDetectionResult(float **pBlob, std::vector<float> &DetectiontRects)
    {
        static auto meshgrid = GenerateMeshgrid();
        int ret = 0;

        int gridIndex = -2;
        float xmin = 0, ymin = 0, xmax = 0, ymax = 0;
        float cls_val = 0;
        float cls_max = 0;
        int cls_index = 0;

        DetectRect temp;
        std::vector<DetectRect> detectRects;

        for (int index = 0; index < headNum; index++)
        {
            float *reg = (float *)pBlob[index * 2 + 0];
            float *cls = (float *)pBlob[index * 2 + 1];

            for (int h = 0; h < mapSize[index][0]; h++)
            {
                for (int w = 0; w < mapSize[index][1]; w++)
                {
                    gridIndex += 2;

                    for (int cl = 0; cl < class_num; cl++)
                    {
                        cls_val = sigmoid(
                            cls[cl * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]);

                        if (0 == cl)
                        {
                            cls_max = cls_val;
                            cls_index = cl;
                        }
                        else
                        {
                            if (cls_val > cls_max)
                            {
                                cls_max = cls_val;
                                cls_index = cl;
                            }
                        }
                    }

                    if (cls_max > objectThreshold)
                    {
                        xmin = (meshgrid[gridIndex + 0] -
                                reg[0 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) *
                               strides[index];
                        ymin = (meshgrid[gridIndex + 1] -
                                reg[1 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) *
                               strides[index];
                        xmax = (meshgrid[gridIndex + 0] +
                                reg[2 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) *
                               strides[index];
                        ymax = (meshgrid[gridIndex + 1] +
                                reg[3 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) *
                               strides[index];

                        xmin = xmin > 0 ? xmin : 0;
                        ymin = ymin > 0 ? ymin : 0;
                        xmax = xmax < input_w ? xmax : input_w;
                        ymax = ymax < input_h ? ymax : input_h;

                        if (xmin >= 0 && ymin >= 0 && xmax <= input_w && ymax <= input_h)
                        {
                            temp.xmin = xmin / input_w;
                            temp.ymin = ymin / input_h;
                            temp.xmax = xmax / input_w;
                            temp.ymax = ymax / input_h;
                            temp.classId = cls_index;
                            temp.score = cls_max;
                            detectRects.push_back(temp);
                        }
                    }
                }
            }
        }

        std::sort(detectRects.begin(), detectRects.end(),
                  [](DetectRect &Rect1, DetectRect &Rect2) -> bool
                  { return (Rect1.score > Rect2.score); });

        NN_LOG_DEBUG("NMS Before num :%ld", detectRects.size());
        for (int i = 0; i < detectRects.size(); ++i)
        {
            float xmin1 = detectRects[i].xmin;
            float ymin1 = detectRects[i].ymin;
            float xmax1 = detectRects[i].xmax;
            float ymax1 = detectRects[i].ymax;
            int classId = detectRects[i].classId;
            float score = detectRects[i].score;

            if (classId != -1)
            {
                // 将检测结果按照classId、score、xmin1、ymin1、xmax1、ymax1 的格式存放在vector<float>中
                DetectiontRects.push_back(float(classId));
                DetectiontRects.push_back(float(score));
                DetectiontRects.push_back(float(xmin1));
                DetectiontRects.push_back(float(ymin1));
                DetectiontRects.push_back(float(xmax1));
                DetectiontRects.push_back(float(ymax1));

                for (int j = i + 1; j < detectRects.size(); ++j)
                {
                    float xmin2 = detectRects[j].xmin;
                    float ymin2 = detectRects[j].ymin;
                    float xmax2 = detectRects[j].xmax;
                    float ymax2 = detectRects[j].ymax;
                    float iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);
                    if (iou > nmsThreshold)
                    {
                        detectRects[j].classId = -1;
                    }
                }
            }
        }

        return ret;
    }

} */


/*这是适用于float32和float16版本的完整代码

#include "postprocess.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace yolo {

typedef struct {
    float xmin, ymin, xmax, ymax;
    float score;
    int classId;
} DetectRect;

static int input_w = 640;
static int input_h = 640;
static float objectThreshold = 0.6;
static float nmsThreshold = 0.45;
static int num_anchors = 8400;
static int num_attrs = 5;  // x, y, w, h, obj_conf

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static inline float IOU(const DetectRect& a, const DetectRect& b) {
    float x1 = std::max(a.xmin, b.xmin);
    float y1 = std::max(a.ymin, b.ymin);
    float x2 = std::min(a.xmax, b.xmax);
    float y2 = std::min(a.ymax, b.ymax);

    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float inter = w * h;
    float union_area = (a.xmax - a.xmin) * (a.ymax - a.ymin) +
                       (b.xmax - b.xmin) * (b.ymax - b.ymin) - inter;
    return inter / union_area;
}

// ✅ FP32 模型后处理（与 ONNX 推理对齐）
int PostProcessSingleOutput(float* output, std::vector<float>& DetectiontRects)
{
    std::vector<DetectRect> proposals;

    for (int i = 0; i < num_anchors; ++i)
    {
        float x = output[0 * num_anchors + i];
        float y = output[1 * num_anchors + i];
        float w = output[2 * num_anchors + i];
        float h = output[3 * num_anchors + i];
        float obj_conf = sigmoid(output[4 * num_anchors + i]);

        if (obj_conf > objectThreshold)
        {
            float x1 = std::max(0.0f, x - w / 2);
            float y1 = std::max(0.0f, y - h / 2);
            float x2 = std::min((float)input_w, x + w / 2);
            float y2 = std::min((float)input_h, y + h / 2);

            // ❗ 跳过过大、无效、负数坐标框
            if ((x2 - x1) < 5 || (y2 - y1) < 5 || (x2 - x1) > input_w || (y2 - y1) > input_h)
            continue;


            DetectRect r = {x1, y1, x2, y2, obj_conf, 0};  // ❗不再归一化
            proposals.push_back(r);
        }
    }

    std::sort(proposals.begin(), proposals.end(), [](const DetectRect& a, const DetectRect& b) {
        return a.score > b.score;
    });

    std::vector<bool> suppressed(proposals.size(), false);
    for (size_t i = 0; i < proposals.size(); ++i)
    {
        if (suppressed[i]) continue;
        DetectiontRects.push_back((float)proposals[i].classId);
        DetectiontRects.push_back(proposals[i].score);
        DetectiontRects.push_back(proposals[i].xmin);
        DetectiontRects.push_back(proposals[i].ymin);
        DetectiontRects.push_back(proposals[i].xmax);
        DetectiontRects.push_back(proposals[i].ymax);

        for (size_t j = i + 1; j < proposals.size(); ++j)
        {
            if (!suppressed[j] && IOU(proposals[i], proposals[j]) > nmsThreshold)
            {
                suppressed[j] = true;
            }
        }
    }

    return 0;
}

// ✅ INT8 后处理（保持不变，仅取消归一化）
int PostProcessSingleOutputInt8(int8_t* output, int zp, float scale, std::vector<float>& DetectiontRects)
{
    std::vector<DetectRect> proposals;

    for (int i = 0; i < num_anchors; ++i)
    {
        float raw = (output[4 * num_anchors + i] - zp) * scale;
        float obj_conf = sigmoid(raw);

        if (obj_conf > objectThreshold)
        {
            float x = (output[0 * num_anchors + i] - zp) * scale;
            float y = (output[1 * num_anchors + i] - zp) * scale;
            float w = (output[2 * num_anchors + i] - zp) * scale;
            float h = (output[3 * num_anchors + i] - zp) * scale;

            float x1 = std::max(0.0f, x - w / 2);
            float y1 = std::max(0.0f, y - h / 2);
            float x2 = std::min((float)input_w, x + w / 2);
            float y2 = std::min((float)input_h, y + h / 2);

            DetectRect r = {x1, y1, x2, y2, obj_conf, 0};  // ❗不再归一化
            proposals.push_back(r);
        }
    }

    std::sort(proposals.begin(), proposals.end(), [](const DetectRect& a, const DetectRect& b) {
        return a.score > b.score;
    });

    std::vector<bool> suppressed(proposals.size(), false);
    for (size_t i = 0; i < proposals.size(); ++i)
    {
        if (suppressed[i]) continue;
        DetectiontRects.push_back((float)proposals[i].classId);
        DetectiontRects.push_back(proposals[i].score);
        DetectiontRects.push_back(proposals[i].xmin);
        DetectiontRects.push_back(proposals[i].ymin);
        DetectiontRects.push_back(proposals[i].xmax);
        DetectiontRects.push_back(proposals[i].ymax);

        for (size_t j = i + 1; j < proposals.size(); ++j)
        {
            if (!suppressed[j] && IOU(proposals[i], proposals[j]) > nmsThreshold)
            {
                suppressed[j] = true;
            }
        }
    }
    

    return 0;
}

}  // namespace yolo
  */


 /*  // ✅ 修改后的支持 letterbox 自动还原的后处理逻辑

#include "postprocess.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace yolo {

typedef struct {
    float xmin, ymin, xmax, ymax;
    float score;
    int classId;
} DetectRect;

static float objectThreshold = 0.6f;
static float nmsThreshold = 0.45f;
static int num_anchors = 8400;

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static inline float IOU(const DetectRect& a, const DetectRect& b) {
    float x1 = std::max(a.xmin, b.xmin);
    float y1 = std::max(a.ymin, b.ymin);
    float x2 = std::min(a.xmax, b.xmax);
    float y2 = std::min(a.ymax, b.ymax);

    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float inter = w * h;
    float union_area = (a.xmax - a.xmin) * (a.ymax - a.ymin) +
                       (b.xmax - b.xmin) * (b.ymax - b.ymin) - inter;
    return inter / union_area;
}

// ✅ Float32/Float16 后处理（支持还原原图坐标）
int PostProcessWithLetterBox(float* output, std::vector<Detection>& results,
                              int num_attrs, int input_w, int input_h,
                              const LetterBoxInfo& info, int src_width, int src_height) {
    std::vector<DetectRect> proposals;

    for (int i = 0; i < num_anchors; ++i) {
        float x = output[0 * num_anchors + i];
        float y = output[1 * num_anchors + i];
        float w = output[2 * num_anchors + i];
        float h = output[3 * num_anchors + i];
        float obj_conf = sigmoid(output[4 * num_anchors + i]);

        if (obj_conf > objectThreshold) {
            float x1 = std::max(0.0f, x - w / 2);
            float y1 = std::max(0.0f, y - h / 2);
            float x2 = std::min((float)input_w, x + w / 2);
            float y2 = std::min((float)input_h, y + h / 2);

            // 排除异常框
            if ((x2 - x1) < 5 || (y2 - y1) < 5 || (x2 - x1) > input_w || (y2 - y1) > input_h)
                continue;

            proposals.push_back({x1, y1, x2, y2, obj_conf, 0});
        }
    }

    std::sort(proposals.begin(), proposals.end(), [](const DetectRect& a, const DetectRect& b) {
        return a.score > b.score;
    });

    std::vector<bool> suppressed(proposals.size(), false);
    for (size_t i = 0; i < proposals.size(); ++i) {
        if (suppressed[i]) continue;

        float x1 = proposals[i].xmin;
        float y1 = proposals[i].ymin;
        float x2 = proposals[i].xmax;
        float y2 = proposals[i].ymax;

        // 🧠 将坐标还原回原图：去除 padding + 缩放
        float scale = info.hor ? ((float)src_height / input_h) : ((float)src_width / input_w);
        float pad = info.hor ? info.pad_w : info.pad_h;


        if (info.hor) {
            x1 = std::max(0.0f, (x1 - pad)) * scale;
            x2 = std::max(0.0f, (x2 - pad)) * scale;
            y1 *= scale;
            y2 *= scale;
        } else {
            x1 *= scale;
            x2 *= scale;
            y1 = std::max(0.0f, (y1 - pad)) * scale;
            y2 = std::max(0.0f, (y2 - pad)) * scale;
        }

        Detection det;
        det.class_id = 0;
        det.confidence = proposals[i].score;
        det.box = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
        det.has_letterbox = true;
        det.letterbox_info = info;
        results.push_back(det);

        for (size_t j = i + 1; j < proposals.size(); ++j) {
            if (!suppressed[j] && IOU(proposals[i], proposals[j]) > nmsThreshold) {
                suppressed[j] = true;
            }
        }
    }
    return 0;
}

}  // namespace yolo
 */

 // postprocess.cpp

/* #include "postprocess.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>

// 确保您的 Detection 结构体定义可以被此文件访问
// #include "path/to/your_detection_struct.h"

// 确保您的 LetterBoxInfo 结构体定义可以被此文件访问
// #include "path/to/your_letterbox_info.h"


namespace yolo {

// 这个辅助结构体可以保留在 postprocess.cpp 内部
typedef struct {
    float xmin, ymin, xmax, ymax;
    float score;
    int classId;
} DetectRect;

// 这些辅助函数和静态变量也保留
static float objectThreshold = 0.6f;
static float nmsThreshold = 0.45f;
static int num_anchors = 8400; // 假设您的模型输出是 8400 个锚点

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static inline float IOU(const DetectRect& a, const DetectRect& b) {
    float x1 = std::max(a.xmin, b.xmin);
    float y1 = std::max(a.ymin, b.ymin);
    float x2 = std::min(a.xmax, b.xmax);
    float y2 = std::min(a.ymax, b.ymax);

    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float inter = w * h;
    float union_area = (a.xmax - a.xmin) * (a.ymax - a.ymin) +
                       (b.xmax - b.xmin) * (b.ymax - b.ymin) - inter;
    return inter / union_area;
}


// ===================================================================================
// ✅ 以下是修改后的、作为最终版本的后处理函数
// ===================================================================================

int PostProcessWithLetterBox(float* output, std::vector<Detection>& results,
                              int num_attrs, int input_w, int input_h,
                              const LetterBoxInfo& info, int src_width, int src_height) {
    
    // 清空上次的结果
    results.clear();

    std::vector<DetectRect> proposals;

    // 1. 解码 + 筛选阈值
    // 这个循环将模型输出的扁平化 float* 数组解析成一个个候选框
    for (int i = 0; i < num_anchors; ++i) {
        float x = output[0 * num_anchors + i];
        float y = output[1 * num_anchors + i];
        float w = output[2 * num_anchors + i];
        float h = output[3 * num_anchors + i];
        float obj_conf = sigmoid(output[4 * num_anchors + i]); // 第5个数据是置信度

        if (obj_conf > objectThreshold) {
            float x1 = std::max(0.0f, x - w / 2);
            float y1 = std::max(0.0f, y - h / 2);
            float x2 = std::min((float)input_w, x + w / 2);
            float y2 = std::min((float)input_h, y + h / 2);

            // 排除异常的小框或无效框
            if ((x2 - x1) < 5 || (y2 - y1) < 5 || (x2 - x1) > input_w || (y2 - y1) > input_h)
                continue;
            
            // 注意：这里只处理了一个类别，classId 硬编码为 0
            // 如果您的模型有多个类别，需要在这里加入对类别分数的解析
            proposals.push_back({x1, y1, x2, y2, obj_conf, 0});
        }
    }

    // 2. 按置信度排序
    std::sort(proposals.begin(), proposals.end(), [](const DetectRect& a, const DetectRect& b) {
        return a.score > b.score;
    });

    // 3. NMS (非极大值抑制)
    std::vector<bool> suppressed(proposals.size(), false);
    for (size_t i = 0; i < proposals.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        // =======================================================================
        // ❌ 我们删除了这里所有的坐标变换代码
        // =======================================================================
        // 现在的 box 坐标是相对于模型输入(640x640)的，这正是我们想要的
        float x1 = proposals[i].xmin;
        float y1 = proposals[i].ymin;
        float x2 = proposals[i].xmax;
        float y2 = proposals[i].ymax;

        Detection det;
        det.class_id = proposals[i].classId; // 使用候选框中的 classId
        det.confidence = proposals[i].score;
        det.box = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
        
        // 关键：将 letterbox 信息附加到每一个检测结果上
        // 这样绘图函数才能根据这个信息进行坐标还原
        det.has_letterbox = true;
        det.letterbox_info = info; 

        results.push_back(det);

        // 执行 NMS
        for (size_t j = i + 1; j < proposals.size(); ++j) {
            if (!suppressed[j] && IOU(proposals[i], proposals[j]) > nmsThreshold) {
                suppressed[j] = true;
            }
        }
    }

    return 0;
}

}  // namespace yolo */

// 引入 "postprocess.h" 头文件，该文件可能包含了此文件中函数的声明
#include "postprocess.h"
// 引入 C++ 标准库头文件 <vector>，以使用 std::vector 容器
#include <vector>
// 引入 C++ 标准库头文件 <algorithm>，以使用 std::sort, std::max, std::min 等算法
#include <algorithm>
// 引入 C++ 标准库头文件 <cmath>，以使用 std::exp 等数学函数
#include <cmath>
// 引入 C 标准库头文件 <cstdio>，提供C风格的输入输出功能（此代码段未直接使用）
#include <cstdio>

// 这是一条注释，提醒开发者需要确保 Detection 结构体的定义对本文件可见
// 比如通过取消下面的注释并提供正确的路径来引入头文件
// #include "path/to/your_detection_struct.h"

// 这是一条注释，提醒开发者需要确保 LetterBoxInfo 结构体的定义对本文件可见
// #include "path/to/your_letterbox_info.h"


// 定义一个名为 yolo 的命名空间，用于封装与YOLO模型相关的代码，避免命名冲突
namespace yolo {

// 这是一个已有的注释，说明下面的结构体是内部使用的辅助类型
// 这个辅助结构体可以保留在 postprocess.cpp 内部
typedef struct {
    float xmin, ymin, xmax, ymax; // 边界框左上角和右下角的坐标
    float score;                  // 目标的置信度分数
    int classId;                  // 目标的类别ID
} DetectRect;                     // 定义一个名为 DetectRect 的结构体，用于临时存储解码后的检测框信息

// 这是一个已有的注释，说明下面的变量和函数也是内部使用的
// 这些辅助函数和静态变量也保留
static float objectThreshold = 0.6f;     // 定义一个静态变量，作为目标置信度的阈值，低于此值的检测框将被过滤
static float nmsThreshold = 0.45f;       // 定义一个静态变量，作为非极大值抑制（NMS）的IoU阈值
static int num_anchors = 8400;           // 定义一个静态变量，表示模型输出的总锚点/预测框数量。注释说明了其假设值

// 定义一个静态内联函数 sigmoid，用于计算 sigmoid 激活函数，常用于将置信度分数归一化到 (0, 1) 区间
static inline float sigmoid(float x) {
    // 返回 sigmoid 函数的计算结果
    return 1.0f / (1.0f + std::exp(-x));
}

// 定义一个静态内联函数 IOU，用于计算两个检测框（DetectRect）之间的交并比（Intersection over Union）
static inline float IOU(const DetectRect& a, const DetectRect& b) {
    // 计算两个框相交区域的左上角 x 坐标
    float x1 = std::max(a.xmin, b.xmin);
    // 计算两个框相交区域的左上角 y 坐标
    float y1 = std::max(a.ymin, b.ymin);
    // 计算两个框相交区域的右下角 x 坐标
    float x2 = std::min(a.xmax, b.xmax);
    // 计算两个框相交区域的右下角 y 坐标
    float y2 = std::min(a.ymax, b.ymax);

    // 计算相交区域的宽度，如果不相交则宽度为0
    float w = std::max(0.0f, x2 - x1);
    // 计算相交区域的高度，如果不相交则高度为0
    float h = std::max(0.0f, y2 - y1);
    // 计算相交区域的面积
    float inter = w * h;
    // 计算两个框的并集面积
    float union_area = (a.xmax - a.xmin) * (a.ymax - a.ymin) +
                       (b.xmax - b.xmin) * (b.ymax - b.ymin) - inter;
    // 返回交并比
    return inter / union_area;
}


// ===================================================================================
// ✅ 以下是修改后的、作为最终版本的后处理函数
// ===================================================================================

// 定义后处理函数，专门处理经过 LetterBox 预处理的图像输出
// output: 模型的原始浮点数输出数组
// results: 用于存储最终检测结果的向量
// num_attrs: 每个锚点的属性数量（例如 x,y,w,h,conf,class_scores...）
// input_w, input_h: 模型输入的宽度和高度（例如 640x640）
// info: LetterBox 预处理时保存的缩放和平移信息
// src_width, src_height: 原始图像的宽度和高度
int PostProcessWithLetterBox(float* output, std::vector<Detection>& results,
                              int num_attrs, int input_w, int input_h,
                              const LetterBoxInfo& info, int src_width, int src_height) {
    
    // 清空上一次的检测结果，为本次处理做准备
    results.clear();

    // 创建一个向量，用于存储通过置信度阈值筛选后的候选框
    std::vector<DetectRect> proposals;

    // 步骤1：解码模型输出并根据阈值进行初步筛选
    // 这个循环将模型输出的扁平化 float* 数组解析成一个个候选框
    for (int i = 0; i < num_anchors; ++i) {
        // 从模型输出中解码中心点 x 坐标
        float x = output[0 * num_anchors + i];
        // 从模型输出中解码中心点 y 坐标
        float y = output[1 * num_anchors + i];
        // 从模型输出中解码框的宽度
        float w = output[2 * num_anchors + i];
        // 从模型输出中解码框的高度
        float h = output[3 * num_anchors + i];
        // 从模型输出中解码目标置信度，并通过 sigmoid 函数处理
        float obj_conf = sigmoid(output[4 * num_anchors + i]);

        // 如果置信度高于设定的阈值，则处理该预测框
        if (obj_conf > objectThreshold) {
            // 将中心点-宽高格式(x,y,w,h)转换为左上角-右下角格式(x1,y1,x2,y2)
            float x1 = std::max(0.0f, x - w / 2); // 计算并限制 xmin 不小于0
            float y1 = std::max(0.0f, y - h / 2); // 计算并限制 ymin 不小于0
            float x2 = std::min((float)input_w, x + w / 2); // 计算并限制 xmax 不超过模型输入宽度
            float y2 = std::min((float)input_h, y + h / 2); // 计算并限制 ymax 不超过模型输入高度

            // 排除尺寸过小或尺寸异常的无效检测框
            if ((x2 - x1) < 5 || (y2 - y1) < 5 || (x2 - x1) > input_w || (y2 - y1) > input_h)
                continue; // 如果是无效框，则跳过本次循环，处理下一个锚点
            
            // 注释：这里只处理单类别检测，类别ID硬编码为0
            // 如果您的模型有多个类别，需要在这里加入对类别分数的解析
            // 将有效的候选框信息存入 proposals 向量
            proposals.push_back({x1, y1, x2, y2, obj_conf, 0});
        }
    }

    // 步骤2：按置信度分数对所有候选框进行降序排序
    std::sort(proposals.begin(), proposals.end(), [](const DetectRect& a, const DetectRect& b) {
        // lambda 表达式定义了排序规则：分数高的在前
        return a.score > b.score;
    });

    // 步骤3：执行非极大值抑制（NMS）
    // 创建一个布尔向量，用于标记哪些框被抑制了
    std::vector<bool> suppressed(proposals.size(), false);
    // 遍历所有已排序的候选框
    for (size_t i = 0; i < proposals.size(); ++i) {
        // 如果当前框已经被抑制，则跳过
        if (suppressed[i]) {
            continue;
        }

        // =======================================================================
        // ❌ 我们删除了这里所有的坐标变换代码
        // =======================================================================
        // 获取当前框的坐标，这些坐标是相对于模型输入尺寸(例如640x640)的
        float x1 = proposals[i].xmin;
        float y1 = proposals[i].ymin;
        float x2 = proposals[i].xmax;
        float y2 = proposals[i].ymax;

        // 创建一个最终的 Detection 对象
        Detection det;
        // 设置类别ID
        det.class_id = proposals[i].classId;
        // 设置置信度
        det.confidence = proposals[i].score;
        // 设置边界框，使用 OpenCV 的 Rect 结构
        det.box = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
        
        // 关键步骤：将 letterbox 信息附加到每一个检测结果上
        // 这样后续的绘图函数等模块就可以根据这些信息将坐标正确地还原到原图上
        det.has_letterbox = true;       // 标记这个检测结果的坐标是基于 letterbox 图像的
        det.letterbox_info = info;      // 附加 letterbox 的具体信息

        // 将这个有效的检测结果添加到最终的 results 向量中
        results.push_back(det);

        // 执行 NMS：将当前框与所有分数比它低的框进行比较
        for (size_t j = i + 1; j < proposals.size(); ++j) {
            // 如果框j未被抑制，且与当前框i的IoU大于NMS阈值
            if (!suppressed[j] && IOU(proposals[i], proposals[j]) > nmsThreshold) {
                // 则抑制框j
                suppressed[j] = true;
            }
        }
    }

    // 函数正常结束，返回0（通常表示成功）
    return 0;
}

}  // 结束 yolo 命名空间