

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

int PostProcessWithLetterBox(float* output, std::vector<Detection>& results,
                              int num_attrs, int input_w, int input_h,
                              const LetterBoxInfo& info, int src_width, int src_height) {
    
    results.clear();

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
        if (suppressed[i]) {
            continue;
        }

        float x1 = proposals[i].xmin;
        float y1 = proposals[i].ymin;
        float x2 = proposals[i].xmax;
        float y2 = proposals[i].ymax;

        Detection det;
        det.class_id = proposals[i].classId;
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

}  
