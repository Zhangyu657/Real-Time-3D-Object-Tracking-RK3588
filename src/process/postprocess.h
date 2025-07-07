#ifndef RK3588_DEMO_POSTPROCESS_H
#define RK3588_DEMO_POSTPROCESS_H

#include <stdint.h>
#include <vector>
#include "types/yolo_datatype.h"
#include "process/preprocess.h"

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
