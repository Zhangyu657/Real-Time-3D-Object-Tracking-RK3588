#ifndef RK3588_DEMO_YOLOV8_CUSTOM_H
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
    std::optional<rs2::depth_frame> depth_frame_;
    rs2_intrinsics intrinsics_{};
    bool has_depth_ = false;
};

#endif // RK3588_DEMO_YOLOV8_CUSTOM_H
