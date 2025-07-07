#include "yolov8_custom.h"
#include <random>
#include "utils/logging.h"
#include "process/preprocess.h"
#include "process/postprocess.h"

static std::vector<std::string> g_classes = {
    "ball"
};

Yolov8Custom::Yolov8Custom()
{
    engine_ = CreateRKNNEngine();
    input_tensor_.data = nullptr;
    want_float_ = false;
    ready_ = false;
}

Yolov8Custom::~Yolov8Custom()
{
    if (input_tensor_.data != nullptr)
    {
        free(input_tensor_.data);
        input_tensor_.data = nullptr;
    }
    if (output_tensor_.data != nullptr)
    {
        free(output_tensor_.data);
        output_tensor_.data = nullptr;
    }
}

nn_error_e Yolov8Custom::LoadModel(const char *model_path)
{
    auto ret = engine_->LoadModelFile(model_path);
    if (ret != NN_SUCCESS)
    {
        NN_LOG_ERROR("yolov8 load model file failed");
        return ret;
    }

    auto input_shapes = engine_->GetInputShapes();
    if (input_shapes.size() != 1)
    {
        NN_LOG_ERROR("yolov8 input tensor number is not 1, but %ld", input_shapes.size());
        return NN_RKNN_INPUT_ATTR_ERROR;
    }
    nn_tensor_attr_to_cvimg_input_data(input_shapes[0], input_tensor_);
    input_tensor_.data = malloc(input_tensor_.attr.size);

    auto output_shapes = engine_->GetOutputShapes();
    if (output_shapes.size() != 1)
    {
        NN_LOG_ERROR("yolov8 output tensor number is not 1, but %ld", output_shapes.size());
        return NN_RKNN_OUTPUT_ATTR_ERROR;
    }

    want_float_ = (output_shapes[0].type == NN_TENSOR_FLOAT16);
    output_tensor_.attr = output_shapes[0];
    output_tensor_.attr.type = want_float_ ? NN_TENSOR_FLOAT : output_shapes[0].type;
    output_tensor_.attr.index = 0;
    output_tensor_.attr.size = output_shapes[0].n_elems * nn_tensor_type_to_size(output_tensor_.attr.type);
    output_tensor_.data = malloc(output_tensor_.attr.size);

    out_zp_ = output_shapes[0].zp;
    out_scale_ = output_shapes[0].scale;

    ready_ = true;
    return NN_SUCCESS;
}

nn_error_e Yolov8Custom::Preprocess(const cv::Mat &img, const std::string process_type, cv::Mat &image_letterbox)
{
    int input_w = input_tensor_.attr.dims[2];
    int input_h = input_tensor_.attr.dims[1];
    float wh_ratio = static_cast<float>(input_w) / input_h;

    if (process_type == "opencv")
    {
        letterbox_info_ = letterbox(img, image_letterbox, wh_ratio);
        cvimg2tensor(image_letterbox, input_w, input_h, input_tensor_);
    }
    else if (process_type == "rga")
    {
        letterbox_info_ = letterbox_rga(img, image_letterbox, wh_ratio);
        cvimg2tensor_rga(image_letterbox, input_w, input_h, input_tensor_);
    }
    return NN_SUCCESS;
}

nn_error_e Yolov8Custom::Inference()
{
    std::vector<tensor_data_s> inputs = {input_tensor_};
    std::vector<tensor_data_s> outputs = {output_tensor_};
    return engine_->Run(inputs, outputs, want_float_);
}

nn_error_e Yolov8Custom::Postprocess(const cv::Mat &img, std::vector<Detection> &objects)
{
    int input_w = input_tensor_.attr.dims[2];
    int input_h = input_tensor_.attr.dims[1];
    int num_attrs = output_tensor_.attr.dims[1];

    if (want_float_)
    {
        yolo::PostProcessWithLetterBox(
            (float *)output_tensor_.data, objects,
            num_attrs, input_w, input_h,
            letterbox_info_, img.cols, img.rows);
    }
    else
    {
        NN_LOG_ERROR("INT8 letterbox后处理未实现！");
        return NN_RKNN_OUTPUT_ATTR_ERROR;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(100, 255);

    for (auto &result : objects)
    {
        result.color = cv::Scalar(dis(gen), dis(gen), dis(gen));
        result.className = (result.class_id >= 0 && result.class_id < g_classes.size()) ? g_classes[result.class_id] : "unknown";
        result.letterbox_info = letterbox_info_;
        result.has_letterbox = true;
    }

    return NN_SUCCESS;
}

nn_error_e Yolov8Custom::Run(const cv::Mat &img, std::vector<Detection> &objects)
{
    cv::Mat image_letterbox;
    Preprocess(img, "opencv", image_letterbox);
    Inference();
    Postprocess(image_letterbox, objects);
    return NN_SUCCESS;
}

nn_error_e Yolov8Custom::Run(const cv::Mat &img, std::vector<Detection> &objects, LetterBoxInfo &info)
{
    cv::Mat image_letterbox;
    Preprocess(img, "opencv", image_letterbox);
    Inference();
    Postprocess(image_letterbox, objects);
    info = letterbox_info_;
    return NN_SUCCESS;
}
