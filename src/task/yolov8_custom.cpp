/* #include "yolov8_custom.h"
#include <random>
#include "utils/logging.h"
#include "process/preprocess.h"
#include "process/postprocess.h"

// define global classes
// static std::vector<std::string> g_classes = {
//     "person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
//     "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
//     "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
//     "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
//     "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
//     "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
//     "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush "};
static std::vector<std::string> g_classes = {
"pedestrians", "riders", "partially-visible-person", "ignore-regions", "crowd"};

Yolov8Custom::Yolov8Custom()
{
    engine_ = CreateRKNNEngine();
    input_tensor_.data = nullptr;
    want_float_ = false; // 是否使用浮点数版本的后处理
    ready_ = false;
}

Yolov8Custom::~Yolov8Custom()
{
    // release input tensor and output tensor
    NN_LOG_DEBUG("release input tensor");
    if (input_tensor_.data != nullptr)
    {
        free(input_tensor_.data);
        input_tensor_.data = nullptr;
    }
    NN_LOG_DEBUG("release output tensor");
    for (auto &tensor : output_tensors_)
    {
        if (tensor.data != nullptr)
        {
            free(tensor.data);
            tensor.data = nullptr;
        }
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
    // get input tensor
    auto input_shapes = engine_->GetInputShapes();

    // check number of input and n_dims
    if (input_shapes.size() != 1)
    {
        NN_LOG_ERROR("yolov8 input tensor number is not 1, but %ld", input_shapes.size());
        return NN_RKNN_INPUT_ATTR_ERROR;
    }
    nn_tensor_attr_to_cvimg_input_data(input_shapes[0], input_tensor_);
    input_tensor_.data = malloc(input_tensor_.attr.size);

    auto output_shapes = engine_->GetOutputShapes();
    if (output_shapes.size() != 6)
    {
        NN_LOG_ERROR("yolov8 output tensor number is not 6, but %ld", output_shapes.size());
        return NN_RKNN_OUTPUT_ATTR_ERROR;
    }
    if (output_shapes[0].type == NN_TENSOR_FLOAT16)
    {
        want_float_ = true;
        NN_LOG_WARNING("yolov8 output tensor type is float16, want type set to float32");
    }
    for (int i = 0; i < output_shapes.size(); i++)
    {
        tensor_data_s tensor;
        tensor.attr.n_elems = output_shapes[i].n_elems;
        tensor.attr.n_dims = output_shapes[i].n_dims;
        for (int j = 0; j < output_shapes[i].n_dims; j++)
        {
            tensor.attr.dims[j] = output_shapes[i].dims[j];
        }
        // output tensor needs to be float32
        tensor.attr.type = want_float_ ? NN_TENSOR_FLOAT : output_shapes[i].type;
        tensor.attr.index = 0;
        tensor.attr.size = output_shapes[i].n_elems * nn_tensor_type_to_size(tensor.attr.type);
        tensor.data = malloc(tensor.attr.size);
        output_tensors_.push_back(tensor);
        out_zps_.push_back(output_shapes[i].zp);
        out_scales_.push_back(output_shapes[i].scale);
    }

    ready_ = true;
    return NN_SUCCESS;
}

nn_error_e Yolov8Custom::Preprocess(const cv::Mat &img, const std::string process_type, cv::Mat &image_letterbox)
{

    // 预处理包含：letterbox、归一化、BGR2RGB、NCWH
    // 其中RKNN会做：归一化、NCWH转换（详见课程文档），所以这里只需要做letterbox、BGR2RGB

    // 比例
    float wh_ratio = (float)input_tensor_.attr.dims[2] / (float)input_tensor_.attr.dims[1];

    // lettorbox

    if (process_type == "opencv")
    {
        // BGR2RGB，resize，再放入input_tensor_中
        letterbox_info_ = letterbox(img, image_letterbox, wh_ratio);
        cvimg2tensor(image_letterbox, input_tensor_.attr.dims[2], input_tensor_.attr.dims[1], input_tensor_);
    }
    else if (process_type == "rga")
    {
        // rga resize
        letterbox_info_ = letterbox_rga(img, image_letterbox, wh_ratio);
        // save img
        // cv::imwrite("rga.jpg", image_letterbox);
        cvimg2tensor_rga(image_letterbox, input_tensor_.attr.dims[2], input_tensor_.attr.dims[1], input_tensor_);
    }

    return NN_SUCCESS;
}

nn_error_e Yolov8Custom::Inference()
{
    std::vector<tensor_data_s> inputs;
    inputs.push_back(input_tensor_);
    return engine_->Run(inputs, output_tensors_, want_float_);
}

nn_error_e Yolov8Custom::Postprocess(const cv::Mat &img, std::vector<Detection> &objects)
{
    void *output_data[6];
    for (int i = 0; i < 6; i++)
    {
        output_data[i] = (void *)output_tensors_[i].data;
    }
    std::vector<float> DetectiontRects;
    if (want_float_)
    {
        // 使用浮点数版本的后处理，他也支持量化的模型
        yolo::GetConvDetectionResult((float **)output_data, DetectiontRects);
        // NN_LOG_INFO("use float version postprocess");
    }
    else
    {
        // 使用量化版本的后处理，只能处理量化的模型
        yolo::GetConvDetectionResultInt8((int8_t **)output_data, out_zps_, out_scales_, DetectiontRects);
        // NN_LOG_INFO("use int8 version postprocess");
    }

    int img_width = img.cols;
    int img_height = img.rows;
    for (int i = 0; i < DetectiontRects.size(); i += 6)
    {
        int classId = int(DetectiontRects[i + 0]);
        float conf = DetectiontRects[i + 1];
        int xmin = int(DetectiontRects[i + 2] * float(img_width) + 0.5);
        int ymin = int(DetectiontRects[i + 3] * float(img_height) + 0.5);
        int xmax = int(DetectiontRects[i + 4] * float(img_width) + 0.5);
        int ymax = int(DetectiontRects[i + 5] * float(img_height) + 0.5);
        Detection result;
        result.class_id = classId;
        result.confidence = conf;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);
        result.color = cv::Scalar(dis(gen),
                                  dis(gen),
                                  dis(gen));

        result.className = g_classes[result.class_id];
        result.box = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);

        objects.push_back(result);
    }

    return NN_SUCCESS;
}
void letterbox_decode(std::vector<Detection> &objects, bool hor, int pad)
{
    for (auto &obj : objects)
    {
        if (hor)
        {
            obj.box.x -= pad;
        }
        else
        {
            obj.box.y -= pad;
        }
    }
}

nn_error_e Yolov8Custom::Run(const cv::Mat &img, std::vector<Detection> &objects)
{

    // letterbox后的图像
    cv::Mat image_letterbox;
    // 预处理，支持opencv或rga
    Preprocess(img, "opencv", image_letterbox);
    // Preprocess(img, "rga", image_letterbox);
    // 推理
    Inference();
    // 后处理
    Postprocess(image_letterbox, objects);

    letterbox_decode(objects, letterbox_info_.hor, letterbox_info_.pad);

    return NN_SUCCESS;
} */



/* #include "yolov8_custom.h"
#include <random>
#include "utils/logging.h"
#include "process/preprocess.h"
#include "process/postprocess.h"

// define global classes
static std::vector<std::string> g_classes = {
    "pedestrians", "riders", "partially-visible-person", "ignore-regions", "crowd"};

Yolov8Custom::Yolov8Custom()
{
    engine_ = CreateRKNNEngine();
    input_tensor_.data = nullptr;
    want_float_ = false; // 是否使用浮点数版本的后处理
    ready_ = false;
}

Yolov8Custom::~Yolov8Custom()
{
    NN_LOG_DEBUG("release input tensor");
    if (input_tensor_.data != nullptr)
    {
        free(input_tensor_.data);
        input_tensor_.data = nullptr;
    }
    NN_LOG_DEBUG("release output tensor");
    for (auto &tensor : output_tensors_)
    {
        if (tensor.data != nullptr)
        {
            free(tensor.data);
            tensor.data = nullptr;
        }
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

    if (output_shapes[0].type == NN_TENSOR_FLOAT16)
    {
        want_float_ = true;
        NN_LOG_WARNING("yolov8 output tensor type is float16, want type set to float32");
    }

    tensor_data_s tensor;
    tensor.attr = output_shapes[0];
    tensor.attr.type = want_float_ ? NN_TENSOR_FLOAT : output_shapes[0].type;
    tensor.attr.size = tensor.attr.n_elems * nn_tensor_type_to_size(tensor.attr.type);
    tensor.data = malloc(tensor.attr.size);
    output_tensors_.push_back(tensor);
    out_zps_.push_back(output_shapes[0].zp);
    out_scales_.push_back(output_shapes[0].scale);

    ready_ = true;
    return NN_SUCCESS;
}

nn_error_e Yolov8Custom::Preprocess(const cv::Mat &img, const std::string process_type, cv::Mat &image_letterbox)
{
    float wh_ratio = (float)input_tensor_.attr.dims[2] / (float)input_tensor_.attr.dims[1];
    if (process_type == "opencv")
    {
        letterbox_info_ = letterbox(img, image_letterbox, wh_ratio);
        cvimg2tensor(image_letterbox, input_tensor_.attr.dims[2], input_tensor_.attr.dims[1], input_tensor_);
    }
    else if (process_type == "rga")
    {
        letterbox_info_ = letterbox_rga(img, image_letterbox, wh_ratio);
        cvimg2tensor_rga(image_letterbox, input_tensor_.attr.dims[2], input_tensor_.attr.dims[1], input_tensor_);
    }
    return NN_SUCCESS;
}

nn_error_e Yolov8Custom::Inference()
{
    std::vector<tensor_data_s> inputs;
    inputs.push_back(input_tensor_);
    return engine_->Run(inputs, output_tensors_, want_float_);
}

nn_error_e Yolov8Custom::Postprocess(const cv::Mat &img, std::vector<Detection> &objects)
{
    std::vector<float> DetectiontRects;
    if (want_float_)
    {
        yolo::PostProcessSingleOutput((float *)output_tensors_[0].data, DetectiontRects);
    }
    else
    {
        yolo::PostProcessSingleOutputInt8((int8_t *)output_tensors_[0].data, out_zps_[0], out_scales_[0], DetectiontRects);
    }

    for (int i = 0; i < DetectiontRects.size(); i += 6)
    {
        int classId = int(DetectiontRects[i + 0]);
        float conf = DetectiontRects[i + 1];
        int xmin = int(DetectiontRects[i + 2] + 0.5);
        int ymin = int(DetectiontRects[i + 3] + 0.5);
        int xmax = int(DetectiontRects[i + 4] + 0.5);
        int ymax = int(DetectiontRects[i + 5] + 0.5);

        Detection result;
        result.class_id = classId;
        result.confidence = conf;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);
        result.color = cv::Scalar(dis(gen), dis(gen), dis(gen));

        result.className = g_classes[result.class_id];
        result.box = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);
        objects.push_back(result);
    }

    return NN_SUCCESS;
}

void letterbox_decode(std::vector<Detection> &objects, bool hor, int pad)
{
    for (auto &obj : objects)
    {
        if (hor)
            obj.box.x -= pad;
        else
            obj.box.y -= pad;
    }
}

nn_error_e Yolov8Custom::Run(const cv::Mat &img, std::vector<Detection> &objects)
{
    cv::Mat image_letterbox;
    Preprocess(img, "opencv", image_letterbox);
    Inference();
    Postprocess(image_letterbox, objects);
    letterbox_decode(objects, letterbox_info_.hor, letterbox_info_.pad);
    return NN_SUCCESS;
}
 */


 /* #include "yolov8_custom.h"
#include <random>
#include "utils/logging.h"
#include "process/preprocess.h"
#include "process/postprocess.h"

static std::vector<std::string> g_classes = {
    "pedestrians", "riders", "partially-visible-person", "ignore-regions", "crowd"
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
    float wh_ratio = (float)input_tensor_.attr.dims[2] / (float)input_tensor_.attr.dims[1];
    if (process_type == "opencv")
    {
        letterbox_info_ = letterbox(img, image_letterbox, wh_ratio);
        cvimg2tensor(image_letterbox, input_tensor_.attr.dims[2], input_tensor_.attr.dims[1], input_tensor_);
    }
    else if (process_type == "rga")
    {
        letterbox_info_ = letterbox_rga(img, image_letterbox, wh_ratio);
        cvimg2tensor_rga(image_letterbox, input_tensor_.attr.dims[2], input_tensor_.attr.dims[1], input_tensor_);
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
    std::vector<float> DetectiontRects;
    if (want_float_)
    {
        yolo::PostProcessSingleOutput((float *)output_tensor_.data, DetectiontRects);
    }
    else
    {
        yolo::PostProcessSingleOutputInt8((int8_t *)output_tensor_.data, out_zp_, out_scale_, DetectiontRects);
    }

    int img_width = img.cols;
    int img_height = img.rows;
    for (int i = 0; i < DetectiontRects.size(); i += 6)
    {
        int classId = int(DetectiontRects[i + 0]);
        float conf = DetectiontRects[i + 1];
        int xmin = int(DetectiontRects[i + 2] + 0.5);
        int ymin = int(DetectiontRects[i + 3] + 0.5);
        int xmax = int(DetectiontRects[i + 4] + 0.5);
        int ymax = int(DetectiontRects[i + 5] + 0.5);

        Detection result;
        result.class_id = classId;
        result.confidence = conf;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);
        result.color = cv::Scalar(dis(gen), dis(gen), dis(gen));

        result.className = g_classes[result.class_id];
        result.box = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);

        objects.push_back(result);
    }
    return NN_SUCCESS;
}

void letterbox_decode(std::vector<Detection> &objects, bool hor, int pad)
{
    for (auto &obj : objects)
    {
        if (hor)
            obj.box.x -= pad;
        else
            obj.box.y -= pad;
    }
}

nn_error_e Yolov8Custom::Run(const cv::Mat &img, std::vector<Detection> &objects)
{
    cv::Mat image_letterbox;
    Preprocess(img, "opencv", image_letterbox);
    Inference();
    Postprocess(image_letterbox, objects);
    letterbox_decode(objects, letterbox_info_.hor, letterbox_info_.pad);
    return NN_SUCCESS;
}
 */
//这是2d相机可以跑得代码
 /* #include "yolov8_custom.h"
#include <random>
#include "utils/logging.h"
#include "process/preprocess.h"
#include "process/postprocess.h"

static std::vector<std::string> g_classes = {
    "pedestrians", "riders", "partially-visible-person", "ignore-regions", "crowd"
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
        //letterbox_info_ = letterbox(img, image_letterbox, input_w, input_h);
        float wh_ratio = static_cast<float>(input_w) / input_h;
        //letterbox_info = letterbox(img, image_letterbox, wh_ratio);
        letterbox_info_ = letterbox(img, image_letterbox, wh_ratio);


        cvimg2tensor(image_letterbox, input_w, input_h, input_tensor_);
    }
    else if (process_type == "rga")
    {
        //letterbox_info_ = letterbox_rga(img, image_letterbox, input_w, input_h);
        //letterbox_info = letterbox_rga(img, image_letterbox, wh_ratio);
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
    int num_attrs = output_tensor_.attr.dims[1];  // 通常是 5：x,y,w,h,obj_conf

    if (want_float_)
    {
        yolo::PostProcessWithLetterBox(
            (float *)output_tensor_.data, objects,
            num_attrs, input_w, input_h,
            letterbox_info_, img.cols, img.rows);
    }
    else
    {
        // 如果你后续添加了 INT8 的 letterbox 版本处理函数，也应在这里调用
        // yolo::PostProcessWithLetterBoxInt8(...);
        NN_LOG_ERROR("INT8 letterbox后处理未实现！");
        return NN_RKNN_OUTPUT_ATTR_ERROR;
    }

    // 添加类名和颜色
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(100, 255);
    for (auto &result : objects)
    {
        result.color = cv::Scalar(dis(gen), dis(gen), dis(gen));
        result.className = (result.class_id >= 0 && result.class_id < g_classes.size()) ? g_classes[result.class_id] : "unknown";
          // ✅ 添加 letterbox 信息（你需要确保类里有 letterbox_info_ 成员变量）
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
    info = letterbox_info_;  // 将内部保存的 letterbox 信息传出
    return NN_SUCCESS;

} */

/* #include "yolov8_custom.h"
#include <random>
#include "utils/logging.h"
#include "process/preprocess.h"
#include "process/postprocess.h"


static std::vector<std::string> g_classes = {
    "pedestrians", "riders", "partially-visible-person", "ignore-regions", "crowd"
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
 */

 // 引入 "yolov8_custom.h" 头文件，其中包含了 Yolov8Custom 类的声明
#include "yolov8_custom.h"
// 引入 C++ 标准库头文件 <random>，用于生成随机数（为检测框分配随机颜色）
#include <random>
// 引入自定义的日志工具头文件
#include "utils/logging.h"
// 引入自定义的预处理函数头文件（例如 letterbox, cvimg2tensor）
#include "process/preprocess.h"
// 引入自定义的后处理函数头文件（例如 PostProcessWithLetterBox）
#include "process/postprocess.h"


// // 定义一个静态的全局字符串向量，存储模型可以识别的所有类别名称
// static std::vector<std::string> g_classes = {
//     "ball", "riders", "partially-visible-person", "ignore-regions", "crowd"
// };
    // 修改后的代码
    static std::vector<std::string> g_classes = {
        "ball" 
    };

// Yolov8Custom 类的构造函数
Yolov8Custom::Yolov8Custom()
{
    // 调用工厂函数创建一个推理引擎实例，并将其赋值给成员变量 engine_
    engine_ = CreateRKNNEngine();
    // 初始化输入张量的数据指针为空
    input_tensor_.data = nullptr;
    // 初始化标志位，默认不要求将输出转换为 float 类型
    want_float_ = false;
    // 初始化准备状态为 false，表示模型尚未加载成功
    ready_ = false;
}

// Yolov8Custom 类的析构函数
Yolov8Custom::~Yolov8Custom()
{
    // 检查输入张量的数据指针是否已被分配内存
    if (input_tensor_.data != nullptr)
    {
        // 如果已分配，则释放内存
        free(input_tensor_.data);
        // 将指针置为空，防止悬挂指针
        input_tensor_.data = nullptr;
    }
    // 检查输出张量的数据指针是否已被分配内存
    if (output_tensor_.data != nullptr)
    {
        // 如果已分配，则释放内存
        free(output_tensor_.data);
        // 将指针置为空
        output_tensor_.data = nullptr;
    }
}

// 加载模型文件并初始化相关资源
nn_error_e Yolov8Custom::LoadModel(const char *model_path)
{
    // 调用引擎的 LoadModelFile 方法加载模型
    auto ret = engine_->LoadModelFile(model_path);
    // 检查模型加载是否成功
    if (ret != NN_SUCCESS)
    {
        // 如果失败，则记录错误日志
        NN_LOG_ERROR("yolov8 load model file failed");
        // 返回错误码
        return ret;
    }
    // 获取模型的输入张量属性
    auto input_shapes = engine_->GetInputShapes();
    // 检查模型的输入张量数量是否为1
    if (input_shapes.size() != 1)
    {
        // 如果不是1，则记录错误日志
        NN_LOG_ERROR("yolov8 input tensor number is not 1, but %ld", input_shapes.size());
        // 返回错误码
        return NN_RKNN_INPUT_ATTR_ERROR;
    }
    // (假设存在辅助函数)将获取的张量属性转换为适合图像输入的自定义数据结构
    nn_tensor_attr_to_cvimg_input_data(input_shapes[0], input_tensor_);
    // 根据输入张量的属性大小，为其数据区分配内存
    input_tensor_.data = malloc(input_tensor_.attr.size);

    // 获取模型的输出张量属性
    auto output_shapes = engine_->GetOutputShapes();
    // 检查模型的输出张量数量是否为1
    if (output_shapes.size() != 1)
    {
        // 如果不是1，则记录错误日志
        NN_LOG_ERROR("yolov8 output tensor number is not 1, but %ld", output_shapes.size());
        // 返回错误码
        return NN_RKNN_OUTPUT_ATTR_ERROR;
    }
    // 判断是否需要将 float16 类型的输出转换为 float32 类型进行后处理
    want_float_ = (output_shapes[0].type == NN_TENSOR_FLOAT16);
    // 复制输出张量的属性
    output_tensor_.attr = output_shapes[0];
    // 如果需要转为 float32，则更新输出张量的类型属性
    output_tensor_.attr.type = want_float_ ? NN_TENSOR_FLOAT : output_shapes[0].type;
    // 设置输出张量的索引为0
    output_tensor_.attr.index = 0;
    // 重新计算输出张量数据区的大小（如果类型从 float16 变为 float32，大小会翻倍）
    output_tensor_.attr.size = output_shapes[0].n_elems * nn_tensor_type_to_size(output_tensor_.attr.type);
    // 根据计算出的大小，为输出张量的数据区分配内存
    output_tensor_.data = malloc(output_tensor_.attr.size);

    // 保存输出张量的量化参数（零点和缩放因子），用于 int8 模型
    out_zp_ = output_shapes[0].zp;
    out_scale_ = output_shapes[0].scale;

    // 设置准备状态为 true，表示模型已成功加载并准备好进行推理
    ready_ = true;
    // 返回成功状态码
    return NN_SUCCESS;
}

// 对输入图像进行预处理
nn_error_e Yolov8Custom::Preprocess(const cv::Mat &img, const std::string process_type, cv::Mat &image_letterbox)
{
    // 从输入张量属性中获取模型的输入宽度
    int input_w = input_tensor_.attr.dims[2];
    // 从输入张量属性中获取模型的输入高度
    int input_h = input_tensor_.attr.dims[1];
    // 计算模型的输入宽高比
    float wh_ratio = static_cast<float>(input_w) / input_h;

    // 根据指定的处理类型（"opencv" 或 "rga"）选择不同的实现
    if (process_type == "opencv")
    {
        // 使用 OpenCV 进行 letterbox 操作，并保存变换信息
        letterbox_info_ = letterbox(img, image_letterbox, wh_ratio);
        // 将 letterbox 处理后的图像转换为张量格式
        cvimg2tensor(image_letterbox, input_w, input_h, input_tensor_);
    }
    else if (process_type == "rga")
    {
        // 使用 RGA 硬件加速进行 letterbox 操作，并保存变换信息
        letterbox_info_ = letterbox_rga(img, image_letterbox, wh_ratio);
        // 将 letterbox 处理后的图像转换为张量格式（RGA版本）
        cvimg2tensor_rga(image_letterbox, input_w, input_h, input_tensor_);
    }
    // 返回成功状态码
    return NN_SUCCESS;
}

// 执行模型推理
nn_error_e Yolov8Custom::Inference()
{
    // 将输入张量放入一个向量中（因为引擎的 Run 接口需要向量作为参数）
    std::vector<tensor_data_s> inputs = {input_tensor_};
    // 将输出张量放入一个向量中
    std::vector<tensor_data_s> outputs = {output_tensor_};
    // 调用引擎的 Run 方法执行推理
    return engine_->Run(inputs, outputs, want_float_);
}

// 对模型输出进行后处理
nn_error_e Yolov8Custom::Postprocess(const cv::Mat &img, std::vector<Detection> &objects)
{
    // 从输入张量属性中获取模型的输入宽度
    int input_w = input_tensor_.attr.dims[2];
    // 从输入张量属性中获取模型的输入高度
    int input_h = input_tensor_.attr.dims[1];
    // 从输出张量属性中获取每个检测结果的属性数量
    int num_attrs = output_tensor_.attr.dims[1];

    // 检查是否处理的是浮点型输出
    if (want_float_)
    {
        // 调用浮点型的后处理函数
        yolo::PostProcessWithLetterBox(
            (float *)output_tensor_.data, objects, // 传递输出数据和结果容器
            num_attrs, input_w, input_h,            // 传递维度信息
            letterbox_info_, img.cols, img.rows);   // 传递 letterbox 信息和原图尺寸
    }
    else
    {
        // 如果是 INT8 量化模型，记录错误日志，因为此处的后处理尚未实现
        NN_LOG_ERROR("INT8 letterbox后处理未实现！");
        // 返回错误码
        return NN_RKNN_OUTPUT_ATTR_ERROR;
    }

    // --- 为每个检测结果分配随机颜色和类别名称 ---
    // 创建一个随机数设备
    std::random_device rd;
    // 使用梅森旋转算法作为随机数生成器引擎
    std::mt19937 gen(rd());
    // 定义一个均匀分布，用于生成 100 到 255 之间的整数
    std::uniform_int_distribution<int> dis(100, 255);
    // 遍历所有检测到的目标
    for (auto &result : objects)
    {
        // 为当前目标生成一个随机颜色
        result.color = cv::Scalar(dis(gen), dis(gen), dis(gen));
        // 根据类别ID从 g_classes 中查找对应的类别名称，并处理越界情况
        result.className = (result.class_id >= 0 && result.class_id < g_classes.size()) ? g_classes[result.class_id] : "unknown";
        // 将 letterbox 信息附加到检测结果中（用于后续绘图等操作）
        result.letterbox_info = letterbox_info_;
        // 设置标志位，表明此结果的坐标是基于 letterbox 图像的
        result.has_letterbox = true;
    }

    // 返回成功状态码
    return NN_SUCCESS;
}

// 运行完整的检测流程（预处理 -> 推理 -> 后处理）
nn_error_e Yolov8Custom::Run(const cv::Mat &img, std::vector<Detection> &objects)
{
    // 创建一个 cv::Mat 用于存放 letterbox 处理后的图像
    cv::Mat image_letterbox;
    // 执行预处理，默认使用 "opencv"
    Preprocess(img, "opencv", image_letterbox);
    // 执行推理
    Inference();
    // 执行后处理，注意这里传入的是 letterbox 处理后的图像
    Postprocess(image_letterbox, objects);
    // 返回成功状态码
    return NN_SUCCESS;
}

// 运行完整流程的另一个重载版本，可以返回 letterbox 信息
nn_error_e Yolov8Custom::Run(const cv::Mat &img, std::vector<Detection> &objects, LetterBoxInfo &info)
{
    // 创建一个 cv::Mat 用于存放 letterbox 处理后的图像
    cv::Mat image_letterbox;
    // 执行预处理
    Preprocess(img, "opencv", image_letterbox);
    // 执行推理
    Inference();
    // 执行后处理
    Postprocess(image_letterbox, objects);
    // 通过引用参数将本次处理的 letterbox 信息返回给调用者
    info = letterbox_info_;
    // 返回成功状态码
    return NN_SUCCESS;
}