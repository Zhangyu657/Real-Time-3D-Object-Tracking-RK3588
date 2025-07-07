

#include "preprocess.h"
#include "utils/logging.h"
#include "im2d.h"
#include "rga.h"

LetterBoxInfo letterbox(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio)
{
    if (img.channels() != 3)
    {
        NN_LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }

    float img_width = img.cols;
    float img_height = img.rows;

    float scale = std::min(640.0f / img_width, 640.0f / img_height);
    int resized_w = static_cast<int>(round(img_width * scale));
    int resized_h = static_cast<int>(round(img_height * scale));

    int pad_w = (640 - resized_w) / 2;
    int pad_h = (640 - resized_h) / 2;

    LetterBoxInfo info;
    info.scale = scale;
    info.pad_w = pad_w;
    info.pad_h = pad_h;
    info.input_w = 640;
    info.input_h = 640;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resized_w, resized_h));

    cv::copyMakeBorder(resized, img_letterbox,
                       pad_h, 640 - resized_h - pad_h, 
                       pad_w, 640 - resized_w - pad_w, 
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0)); 
    return info;
}

void cvimg2tensor(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor)
{
    if (img.channels() != 3)
    {
        NN_LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }
    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

    cv::Mat img_resized;
    cv::resize(img_rgb, img_resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    memcpy(tensor.data, img_resized.data, tensor.attr.size);
}

void cvimg2tensor_rga(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor)
{
    if (img.channels() != 3)
    {
        NN_LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }

    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));

    rga_buffer_t src = wrapbuffer_virtualaddr((void *)img_rgb.data, img.cols, img.rows, RK_FORMAT_RGB_888);
    rga_buffer_t dst = wrapbuffer_virtualaddr((void *)tensor.data, width, height, RK_FORMAT_RGB_888);

    int ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret)
    {
        NN_LOG_ERROR("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        exit(-1);
    }
    imresize(src, dst);
}

LetterBoxInfo letterbox_rga(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio)
{
    if (img.channels() != 3)
    {
        NN_LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }

    float img_width = img.cols;
    float img_height = img.rows;

    float scale = std::min(640.0f / img_width, 640.0f / img_height);
    int resized_w = static_cast<int>(round(img_width * scale));
    int resized_h = static_cast<int>(round(img_height * scale));

    int pad_w = (640 - resized_w) / 2;
    int pad_h = (640 - resized_h) / 2;

    LetterBoxInfo info;
    info.scale = scale;
    info.pad_w = pad_w;
    info.pad_h = pad_h;
    info.input_w = 640;
    info.input_h = 640;

    img_letterbox = cv::Mat::zeros(640, 640, CV_8UC3);

    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(img_rgb, resized, cv::Size(resized_w, resized_h));

    im_rect src_rect = {0};
    im_rect dst_rect = {0};

    rga_buffer_t src = wrapbuffer_virtualaddr((void *)resized.data, resized.cols, resized.rows, RK_FORMAT_RGB_888);
    rga_buffer_t dst = wrapbuffer_virtualaddr((void *)img_letterbox.data, img_letterbox.cols, img_letterbox.rows, RK_FORMAT_RGB_888);

    int ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret)
    {
        NN_LOG_ERROR("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        exit(-1);
    }

    immakeBorder(src, dst,
                 pad_h, 640 - resized_h - pad_h,
                 pad_w, 640 - resized_w - pad_w,
                 0, 0, 0);

    return info;
}
