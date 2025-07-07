#include <opencv2/opencv.hpp>
#include <iomanip>
#include <sstream>
#include "types/yolo_datatype.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"
#include "process/preprocess.h"

void scale_coords_back(const LetterBoxInfo& info, int net_input_w, int net_input_h,
    int origin_img_w, int origin_img_h, cv::Rect& box)
{
    float scale = info.scale;
    int pad_w = info.pad_w;
    int pad_h = info.pad_h;

    float x = (box.x - pad_w) / scale;
    float y = (box.y - pad_h) / scale;
    float w = box.width / scale;
    float h = box.height / scale;

    int x0 = std::max(int(x), 0);
    int y0 = std::max(int(y), 0);
    int x1 = std::min(int(x + w), origin_img_w);
    int y1 = std::min(int(y + h), origin_img_h);

    box = cv::Rect(cv::Point(x0, y0), cv::Point(x1, y1));
}

void DrawDetections(cv::Mat& img, const std::vector<Detection>& objects)
{
    for (const auto& object : objects)
    {
        cv::Rect box = object.box;
        if (object.has_letterbox)
        {
            scale_coords_back(object.letterbox_info, 640, 640, img.cols, img.rows, box);
        }

        cv::rectangle(img, box, object.color, 2);

        std::ostringstream oss;
        oss << object.className << " " << std::fixed << std::setprecision(2) << object.confidence;
        if (object.has_xyz)
        {
            oss << " | X:" << object.x << " Y:" << object.y << " Z:" << object.z;
        }

        cv::putText(img, oss.str(), cv::Point(box.x, std::max(0, box.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, object.color, 2);
    }
}
