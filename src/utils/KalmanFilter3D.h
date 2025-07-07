
#ifndef KALMAN_FILTER_3D_H
#define KALMAN_FILTER_3D_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "logging.h"

class KalmanFilter3D {
public:
    KalmanFilter3D();
    void init(const cv::Point3f& initial_pos);
    cv::Point3f predict();
    cv::Point3f update(const cv::Point3f& measurement);
    cv::Point3f get_state() const;
    
    // 检查点是否有效
    static bool is_valid_point(const cv::Point3f& pt) {
        return !(std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z)) && 
               pt.z > 0.1f && pt.z < 10.0f;
    }
    
    // 预测未来轨迹
   // std::vector<cv::Point3f> predict_future(int steps);
    std::vector<cv::Point3f> predict_future(int steps, int interval);
   // std::vector<cv::Point3f> predict_future(int steps, float dt);
   
private:
    cv::KalmanFilter kf_;
    bool initialized_;
    cv::Mat measurement_; // 测量向量
};

#endif // KALMAN_FILTER_3D_H