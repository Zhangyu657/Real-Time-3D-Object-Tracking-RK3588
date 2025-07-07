/* #ifndef KALMAN_FILTER_3D_H
#define KALMAN_FILTER_3D_H

#include <opencv2/video/tracking.hpp>

class KalmanFilter3D {
public:
    // 构造函数
    KalmanFilter3D();

    // 用第一个有效的3D点来初始化滤波器
    void init(const cv::Point3f& initial_pos);

    // 预测下一时刻的状态
    cv::Point3f predict();

    // 用新的测量值来更新滤波器
    cv::Point3f update(const cv::Point3f& measurement);

    // 获取当前平滑后的位置状态
    cv::Point3f get_state() const;
    // <<<--- 新增：预测未来多个步骤的状态 ---<<<
    std::vector<cv::Point3f> predict_future(int steps);

private:
    cv::KalmanFilter kf_;        // OpenCV的卡尔曼滤波器对象
    cv::Mat measurement_;    // 用于存放测量值的矩阵
    bool initialized_;       // 标记是否已初始化
};

#endif // KALMAN_FILTER_3D_H */

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