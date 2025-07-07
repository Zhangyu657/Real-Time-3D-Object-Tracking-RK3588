/* #include "KalmanFilter3D.h"

// 构造函数
KalmanFilter3D::KalmanFilter3D() : initialized_(false) {
    // 状态量维度：6 (x, y, z, vx, vy, vz)
    // 测量量维度：3 (x, y, z)
    int stateDim = 6;
    int measureDim = 3;
    kf_ = cv::KalmanFilter(stateDim, measureDim, 0, CV_32F);

    // 状态转移矩阵 F - 匀速运动模型
    // x_k = x_{k-1} + vx_{k-1}*dt
    // y_k = y_{k-1} + vy_{k-1}*dt
    // ...
    // vx_k = vx_{k-1}
    // ...
    // dt=1 (因为我们是逐帧处理)
    cv::setIdentity(kf_.transitionMatrix);
    kf_.transitionMatrix.at<float>(0, 3) = 1.0f;
    kf_.transitionMatrix.at<float>(1, 4) = 1.0f;
    kf_.transitionMatrix.at<float>(2, 5) = 1.0f;

    // 测量矩阵 H - 我们只能测量到位置，测量不到速度
    // [x_m; y_m; z_m] = H * [x; y; z; vx; vy; vz]
    kf_.measurementMatrix = cv::Mat::zeros(measureDim, stateDim, CV_32F);
    kf_.measurementMatrix.at<float>(0, 0) = 1.0f;
    kf_.measurementMatrix.at<float>(1, 1) = 1.0f;
    kf_.measurementMatrix.at<float>(2, 2) = 1.0f;

    // 过程噪声协方差矩阵 Q - 代表我们对运动模型的信任程度
    // 值越小，代表我们越相信物体会保持匀速运动，轨迹会更平滑但可能跟不上突变
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-3));

    // 测量噪声协方差矩阵 R - 代表我们对测量值(3D质心)的信任程度
    // 值越大，代表我们认为测量值噪声越大、越不可信，轨迹会更平滑但响应更慢
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-2));

    // 后验误差协方差矩阵 P
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1));

    // 初始化测量矩阵
    measurement_ = cv::Mat(measureDim, 1, CV_32F);
}

// 初始化滤波器
void KalmanFilter3D::init(const cv::Point3f& initial_pos) {
    // 使用第一个测量点初始化状态向量
    kf_.statePost.at<float>(0) = initial_pos.x;
    kf_.statePost.at<float>(1) = initial_pos.y;
    kf_.statePost.at<float>(2) = initial_pos.z;
    // 初始速度设为0
    kf_.statePost.at<float>(3) = 0;
    kf_.statePost.at<float>(4) = 0;
    kf_.statePost.at<float>(5) = 0;
    initialized_ = true;
}

// 预测
cv::Point3f KalmanFilter3D::predict() {
    if (!initialized_) return cv::Point3f(0, 0, 0);
    cv::Mat prediction = kf_.predict();
    return cv::Point3f(prediction.at<float>(0), prediction.at<float>(1), prediction.at<float>(2));
}

// 更新
cv::Point3f KalmanFilter3D::update(const cv::Point3f& measurement) {
    if (!initialized_) {
        init(measurement);
        return measurement;
    }
    measurement_.at<float>(0) = measurement.x;
    measurement_.at<float>(1) = measurement.y;
    measurement_.at<float>(2) = measurement.z;

    cv::Mat estimated = kf_.correct(measurement_);
    return cv::Point3f(estimated.at<float>(0), estimated.at<float>(1), estimated.at<float>(2));
}

// 获取状态
cv::Point3f KalmanFilter3D::get_state() const {
    if (!initialized_) return cv::Point3f(0, 0, 0);
    cv::Mat state = kf_.statePost;
    return cv::Point3f(state.at<float>(0), state.at<float>(1), state.at<float>(2));
}
// <<<--- 新增：实现 predict_future 函数 ---<<<
std::vector<cv::Point3f> KalmanFilter3D::predict_future(int steps) {
    // 如果滤波器未初始化，返回空向量
    if (!initialized_) return {};

    std::vector<cv::Point3f> future_points;
    // 复制当前最新的状态，作为预测的起点，避免修改滤波器内部的真实状态
    cv::Mat temp_state = kf_.statePost.clone();

    // 循环预测指定的步数
    for (int i = 0; i < steps; ++i) {
        // 使用状态转移矩阵 F 乘以当前状态，得到下一个状态
        temp_state = kf_.transitionMatrix * temp_state;
        // 从预测出的状态向量中提取出位置信息 (x, y, z)
        future_points.emplace_back(temp_state.at<float>(0), temp_state.at<float>(1), temp_state.at<float>(2));
    }

    return future_points;
} */

#include "KalmanFilter3D.h"
#include <cmath>

KalmanFilter3D::KalmanFilter3D() : initialized_(false) {
    // 状态量维度：6 (x, y, z, vx, vy, vz)
    // 测量量维度：3 (x, y, z)
    int stateDim = 6;
    int measureDim = 3;
    kf_ = cv::KalmanFilter(stateDim, measureDim, 0, CV_32F);

    // 状态转移矩阵 F - 匀速运动模型，考虑时间间隔dt
    float dt = 1.0f; // 假设帧率为30fps
    kf_.transitionMatrix = (cv::Mat_<float>(6, 6) <<
        1, 0, 0, dt, 0, 0,
        0, 1, 0, 0, dt, 0,
        0, 0, 1, 0, 0, dt,
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1);

    // 测量矩阵 H - 只测量位置
    kf_.measurementMatrix = cv::Mat::zeros(measureDim, stateDim, CV_32F);
    kf_.measurementMatrix.at<float>(0, 0) = 1.0f;
    kf_.measurementMatrix.at<float>(1, 1) = 1.0f;
    kf_.measurementMatrix.at<float>(2, 2) = 1.0f;

    // 过程噪声协方差矩阵 Q - 设置不同维度的噪声
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-2));
    // 位置噪声比速度噪声大一些
    kf_.processNoiseCov.at<float>(0,0) = 1e-1;
    kf_.processNoiseCov.at<float>(1,1) = 1e-1;
    kf_.processNoiseCov.at<float>(2,2) = 1e-1;
    kf_.processNoiseCov.at<float>(3,3) = 1e-3;
    kf_.processNoiseCov.at<float>(4,4) = 1e-3;
    kf_.processNoiseCov.at<float>(5,5) = 1e-3;

    // 测量噪声协方差矩阵 R - 测量值的不确定性
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-1));

    // 后验误差协方差矩阵 P 初始值 - 设置为0.1而不是1.0
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(0.1));

    // 初始化测量矩阵
    measurement_ = cv::Mat(measureDim, 1, CV_32F);
}

void KalmanFilter3D::init(const cv::Point3f& initial_pos) {
    if (!is_valid_point(initial_pos)) {
        NN_LOG_ERROR("KalmanFilter3D: Trying to initialize with invalid point (%.2f, %.2f, %.2f)", 
                     initial_pos.x, initial_pos.y, initial_pos.z);
        return;
    }

    // 设置初始状态：位置为测量值，速度为0
    kf_.statePost = cv::Mat::zeros(6, 1, CV_32F);
    kf_.statePost.at<float>(0) = initial_pos.x;
    kf_.statePost.at<float>(1) = initial_pos.y;
    kf_.statePost.at<float>(2) = initial_pos.z;
    
    // 重置后验协方差
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(0.1));
    
    initialized_ = true;
    NN_LOG_INFO("KalmanFilter3D: Initialized at (%.2f, %.2f, %.2f)", 
                initial_pos.x, initial_pos.y, initial_pos.z);
}

cv::Point3f KalmanFilter3D::predict() {
    if (!initialized_) {
        return cv::Point3f(0,0,0);
    }
    cv::Mat prediction = kf_.predict();
    return cv::Point3f(prediction.at<float>(0), prediction.at<float>(1), prediction.at<float>(2));
}

cv::Point3f KalmanFilter3D::update(const cv::Point3f& measurement) {
    // 如果测量点无效，则跳过更新，返回预测值
    if (!is_valid_point(measurement)) {
        NN_LOG_WARNING("KalmanFilter3D: Invalid measurement (%.2f, %.2f, %.2f), skipping update", 
                       measurement.x, measurement.y, measurement.z);
        return predict();
    }
    
    // 如果尚未初始化，则用当前测量值初始化
    if (!initialized_) {
        init(measurement);
        return measurement;
    }
    
    // 设置测量值
    measurement_.at<float>(0) = measurement.x;
    measurement_.at<float>(1) = measurement.y;
    measurement_.at<float>(2) = measurement.z;
    
    // 执行卡尔曼更新
    cv::Mat estimated = kf_.correct(measurement_);
    return cv::Point3f(estimated.at<float>(0), estimated.at<float>(1), estimated.at<float>(2));
}

cv::Point3f KalmanFilter3D::get_state() const {
    if (!initialized_) {
        return cv::Point3f(0,0,0);
    }
    return cv::Point3f(kf_.statePost.at<float>(0), kf_.statePost.at<float>(1), kf_.statePost.at<float>(2));
}

/* std::vector<cv::Point3f> KalmanFilter3D::predict_future(int steps) {
    std::vector<cv::Point3f> future_points;
    if (!initialized_) {
        return future_points;
    }
    
    // 保存当前状态，以便预测后恢复
    cv::Mat original_state = kf_.statePost.clone();
    cv::Mat original_cov = kf_.errorCovPost.clone();
    
    for (int i = 0; i < steps; ++i) {
        cv::Mat prediction = kf_.predict(); // 预测下一步
        future_points.emplace_back(prediction.at<float>(0), prediction.at<float>(1), prediction.at<float>(2));
    }
    
    // 恢复状态
    kf_.statePost = original_state;
    kf_.errorCovPost = original_cov;
    
    return future_points;
} */

/* std::vector<cv::Point3f> KalmanFilter3D::predict_future(int steps, int interval) {
    std::vector<cv::Point3f> future_points;
    if (!initialized_) {
        return future_points;
    }

    // 保存当前状态
    cv::Mat original_state = kf_.statePost.clone();
    cv::Mat original_cov = kf_.errorCovPost.clone();

    // 使用一个临时 KalmanFilter 进行多次间隔预测
    cv::Mat temp_state = original_state.clone();
    cv::Mat temp_cov = original_cov.clone();

    for (int i = 0; i < steps; ++i) {
        // 连续预测 interval 步后作为一个预测点
        for (int j = 0; j < interval; ++j) {
            kf_.statePost = temp_state;
            kf_.errorCovPost = temp_cov;
            kf_.predict();
            temp_state = kf_.statePost.clone();
            temp_cov = kf_.errorCovPost.clone();
        }

        // 记录当前预测点
        future_points.emplace_back(
            temp_state.at<float>(0), temp_state.at<float>(1), temp_state.at<float>(2));
    }

    // 恢复原始状态
    kf_.statePost = original_state;
    kf_.errorCovPost = original_cov;

    return future_points;
}
 */

  std::vector<cv::Point3f> KalmanFilter3D::predict_future(int steps, int interval) {
    std::vector<cv::Point3f> future_points;
    if (!initialized_) return future_points;

    // 保存当前状态
    cv::Mat original_state = kf_.statePost.clone();
    cv::Mat original_cov = kf_.errorCovPost.clone();

    for (int i = 0; i < steps; ++i) {
        for (int j = 0; j < interval; ++j) {
            kf_.predict();  // 多次预测，增加时间步长
        }
        cv::Mat prediction = kf_.statePost.clone();  // 手动保存预测点
        future_points.emplace_back(prediction.at<float>(0),
                                   prediction.at<float>(1),
                                   prediction.at<float>(2));
    }

    // 恢复状态
    kf_.statePost = original_state;
    kf_.errorCovPost = original_cov;

    return future_points;
}
 

 /* std::vector<cv::Point3f> KalmanFilter3D::predict_future(int steps, float dt) {
    std::vector<cv::Point3f> future_points;
    if (!initialized_) return future_points;

    // 保存原始状态
    cv::Mat original_state = kf_.statePost.clone();
    cv::Mat original_F = kf_.transitionMatrix.clone();  // 保存原始转移矩阵

    // 设置新的时间步长 (关键修改!)
    cv::Mat F = original_F.clone();
    F.at<float>(0, 3) = dt;  // 位置-速度关系: x = x0 + vx*dt
    F.at<float>(1, 4) = dt;
    F.at<float>(2, 5) = dt;
    kf_.transitionMatrix = F;

    cv::Mat current_state = original_state.clone();
    for (int i = 0; i < steps; ++i) {
        // 单次预测 (不再需要内层循环)
        kf_.statePost = current_state;
        cv::Mat prediction = kf_.predict();
        
        // 提取位置 (假设状态为 [x,y,z,vx,vy,vz])
        future_points.emplace_back(
            prediction.at<float>(0),
            prediction.at<float>(1),
            prediction.at<float>(2)
        );
        
        // 更新状态用于下一步预测
        current_state = prediction;
    }

    // 恢复原始状态和矩阵
    kf_.transitionMatrix = original_F;
    kf_.statePost = original_state;
    
    return future_points;
} */

/* std::vector<cv::Point3f> KalmanFilter3D::predict_future(int steps, float dt) {
    std::vector<cv::Point3f> future_points;
    if (!initialized_) return future_points;

    // 保存原始状态
    cv::Mat original_state = kf_.statePost.clone();
    cv::Mat original_cov = kf_.errorCovPost.clone();
    cv::Mat original_F = kf_.transitionMatrix.clone();

    // 设置新的时间步长 (关键!)
    cv::Mat F = original_F.clone();
    F.at<float>(0, 3) = dt;  // x = x0 + vx*dt
    F.at<float>(1, 4) = dt;  // y = y0 + vy*dt
    F.at<float>(2, 5) = dt;  // z = z0 + vz*dt
    F.at<float>(5, 5) = 1.0; // 考虑重力加速度 g
    kf_.transitionMatrix = F;

    // 添加重力影响 (如果适用)
    cv::Mat control = cv::Mat::zeros(6, 1, CV_32F);
    control.at<float>(5) = -0.5 * 9.8 * dt * dt;  // z方向重力加速度
    
    cv::Mat current_state = original_state.clone();
    for (int i = 0; i < steps; ++i) {
        // 设置当前状态
        kf_.statePost = current_state;
        
        // 预测下一步 (考虑控制量)
        cv::Mat prediction = kf_.predict(control);
        
        // 提取位置 (假设状态为 [x,y,z,vx,vy,vz])
        float x = prediction.at<float>(0);
        float y = prediction.at<float>(1);
        float z = prediction.at<float>(2);
        
        // 调试输出
        std::cout << "预测点 " << i << ": (" << x << ", " << y << ", " << z << ")\n";
        
        future_points.emplace_back(x, y, z);
        
        // 更新状态用于下一步
        current_state = prediction;
    }

    // 恢复原始状态
    kf_.transitionMatrix = original_F;
    kf_.statePost = original_state;
    kf_.errorCovPost = original_cov;
    
    return future_points;
} */