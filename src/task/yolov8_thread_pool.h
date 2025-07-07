

#ifndef RK3588_DEMO_YOLOV8_THREAD_POOL_H
#define RK3588_DEMO_YOLOV8_THREAD_POOL_H

#include "yolov8_custom.h"
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <vector>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>

class Yolov8ThreadPool
{
public:
    Yolov8ThreadPool();
    ~Yolov8ThreadPool();

    nn_error_e setUp(std::string &model_path, int num_threads = 12);
    nn_error_e submitTask(const cv::Mat &img, int id);
    nn_error_e getTargetResult(std::vector<Detection> &objects, int id);
    nn_error_e getTargetImgResult(cv::Mat &img, int id);
    void stopAll();

    void SetDepthFrame(const rs2::depth_frame& frame, const rs2_intrinsics& intrin);
    rs2::depth_frame GetDepth();
    rs2_intrinsics GetIntrinsics() const;

private:
    void worker(int id);

    std::queue<std::pair<int, cv::Mat>> tasks;
    std::vector<std::shared_ptr<Yolov8Custom>> Yolov8_instances;
    std::map<int, std::vector<Detection>> results;
    std::map<int, cv::Mat> img_results;

    std::vector<std::thread> threads;
    std::mutex mtx1;
    std::mutex mtx2;
    std::condition_variable cv_task;
    bool stop;
    mutable std::mutex depth_mtx;
    rs2::frame depth_frame_;
    rs2_intrinsics intrinsics_;
    bool has_depth_ = false;
};

#endif // RK3588_DEMO_YOLOV8_THREAD_POOL_H
