#include "yolov8_thread_pool.h"
#include "draw/cv_draw.h"
#include <librealsense2/rs.hpp>
#include <numeric>
#include "utils/logging.h"

Yolov8ThreadPool::Yolov8ThreadPool() {
    stop = false;
    has_depth_ = false; 
}

Yolov8ThreadPool::~Yolov8ThreadPool() {
    stopAll();
    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

nn_error_e Yolov8ThreadPool::setUp(std::string &model_path, int num_threads) {
    for (int i = 0; i < num_threads; ++i) {
        std::shared_ptr<Yolov8Custom> Yolov8 = std::make_shared<Yolov8Custom>();
        if (Yolov8->LoadModel(model_path.c_str()) != NN_SUCCESS) {
            NN_LOG_ERROR("Thread %d: model load failed", i);
            return NN_RKNN_INPUT_ATTR_ERROR;
        }
        Yolov8_instances.push_back(Yolov8);
    }

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(&Yolov8ThreadPool::worker, this, i);
    }

    return NN_SUCCESS;
}

nn_error_e Yolov8ThreadPool::submitTask(const cv::Mat &img, int id) {
    if (stop) {
        return NN_TIMEOUT; 
    }

    while (tasks.size() > 10) {
        if (stop) return NN_TIMEOUT;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
        std::lock_guard<std::mutex> lock(mtx1);
        tasks.push({id, img});
    }
    cv_task.notify_one();
    return NN_SUCCESS;
}

void Yolov8ThreadPool::worker(int id) {
   NN_LOG_INFO("[Worker %d] Thread started and running.", id);

   while (!stop) {
       std::pair<int, cv::Mat> task;
       std::shared_ptr<Yolov8Custom> instance = Yolov8_instances[id];

       {
           std::unique_lock<std::mutex> lock(mtx1);
           cv_task.wait(lock, [&]() { return !tasks.empty() || stop; });

           if (stop) {
               NN_LOG_INFO("[Worker %d] Stop signal received, exiting.", id);
               return;
           }

           task = tasks.front();
           tasks.pop();
           NN_LOG_INFO("[Worker %d] Got task for frame %d. Remaining tasks in queue: %zu", id, task.first, tasks.size());
       }

       std::vector<Detection> detections;
       LetterBoxInfo info;

       try {
           {
               std::lock_guard<std::mutex> lock(depth_mtx);
               if (has_depth_) {
                   instance->SetDepthContext(depth_frame_, intrinsics_);
               }
           }

           NN_LOG_INFO("[Worker %d] ==> Starting instance->Run() for frame %d.", id, task.first);
           instance->Run(task.second, detections, info);
           NN_LOG_INFO("[Worker %d] <== Finished instance->Run() for frame %d. Detections found: %zu", id, task.first, detections.size());

       } catch (const std::exception& e) {
           NN_LOG_ERROR("[Worker %d] CRITICAL: Exception caught while processing frame %d: %s", id, task.first, e.what());
           continue; 
       } catch (...) {
           NN_LOG_ERROR("[Worker %d] CRITICAL: Unknown exception caught while processing frame %d.", id, task.first);
           continue;
       }

       {
           std::lock_guard<std::mutex> lock(mtx2);
           NN_LOG_INFO("[Worker %d] Storing results for frame %d.", id, task.first);

           results[task.first] = detections;
           cv::Mat vis_img = task.second.clone();
           img_results[task.first] = vis_img;

           NN_LOG_INFO("[Worker %d] Finished storing results for frame %d.", id, task.first);
       }
   }
}

nn_error_e Yolov8ThreadPool::getTargetResult(std::vector<Detection> &objects, int id) {
    while (results.find(id) == results.end()) {
        if (stop) {
            NN_LOG_WARNING("Pool is stopping, getTargetResult aborted.");
            return NN_TIMEOUT; 
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
        std::lock_guard<std::mutex> lock(mtx2);
        if (results.find(id) == results.end()) {
             return NN_TIMEOUT;
        }
        objects = results[id];
        results.erase(id);
    }

    return NN_SUCCESS;
}

nn_error_e Yolov8ThreadPool::getTargetImgResult(cv::Mat &img, int id) {
    int loop_cnt = 0;
    while (img_results.find(id) == img_results.end()) {
        if (stop) {
            NN_LOG_WARNING("Pool is stopping, getTargetImgResult aborted.");
            return NN_TIMEOUT;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (++loop_cnt > 1000) {
            NN_LOG_ERROR("getTargetImgResult timeout for frame %d", id);
            return NN_TIMEOUT;
        }
    }

    {
        std::lock_guard<std::mutex> lock(mtx2);
        if (img_results.find(id) == img_results.end()) {
             return NN_TIMEOUT;
        }
        img = img_results[id].clone();
        img_results.erase(id);
    }

    return NN_SUCCESS;
}

void Yolov8ThreadPool::stopAll() {
    stop = true;
    cv_task.notify_all();
}

void Yolov8ThreadPool::SetDepthFrame(const rs2::depth_frame& frame, const rs2_intrinsics& intrin) {
    std::lock_guard<std::mutex> lock(depth_mtx);
    depth_frame_ = frame;
    intrinsics_ = intrin;
    has_depth_ = true;
}

rs2::depth_frame Yolov8ThreadPool::GetDepth() {
    std::lock_guard<std::mutex> lock(depth_mtx);
    return rs2::depth_frame(depth_frame_);
}

rs2_intrinsics Yolov8ThreadPool::GetIntrinsics() {
    std::lock_guard<std::mutex> lock(depth_mtx);
    return intrinsics_;
}
