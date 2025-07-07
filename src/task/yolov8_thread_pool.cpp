/* 
#include "yolov8_thread_pool.h"
#include "draw/cv_draw.h"
// 构造函数
Yolov8ThreadPool::Yolov8ThreadPool() { stop = false; }

// 析构函数
Yolov8ThreadPool::~Yolov8ThreadPool()
{
    // stop all threads
    stop = true;
    cv_task.notify_all();
    for (auto &thread : threads)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
}
// 初始化：加载模型，创建线程，参数：模型路径，线程数量
nn_error_e Yolov8ThreadPool::setUp(std::string &model_path, int num_threads)
{
    // 遍历线程数量，创建模型实例，放入vector
    // 这些线程加载的模型是同一个
    for (size_t i = 0; i < num_threads; ++i)
    {
        std::shared_ptr<Yolov8Custom> Yolov8 = std::make_shared<Yolov8Custom>();
        Yolov8->LoadModel(model_path.c_str());
        Yolov8_instances.push_back(Yolov8);
    }
    // 遍历线程数量，创建线程
    for (size_t i = 0; i < num_threads; ++i)
    {
        threads.emplace_back(&Yolov8ThreadPool::worker, this, i);
    }
    return NN_SUCCESS;
}

// 线程函数。参数：线程id
void Yolov8ThreadPool::worker(int id)
{
    while (!stop)
    {
        std::pair<int, cv::Mat> task;
        std::shared_ptr<Yolov8Custom> instance = Yolov8_instances[id]; // 获取模型实例
        {
            // 获取任务
            std::unique_lock<std::mutex> lock(mtx1);
            cv_task.wait(lock, [&]
                         { return !tasks.empty() || stop; });

            if (stop)
            {
                return;
            }

            task = tasks.front();
            tasks.pop();
        }
        // 运行模型
        std::vector<Detection> detections;
        instance->Run(task.second, detections);

        {
            // 保存结果
            std::lock_guard<std::mutex> lock(mtx2);
            results.insert({task.first, detections});
            DrawDetections(task.second, detections);
            img_results.insert({task.first, task.second});
            // cv_result.notify_one();
        }
    }
}
// 提交任务，参数：图片，id（帧号）
nn_error_e Yolov8ThreadPool::submitTask(const cv::Mat &img, int id)
{
    // 如果任务队列中的任务数量大于10，等待，避免内存占用过多
    while (tasks.size() > 10)
    {
        // sleep 1ms
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
        // 保存任务
        std::lock_guard<std::mutex> lock(mtx1);
        tasks.push({id, img});
    }
    cv_task.notify_one();
    return NN_SUCCESS;
}

// 获取结果，参数：检测框，id（帧号）
nn_error_e Yolov8ThreadPool::getTargetResult(std::vector<Detection> &objects, int id)
{
    // 如果没有结果，等待
    while (results.find(id) == results.end())
    {
        // sleep 1ms
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::lock_guard<std::mutex> lock(mtx2);
    objects = results[id];
    // remove from map
    results.erase(id);

    return NN_SUCCESS;
}

// 获取结果（图片），参数：图片，id（帧号）
nn_error_e Yolov8ThreadPool::getTargetImgResult(cv::Mat &img, int id)
{
    int loop_cnt = 0;
    // 如果没有结果，等待
    while (img_results.find(id) == img_results.end())
    {
        // 等待 5ms x 1000 = 5s
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        loop_cnt++;
        if (loop_cnt > 1000)
        {
            NN_LOG_ERROR("getTargetImgResult timeout");
            return NN_TIMEOUT;
        }
    }
    std::lock_guard<std::mutex> lock(mtx2);
    img = img_results[id];
    // remove from map
    img_results.erase(id);

    return NN_SUCCESS;
}
// 停止所有线程
void Yolov8ThreadPool::stopAll()
{
    stop = true;
    cv_task.notify_all();
} */

/* #include "yolov8_thread_pool.h"
#include "draw/cv_draw.h"

// 构造函数
Yolov8ThreadPool::Yolov8ThreadPool() { stop = false; }

// 析构函数
Yolov8ThreadPool::~Yolov8ThreadPool()
{
    stop = true;
    cv_task.notify_all();
    for (auto &thread : threads)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
}

// 初始化：加载模型，创建线程，参数：模型路径，线程数量
nn_error_e Yolov8ThreadPool::setUp(std::string &model_path, int num_threads)
{
    for (size_t i = 0; i < num_threads; ++i)
    {
        std::shared_ptr<Yolov8Custom> Yolov8 = std::make_shared<Yolov8Custom>();
        Yolov8->LoadModel(model_path.c_str());
        Yolov8_instances.push_back(Yolov8);
    }
    for (size_t i = 0; i < num_threads; ++i)
    {
        threads.emplace_back(&Yolov8ThreadPool::worker, this, i);
    }
    return NN_SUCCESS;
}

// 线程函数。参数：线程id
void Yolov8ThreadPool::worker(int id)
{
    while (!stop)
    {
        std::pair<int, cv::Mat> task;
        std::shared_ptr<Yolov8Custom> instance = Yolov8_instances[id];
        {
            std::unique_lock<std::mutex> lock(mtx1);
            cv_task.wait(lock, [&] { return !tasks.empty() || stop; });

            if (stop)
            {
                return;
            }

            task = tasks.front();
            tasks.pop();
        }

        std::vector<Detection> detections;
        instance->Run(task.second, detections);

        {
            std::lock_guard<std::mutex> lock(mtx2);
            results.insert({task.first, detections});

            // 直接使用 cv_draw 中的 DrawDetections 绘制框
            cv::Mat vis_img = task.second.clone();
            DrawDetections(vis_img, detections);
            img_results.insert({task.first, vis_img});
        }
    }
}

nn_error_e Yolov8ThreadPool::submitTask(const cv::Mat &img, int id)
{
    while (tasks.size() > 10)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
        std::lock_guard<std::mutex> lock(mtx1);
        tasks.push({id, img});
    }
    cv_task.notify_one();
    return NN_SUCCESS;
}

nn_error_e Yolov8ThreadPool::getTargetResult(std::vector<Detection> &objects, int id)
{
    while (results.find(id) == results.end())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::lock_guard<std::mutex> lock(mtx2);
    objects = results[id];
    results.erase(id);

    return NN_SUCCESS;
}

nn_error_e Yolov8ThreadPool::getTargetImgResult(cv::Mat &img, int id)
{
    int loop_cnt = 0;
    while (img_results.find(id) == img_results.end())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        loop_cnt++;
        if (loop_cnt > 1000)
        {
            NN_LOG_ERROR("getTargetImgResult timeout");
            return NN_TIMEOUT;
        }
    }
    std::lock_guard<std::mutex> lock(mtx2);
    img = img_results[id];
    img_results.erase(id);

    return NN_SUCCESS;
}

void Yolov8ThreadPool::stopAll()
{
    stop = true;
    cv_task.notify_all();
}
 */


 //这是2d相机运行代码
/*  #include "yolov8_thread_pool.h"
#include "draw/cv_draw.h"

// 构造函数
Yolov8ThreadPool::Yolov8ThreadPool() {
    stop = false;
}

// 析构函数：释放线程资源
Yolov8ThreadPool::~Yolov8ThreadPool() {
    stop = true;
    cv_task.notify_all();
    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

// 初始化：加载模型，创建线程
nn_error_e Yolov8ThreadPool::setUp(std::string &model_path, int num_threads) {
    for (int i = 0; i < num_threads; ++i) {
        std::shared_ptr<Yolov8Custom> Yolov8 = std::make_shared<Yolov8Custom>();
        if (Yolov8->LoadModel(model_path.c_str()) != NN_SUCCESS) {
            NN_LOG_ERROR("Thread %d: model load failed", i);
            //return NN_MODEL_LOAD_FAIL;
            return NN_RKNN_INPUT_ATTR_ERROR;  // 根据你的语义选择一个已有错误码


        }
        Yolov8_instances.push_back(Yolov8);
    }

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(&Yolov8ThreadPool::worker, this, i);
    }

    return NN_SUCCESS;
}

// 线程执行函数
void Yolov8ThreadPool::worker(int id) {
    while (!stop) {
        std::pair<int, cv::Mat> task;
        std::shared_ptr<Yolov8Custom> instance = Yolov8_instances[id];

        {
            std::unique_lock<std::mutex> lock(mtx1);
            cv_task.wait(lock, [&]() { return !tasks.empty() || stop; });

            if (stop) return;

            task = tasks.front();
            tasks.pop();
        }

      
        std::vector<Detection> detections;
        LetterBoxInfo info;
        instance->Run(task.second, detections, info);  // 注意这里多传入 info
        
        // 可视化图像绘制
        {
            std::lock_guard<std::mutex> lock(mtx2);
            results[task.first] = detections;
        
            cv::Mat vis_img = task.second.clone();
            //DrawDetections(vis_img, detections, info);  // 用 info 恢复坐标
            DrawDetections(vis_img, detections);        // ✅ 正确，只传两个参数

            img_results[task.first] = vis_img;
        }
        
    }
}

// 提交任务
nn_error_e Yolov8ThreadPool::submitTask(const cv::Mat &img, int id) {
    while (tasks.size() > 10) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
        std::lock_guard<std::mutex> lock(mtx1);
        tasks.push({id, img});
    }
    cv_task.notify_one();
    return NN_SUCCESS;
}

// 获取推理结果（Detection结构体）
nn_error_e Yolov8ThreadPool::getTargetResult(std::vector<Detection> &objects, int id) {
    while (results.find(id) == results.end()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
        std::lock_guard<std::mutex> lock(mtx2);
        objects = results[id];
        results.erase(id);
    }

    return NN_SUCCESS;
}

// 获取可视化图像结果
nn_error_e Yolov8ThreadPool::getTargetImgResult(cv::Mat &img, int id) {
    int loop_cnt = 0;
    while (img_results.find(id) == img_results.end()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (++loop_cnt > 1000) {
            NN_LOG_ERROR("getTargetImgResult timeout");
            return NN_TIMEOUT;
        }
    }

    {
        std::lock_guard<std::mutex> lock(mtx2);
        img = img_results[id];
        img_results.erase(id);
    }

    return NN_SUCCESS;
}

// 停止所有线程
void Yolov8ThreadPool::stopAll() {
    stop = true;
    cv_task.notify_all();
}
 */

 /* //这是3d
 #include "yolov8_thread_pool.h"
#include "draw/cv_draw.h"
#include <librealsense2/rs.hpp>

// 构造函数
Yolov8ThreadPool::Yolov8ThreadPool() {
    stop = false;
}

// 析构函数：释放线程资源
Yolov8ThreadPool::~Yolov8ThreadPool() {
    stop = true;
    cv_task.notify_all();
    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

// 初始化：加载模型，创建线程
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

// 线程执行函数
void Yolov8ThreadPool::worker(int id) {
    while (!stop) {
        std::pair<int, cv::Mat> task;
        std::shared_ptr<Yolov8Custom> instance = Yolov8_instances[id];

        {
            std::unique_lock<std::mutex> lock(mtx1);
            cv_task.wait(lock, [&]() { return !tasks.empty() || stop; });

            if (stop) return;

            task = tasks.front();
            tasks.pop();
        }

        std::vector<Detection> detections;
        LetterBoxInfo info;

        {
            std::lock_guard<std::mutex> lock(depth_mtx);
            if (has_depth_) {
                instance->SetDepthContext(depth_frame_, intrinsics_);
            }
        }

        instance->Run(task.second, detections, info);

        // 可视化图像绘制
        {
            std::lock_guard<std::mutex> lock(mtx2);
            results[task.first] = detections;

            cv::Mat vis_img = task.second.clone();
            DrawDetections(vis_img, detections);
            img_results[task.first] = vis_img;
        }
    }
}

// 提交任务
nn_error_e Yolov8ThreadPool::submitTask(const cv::Mat &img, int id) {
    while (tasks.size() > 10) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
        std::lock_guard<std::mutex> lock(mtx1);
        tasks.push({id, img});
    }
    cv_task.notify_one();
    return NN_SUCCESS;
}

// 获取推理结果（Detection结构体）
nn_error_e Yolov8ThreadPool::getTargetResult(std::vector<Detection> &objects, int id) {
    while (results.find(id) == results.end()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
        std::lock_guard<std::mutex> lock(mtx2);
        objects = results[id];
        results.erase(id);
    }

    return NN_SUCCESS;
}

// 获取可视化图像结果
nn_error_e Yolov8ThreadPool::getTargetImgResult(cv::Mat &img, int id) {
    int loop_cnt = 0;
    while (img_results.find(id) == img_results.end()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (++loop_cnt > 1000) {
            NN_LOG_ERROR("getTargetImgResult timeout");
            return NN_TIMEOUT;
        }
    }

    {
        std::lock_guard<std::mutex> lock(mtx2);
        img = img_results[id];
        img_results.erase(id);
    }

    return NN_SUCCESS;
}

// 停止所有线程
void Yolov8ThreadPool::stopAll() {
    stop = true;
    cv_task.notify_all();
}

// 设置深度信息（外部调用）
void Yolov8ThreadPool::SetDepthFrame(const rs2::depth_frame& frame, const rs2_intrinsics& intrin) {
    std::lock_guard<std::mutex> lock(depth_mtx);
    depth_frame_ = frame;
    intrinsics_ = intrin;
    has_depth_ = true;
}
 */


 /* #include "yolov8_thread_pool.h"
#include "draw/cv_draw.h"
#include <librealsense2/rs.hpp>
#include <numeric>

// 构造函数
Yolov8ThreadPool::Yolov8ThreadPool() {
    stop = false;
}

// 析构函数：释放线程资源
Yolov8ThreadPool::~Yolov8ThreadPool() {
    stop = true;
    cv_task.notify_all();
    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

// 初始化：加载模型，创建线程
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

// 线程执行函数
void Yolov8ThreadPool::worker(int id) {
    while (!stop) {
        std::pair<int, cv::Mat> task;
        std::shared_ptr<Yolov8Custom> instance = Yolov8_instances[id];

        {
            std::unique_lock<std::mutex> lock(mtx1);
            cv_task.wait(lock, [&]() { return !tasks.empty() || stop; });

            if (stop) return;

            task = tasks.front();
            tasks.pop();
        }

        std::vector<Detection> detections;
        LetterBoxInfo info;

        {
            std::lock_guard<std::mutex> lock(depth_mtx);
            if (has_depth_) {
                instance->SetDepthContext(depth_frame_, intrinsics_);
            }
        }

        instance->Run(task.second, detections, info);

        // 可视化图像绘制
        {
            std::lock_guard<std::mutex> lock(mtx2);
            results[task.first] = detections;

            cv::Mat vis_img = task.second.clone();
            DrawDetections(vis_img, detections);
            img_results[task.first] = vis_img;
        }
    }
}

// 提交任务
nn_error_e Yolov8ThreadPool::submitTask(const cv::Mat &img, int id) {
    while (tasks.size() > 10) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
        std::lock_guard<std::mutex> lock(mtx1);
        tasks.push({id, img});
    }
    cv_task.notify_one();
    return NN_SUCCESS;
}

// 获取推理结果（Detection结构体）
nn_error_e Yolov8ThreadPool::getTargetResult(std::vector<Detection> &objects, int id) {
    while (results.find(id) == results.end()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
        std::lock_guard<std::mutex> lock(mtx2);
        objects = results[id];
        results.erase(id);
    }

    return NN_SUCCESS;
}

// 获取可视化图像结果
nn_error_e Yolov8ThreadPool::getTargetImgResult(cv::Mat &img, int id) {
    int loop_cnt = 0;
    while (img_results.find(id) == img_results.end()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (++loop_cnt > 1000) {
            NN_LOG_ERROR("getTargetImgResult timeout");
            return NN_TIMEOUT;
        }
    }

    {
        std::lock_guard<std::mutex> lock(mtx2);
        img = img_results[id];
        img_results.erase(id);
    }

    return NN_SUCCESS;
}

// 停止所有线程
void Yolov8ThreadPool::stopAll() {
    stop = true;
    cv_task.notify_all();
}

// 设置深度信息（外部调用）
void Yolov8ThreadPool::SetDepthFrame(const rs2::depth_frame& frame, const rs2_intrinsics& intrin) {
    std::lock_guard<std::mutex> lock(depth_mtx);
    depth_frame_ = frame;
    intrinsics_ = intrin;
    has_depth_ = true;
}

// 获取深度帧
rs2::depth_frame Yolov8ThreadPool::GetDepth() const {
    std::lock_guard<std::mutex> lock(depth_mtx);
    return rs2::depth_frame(depth_frame_);
}


// 获取相机内参
rs2_intrinsics Yolov8ThreadPool::GetIntrinsics() {
    std::lock_guard<std::mutex> lock(depth_mtx);
    return intrinsics_;
}
 */

 /* #include "yolov8_thread_pool.h"
 #include "draw/cv_draw.h"
 #include <librealsense2/rs.hpp>
 #include <numeric>
 #include "utils/logging.h" // 确保包含了您的日志头文件
 // 构造函数
 Yolov8ThreadPool::Yolov8ThreadPool() {
     stop = false;
     has_depth_ = false; 
 }
 
 // 析构函数：释放线程资源
 Yolov8ThreadPool::~Yolov8ThreadPool() {
     stopAll();
     for (auto &thread : threads) {
         if (thread.joinable()) {
             thread.join();
         }
     }
 }
 
 // 初始化：加载模型，创建线程
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
 
 // 线程执行函数
 
 // 提交任务
 nn_error_e Yolov8ThreadPool::submitTask(const cv::Mat &img, int id) {
     if (stop) {
         // 【修正】使用已知的错误码
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
    // 线程启动时打印一条日志，方便确认线程已创建
    NN_LOG_INFO("[Worker %d] Thread started and running.", id);

    while (!stop) {
        std::pair<int, cv::Mat> task;
        std::shared_ptr<Yolov8Custom> instance = Yolov8_instances[id];

        {
            std::unique_lock<std::mutex> lock(mtx1);
            
            // 在等待前可以加一条日志，用于调试长时间无任务的场景
            // NN_LOG_INFO("[Worker %d] Waiting for a task...", id);
            
            cv_task.wait(lock, [&]() { return !tasks.empty() || stop; });

            if (stop) {
                NN_LOG_INFO("[Worker %d] Stop signal received, exiting.", id);
                return;
            }

            // 从队列中获取任务
            task = tasks.front();
            tasks.pop();
            NN_LOG_INFO("[Worker %d] Got task for frame %d. Remaining tasks in queue: %zu", id, task.first, tasks.size());
        }

        std::vector<Detection> detections;
        LetterBoxInfo info;

        // 【【【核心修改：添加 try-catch 块】】】
        // 这是最重要的部分，用于捕获推理过程中可能出现的任何异常
        try {
            {
                std::lock_guard<std::mutex> lock(depth_mtx);
                if (has_depth_) {
                    // 日志：记录是否设置了深度上下文
                    // NN_LOG_INFO("[Worker %d] Setting depth context for frame %d.", id, task.first);
                    instance->SetDepthContext(depth_frame_, intrinsics_);
                }
            }
            
            NN_LOG_INFO("[Worker %d] ==> Starting instance->Run() for frame %d.", id, task.first);

            // 执行核心的推理和处理
            instance->Run(task.second, detections, info);

            NN_LOG_INFO("[Worker %d] <== Finished instance->Run() for frame %d. Detections found: %zu", id, task.first, detections.size());

        } catch (const std::exception& e) {
            // 如果捕获到标准异常 (如 std::runtime_error)
            NN_LOG_ERROR("[Worker %d] CRITICAL: Exception caught while processing frame %d: %s", id, task.first, e.what());
            // 发生异常后，我们应该跳过这一帧的后续处理，直接进入下一次循环
            // 这样主程序不会因为一个坏帧而完全卡死
            continue; 
        } catch (...) {
            // 如果捕获到非标准异常
            NN_LOG_ERROR("[Worker %d] CRITICAL: Unknown exception caught while processing frame %d.", id, task.first);
            continue;
        }

        // 将结果安全地放入结果队列
        {
            std::lock_guard<std::mutex> lock(mtx2);
            NN_LOG_INFO("[Worker %d] Storing results for frame %d.", id, task.first);

            results[task.first] = detections;

            cv::Mat vis_img = task.second.clone();
            // 使用您项目中存在的两参数版本DrawDetections
            DrawDetections(vis_img, detections); 
            img_results[task.first] = vis_img;
            
            NN_LOG_INFO("[Worker %d] Finished storing results for frame %d.", id, task.first);
        }
        
        // 【【【重要补充：通知结果线程】】】
        // 您的代码中缺少了这一步。当一个任务处理完成后，需要通知等待结果的线程。
        // 如果您的线程池实现依赖于此，请取消下面这行代码的注释。
        // cv_result.notify_all(); // 假设您有一个名为 cv_result 的 std::condition_variable
    }
}
 // 获取推理结果（Detection结构体）
 nn_error_e Yolov8ThreadPool::getTargetResult(std::vector<Detection> &objects, int id) {
     while (results.find(id) == results.end()) {
         if (stop) {
             // 【修正】使用正确的日志宏
             NN_LOG_WARNING("Pool is stopping, getTargetResult aborted.");
             // 【修正】使用已知的错误码
             return NN_TIMEOUT; 
         }
         std::this_thread::sleep_for(std::chrono::milliseconds(1));
     }
 
     {
         std::lock_guard<std::mutex> lock(mtx2);
         if (results.find(id) == results.end()) {
              // 【修正】使用已知的错误码
              return NN_TIMEOUT;
         }
         objects = results[id];
         results.erase(id);
     }
 
     return NN_SUCCESS;
 }
 
 // 获取可视化图像结果
 nn_error_e Yolov8ThreadPool::getTargetImgResult(cv::Mat &img, int id) {
     int loop_cnt = 0;
     while (img_results.find(id) == img_results.end()) {
         if (stop) {
             // 【修正】使用正确的日志宏
             NN_LOG_WARNING("Pool is stopping, getTargetImgResult aborted.");
             // 【修正】使用已知的错误码
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
              // 【修正】使用已知的错误码
              return NN_TIMEOUT;
         }
         img = img_results[id].clone();
         img_results.erase(id);
     }
 
     return NN_SUCCESS;
 }
 
 // 停止所有线程
 void Yolov8ThreadPool::stopAll() {
     stop = true;
     cv_task.notify_all();
 }
 
 // 设置深度信息（外部调用）
 void Yolov8ThreadPool::SetDepthFrame(const rs2::depth_frame& frame, const rs2_intrinsics& intrin) {
     std::lock_guard<std::mutex> lock(depth_mtx);
     depth_frame_ = frame;
     intrinsics_ = intrin;
     has_depth_ = true;
 }
 
 // 获取深度帧
 rs2::depth_frame Yolov8ThreadPool::GetDepth() {
     // 【修正】移除了const，以匹配对非mutable互斥量的加锁操作
     std::lock_guard<std::mutex> lock(depth_mtx);
     return rs2::depth_frame(depth_frame_);
 }
 
 
 // 获取相机内参
 rs2_intrinsics Yolov8ThreadPool::GetIntrinsics() {
     std::lock_guard<std::mutex> lock(depth_mtx);
     return intrinsics_;
 } */


 // 引入 "yolov8_thread_pool.h" 头文件，其中包含了 Yolov8ThreadPool 类的声明
#include "yolov8_thread_pool.h"
// 引入自定义的绘图功能头文件，用于在图像上绘制检测结果
#include "draw/cv_draw.h"
// 引入 Intel RealSense SDK 的主头文件，以支持深度相机
#include <librealsense2/rs.hpp>
// 引入 C++ 标准库头文件 <numeric>，可能用于数值计算
#include <numeric>
// 引入自定义的日志工具头文件
#include "utils/logging.h" 

// Yolov8ThreadPool 类的构造函数
Yolov8ThreadPool::Yolov8ThreadPool() {
    // 初始化停止标志为 false，表示线程池启动后应处于运行状态
    stop = false;
    // 初始化深度信息标志为 false，表示初始时没有深度数据
    has_depth_ = false; 
}

// Yolov8ThreadPool 类的析构函数，负责清理资源
Yolov8ThreadPool::~Yolov8ThreadPool() {
    // 调用 stopAll 方法，向所有线程发送停止信号
    stopAll();
    // 遍历所有线程对象
    for (auto &thread : threads) {
        // 检查线程是否可以被 join（即线程是否仍在运行）
        if (thread.joinable()) {
            // 等待线程执行完毕，回收线程资源
            thread.join();
        }
    }
}

// 初始化线程池：加载模型，创建指定数量的工作线程
nn_error_e Yolov8ThreadPool::setUp(std::string &model_path, int num_threads) {
    // 循环创建指定数量的 Yolov8Custom 实例
    for (int i = 0; i < num_threads; ++i) {
        // 使用智能指针创建一个 Yolov8Custom 对象
        std::shared_ptr<Yolov8Custom> Yolov8 = std::make_shared<Yolov8Custom>();
        // 为该实例加载模型文件，每个线程拥有自己独立的模型实例
        if (Yolov8->LoadModel(model_path.c_str()) != NN_SUCCESS) {
            // 如果模型加载失败，记录错误日志
            NN_LOG_ERROR("Thread %d: model load failed", i);
            // 返回错误码
            return NN_RKNN_INPUT_ATTR_ERROR;
        }
        // 将成功创建的实例存入向量中
        Yolov8_instances.push_back(Yolov8);
    }

    // 循环创建并启动指定数量的工作线程
    for (int i = 0; i < num_threads; ++i) {
        // 创建一个线程，执行 worker 方法，并传入线程ID作为参数
        threads.emplace_back(&Yolov8ThreadPool::worker, this, i);
    }

    // 初始化成功，返回成功状态码
    return NN_SUCCESS;
}

// 这是一个被注释掉的旧版或备用版的 worker 函数
/* void Yolov8ThreadPool::worker(int id) {
     ...
}
*/

// 提交任务到任务队列（生产者）
nn_error_e Yolov8ThreadPool::submitTask(const cv::Mat &img, int id) {
    // 如果线程池已停止，则不再接受新任务
    if (stop) {
        // 返回超时错误码，表示任务无法提交
        return NN_TIMEOUT; 
    }

    // 当任务队列中的任务数量超过10个时，进行等待，防止任务积压过多消耗内存
    while (tasks.size() > 10) {
        // 再次检查停止标志，以便能及时退出等待
        if (stop) return NN_TIMEOUT;
        // 短暂休眠，避免忙等待消耗过多CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // 使用互斥锁保护任务队列的访问
    {
        // lock_guard 在其作用域结束时会自动解锁
        std::lock_guard<std::mutex> lock(mtx1);
        // 将任务（由帧ID和图像数据组成）推入队列
        tasks.push({id, img});
    }
    // 通知一个正在等待的 worker 线程，有新任务可以处理了
    cv_task.notify_one();
    // 返回成功状态码
    return NN_SUCCESS;
}
 

// 工作线程的主函数（消费者）
void Yolov8ThreadPool::worker(int id) {
   // 线程启动时打印一条日志，方便调试，确认线程已成功创建并运行
   NN_LOG_INFO("[Worker %d] Thread started and running.", id);

   // 循环运行，直到收到停止信号
   while (!stop) {
       // 定义一个 pair 用于存储从队列中取出的任务（帧ID, 图像）
       std::pair<int, cv::Mat> task;
       // 获取当前线程对应的 Yolov8Custom 实例的智能指针
       std::shared_ptr<Yolov8Custom> instance = Yolov8_instances[id];

       // 使用互斥锁和条件变量来安全地从任务队列获取任务
       {
           // unique_lock 比 lock_guard 更灵活，是与条件变量配合使用的标准方式
           std::unique_lock<std::mutex> lock(mtx1);
           
           // 这是一个可选的调试日志，用于监控线程是否在等待任务
           // NN_LOG_INFO("[Worker %d] Waiting for a task...", id);
           
           // 等待条件满足：任务队列不为空 或 线程池已停止
           cv_task.wait(lock, [&]() { return !tasks.empty() || stop; });

           // 再次检查停止标志（因为可能是被 stop 信号唤醒的）
           if (stop) {
               // 如果收到停止信号，打印日志并退出线程
               NN_LOG_INFO("[Worker %d] Stop signal received, exiting.", id);
               return;
           }

           // 从任务队列的前端获取一个任务
           task = tasks.front();
           // 从队列中移除该任务
           tasks.pop();
           // 记录日志，表明已获取任务，并显示队列中剩余任务数
           NN_LOG_INFO("[Worker %d] Got task for frame %d. Remaining tasks in queue: %zu", id, task.first, tasks.size());
       }

       // 定义用于存储检测结果的向量
       std::vector<Detection> detections;
       // 定义用于存储 letterbox 信息的结构体
       LetterBoxInfo info;

       // 【【【核心修改：添加 try-catch 块以增强鲁棒性】】】
       // 使用 try-catch 块来捕获推理过程中可能发生的任何异常，防止单个坏帧导致整个程序崩溃
       try {
           // 使用互斥锁保护对共享深度数据的访问
           {
               std::lock_guard<std::mutex> lock(depth_mtx);
               // 如果当前有可用的深度数据
               if (has_depth_) {
                   // 这是一个可选的调试日志
                   // NN_LOG_INFO("[Worker %d] Setting depth context for frame %d.", id, task.first);
                   // 将深度帧和相机内参设置到 Yolov8 实例中
                   instance->SetDepthContext(depth_frame_, intrinsics_);
               }
           }
           
           // 记录开始执行推理的日志
           NN_LOG_INFO("[Worker %d] ==> Starting instance->Run() for frame %d.", id, task.first);

           // 调用 Yolov8 实例的 Run 方法，执行完整的预处理、推理和后处理流程
           instance->Run(task.second, detections, info);

           // 记录推理完成的日志，并显示检测到的目标数量
           NN_LOG_INFO("[Worker %d] <== Finished instance->Run() for frame %d. Detections found: %zu", id, task.first, detections.size());

       } catch (const std::exception& e) {
           // 如果捕获到 C++ 标准异常（例如 std::runtime_error）
           NN_LOG_ERROR("[Worker %d] CRITICAL: Exception caught while processing frame %d: %s", id, task.first, e.what());
           // 使用 continue 跳过当前帧的后续处理，直接开始下一次循环，以处理下一个任务
           continue; 
       } catch (...) {
           // 如果捕获到任何其他类型的未知异常
           NN_LOG_ERROR("[Worker %d] CRITICAL: Unknown exception caught while processing frame %d.", id, task.first);
           // 同样跳过当前帧
           continue;
       }

       // 使用互斥锁保护结果映射表的访问，将处理结果安全地放入
       {
           std::lock_guard<std::mutex> lock(mtx2);
           // 记录开始存储结果的日志
           NN_LOG_INFO("[Worker %d] Storing results for frame %d.", id, task.first);

           // 以帧ID为键，将检测结果向量存入 results 映射表
           results[task.first] = detections;

           // 创建一个图像副本用于可视化
           cv::Mat vis_img = task.second.clone();
           // 在副本上绘制检测框
           //DrawDetections(vis_img, detections); 
           // 以帧ID为键，将可视化后的图像存入 img_results 映射表
           img_results[task.first] = vis_img;
           
           // 记录存储结果完成的日志
           NN_LOG_INFO("[Worker %d] Finished storing results for frame %d.", id, task.first);
       }
       
       // 【【【重要补充：通知结果线程】】】
       // 如果外部有线程在等待结果，需要在这里通知它们。
       // cv_result.notify_all(); // 例如，唤醒等待在名为 cv_result 的条件变量上的线程
   }
}

// 从结果队列中获取指定ID的推理结果（Detection结构体）
nn_error_e Yolov8ThreadPool::getTargetResult(std::vector<Detection> &objects, int id) {
    // 循环等待，直到找到指定ID的结果
    while (results.find(id) == results.end()) {
        // 如果在等待期间收到停止信号
        if (stop) {
            // 记录警告日志
            NN_LOG_WARNING("Pool is stopping, getTargetResult aborted.");
            // 返回超时错误码
            return NN_TIMEOUT; 
        }
        // 短暂休眠，避免忙等待
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // 使用互斥锁保护结果映射表的访问
    {
        std::lock_guard<std::mutex> lock(mtx2);
        // 再次检查结果是否存在（双重检查，防止在等待锁的过程中结果被其他线程取走）
        if (results.find(id) == results.end()) {
             // 如果不存在，返回超时错误码
             return NN_TIMEOUT;
        }
        // 复制结果到输出参数
        objects = results[id];
        // 从映射表中移除已经取走的结果，防止重复获取
        results.erase(id);
    }

    // 返回成功状态码
    return NN_SUCCESS;
}

// 从结果队列中获取指定ID的可视化图像结果
nn_error_e Yolov8ThreadPool::getTargetImgResult(cv::Mat &img, int id) {
    // 定义一个循环计数器，用于实现超时逻辑
    int loop_cnt = 0;
    // 循环等待，直到找到指定ID的图像结果
    while (img_results.find(id) == img_results.end()) {
        // 如果在等待期间收到停止信号
        if (stop) {
            // 记录警告日志
            NN_LOG_WARNING("Pool is stopping, getTargetImgResult aborted.");
            // 返回超时错误码
            return NN_TIMEOUT;
        }

        // 休眠稍长一点的时间
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        // 如果等待时间过长（超过1000*5ms = 5秒），则判断为超时
        if (++loop_cnt > 1000) {
            // 记录超时错误
            NN_LOG_ERROR("getTargetImgResult timeout for frame %d", id);
            // 返回超时错误码
            return NN_TIMEOUT;
        }
    }

    // 使用互斥锁保护结果映射表的访问
    {
        std::lock_guard<std::mutex> lock(mtx2);
        // 再次检查结果是否存在
        if (img_results.find(id) == img_results.end()) {
             // 如果不存在，返回超时错误码
             return NN_TIMEOUT;
        }
        // 将图像结果深拷贝到输出参数
        img = img_results[id].clone();
        // 从映射表中移除已经取走的结果
        img_results.erase(id);
    }

    // 返回成功状态码
    return NN_SUCCESS;
}

// 停止所有工作线程
void Yolov8ThreadPool::stopAll() {
    // 将全局停止标志设为 true
    stop = true;
    // 唤醒所有可能因任务队列为空而正在等待的线程，以便它们能检查 stop 标志并退出
    cv_task.notify_all();
}

// 线程安全地设置深度信息（由外部线程，如相机采集线程调用）
void Yolov8ThreadPool::SetDepthFrame(const rs2::depth_frame& frame, const rs2_intrinsics& intrin) {
    // 使用互斥锁保护对共享深度数据的写入
    std::lock_guard<std::mutex> lock(depth_mtx);
    // 更新深度帧
    depth_frame_ = frame;
    // 更新相机内参
    intrinsics_ = intrin;
    // 更新标志位，表示现在有可用的深度数据
    has_depth_ = true;
}

// 线程安全地获取当前的深度帧
rs2::depth_frame Yolov8ThreadPool::GetDepth() {
    // 使用互斥锁保护对共享深度数据的读取
    std::lock_guard<std::mutex> lock(depth_mtx);
    // 返回深度帧的副本
    return rs2::depth_frame(depth_frame_);
}


// 线程安全地获取当前的相机内参
rs2_intrinsics Yolov8ThreadPool::GetIntrinsics() {
    // 使用互斥锁保护对共享相机内参的读取
    std::lock_guard<std::mutex> lock(depth_mtx);
    // 返回内参的副本
    return intrinsics_;
}