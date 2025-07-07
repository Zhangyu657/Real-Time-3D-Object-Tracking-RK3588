/* 

#ifndef RK3588_DEMO_Yolov8_THREAD_POOL_H
#define RK3588_DEMO_Yolov8_THREAD_POOL_H

#include "yolov8_custom.h"

#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>

class Yolov8ThreadPool
{
private:
    std::queue<std::pair<int, cv::Mat>> tasks;             // <id, img>用来存放任务
    std::vector<std::shared_ptr<Yolov8Custom>> Yolov8_instances; // 模型实例
    std::map<int, std::vector<Detection>> results;         // <id, objects>用来存放结果（检测框）
    std::map<int, cv::Mat> img_results;                    // <id, img>用来存放结果（图片）
    std::vector<std::thread> threads;                      // 线程池
    std::mutex mtx1;
    std::mutex mtx2;
    std::condition_variable cv_task;
    bool stop;

    void worker(int id);

public:
    Yolov8ThreadPool();  // 构造函数
    ~Yolov8ThreadPool(); // 析构函数

    nn_error_e setUp(std::string &model_path, int num_threads = 12);     // 初始化
    nn_error_e submitTask(const cv::Mat &img, int id);                   // 提交任务
    nn_error_e getTargetResult(std::vector<Detection> &objects, int id); // 获取结果（检测框）
    nn_error_e getTargetImgResult(cv::Mat &img, int id);                 // 获取结果（图片）
    void stopAll();                                                      // 停止所有线程
};

#endif // RK3588_DEMO_Yolov8_THREAD_POOL_H
 */

/*  #ifndef RK3588_DEMO_YOLOV8_THREAD_POOL_H
#define RK3588_DEMO_YOLOV8_THREAD_POOL_H

#include "yolov8_custom.h"
#include <opencv2/opencv.hpp>

#include <vector>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>

class Yolov8ThreadPool
{
public:
    Yolov8ThreadPool();  // 构造函数
    ~Yolov8ThreadPool(); // 析构函数

    nn_error_e setUp(std::string &model_path, int num_threads = 12);     // 初始化模型和线程
    nn_error_e submitTask(const cv::Mat &img, int id);                   // 提交推理任务
    nn_error_e getTargetResult(std::vector<Detection> &objects, int id); // 获取结果（检测框）
    nn_error_e getTargetImgResult(cv::Mat &img, int id);                 // 获取结果（绘制图）
    void stopAll();                                                      // 停止所有线程

private:
    void worker(int id);                                                 // 线程工作函数

    std::queue<std::pair<int, cv::Mat>> tasks;                           // 任务队列 <任务ID, 图像>
    std::vector<std::shared_ptr<Yolov8Custom>> Yolov8_instances;         // 每个线程独立模型实例
    std::map<int, std::vector<Detection>> results;                       // <任务ID, 检测框列表>
    std::map<int, cv::Mat> img_results;                                  // <任务ID, 可视化图像>

    std::vector<std::thread> threads;                                    // 工作线程池
    std::mutex mtx1;                                                     // 任务队列锁
    std::mutex mtx2;                                                     // 结果队列锁
    std::condition_variable cv_task;                                     // 任务通知
    bool stop;                                                           // 停止标志
};

#endif // RK3588_DEMO_YOLOV8_THREAD_POOL_H
 */

//3d
 /* #ifndef RK3588_DEMO_YOLOV8_THREAD_POOL_H
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
    Yolov8ThreadPool();  // 构造函数
    ~Yolov8ThreadPool(); // 析构函数

    nn_error_e setUp(std::string &model_path, int num_threads = 12);     // 初始化模型和线程
    nn_error_e submitTask(const cv::Mat &img, int id);                   // 提交推理任务
    nn_error_e getTargetResult(std::vector<Detection> &objects, int id); // 获取结果（检测框）
    nn_error_e getTargetImgResult(cv::Mat &img, int id);                 // 获取结果（绘制图）
    void stopAll();                                                      // 停止所有线程

    void SetDepthFrame(const rs2::depth_frame& frame, const rs2_intrinsics& intrin); // 设置深度帧和相机内参
     // 新增接口：外部访问 depth frame 和相机内参
     rs2::depth_frame GetDepth() ;
     rs2_intrinsics GetIntrinsics();
      //rs2::depth_frame& GetDepth() const { return depth_frame_; }
     const rs2_intrinsics& GetIntrinsics() const { return intrinsics_; }
 
private:
    void worker(int id);                                                 // 线程工作函数

    std::queue<std::pair<int, cv::Mat>> tasks;                           // 任务队列 <任务ID, 图像>
    std::vector<std::shared_ptr<Yolov8Custom>> Yolov8_instances;         // 每个线程独立模型实例
    std::map<int, std::vector<Detection>> results;                       // <任务ID, 检测框列表>
    std::map<int, cv::Mat> img_results;                                  // <任务ID, 可视化图像>

    std::vector<std::thread> threads;                                    // 工作线程池
    std::mutex mtx1;                                                     // 任务队列锁
    std::mutex mtx2;                                                     // 结果队列锁
    std::condition_variable cv_task;                                     // 任务通知
    bool stop;                                                           // 停止标志
    mutable std::mutex depth_mtx;
    // 深度图像和内参，用于 XYZ 恢复
    
    //rs2::depth_frame depth_frame_;
    rs2::frame depth_frame_;    
    rs2_intrinsics intrinsics_;
    bool has_depth_ = false;
};

#endif // RK3588_DEMO_YOLOV8_THREAD_POOL_H
 */

 // 这是 "Include Guard"（头文件保护宏），用于防止本头文件在一次编译中被重复包含
#ifndef RK3588_DEMO_YOLOV8_THREAD_POOL_H
// 如果宏 RK3588_DEMO_YOLOV8_THREAD_POOL_H 未被定义，则定义它，与 #ifndef 配对使用
#define RK3588_DEMO_YOLOV8_THREAD_POOL_H

// 引入 "yolov8_custom.h" 头文件，这是线程池中每个线程要运行的核心工作类
#include "yolov8_custom.h"
// 引入 OpenCV 主头文件，以使用 cv::Mat 等
#include <opencv2/opencv.hpp>
// 引入 Intel RealSense SDK 的主头文件，以支持深度相机功能
#include <librealsense2/rs.hpp>

// 引入 C++ 标准库头文件 <vector>，以使用 std::vector
#include <vector>
// 引入 C++ 标准库头文件 <queue>，以使用 std::queue 作为任务队列
#include <queue>
// 引入 C++ 标准库头文件 <map>，以使用 std::map 作为结果存储
#include <map>
// 引入 C++ 标准库头文件 <thread>，以支持多线程
#include <thread>
// 引入 C++ 标准库头文件 <mutex>，以使用互斥锁进行线程同步
#include <mutex>
// 引入 C++ 标准库头文件 <condition_variable>，以使用条件变量进行高效的线程等待和唤醒
#include <condition_variable>


// 定义一个名为 Yolov8ThreadPool 的类，用于管理一个YOLOv8推理任务的线程池
class Yolov8ThreadPool
{
// public 访问修饰符，表示接下来的成员是类的公开接口
public:
    Yolov8ThreadPool();  // 声明类的构造函数
    ~Yolov8ThreadPool(); // 声明类的析构函数

    nn_error_e setUp(std::string &model_path, int num_threads = 12);     // 声明初始化方法，用于加载模型并创建线程，线程数默认为12
    nn_error_e submitTask(const cv::Mat &img, int id);                   // 声明提交任务的方法，任务由图像和唯一ID组成
    nn_error_e getTargetResult(std::vector<Detection> &objects, int id); // 声明获取检测结果（Detection对象列表）的方法
    nn_error_e getTargetImgResult(cv::Mat &img, int id);                 // 声明获取可视化图像结果的方法
    void stopAll();                                                      // 声明停止所有线程的方法

    void SetDepthFrame(const rs2::depth_frame& frame, const rs2_intrinsics& intrin); // 声明设置深度帧和相机内参的方法
     // 这是一个已有的注释，说明下方是新增的接口
     // 新增接口：外部访问 depth frame 和相机内参
     rs2::depth_frame GetDepth();           // 声明获取深度帧的方法
     rs2_intrinsics GetIntrinsics();      // 声明获取相机内参的方法
     // 这是一个被注释掉的旧版或备用版 GetDepth 接口
      //rs2::depth_frame& GetDepth() const { return depth_frame_; }
     // 声明一个 const 版本的 GetIntrinsics 方法，返回一个常量引用
     const rs2_intrinsics& GetIntrinsics() const { return intrinsics_; }
 
// private 访问修饰符，表示接下来的成员只能被本类的成员函数访问
private:
    void worker(int id);                                                 // 声明私有的工作线程主函数

    std::queue<std::pair<int, cv::Mat>> tasks;                           // 定义一个任务队列，元素是<任务ID, 图像>的配对
    std::vector<std::shared_ptr<Yolov8Custom>> Yolov8_instances;         // 定义一个向量，用于存放每个线程独立的Yolov8模型实例（使用智能指针管理）
    std::map<int, std::vector<Detection>> results;                       // 定义一个映射表，用于存放<任务ID, 检测框列表>的结果
    std::map<int, cv::Mat> img_results;                                  // 定义一个映射表，用于存放<任务ID, 可视化图像>的结果

    std::vector<std::thread> threads;                                    // 定义一个向量，用于存放所有工作线程的 std::thread 对象
    std::mutex mtx1;                                                     // 定义一个互斥锁，专门用于保护任务队列 `tasks` 的访问
    std::mutex mtx2;                                                     // 定义一个互斥锁，专门用于保护结果映射表 `results` 和 `img_results` 的访问
    std::condition_variable cv_task;                                     // 定义一个条件变量，用于在任务队列为空时挂起线程，有新任务时唤醒线程
    bool stop;                                                           // 定义一个布尔型标志，用于向所有线程发送停止信号
    mutable std::mutex depth_mtx;                                        // 定义一个可变的互斥锁，用于保护深度数据。`mutable` 关键字允许在 const 成员函数中修改它（加锁/解锁）
    // 这是一个已有的注释
    // 深度图像和内参，用于 XYZ 恢复
    
    // 这是一个被注释掉的旧版 depth_frame_ 成员变量
    //rs2::depth_frame depth_frame_;
    rs2::frame depth_frame_;                                             // 定义一个通用的 RealSense 帧对象，用于存储深度帧
    rs2_intrinsics intrinsics_;                                          // 定义一个结构体，用于存储相机的内参信息
    bool has_depth_ = false;                                             // 定义一个布尔型标志，表示当前是否有可用的深度数据
}; // 类定义结束

// 结束 #ifndef RK3588_DEMO_YOLOV8_THREAD_POOL_H 定义的条件编译块
#endif // RK3588_DEMO_YOLOV8_THREAD_POOL_H