/* /// 引入OpenCV主头文件，用于图像处理
#include <opencv2/opencv.hpp>
// 引入OpenCV的视频跟踪功能头文件
#include <opencv2/video/tracking.hpp>
// 引入Intel RealSense SDK的主头文件，用于与深度相机交互
#include <librealsense2/rs.hpp>
// 引入C++17的文件系统库，用于创建目录等操作
#include <filesystem>
// 引入C++的IO流控制库，用于格式化输出
#include <iomanip>
// 引入C++的字符串流库，用于在内存中构建字符串
#include <sstream>
// 引入C++的多线程库
#include <thread>
// 引入C语言的信号处理库，用于捕获Ctrl+C等信号
#include <csignal>
// 引入C++的map, deque, set, vector, numeric, iostream等标准库
#include <map>
#include <deque>
#include <set>
#include <vector>
#include <numeric>
#include <iostream>
#include <chrono>

// 引入SORT算法的核心组件
#include "utils/KalmanTracker.h"
#include "utils/Hungarian.h"
#include "utils/KalmanFilter3D.h"

// 引入YOLOv8相关的自定义模块
#include "task/yolov8_custom.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"
#include "task/yolov8_thread_pool.h"

// 为 std::filesystem 创建一个简短的命名空间别名 fs
namespace fs = std::filesystem;

// ------------------- 全局变量定义 -------------------
static int g_frame_start_id = 0;
static int g_frame_end_id = 0;
static Yolov8ThreadPool* g_pool = nullptr;
static volatile bool g_end = false;

// 跟踪器、历史记录等相关数据结构
std::map<int, KalmanTracker> trackers;
std::map<int, KalmanFilter3D> trackers_3d;
std::map<int, std::deque<cv::Point3f>> point3D_history;
std::map<int, std::deque<float>> diameter_history;
std::map<int, std::chrono::steady_clock::time_point> last_seen;
std::map<int, cv::Rect> last_known_boxes;
std::map<int, std::string> tracker_class_names;  // 新增：存储每个跟踪目标的类别名称
const int MAX_HISTORY = 30;
const int MAX_DIAMETER_HISTORY = 10;

// ------------------- 信号处理函数 -------------------
void signal_handler(int) {
    g_end = true;
    NN_LOG_INFO("Received Ctrl+C signal, stopping...");
}

// 辅助函数
cv::Scalar get_track_color(int id) { 
    cv::RNG rng(id); 
    return cv::Scalar(rng.uniform(80,255), rng.uniform(80,255), rng.uniform(80,255)); 
}

double calculate_iou(const cv::Rect& box1, const cv::Rect& box2) { 
    cv::Rect intersection = box1 & box2; 
    double intersection_area = intersection.area(); 
    double union_area = box1.area() + box2.area() - intersection_area; 
    if (union_area < 1e-6) return 0; 
    return intersection_area / union_area; 
}

// 清理旧跟踪器
void cleanup_old_tracks() {
    const auto now = std::chrono::steady_clock::now();
    std::vector<int> ids_to_remove;
    
    for (const auto& [id, last_time] : last_seen) {
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - last_time).count();
        if (duration > 5) { // 超过5秒未更新则移除
            ids_to_remove.push_back(id);
        }
    }
    
    for (int id : ids_to_remove) {
        trackers.erase(id);
        trackers_3d.erase(id);
        point3D_history.erase(id);
        diameter_history.erase(id);
        last_seen.erase(id);
        last_known_boxes.erase(id);
        tracker_class_names.erase(id); // 清理类别名称
    }
}

// 3D边缘点计算
cv::Point3f get_3d_edge_point(const cv::Rect& box, bool is_left_edge, const rs2::depth_frame& depth, 
                             const rs2_intrinsics& intrin, int width, int height) {
    int x = is_left_edge ? box.x : box.x + box.width - 1;
    int y = box.y + box.height / 2;
    
    if (x < 0 || x >= width || y < 0 || y >= height) 
        return cv::Point3f(0, 0, 0);
    
    float d = depth.get_distance(x, y);
    if (d <= 0.1f || d >= 10.0f) 
        return cv::Point3f(0, 0, 0);
    
    float point3d[3], pixel[2] = {(float)x, (float)y};
    rs2_deproject_pixel_to_point(point3d, &intrin, pixel, d);
    return cv::Point3f(point3d[0], point3d[1], point3d[2]);
}

// 3D质心计算
cv::Point3f compute_3d_centroid(const cv::Rect& box, const rs2::depth_frame& depth, 
                               const rs2_intrinsics& intrin, int width, int height) {
    int valid_points = 0;
    cv::Point3f sum(0, 0, 0);
    
    for (int y = box.y; y < box.y + box.height; y += 4) {
        for (int x = box.x; x < box.x + box.width; x += 4) {
            if (x < 0 || x >= width || y < 0 || y >= height) 
                continue;
                
            float d = depth.get_distance(x, y);
            if (d >= 0.15f && d <= 8.0f) {
                float point3d[3];
                float pixel[2] = {(float)x, (float)y};
                rs2_deproject_pixel_to_point(point3d, &intrin, pixel, d);
                sum += cv::Point3f(point3d[0], point3d[1], point3d[2]);
                valid_points++;
            }
        }
    }
    
    if (valid_points > 10) {
        return sum * (1.0f / valid_points);
    }
    return cv::Point3f(0, 0, 0);
}

// ------------------- 跟踪与可视化主流程线程函数 (添加类别显示) -------------------
void get_results(int width = 640, int height = 480, int fps = 30) {
    auto start_all = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    static int next_track_id = 0;
    const double iou_threshold = 0.3;
    std::string output_dir = "output";
    fs::create_directories(output_dir);
    HungarianAlgorithm hungarian_solver;

    const float model_w = 640.0f;
    const float model_h = 640.0f;
    float scale = std::min(model_w / (float)width, model_h / (float)height);
    int pad_x = (model_w - (float)width * scale) / 2;
    int pad_y = (model_h - (float)height * scale) / 2;

    while (!g_end) {
        cv::Mat img;
        auto ret_img = g_pool->getTargetImgResult(img, g_frame_end_id);
        if (ret_img != NN_SUCCESS) { 
            if (g_end) break; 
            std::this_thread::sleep_for(std::chrono::milliseconds(5)); 
            continue; 
        }

        std::vector<Detection> dets_from_pool;
        auto ret_det = g_pool->getTargetResult(dets_from_pool, g_frame_end_id);

        std::vector<Detection> restored_dets;
        if (ret_det == NN_SUCCESS) {
            for (const auto& det : dets_from_pool) {
                Detection restored_det = det;
                restored_det.box.x = (det.box.x - pad_x) / scale;
                restored_det.box.y = (det.box.y - pad_y) / scale;
                restored_det.box.width = det.box.width / scale;
                restored_det.box.height = det.box.height / scale;
                if (restored_det.confidence < 0.5) continue;
                restored_dets.push_back(restored_det);
            }
        }

        auto depth = g_pool->GetDepth();
        rs2_intrinsics intrin = g_pool->GetIntrinsics();

        std::vector<int> track_ids;
        std::vector<cv::Rect> predicted_boxes;
        for (auto& [id, tracker] : trackers) {
            cv::Rect2f pred = tracker.predict();
            track_ids.push_back(id);
            predicted_boxes.push_back(pred);
        }

        std::vector<std::vector<double>> cost_matrix(track_ids.size(), std::vector<double>(restored_dets.size(), 1.0));
        for (size_t i = 0; i < track_ids.size(); ++i)
            for (size_t j = 0; j < restored_dets.size(); ++j) {
                double iou = calculate_iou(predicted_boxes[i], restored_dets[j].box);
                if (iou > iou_threshold) cost_matrix[i][j] = 1.0 - iou;
            }
        std::vector<int> assignment;
        if (!track_ids.empty() && !restored_dets.empty())
            hungarian_solver.Solve(cost_matrix, assignment);

        std::set<int> matched_det_indices;
        for (size_t i = 0; i < assignment.size(); ++i) {
            int det_idx = assignment[i];
            if (det_idx >= 0 && cost_matrix[i][det_idx] < 1.0 - iou_threshold) {
                int id = track_ids[i];
                trackers[id].update(restored_dets[det_idx].box);
                
                // 更新类别名称
                if (det_idx < restored_dets.size()) {
                    tracker_class_names[id] = restored_dets[det_idx].className;
                }
                
                cv::Rect smoothed_box = trackers[id].get_state();

                cv::Rect analysis_box = smoothed_box;
                int shrink_x = smoothed_box.width * 0.15;
                int shrink_y = smoothed_box.height * 0.15;
                analysis_box.x += shrink_x;
                analysis_box.y += shrink_y;
                analysis_box.width -= 2 * shrink_x;
                analysis_box.height -= 2 * shrink_y;
                
                cv::Point3f noisy_centroid(0,0,0);
                if (analysis_box.width > 0 && analysis_box.height > 0) {
                    noisy_centroid = compute_3d_centroid(analysis_box, depth, intrin, width, height);
                }

                if (noisy_centroid.x == 0 && noisy_centroid.y == 0 && noisy_centroid.z == 0) {
                    int cx = smoothed_box.x + smoothed_box.width / 2;
                    int cy = smoothed_box.y + smoothed_box.height / 2;
                    if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
                        float d = depth.get_distance(cx, cy);
                        if (d > 0.1f && d < 10.0f) {
                            float point3d[3], pixel[2] = {(float)cx, (float)cy};
                            rs2_deproject_pixel_to_point(point3d, &intrin, pixel, d);
                            noisy_centroid = cv::Point3f(point3d[0], point3d[1], point3d[2]);
                        }
                    }
                }

                if (noisy_centroid.x != 0 || noisy_centroid.y != 0 || noisy_centroid.z != 0) {
                    cv::Point3f smoothed_centroid = trackers_3d[id].update(noisy_centroid);
                    point3D_history[id].push_back(smoothed_centroid);
                    if (point3D_history[id].size() > MAX_HISTORY) 
                        point3D_history[id].pop_front();
                }
                
                last_seen[id] = std::chrono::steady_clock::now();
                last_known_boxes[id] = restored_dets[det_idx].box;
                matched_det_indices.insert(det_idx);
            }
        }
        
        for (size_t j = 0; j < restored_dets.size(); ++j) {
            if (matched_det_indices.find(j) == matched_det_indices.end()) {
                int new_id = next_track_id++;
                trackers[new_id] = KalmanTracker(restored_dets[j].box);
                last_known_boxes[new_id] = restored_dets[j].box;
                last_seen[new_id] = std::chrono::steady_clock::now();
                
                // 保存类别名称
                tracker_class_names[new_id] = restored_dets[j].className;
                
                cv::Point3f initial_centroid = compute_3d_centroid(restored_dets[j].box, depth, intrin, width, height);
                if (initial_centroid.x != 0 || initial_centroid.y != 0 || initial_centroid.z != 0) {
                    trackers_3d[new_id].init(initial_centroid);
                    point3D_history[new_id].push_back(initial_centroid);
                }
            }
        }

        cleanup_old_tracks();

        for (auto const& [id, tracker] : trackers) {
            if (tracker.m_time_since_update > 2 && tracker.m_hit_streak < 3) continue;

            cv::Rect box = tracker.get_state();
            cv::Scalar color = get_track_color(id);
            cv::rectangle(img, box, color, 2);
            
            // 获取类别名称
            std::string class_name = "object";
            if (tracker_class_names.find(id) != tracker_class_names.end()) {
                class_name = tracker_class_names[id];
                if (class_name.empty()) {
                    class_name = "object";
                }
            }
            
            // 显示类别、ID和置信度
            std::ostringstream label_ss;
            label_ss << class_name << " ID:" << id;
            cv::putText(img, label_ss.str(), cv::Point(box.x, box.y - 10), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);

            if (point3D_history.count(id) && !point3D_history[id].empty()) {
                const auto& pt = point3D_history[id].back();
                std::ostringstream oss3d;
                oss3d << std::fixed << std::setprecision(2) << "X:" << pt.x << " Y:" << pt.y << " Z:" << pt.z;
                cv::putText(img, oss3d.str(), cv::Point(box.x, box.y + box.height + 20), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
            }
            
            cv::Point3f left_3d = get_3d_edge_point(box, true, depth, intrin, width, height);
            cv::Point3f right_3d = get_3d_edge_point(box, false, depth, intrin, width, height);
            if (left_3d.x != 0 && right_3d.x != 0) {
                float current_diameter_mm = cv::norm(left_3d - right_3d) * 1000.0f;
                diameter_history[id].push_back(current_diameter_mm);
                if (diameter_history[id].size() > MAX_DIAMETER_HISTORY) 
                    diameter_history[id].pop_front();
                float sum = std::accumulate(diameter_history[id].begin(), diameter_history[id].end(), 0.0f);
                float smoothed_diameter_mm = sum / diameter_history[id].size();
                std::ostringstream oss_dia;
                oss_dia << std::fixed << std::setprecision(1) << "Dia:" << smoothed_diameter_mm << "mm";
                cv::putText(img, oss_dia.str(), cv::Point(box.x, box.y + box.height + 40), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
            }

            if (trackers_3d.count(id)) {
                std::vector<cv::Point3f> future_points = trackers_3d[id].predict_future(10);
                for (const auto& pt : future_points) {
                    float point3d[3] = {pt.x, pt.y, pt.z};
                    float pixel[2];
                    rs2_project_point_to_pixel(pixel, &intrin, point3d);
                    if (pixel[0] > 0 && pixel[0] < width && pixel[1] > 0 && pixel[1] < height) {
                        cv::circle(img, cv::Point((int)pixel[0], (int)pixel[1]), 2, cv::Scalar(0, 255, 0), -1);
                    }
                }
            }
            
            if (point3D_history.count(id) && point3D_history[id].size() > 1) {
                const auto& hist = point3D_history[id];
                for (size_t j = 1; j < hist.size(); ++j) {
                    float p1[3] = {hist[j-1].x, hist[j-1].y, hist[j-1].z};
                    float p2[3] = {hist[j].x,   hist[j].y,   hist[j].z};
                    float px1[2], px2[2];
                    rs2_project_point_to_pixel(px1, &intrin, p1);
                    rs2_project_point_to_pixel(px2, &intrin, p2);
                    if (px1[0] > 0 && px1[1] > 0 && px2[0] > 0 && px2[1] > 0)
                        cv::line(img, cv::Point((int)px1[0], (int)px1[1]), 
                                cv::Point((int)px2[0], (int)px2[1]), color, 2);
                }
            }
        }

        std::ostringstream filename;
        filename << output_dir << "/" << std::setw(6) << std::setfill('0') << g_frame_end_id << ".jpg";
        cv::imwrite(filename.str(), img);
        g_frame_end_id++;
        frame_count++;
        
        auto end_all = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count();
        if (elapsed > 1000.0f) {
            NN_LOG_INFO("FPS: %.2f", frame_count / (elapsed / 1000.0f));
            frame_count = 0;
            start_all = std::chrono::high_resolution_clock::now();
        }
    }
    g_pool->stopAll();
    NN_LOG_INFO("Result thread exited.");
}

// ------------------- RealSense数据流读取线程函数 -------------------
void read_stream() {
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    try {
        rs2::pipeline_profile profile = pipe.start(cfg);
        rs2::align align_to_color(RS2_STREAM_COLOR);
        
        // 获取深度传感器配置
        auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
        if (depth_sensor.supports(RS2_OPTION_EMITTER_ENABLED)) {
            depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 1.f);
        }
        
        while (!g_end) {
            rs2::frameset frameset;
            if (!pipe.try_wait_for_frames(&frameset, 1000)) continue;
            frameset = align_to_color.process(frameset);
            rs2::frame color_frame = frameset.get_color_frame();
            rs2::depth_frame depth_frame = frameset.get_depth_frame();
            if (!color_frame || !depth_frame) continue;
            
            rs2_intrinsics intrin = color_frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
            cv::Mat img(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
            
            g_pool->SetDepthFrame(depth_frame, intrin);
            auto submit_ret = g_pool->submitTask(img.clone(), g_frame_start_id);
            
            if (submit_ret != NN_SUCCESS) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            } else {
                g_frame_start_id++;
            }
        }
        pipe.stop();
    } catch (const rs2::error & e) {
        NN_LOG_ERROR("RealSense camera could not be started: %s", e.what()); 
        g_end = true;
    }
}

// ------------------- 程序主函数 -------------------
int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <model_file> [num_threads]\n", argv[0]);
        return 1;
    }
    std::string model_file = argv[1];
    const int num_threads = (argc > 2) ? std::stoi(argv[2]) : 4;
    signal(SIGINT, signal_handler);
    g_pool = new Yolov8ThreadPool();
    
    if (g_pool->setUp(model_file, num_threads) != NN_SUCCESS) {
        NN_LOG_ERROR("Failed to initialize YOLOv8 thread pool");
        delete g_pool;
        return -1;
    }
    
    std::thread reader(read_stream);
    std::thread writer(get_results, 640, 480, 30);
    
    reader.join();
    writer.join();
    
    delete g_pool;
    return 0;
} */


/* /// 引入OpenCV主头文件，用于图像处理
#include <opencv2/opencv.hpp>
// 引入OpenCV的视频跟踪功能头文件
#include <opencv2/video/tracking.hpp>
// 引入Intel RealSense SDK的主头文件，用于与深度相机交互
#include <librealsense2/rs.hpp>
// 引入C++17的文件系统库，用于创建目录等操作
#include <filesystem>
// 引入C++的IO流控制库，用于格式化输出
#include <iomanip>
// 引入C++的字符串流库，用于在内存中构建字符串
#include <sstream>
// 引入C++的多线程库
#include <thread>
// 引入C语言的信号处理库，用于捕获Ctrl+C等信号
#include <csignal>
// 引入C++的map, deque, set, vector, numeric, iostream等标准库
#include <map>
#include <deque>
#include <set>
#include <vector>
#include <numeric>
#include <iostream>
#include <chrono>

// 引入SORT算法的核心组件
#include "utils/KalmanTracker.h"
#include "utils/Hungarian.h"
#include "utils/KalmanFilter3D.h"

// 引入YOLOv8相关的自定义模块
#include "task/yolov8_custom.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"
#include "task/yolov8_thread_pool.h"

// 为 std::filesystem 创建一个简短的命名空间别名 fs
namespace fs = std::filesystem;

// ------------------- 全局变量定义 -------------------
static int g_frame_start_id = 0;
static int g_frame_end_id = 0;
static Yolov8ThreadPool* g_pool = nullptr;
static volatile bool g_end = false;

// 跟踪器、历史记录等相关数据结构
std::map<int, KalmanTracker> trackers;
std::map<int, KalmanFilter3D> trackers_3d;
std::map<int, std::deque<cv::Point3f>> point3D_history;
std::map<int, std::deque<float>> diameter_history;
std::map<int, std::chrono::steady_clock::time_point> last_seen;
std::map<int, cv::Rect> last_known_boxes;
std::map<int, std::string> tracker_class_names;  // 存储每个跟踪目标的类别名称
std::map<int, float> tracker_confidences;        // 存储每个跟踪目标的最新置信度
const int MAX_HISTORY = 30;
const int MAX_DIAMETER_HISTORY = 10;

// ------------------- 信号处理函数 -------------------
void signal_handler(int) {
    g_end = true;
    NN_LOG_INFO("Received Ctrl+C signal, stopping...");
}

// 辅助函数
cv::Scalar get_track_color(int id) { 
    cv::RNG rng(id); 
    return cv::Scalar(rng.uniform(80,255), rng.uniform(80,255), rng.uniform(80,255)); 
}

double calculate_iou(const cv::Rect& box1, const cv::Rect& box2) { 
    cv::Rect intersection = box1 & box2; 
    double intersection_area = intersection.area(); 
    double union_area = box1.area() + box2.area() - intersection_area; 
    if (union_area < 1e-6) return 0; 
    return intersection_area / union_area; 
}

// 清理旧跟踪器
void cleanup_old_tracks() {
    const auto now = std::chrono::steady_clock::now();
    std::vector<int> ids_to_remove;
    
    for (const auto& [id, last_time] : last_seen) {
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - last_time).count();
        if (duration > 5) { // 超过5秒未更新则移除
            ids_to_remove.push_back(id);
        }
    }
    
    for (int id : ids_to_remove) {
        trackers.erase(id);
        trackers_3d.erase(id);
        point3D_history.erase(id);
        diameter_history.erase(id);
        last_seen.erase(id);
        last_known_boxes.erase(id);
        tracker_class_names.erase(id); // 清理类别名称
        tracker_confidences.erase(id); // 清理置信度
    }
}

// 3D边缘点计算
cv::Point3f get_3d_edge_point(const cv::Rect& box, bool is_left_edge, const rs2::depth_frame& depth, 
                             const rs2_intrinsics& intrin, int width, int height) {
    int x = is_left_edge ? box.x : box.x + box.width - 1;
    int y = box.y + box.height / 2;
    
    if (x < 0 || x >= width || y < 0 || y >= height) 
        return cv::Point3f(0, 0, 0);
    
    float d = depth.get_distance(x, y);
    if (d <= 0.1f || d >= 10.0f) 
        return cv::Point3f(0, 0, 0);
    
    float point3d[3], pixel[2] = {(float)x, (float)y};
    rs2_deproject_pixel_to_point(point3d, &intrin, pixel, d);
    return cv::Point3f(point3d[0], point3d[1], point3d[2]);
}

// 3D质心计算
cv::Point3f compute_3d_centroid(const cv::Rect& box, const rs2::depth_frame& depth, 
                               const rs2_intrinsics& intrin, int width, int height) {
    int valid_points = 0;
    cv::Point3f sum(0, 0, 0);
    
    for (int y = box.y; y < box.y + box.height; y += 4) {
        for (int x = box.x; x < box.x + box.width; x += 4) {
            if (x < 0 || x >= width || y < 0 || y >= height) 
                continue;
                
            float d = depth.get_distance(x, y);
            if (d >= 0.15f && d <= 8.0f) {
                float point3d[3];
                float pixel[2] = {(float)x, (float)y};
                rs2_deproject_pixel_to_point(point3d, &intrin, pixel, d);
                sum += cv::Point3f(point3d[0], point3d[1], point3d[2]);
                valid_points++;
            }
        }
    }
    
    if (valid_points > 10) {
        return sum * (1.0f / valid_points);
    }
    return cv::Point3f(0, 0, 0);
}

// ------------------- 跟踪与可视化主流程线程函数 (优化字体显示) -------------------
void get_results(int width = 640, int height = 480, int fps = 30) {
    auto start_all = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    static int next_track_id = 0;
    const double iou_threshold = 0.3;
    std::string output_dir = "output";
    fs::create_directories(output_dir);
    HungarianAlgorithm hungarian_solver;

    const float model_w = 640.0f;
    const float model_h = 640.0f;
    float scale = std::min(model_w / (float)width, model_h / (float)height);
    int pad_x = (model_w - (float)width * scale) / 2;
    int pad_y = (model_h - (float)height * scale) / 2;

    // 定义字体参数（全局使用）
    const float font_scale = 0.4; // 减小字体大小（更小）
    const int font_thickness = 1;  // 保持字体厚度（不细）
    const int line_spacing = 12;   // 行间距（更紧凑）

    while (!g_end) {
        cv::Mat img;
        auto ret_img = g_pool->getTargetImgResult(img, g_frame_end_id);
        if (ret_img != NN_SUCCESS) { 
            if (g_end) break; 
            std::this_thread::sleep_for(std::chrono::milliseconds(5)); 
            continue; 
        }

        std::vector<Detection> dets_from_pool;
        auto ret_det = g_pool->getTargetResult(dets_from_pool, g_frame_end_id);

        std::vector<Detection> restored_dets;
        if (ret_det == NN_SUCCESS) {
            for (const auto& det : dets_from_pool) {
                Detection restored_det = det;
                restored_det.box.x = (det.box.x - pad_x) / scale;
                restored_det.box.y = (det.box.y - pad_y) / scale;
                restored_det.box.width = det.box.width / scale;
                restored_det.box.height = det.box.height / scale;
                if (restored_det.confidence < 0.5) continue;
                restored_dets.push_back(restored_det);
            }
        }

        auto depth = g_pool->GetDepth();
        rs2_intrinsics intrin = g_pool->GetIntrinsics();

        std::vector<int> track_ids;
        std::vector<cv::Rect> predicted_boxes;
        for (auto& [id, tracker] : trackers) {
            cv::Rect2f pred = tracker.predict();
            track_ids.push_back(id);
            predicted_boxes.push_back(pred);
        }

        std::vector<std::vector<double>> cost_matrix(track_ids.size(), std::vector<double>(restored_dets.size(), 1.0));
        for (size_t i = 0; i < track_ids.size(); ++i)
            for (size_t j = 0; j < restored_dets.size(); ++j) {
                double iou = calculate_iou(predicted_boxes[i], restored_dets[j].box);
                if (iou > iou_threshold) cost_matrix[i][j] = 1.0 - iou;
            }
        std::vector<int> assignment;
        if (!track_ids.empty() && !restored_dets.empty())
            hungarian_solver.Solve(cost_matrix, assignment);

        std::set<int> matched_det_indices;
        for (size_t i = 0; i < assignment.size(); ++i) {
            int det_idx = assignment[i];
            if (det_idx >= 0 && cost_matrix[i][det_idx] < 1.0 - iou_threshold) {
                int id = track_ids[i];
                trackers[id].update(restored_dets[det_idx].box);
                
                // 更新类别名称和置信度
                if (det_idx < restored_dets.size()) {
                    tracker_class_names[id] = restored_dets[det_idx].className;
                    tracker_confidences[id] = restored_dets[det_idx].confidence;
                }
                
                cv::Rect smoothed_box = trackers[id].get_state();

                cv::Rect analysis_box = smoothed_box;
                int shrink_x = smoothed_box.width * 0.15;
                int shrink_y = smoothed_box.height * 0.15;
                analysis_box.x += shrink_x;
                analysis_box.y += shrink_y;
                analysis_box.width -= 2 * shrink_x;
                analysis_box.height -= 2 * shrink_y;
                
                cv::Point3f noisy_centroid(0,0,0);
                if (analysis_box.width > 0 && analysis_box.height > 0) {
                    noisy_centroid = compute_3d_centroid(analysis_box, depth, intrin, width, height);
                }

                if (noisy_centroid.x == 0 && noisy_centroid.y == 0 && noisy_centroid.z == 0) {
                    int cx = smoothed_box.x + smoothed_box.width / 2;
                    int cy = smoothed_box.y + smoothed_box.height / 2;
                    if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
                        float d = depth.get_distance(cx, cy);
                        if (d > 0.1f && d < 10.0f) {
                            float point3d[3], pixel[2] = {(float)cx, (float)cy};
                            rs2_deproject_pixel_to_point(point3d, &intrin, pixel, d);
                            noisy_centroid = cv::Point3f(point3d[0], point3d[1], point3d[2]);
                        }
                    }
                }

                if (noisy_centroid.x != 0 || noisy_centroid.y != 0 || noisy_centroid.z != 0) {
                    cv::Point3f smoothed_centroid = trackers_3d[id].update(noisy_centroid);
                    point3D_history[id].push_back(smoothed_centroid);
                    if (point3D_history[id].size() > MAX_HISTORY) 
                        point3D_history[id].pop_front();
                }
                
                last_seen[id] = std::chrono::steady_clock::now();
                last_known_boxes[id] = restored_dets[det_idx].box;
                matched_det_indices.insert(det_idx);
            }
        }
        
        for (size_t j = 0; j < restored_dets.size(); ++j) {
            if (matched_det_indices.find(j) == matched_det_indices.end()) {
                int new_id = next_track_id++;
                trackers[new_id] = KalmanTracker(restored_dets[j].box);
                last_known_boxes[new_id] = restored_dets[j].box;
                last_seen[new_id] = std::chrono::steady_clock::now();
                
                // 保存类别名称和置信度
                tracker_class_names[new_id] = restored_dets[j].className;
                tracker_confidences[new_id] = restored_dets[j].confidence;
                
                cv::Point3f initial_centroid = compute_3d_centroid(restored_dets[j].box, depth, intrin, width, height);
                if (initial_centroid.x != 0 || initial_centroid.y != 0 || initial_centroid.z != 0) {
                    trackers_3d[new_id].init(initial_centroid);
                    point3D_history[new_id].push_back(initial_centroid);
                }
            }
        }

        cleanup_old_tracks();

        for (auto const& [id, tracker] : trackers) {
            if (tracker.m_time_since_update > 2 && tracker.m_hit_streak < 3) continue;

            cv::Rect box = tracker.get_state();
            cv::Scalar color = get_track_color(id);
            cv::rectangle(img, box, color, 2);
            
            // 获取类别名称和置信度
            std::string class_name = "object";
            float confidence = 0.0f;
            
            if (tracker_class_names.find(id) != tracker_class_names.end()) {
                class_name = tracker_class_names[id];
            }
            
            if (tracker_confidences.find(id) != tracker_confidences.end()) {
                confidence = tracker_confidences[id];
            }
            
            // 显示主要信息（框上方）
            std::ostringstream main_label;
            main_label << "id:" << id << " " << class_name << " " 
                       << std::fixed << std::setprecision(2) << confidence;
            cv::putText(img, main_label.str(), 
                       cv::Point(box.x, box.y - 5), 
                       cv::FONT_HERSHEY_SIMPLEX, 
                       font_scale, 
                       color, 
                       font_thickness);
            
            // 显示3D坐标（框下方第一行）
            if (point3D_history.count(id) && !point3D_history[id].empty()) {
                const auto& pt = point3D_history[id].back();
                std::ostringstream pos_label;
                pos_label << "x:" << std::fixed << std::setprecision(2) << pt.x 
                         << " y:" << std::fixed << std::setprecision(2) << pt.y 
                         << " z:" << std::fixed << std::setprecision(2) << pt.z;
                cv::putText(img, pos_label.str(), 
                           cv::Point(box.x, box.y + box.height + line_spacing), 
                           cv::FONT_HERSHEY_SIMPLEX, 
                           font_scale, 
                           color, 
                           font_thickness);
            }
            
            // 显示直径（框下方第二行）
            cv::Point3f left_3d = get_3d_edge_point(box, true, depth, intrin, width, height);
            cv::Point3f right_3d = get_3d_edge_point(box, false, depth, intrin, width, height);
            if (left_3d.x != 0 && right_3d.x != 0) {
                float current_diameter_mm = cv::norm(left_3d - right_3d) * 1000.0f;
                diameter_history[id].push_back(current_diameter_mm);
                if (diameter_history[id].size() > MAX_DIAMETER_HISTORY) 
                    diameter_history[id].pop_front();
                float sum = std::accumulate(diameter_history[id].begin(), diameter_history[id].end(), 0.0f);
                float smoothed_diameter_mm = sum / diameter_history[id].size();
                std::ostringstream dia_label;
                dia_label << "Dia:" << std::fixed << std::setprecision(1) << smoothed_diameter_mm << "mm";
                cv::putText(img, dia_label.str(), 
                           cv::Point(box.x, box.y + box.height + line_spacing * 2), 
                           cv::FONT_HERSHEY_SIMPLEX, 
                           font_scale, 
                           color, 
                           font_thickness);
            }

            // 显示速度（框下方第三行） - 示例，实际需要计算
            std::ostringstream speed_label;
            speed_label << "Speed: 0.163 m/s";
            cv::putText(img, speed_label.str(), 
                       cv::Point(box.x, box.y + box.height + line_spacing * 3), 
                       cv::FONT_HERSHEY_SIMPLEX, 
                       font_scale, 
                       color, 
                       font_thickness);

            // 显示未来轨迹
            if (trackers_3d.count(id)) {
                std::vector<cv::Point3f> future_points = trackers_3d[id].predict_future(10);
                for (const auto& pt : future_points) {
                    float point3d[3] = {pt.x, pt.y, pt.z};
                    float pixel[2];
                    rs2_project_point_to_pixel(pixel, &intrin, point3d);
                    if (pixel[0] > 0 && pixel[0] < width && pixel[1] > 0 && pixel[1] < height) {
                        cv::circle(img, cv::Point((int)pixel[0], (int)pixel[1]), 2, cv::Scalar(0, 255, 0), -1);
                    }
                }
            }
            
            // 显示历史轨迹
            if (point3D_history.count(id) && point3D_history[id].size() > 1) {
                const auto& hist = point3D_history[id];
                for (size_t j = 1; j < hist.size(); ++j) {
                    float p1[3] = {hist[j-1].x, hist[j-1].y, hist[j-1].z};
                    float p2[3] = {hist[j].x,   hist[j].y,   hist[j].z};
                    float px1[2], px2[2];
                    rs2_project_point_to_pixel(px1, &intrin, p1);
                    rs2_project_point_to_pixel(px2, &intrin, p2);
                    if (px1[0] > 0 && px1[1] > 0 && px2[0] > 0 && px2[1] > 0)
                        cv::line(img, cv::Point((int)px1[0], (int)px1[1]), 
                                cv::Point((int)px2[0], (int)px2[1]), color, 1);
                }
            }
        }

        std::ostringstream filename;
        filename << output_dir << "/" << std::setw(6) << std::setfill('0') << g_frame_end_id << ".jpg";
        cv::imwrite(filename.str(), img);
        g_frame_end_id++;
        frame_count++;
        
        auto end_all = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count();
        if (elapsed > 1000.0f) {
            NN_LOG_INFO("FPS: %.2f", frame_count / (elapsed / 1000.0f));
            frame_count = 0;
            start_all = std::chrono::high_resolution_clock::now();
        }
    }
    g_pool->stopAll();
    NN_LOG_INFO("Result thread exited.");
}

// ------------------- RealSense数据流读取线程函数 -------------------
void read_stream() {
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    try {
        rs2::pipeline_profile profile = pipe.start(cfg);
        rs2::align align_to_color(RS2_STREAM_COLOR);
        
        // 获取深度传感器配置
        auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
        if (depth_sensor.supports(RS2_OPTION_EMITTER_ENABLED)) {
            depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 1.f);
        }
        
        while (!g_end) {
            rs2::frameset frameset;
            if (!pipe.try_wait_for_frames(&frameset, 1000)) continue;
            frameset = align_to_color.process(frameset);
            rs2::frame color_frame = frameset.get_color_frame();
            rs2::depth_frame depth_frame = frameset.get_depth_frame();
            if (!color_frame || !depth_frame) continue;
            
            rs2_intrinsics intrin = color_frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
            cv::Mat img(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
            
            g_pool->SetDepthFrame(depth_frame, intrin);
            auto submit_ret = g_pool->submitTask(img.clone(), g_frame_start_id);
            
            if (submit_ret != NN_SUCCESS) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            } else {
                g_frame_start_id++;
            }
        }
        pipe.stop();
    } catch (const rs2::error & e) {
        NN_LOG_ERROR("RealSense camera could not be started: %s", e.what()); 
        g_end = true;
    }
}

// ------------------- 程序主函数 -------------------
int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <model_file> [num_threads]\n", argv[0]);
        return 1;
    }
    std::string model_file = argv[1];
    const int num_threads = (argc > 2) ? std::stoi(argv[2]) : 4;
    signal(SIGINT, signal_handler);
    g_pool = new Yolov8ThreadPool();
    
    if (g_pool->setUp(model_file, num_threads) != NN_SUCCESS) {
        NN_LOG_ERROR("Failed to initialize YOLOv8 thread pool");
        delete g_pool;
        return -1;
    }
    
    std::thread reader(read_stream);
    std::thread writer(get_results, 640, 480, 30);
    
    reader.join();
    writer.join();
    
    delete g_pool;
    return 0;
} */


/* //功能都有，但不精确
/// 引入OpenCV主头文件，用于图像处理
#include <opencv2/opencv.hpp>
// 引入OpenCV的视频跟踪功能头文件
#include <opencv2/video/tracking.hpp>
// 引入Intel RealSense SDK的主头文件，用于与深度相机交互
#include <librealsense2/rs.hpp>
// 引入C++17的文件系统库，用于创建目录等操作
#include <filesystem>
// 引入C++的IO流控制库，用于格式化输出
#include <iomanip>
// 引入C++的字符串流库，用于在内存中构建字符串
#include <sstream>
// 引入C++的多线程库
#include <thread>
// 引入C语言的信号处理库，用于捕获Ctrl+C等信号
#include <csignal>
// 引入C++的map, deque, set, vector, numeric, iostream等标准库
#include <map>
#include <deque>
#include <set>
#include <vector>
#include <numeric>
#include <iostream>
#include <chrono>

// 引入SORT算法的核心组件
#include "utils/KalmanTracker.h"
#include "utils/Hungarian.h"
#include "utils/KalmanFilter3D.h"

// 引入YOLOv8相关的自定义模块
#include "task/yolov8_custom.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"
#include "task/yolov8_thread_pool.h"

// 为 std::filesystem 创建一个简短的命名空间别名 fs
namespace fs = std::filesystem;

// ------------------- 全局变量定义 -------------------
static int g_frame_start_id = 0;
static int g_frame_end_id = 0;
static Yolov8ThreadPool* g_pool = nullptr;
static volatile bool g_end = false;

// 跟踪器、历史记录等相关数据结构
std::map<int, KalmanTracker> trackers;
std::map<int, KalmanFilter3D> trackers_3d;
std::map<int, std::deque<cv::Point3f>> point3D_history;
std::map<int, std::deque<float>> diameter_history;
std::map<int, std::chrono::steady_clock::time_point> last_seen;
std::map<int, cv::Rect> last_known_boxes;
std::map<int, std::string> tracker_class_names;  // 存储每个跟踪目标的类别名称
std::map<int, float> tracker_confidences;        // 存储每个跟踪目标的最新置信度
const int MAX_HISTORY = 30;
const int MAX_DIAMETER_HISTORY = 10;

// ------------------- 信号处理函数 -------------------
void signal_handler(int) {
    g_end = true;
    NN_LOG_INFO("Received Ctrl+C signal, stopping...");
}

// 辅助函数
cv::Scalar get_track_color(int id) { 
    cv::RNG rng(id); 
    return cv::Scalar(rng.uniform(80,255), rng.uniform(80,255), rng.uniform(80,255)); 
}

double calculate_iou(const cv::Rect& box1, const cv::Rect& box2) { 
    cv::Rect intersection = box1 & box2; 
    double intersection_area = intersection.area(); 
    double union_area = box1.area() + box2.area() - intersection_area; 
    if (union_area < 1e-6) return 0; 
    return intersection_area / union_area; 
}

// 清理旧跟踪器
void cleanup_old_tracks() {
    const auto now = std::chrono::steady_clock::now();
    std::vector<int> ids_to_remove;
    
    for (const auto& [id, last_time] : last_seen) {
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - last_time).count();
        if (duration > 5) { // 超过5秒未更新则移除
            ids_to_remove.push_back(id);
        }
    }
    
    for (int id : ids_to_remove) {
        trackers.erase(id);
        trackers_3d.erase(id);
        point3D_history.erase(id);
        diameter_history.erase(id);
        last_seen.erase(id);
        last_known_boxes.erase(id);
        tracker_class_names.erase(id); // 清理类别名称
        tracker_confidences.erase(id); // 清理置信度
    }
}

// 3D边缘点计算
cv::Point3f get_3d_edge_point(const cv::Rect& box, bool is_left_edge, const rs2::depth_frame& depth, 
                             const rs2_intrinsics& intrin, int width, int height) {
    int x = is_left_edge ? box.x : box.x + box.width - 1;
    int y = box.y + box.height / 2;
    
    if (x < 0 || x >= width || y < 0 || y >= height) 
        return cv::Point3f(0, 0, 0);
    
    float d = depth.get_distance(x, y);
    if (d <= 0.1f || d >= 10.0f) 
        return cv::Point3f(0, 0, 0);
    
    float point3d[3], pixel[2] = {(float)x, (float)y};
    rs2_deproject_pixel_to_point(point3d, &intrin, pixel, d);
    return cv::Point3f(point3d[0], point3d[1], point3d[2]);
}

// 3D质心计算
cv::Point3f compute_3d_centroid(const cv::Rect& box, const rs2::depth_frame& depth, 
                               const rs2_intrinsics& intrin, int width, int height) {
    int valid_points = 0;
    cv::Point3f sum(0, 0, 0);
    
    for (int y = box.y; y < box.y + box.height; y += 4) {
        for (int x = box.x; x < box.x + box.width; x += 4) {
            if (x < 0 || x >= width || y < 0 || y >= height) 
                continue;
                
            float d = depth.get_distance(x, y);
            if (d >= 0.15f && d <= 8.0f) {
                float point3d[3];
                float pixel[2] = {(float)x, (float)y};
                rs2_deproject_pixel_to_point(point3d, &intrin, pixel, d);
                sum += cv::Point3f(point3d[0], point3d[1], point3d[2]);
                valid_points++;
            }
        }
    }
    
    if (valid_points > 10) {
        return sum * (1.0f / valid_points);
    }
    return cv::Point3f(0, 0, 0);
}

// ------------------- 跟踪与可视化主流程线程函数 (加粗字体+轨迹预测) -------------------
void get_results(int width = 640, int height = 480, int fps = 30) {
    auto start_all = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    static int next_track_id = 0;
    const double iou_threshold = 0.3;
    std::string output_dir = "output";
    fs::create_directories(output_dir);
    HungarianAlgorithm hungarian_solver;

    const float model_w = 640.0f;
    const float model_h = 640.0f;
    float scale = std::min(model_w / (float)width, model_h / (float)height);
    int pad_x = (model_w - (float)width * scale) / 2;
    int pad_y = (model_h - (float)height * scale) / 2;

    // 定义字体参数（全局使用）
    const float font_scale = 0.4;  // 字体大小
    const int font_thickness = 1.5;  // 加粗字体（从1改为2）
    const int line_spacing = 10;   // 行间距

    // 速度计算相关变量
    std::map<int, cv::Point3f> last_positions;
    std::map<int, std::deque<float>> speed_history;
    const int MAX_SPEED_HISTORY = 10;
    const float fps_float = static_cast<float>(fps); // 帧率用于速度计算
    auto last_time = std::chrono::steady_clock::now();

    while (!g_end) {
        cv::Mat img;
        auto ret_img = g_pool->getTargetImgResult(img, g_frame_end_id);
        if (ret_img != NN_SUCCESS) { 
            if (g_end) break; 
            std::this_thread::sleep_for(std::chrono::milliseconds(5)); 
            continue; 
        }

        std::vector<Detection> dets_from_pool;
        auto ret_det = g_pool->getTargetResult(dets_from_pool, g_frame_end_id);

        std::vector<Detection> restored_dets;
        if (ret_det == NN_SUCCESS) {
            for (const auto& det : dets_from_pool) {
                Detection restored_det = det;
                restored_det.box.x = (det.box.x - pad_x) / scale;
                restored_det.box.y = (det.box.y - pad_y) / scale;
                restored_det.box.width = det.box.width / scale;
                restored_det.box.height = det.box.height / scale;
                if (restored_det.confidence < 0.5) continue;
                restored_dets.push_back(restored_det);
            }
        }

        auto depth = g_pool->GetDepth();
        rs2_intrinsics intrin = g_pool->GetIntrinsics();

        std::vector<int> track_ids;
        std::vector<cv::Rect> predicted_boxes;
        for (auto& [id, tracker] : trackers) {
            cv::Rect2f pred = tracker.predict();
            track_ids.push_back(id);
            predicted_boxes.push_back(pred);
        }

        std::vector<std::vector<double>> cost_matrix(track_ids.size(), std::vector<double>(restored_dets.size(), 1.0));
        for (size_t i = 0; i < track_ids.size(); ++i)
            for (size_t j = 0; j < restored_dets.size(); ++j) {
                double iou = calculate_iou(predicted_boxes[i], restored_dets[j].box);
                if (iou > iou_threshold) cost_matrix[i][j] = 1.0 - iou;
            }
        std::vector<int> assignment;
        if (!track_ids.empty() && !restored_dets.empty())
            hungarian_solver.Solve(cost_matrix, assignment);

        std::set<int> matched_det_indices;
        for (size_t i = 0; i < assignment.size(); ++i) {
            int det_idx = assignment[i];
            if (det_idx >= 0 && cost_matrix[i][det_idx] < 1.0 - iou_threshold) {
                int id = track_ids[i];
                trackers[id].update(restored_dets[det_idx].box);
                
                // 更新类别名称和置信度
                if (det_idx < restored_dets.size()) {
                    tracker_class_names[id] = restored_dets[det_idx].className;
                    tracker_confidences[id] = restored_dets[det_idx].confidence;
                }
                
                cv::Rect smoothed_box = trackers[id].get_state();

                cv::Rect analysis_box = smoothed_box;
                int shrink_x = smoothed_box.width * 0.15;
                int shrink_y = smoothed_box.height * 0.15;
                analysis_box.x += shrink_x;
                analysis_box.y += shrink_y;
                analysis_box.width -= 2 * shrink_x;
                analysis_box.height -= 2 * shrink_y;
                
                cv::Point3f noisy_centroid(0,0,0);
                if (analysis_box.width > 0 && analysis_box.height > 0) {
                    noisy_centroid = compute_3d_centroid(analysis_box, depth, intrin, width, height);
                }

                if (noisy_centroid.x == 0 && noisy_centroid.y == 0 && noisy_centroid.z == 0) {
                    int cx = smoothed_box.x + smoothed_box.width / 2;
                    int cy = smoothed_box.y + smoothed_box.height / 2;
                    if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
                        float d = depth.get_distance(cx, cy);
                        if (d > 0.1f && d < 10.0f) {
                            float point3d[3], pixel[2] = {(float)cx, (float)cy};
                            rs2_deproject_pixel_to_point(point3d, &intrin, pixel, d);
                            noisy_centroid = cv::Point3f(point3d[0], point3d[1], point3d[2]);
                        }
                    }
                }

                if (noisy_centroid.x != 0 || noisy_centroid.y != 0 || noisy_centroid.z != 0) {
                    cv::Point3f smoothed_centroid = trackers_3d[id].update(noisy_centroid);
                    point3D_history[id].push_back(smoothed_centroid);
                    if (point3D_history[id].size() > MAX_HISTORY) 
                        point3D_history[id].pop_front();
                    
                    // 计算速度
                    if (last_positions.find(id) != last_positions.end()) {
                        auto current_time = std::chrono::steady_clock::now();
                        float time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
                            current_time - last_time).count() / 1000.0f;
                        
                        if (time_diff > 0) {
                            cv::Point3f displacement = smoothed_centroid - last_positions[id];
                            float distance = std::sqrt(displacement.x * displacement.x + 
                                                     displacement.y * displacement.y + 
                                                     displacement.z * displacement.z);
                            float speed = distance / time_diff; // m/s
                            
                            // 存储速度历史
                            speed_history[id].push_back(speed);
                            if (speed_history[id].size() > MAX_SPEED_HISTORY) 
                                speed_history[id].pop_front();
                        }
                    }
                    last_positions[id] = smoothed_centroid;
                }
                
                last_seen[id] = std::chrono::steady_clock::now();
                last_known_boxes[id] = restored_dets[det_idx].box;
                matched_det_indices.insert(det_idx);
            }
        }
        
        for (size_t j = 0; j < restored_dets.size(); ++j) {
            if (matched_det_indices.find(j) == matched_det_indices.end()) {
                int new_id = next_track_id++;
                trackers[new_id] = KalmanTracker(restored_dets[j].box);
                last_known_boxes[new_id] = restored_dets[j].box;
                last_seen[new_id] = std::chrono::steady_clock::now();
                
                // 保存类别名称和置信度
                tracker_class_names[new_id] = restored_dets[j].className;
                tracker_confidences[new_id] = restored_dets[j].confidence;
                
                cv::Point3f initial_centroid = compute_3d_centroid(restored_dets[j].box, depth, intrin, width, height);
                if (initial_centroid.x != 0 || initial_centroid.y != 0 || initial_centroid.z != 0) {
                    trackers_3d[new_id].init(initial_centroid);
                    point3D_history[new_id].push_back(initial_centroid);
                    last_positions[new_id] = initial_centroid;
                }
            }
        }

        last_time = std::chrono::steady_clock::now();
        cleanup_old_tracks();

        for (auto const& [id, tracker] : trackers) {
            if (tracker.m_time_since_update > 2 && tracker.m_hit_streak < 3) continue;

            cv::Rect box = tracker.get_state();
            cv::Scalar color = get_track_color(id);
            cv::rectangle(img, box, color, 2);
            
            // 获取类别名称和置信度
            std::string class_name = "object";
            float confidence = 0.0f;
            
            if (tracker_class_names.find(id) != tracker_class_names.end()) {
                class_name = tracker_class_names[id];
            }
            
            if (tracker_confidences.find(id) != tracker_confidences.end()) {
                confidence = tracker_confidences[id];
            }
            
            // 显示主要信息（框上方）
            std::ostringstream main_label;
            main_label << "id:" << id << " " << class_name << " " 
                       << std::fixed << std::setprecision(2) << confidence;
            cv::putText(img, main_label.str(), 
                       cv::Point(box.x, box.y - 5), 
                       cv::FONT_HERSHEY_SIMPLEX, 
                       font_scale, 
                       color, 
                       font_thickness);
            
            // 显示3D坐标（框下方第一行）
            if (point3D_history.count(id) && !point3D_history[id].empty()) {
                const auto& pt = point3D_history[id].back();
                std::ostringstream pos_label;
                pos_label << "x:" << std::fixed << std::setprecision(2) << pt.x 
                         << " y:" << std::fixed << std::setprecision(2) << pt.y 
                         << " z:" << std::fixed << std::setprecision(2) << pt.z;
                cv::putText(img, pos_label.str(), 
                           cv::Point(box.x, box.y + box.height + line_spacing), 
                           cv::FONT_HERSHEY_SIMPLEX, 
                           font_scale, 
                           color, 
                           font_thickness);
            }
            
            // 显示直径（框下方第二行）
            cv::Point3f left_3d = get_3d_edge_point(box, true, depth, intrin, width, height);
            cv::Point3f right_3d = get_3d_edge_point(box, false, depth, intrin, width, height);
            if (left_3d.x != 0 && right_3d.x != 0) {
                float current_diameter_mm = cv::norm(left_3d - right_3d) * 1000.0f;
                diameter_history[id].push_back(current_diameter_mm);
                if (diameter_history[id].size() > MAX_DIAMETER_HISTORY) 
                    diameter_history[id].pop_front();
                float sum = std::accumulate(diameter_history[id].begin(), diameter_history[id].end(), 0.0f);
                float smoothed_diameter_mm = sum / diameter_history[id].size();
                std::ostringstream dia_label;
                dia_label << "Dia:" << std::fixed << std::setprecision(1) << smoothed_diameter_mm << "mm";
                cv::putText(img, dia_label.str(), 
                           cv::Point(box.x, box.y + box.height + line_spacing * 2), 
                           cv::FONT_HERSHEY_SIMPLEX, 
                           font_scale, 
                           color, 
                           font_thickness);
            }

            // 显示速度（框下方第三行）
            if (speed_history.find(id) != speed_history.end() && !speed_history[id].empty()) {
                float sum_speed = std::accumulate(speed_history[id].begin(), speed_history[id].end(), 0.0f);
                float avg_speed = sum_speed / speed_history[id].size();
                std::ostringstream speed_label;
                speed_label << "Speed:" << std::fixed << std::setprecision(3) << avg_speed << " m/s";
                cv::putText(img, speed_label.str(), 
                           cv::Point(box.x, box.y + box.height + line_spacing * 3), 
                           cv::FONT_HERSHEY_SIMPLEX, 
                           font_scale, 
                           color, 
                           font_thickness);
            }

            // ============== 轨迹预测功能 ==============
            if (trackers_3d.count(id)) {
                // 预测未来轨迹
                std::vector<cv::Point3f> future_points = trackers_3d[id].predict_future(20);
                std::vector<cv::Point> future_pixels;
                
                // 转换3D点到2D像素坐标
                for (const auto& pt : future_points) {
                    float point3d[3] = {pt.x, pt.y, pt.z};
                    float pixel[2];
                    rs2_project_point_to_pixel(pixel, &intrin, point3d);
                    if (pixel[0] > 0 && pixel[0] < width && pixel[1] > 0 && pixel[1] < height) {
                        future_pixels.emplace_back(static_cast<int>(pixel[0]), static_cast<int>(pixel[1]));
                    }
                }
                
                // 绘制预测轨迹（带箭头的线）
                if (future_pixels.size() > 1) {
                    for (size_t i = 0; i < future_pixels.size() - 1; i++) {
                        // 绘制轨迹线
                        cv::line(img, future_pixels[i], future_pixels[i+1], 
                                cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
                        
                        // 每隔5个点绘制一个箭头
                        if (i % 5 == 0 && (i + 1) < future_pixels.size()) {
                            cv::arrowedLine(img, future_pixels[i], future_pixels[i+1], 
                                          cv::Scalar(0, 200, 0), 2, cv::LINE_AA, 0, 0.1);
                        }
                    }
                    
                    // 在轨迹终点绘制目标图标
                    if (!future_pixels.empty()) {
                        cv::drawMarker(img, future_pixels.back(), cv::Scalar(0, 255, 0), 
                                      cv::MARKER_TILTED_CROSS, 10, 2);
                    }
                }
            }
            
            // 显示历史轨迹（实际运动轨迹）
            if (point3D_history.count(id) && point3D_history[id].size() > 1) {
                const auto& hist = point3D_history[id];
                std::vector<cv::Point> history_pixels;
                
                // 转换历史点到2D像素坐标
                for (const auto& pt : hist) {
                    float point3d[3] = {pt.x, pt.y, pt.z};
                    float pixel[2];
                    rs2_project_point_to_pixel(pixel, &intrin, point3d);
                    if (pixel[0] > 0 && pixel[0] < width && pixel[1] > 0 && pixel[1] < height) {
                        history_pixels.emplace_back(static_cast<int>(pixel[0]), static_cast<int>(pixel[1]));
                    }
                }
                
                // 绘制历史轨迹（实线）
                if (history_pixels.size() > 1) {
                    for (size_t j = 1; j < history_pixels.size(); ++j) {
                        cv::line(img, history_pixels[j-1], history_pixels[j], 
                                color, 2, cv::LINE_AA);
                    }
                    
                    // 在当前位置绘制目标图标
                    if (!history_pixels.empty()) {
                        cv::circle(img, history_pixels.back(), 5, color, -1);
                    }
                }
            }
        }

        std::ostringstream filename;
        filename << output_dir << "/" << std::setw(6) << std::setfill('0') << g_frame_end_id << ".jpg";
        cv::imwrite(filename.str(), img);
        g_frame_end_id++;
        frame_count++;
        
        auto end_all = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count();
        if (elapsed > 1000.0f) {
            NN_LOG_INFO("FPS: %.2f", frame_count / (elapsed / 1000.0f));
            frame_count = 0;
            start_all = std::chrono::high_resolution_clock::now();
        }
    }
    g_pool->stopAll();
    NN_LOG_INFO("Result thread exited.");
}

// ------------------- RealSense数据流读取线程函数 -------------------
void read_stream() {
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    try {
        rs2::pipeline_profile profile = pipe.start(cfg);
        rs2::align align_to_color(RS2_STREAM_COLOR);
        
        // 获取深度传感器配置
        auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
        if (depth_sensor.supports(RS2_OPTION_EMITTER_ENABLED)) {
            depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 1.f);
        }
        
        while (!g_end) {
            rs2::frameset frameset;
            if (!pipe.try_wait_for_frames(&frameset, 1000)) continue;
            frameset = align_to_color.process(frameset);
            rs2::frame color_frame = frameset.get_color_frame();
            rs2::depth_frame depth_frame = frameset.get_depth_frame();
            if (!color_frame || !depth_frame) continue;
            
            rs2_intrinsics intrin = color_frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
            cv::Mat img(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
            
            g_pool->SetDepthFrame(depth_frame, intrin);
            auto submit_ret = g_pool->submitTask(img.clone(), g_frame_start_id);
            
            if (submit_ret != NN_SUCCESS) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            } else {
                g_frame_start_id++;
            }
        }
        pipe.stop();
    } catch (const rs2::error & e) {
        NN_LOG_ERROR("RealSense camera could not be started: %s", e.what()); 
        g_end = true;
    }
}

// ------------------- 程序主函数 -------------------
int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <model_file> [num_threads]\n", argv[0]);
        return 1;
    }
    std::string model_file = argv[1];
    const int num_threads = (argc > 2) ? std::stoi(argv[2]) : 4;
    signal(SIGINT, signal_handler);
    g_pool = new Yolov8ThreadPool();
    
    if (g_pool->setUp(model_file, num_threads) != NN_SUCCESS) {
        NN_LOG_ERROR("Failed to initialize YOLOv8 thread pool");
        delete g_pool;
        return -1;
    }
    
    std::thread reader(read_stream);
    std::thread writer(get_results, 640, 480, 30);
    
    reader.join();
    writer.join();
    
    delete g_pool;
    return 0;
} */


/* //完整的代码第一版
/// 引入OpenCV主头文件，用于图像处理
#include <opencv2/opencv.hpp>
// 引入OpenCV的视频跟踪功能头文件
#include <opencv2/video/tracking.hpp>
// 引入Intel RealSense SDK的主头文件，用于与深度相机交互
#include <librealsense2/rs.hpp>
// 引入C++17的文件系统库，用于创建目录等操作
#include <filesystem>
// 引入C++的IO流控制库，用于格式化输出
#include <iomanip>
// 引入C++的字符串流库，用于在内存中构建字符串
#include <sstream>
// 引入C++的多线程库
#include <thread>
// 引入C语言的信号处理库，用于捕获Ctrl+C等信号
#include <csignal>
// 引入C++的map, deque, set, vector, numeric, iostream等标准库
#include <map>
#include <deque>
#include <set>
#include <vector>
#include <numeric>
#include <iostream>
#include <chrono>

// 引入SORT算法的核心组件
#include "utils/KalmanTracker.h"
#include "utils/Hungarian.h"
#include "utils/KalmanFilter3D.h"  // 使用头文件中定义的KalmanFilter3D

// 引入YOLOv8相关的自定义模块
#include "task/yolov8_custom.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"
#include "task/yolov8_thread_pool.h"

// 为 std::filesystem 创建一个简短的命名空间别名 fs
namespace fs = std::filesystem;

// ------------------- 全局变量定义 -------------------
static int g_frame_start_id = 0;
static int g_frame_end_id = 0;
static Yolov8ThreadPool* g_pool = nullptr;
static volatile bool g_end = false;

// 跟踪器、历史记录等相关数据结构
std::map<int, KalmanTracker> trackers;
std::map<int, KalmanFilter3D> trackers_3d;  // 使用头文件中的KalmanFilter3D
std::map<int, std::deque<cv::Point3f>> point3D_history;
std::map<int, std::deque<float>> diameter_history;
std::map<int, std::chrono::steady_clock::time_point> last_seen;
std::map<int, cv::Rect> last_known_boxes;
std::map<int, std::string> tracker_class_names;  // 存储每个跟踪目标的类别名称
std::map<int, float> tracker_confidences;        // 存储每个跟踪目标的最新置信度
const int MAX_HISTORY = 30;
const int MAX_DIAMETER_HISTORY = 1;  // 只使用最新直径值

// ------------------- 信号处理函数 -------------------
void signal_handler(int) {
    g_end = true;
    NN_LOG_INFO("Received Ctrl+C signal, stopping...");
}

// 辅助函数
// cv::Scalar get_track_color(int id) { 
//     cv::RNG rng(id); 
//     return cv::Scalar(rng.uniform(80,255), rng.uniform(80,255), rng.uniform(80,255)); 
// }
cv::Scalar get_track_color(int id) {
    return cv::Scalar(255, 0, 0);  // BGR: 黄色
}


double calculate_iou(const cv::Rect& box1, const cv::Rect& box2) { 
    cv::Rect intersection = box1 & box2; 
    double intersection_area = intersection.area(); 
    double union_area = box1.area() + box2.area() - intersection_area; 
    if (union_area < 1e-6) return 0; 
    return intersection_area / union_area; 
}

// 清理旧跟踪器
void cleanup_old_tracks() {
    const auto now = std::chrono::steady_clock::now();
    std::vector<int> ids_to_remove;
    
    for (const auto& [id, last_time] : last_seen) {
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - last_time).count();
        if (duration > 5) { // 超过5秒未更新则移除
            ids_to_remove.push_back(id);
        }
    }
    
    for (int id : ids_to_remove) {
        trackers.erase(id);
        trackers_3d.erase(id);
        point3D_history.erase(id);
        diameter_history.erase(id);
        last_seen.erase(id);
        last_known_boxes.erase(id);
        tracker_class_names.erase(id); // 清理类别名称
        tracker_confidences.erase(id); // 清理置信度
    }
}

// 3D边缘点计算
cv::Point3f get_3d_edge_point(const cv::Rect& box, bool is_left_edge, const rs2::depth_frame& depth, 
                             const rs2_intrinsics& intrin, int width, int height) {
    int x = is_left_edge ? box.x : box.x + box.width - 1;
    int y = box.y + box.height / 2;
    
    if (x < 0 || x >= width || y < 0 || y >= height) 
        return cv::Point3f(0, 0, 0);
    
    float d = depth.get_distance(x, y);
    if (d <= 0.1f || d >= 10.0f) 
        return cv::Point3f(0, 0, 0);
    
    float point3d[3], pixel[2] = {(float)x, (float)y};
    rs2_deproject_pixel_to_point(point3d, &intrin, pixel, d);
    return cv::Point3f(point3d[0], point3d[1], point3d[2]);
}

// 3D质心计算
cv::Point3f compute_3d_centroid(const cv::Rect& box, const rs2::depth_frame& depth, 
                               const rs2_intrinsics& intrin, int width, int height) {
    int valid_points = 0;
    cv::Point3f sum(0, 0, 0);
    
    for (int y = box.y; y < box.y + box.height; y += 4) {
        for (int x = box.x; x < box.x + box.width; x += 4) {
            if (x < 0 || x >= width || y < 0 || y >= height) 
                continue;
                
            float d = depth.get_distance(x, y);
            if (d >= 0.15f && d <= 8.0f) {
                float point3d[3];
                float pixel[2] = {(float)x, (float)y};
                rs2_deproject_pixel_to_point(point3d, &intrin, pixel, d);
                sum += cv::Point3f(point3d[0], point3d[1], point3d[2]);
                valid_points++;
            }
        }
    }
    
    if (valid_points > 10) {
        return sum * (1.0f / valid_points);
    }
    return cv::Point3f(0, 0, 0);
}

// ------------------- 跟踪与可视化主流程线程函数 (加粗字体+轨迹预测) -------------------
void get_results(int width = 640, int height = 480, int fps = 30) {
    auto start_all = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    static int next_track_id = 0;
    const double iou_threshold = 0.3;
    std::string output_dir = "output";
    fs::create_directories(output_dir);
    HungarianAlgorithm hungarian_solver;

    const float model_w = 640.0f;
    const float model_h = 640.0f;
    float scale = std::min(model_w / (float)width, model_h / (float)height);
    int pad_x = (model_w - (float)width * scale) / 2;
    int pad_y = (model_h - (float)height * scale) / 2;

    // 定义字体参数（全局使用）
    const float font_scale = 0.4;  // 字体大小
    const int font_thickness = 1.5;  // 加粗字体
    const int line_spacing = 10;   // 行间距

    // 速度计算相关变量
    std::map<int, cv::Point3f> last_positions;
    std::map<int, float> current_speeds; // 当前帧计算的速度
    const float fps_float = static_cast<float>(fps); // 帧率用于速度计算
    auto last_time = std::chrono::steady_clock::now();

    while (!g_end) {
        cv::Mat img;
        auto ret_img = g_pool->getTargetImgResult(img, g_frame_end_id);
        if (ret_img != NN_SUCCESS) { 
            if (g_end) break; 
            std::this_thread::sleep_for(std::chrono::milliseconds(5)); 
            continue; 
        }

        std::vector<Detection> dets_from_pool;
        auto ret_det = g_pool->getTargetResult(dets_from_pool, g_frame_end_id);

        std::vector<Detection> restored_dets;
        if (ret_det == NN_SUCCESS) {
            for (const auto& det : dets_from_pool) {
                Detection restored_det = det;
                restored_det.box.x = (det.box.x - pad_x) / scale;
                restored_det.box.y = (det.box.y - pad_y) / scale;
                restored_det.box.width = det.box.width / scale;
                restored_det.box.height = det.box.height / scale;
                if (restored_det.confidence < 0.5) continue;
                restored_dets.push_back(restored_det);
            }
        }

        auto depth = g_pool->GetDepth();
        rs2_intrinsics intrin = g_pool->GetIntrinsics();

        std::vector<int> track_ids;
        std::vector<cv::Rect> predicted_boxes;
        for (auto& [id, tracker] : trackers) {
            cv::Rect2f pred = tracker.predict();
            track_ids.push_back(id);
            predicted_boxes.push_back(pred);
        }

        std::vector<std::vector<double>> cost_matrix(track_ids.size(), std::vector<double>(restored_dets.size(), 1.0));
        for (size_t i = 0; i < track_ids.size(); ++i)
            for (size_t j = 0; j < restored_dets.size(); ++j) {
                double iou = calculate_iou(predicted_boxes[i], restored_dets[j].box);
                if (iou > iou_threshold) cost_matrix[i][j] = 1.0 - iou;
            }
        std::vector<int> assignment;
        if (!track_ids.empty() && !restored_dets.empty())
            hungarian_solver.Solve(cost_matrix, assignment);

        std::set<int> matched_det_indices;
        for (size_t i = 0; i < assignment.size(); ++i) {
            int det_idx = assignment[i];
            if (det_idx >= 0 && cost_matrix[i][det_idx] < 1.0 - iou_threshold) {
                int id = track_ids[i];
                trackers[id].update(restored_dets[det_idx].box);
                
                // 更新类别名称和置信度
                if (det_idx < restored_dets.size()) {
                    tracker_class_names[id] = restored_dets[det_idx].className;
                    tracker_confidences[id] = restored_dets[det_idx].confidence;
                }
                
                cv::Rect smoothed_box = trackers[id].get_state();

                cv::Rect analysis_box = smoothed_box;
                int shrink_x = smoothed_box.width * 0.15;
                int shrink_y = smoothed_box.height * 0.15;
                analysis_box.x += shrink_x;
                analysis_box.y += shrink_y;
                analysis_box.width -= 2 * shrink_x;
                analysis_box.height -= 2 * shrink_y;
                
                cv::Point3f noisy_centroid(0,0,0);
                if (analysis_box.width > 0 && analysis_box.height > 0) {
                    noisy_centroid = compute_3d_centroid(analysis_box, depth, intrin, width, height);
                }

                if (noisy_centroid.x == 0 && noisy_centroid.y == 0 && noisy_centroid.z == 0) {
                    int cx = smoothed_box.x + smoothed_box.width / 2;
                    int cy = smoothed_box.y + smoothed_box.height / 2;
                    if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
                        float d = depth.get_distance(cx, cy);
                        if (d > 0.1f && d < 10.0f) {
                            float point3d[3], pixel[2] = {(float)cx, (float)cy};
                            rs2_deproject_pixel_to_point(point3d, &intrin, pixel, d);
                            noisy_centroid = cv::Point3f(point3d[0], point3d[1], point3d[2]);
                        }
                    }
                }

                if (noisy_centroid.x != 0 || noisy_centroid.y != 0 || noisy_centroid.z != 0) {
                    cv::Point3f smoothed_centroid = trackers_3d[id].update(noisy_centroid);
                    point3D_history[id].push_back(smoothed_centroid);
                    if (point3D_history[id].size() > MAX_HISTORY) 
                        point3D_history[id].pop_front();
                    
                    // 计算速度
                    if (last_positions.find(id) != last_positions.end()) {
                        auto current_time = std::chrono::steady_clock::now();
                        float time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
                            current_time - last_time).count() / 1000.0f;
                        
                        if (time_diff > 0) {
                            cv::Point3f displacement = smoothed_centroid - last_positions[id];
                            float distance = std::sqrt(displacement.x * displacement.x + 
                                                     displacement.y * displacement.y + 
                                                     displacement.z * displacement.z);
                            current_speeds[id] = distance / time_diff; // m/s
                        }
                    }
                    last_positions[id] = smoothed_centroid;
                }
                
                last_seen[id] = std::chrono::steady_clock::now();
                last_known_boxes[id] = restored_dets[det_idx].box;
                matched_det_indices.insert(det_idx);
            }
        }
        
        for (size_t j = 0; j < restored_dets.size(); ++j) {
            if (matched_det_indices.find(j) == matched_det_indices.end()) {
                int new_id = next_track_id++;
                trackers[new_id] = KalmanTracker(restored_dets[j].box);
                last_known_boxes[new_id] = restored_dets[j].box;
                last_seen[new_id] = std::chrono::steady_clock::now();
                
                // 保存类别名称和置信度
                tracker_class_names[new_id] = restored_dets[j].className;
                tracker_confidences[new_id] = restored_dets[j].confidence;
                
                cv::Point3f initial_centroid = compute_3d_centroid(restored_dets[j].box, depth, intrin, width, height);
                if (initial_centroid.x != 0 || initial_centroid.y != 0 || initial_centroid.z != 0) {
                    trackers_3d[new_id].init(initial_centroid);
                    point3D_history[new_id].push_back(initial_centroid);
                    last_positions[new_id] = initial_centroid;
                }
            }
        }

        last_time = std::chrono::steady_clock::now();
        cleanup_old_tracks();

        for (auto const& [id, tracker] : trackers) {
            if (tracker.m_time_since_update > 2 && tracker.m_hit_streak < 3) continue;

            cv::Rect box = tracker.get_state();
            cv::Scalar color = get_track_color(id);
            cv::rectangle(img, box, color, 2);
            
            // 获取类别名称和置信度
            std::string class_name = "object";
            float confidence = 0.0f;
            
            if (tracker_class_names.find(id) != tracker_class_names.end()) {
                class_name = tracker_class_names[id];
            }
            
            if (tracker_confidences.find(id) != tracker_confidences.end()) {
                confidence = tracker_confidences[id];
            }
            
            // 显示主要信息（框上方）- 格式: id:6 ball 0.799
            std::ostringstream main_label;
            main_label << "id:" << id << " " << class_name << " " 
                       << std::fixed << std::setprecision(3) << confidence;
            // cv::putText(img, main_label.str(), 
            //            cv::Point(box.x, box.y - 5), 
            //            cv::FONT_HERSHEY_SIMPLEX, 
            //            font_scale, 
            //            color, 
            //            font_thickness);
            cv::putText(img, main_label.str(), 
                        cv::Point(box.x, box.y - 5), 
                        cv::FONT_HERSHEY_SIMPLEX, 
                        font_scale, 
                        cv::Scalar(255, 255, 255),  // 白色
                        font_thickness);


            
            // 显示3D坐标（框下方第一行）- 格式: xè=0.07 Yè=0.06 Zs0.30 m
            if (point3D_history.count(id) && !point3D_history[id].empty()) {
                const auto& pt = point3D_history[id].back();
                std::ostringstream pos_label;
                pos_label << "    x=" << std::fixed << std::setprecision(2) << pt.x 
                         << " Y=" << std::fixed << std::setprecision(2) << pt.y 
                         << " Z=" << std::fixed << std::setprecision(2) << pt.z << " m";
                // cv::putText(img, pos_label.str(), 
                //            cv::Point(box.x, box.y + box.height + line_spacing), 
                //            cv::FONT_HERSHEY_SIMPLEX, 
                //            font_scale, 
                //            color, 
                //            font_thickness);
                //cv::putText(img, pos_label.str(), ..., cv::Scalar(0, 255, 255), font_thickness);  // 黄色
                cv::putText(img, pos_label.str(), 
                            cv::Point(box.x, box.y + box.height + line_spacing), 
                            cv::FONT_HERSHEY_SIMPLEX, 
                            font_scale, 
                            cv::Scalar(0, 255, 255),  // 浅灰色
                            font_thickness);


            }
            
            // 显示直径（框下方第二行）- 格式: Dia: 38.1 mm
            cv::Point3f left_3d = get_3d_edge_point(box, true, depth, intrin, width, height);
            cv::Point3f right_3d = get_3d_edge_point(box, false, depth, intrin, width, height);
            if (left_3d.x != 0 && right_3d.x != 0) {
                float current_diameter_mm = cv::norm(left_3d - right_3d) * 1000.0f;
                diameter_history[id].push_back(current_diameter_mm);
                if (diameter_history[id].size() > MAX_DIAMETER_HISTORY) 
                    diameter_history[id].pop_front();
                
                if (!diameter_history[id].empty()) {
                    std::ostringstream dia_label;
                    dia_label << "    Dia: " << std::fixed << std::setprecision(1) 
                             << diameter_history[id].back() << " mm";
                    // cv::putText(img, dia_label.str(), 
                    //            cv::Point(box.x, box.y + box.height + line_spacing * 2), 
                    //            cv::FONT_HERSHEY_SIMPLEX, 
                    //            font_scale, 
                    //            color, 
                    //            font_thickness);
                    //cv::putText(img, dia_label.str(), ..., cv::Scalar(0, 0, 255), font_thickness);  // 红色
                    cv::putText(img, dia_label.str(), 
                            cv::Point(box.x, box.y + box.height + line_spacing * 2), 
                            cv::FONT_HERSHEY_SIMPLEX, 
                            font_scale, 
                            cv::Scalar(20, 0, 255),  // 浅灰色
                            font_thickness);


                }
            }

            // 显示速度（框下方第三行）- 格式: Speed: 0.110 m/s
            if (current_speeds.find(id) != current_speeds.end()) {
                std::ostringstream speed_label;
                speed_label << "    Speed: " << std::fixed << std::setprecision(3) 
                           << current_speeds[id] << " m/s";
                // cv::putText(img, speed_label.str(), 
                //            cv::Point(box.x, box.y + box.height + line_spacing * 3), 
                //            cv::FONT_HERSHEY_SIMPLEX, 
                //            font_scale, 
                //            color, 
                //            font_thickness);
                    cv::putText(img, speed_label.str(), 
                                cv::Point(box.x, box.y + box.height + line_spacing * 3), 
                                cv::FONT_HERSHEY_SIMPLEX, 
                                font_scale, 
                                cv::Scalar(200, 200, 200),  // 浅灰色 (BGR)
                                font_thickness);


            }

            // ============== 轨迹预测功能 ==============
            if (trackers_3d.count(id)) {
                // 预测未来轨迹 (20步)
                std::vector<cv::Point3f> future_points = trackers_3d[id].predict_future(40);
                std::vector<cv::Point> future_pixels;
                
                // 转换3D点到2D像素坐标
                for (const auto& pt : future_points) {
                    float point3d[3] = {pt.x, pt.y, pt.z};
                    float pixel[2];
                    rs2_project_point_to_pixel(pixel, &intrin, point3d);
                    if (pixel[0] > 0 && pixel[0] < width && pixel[1] > 0 && pixel[1] < height) {
                        future_pixels.emplace_back(static_cast<int>(pixel[0]), static_cast<int>(pixel[1]));
                    }
                }
                
                // 绘制预测轨迹点 (绿色小圆点)
                for (const auto& px : future_pixels) {
                    cv::circle(img, px, 4, cv::Scalar(255, 0, 255), -1);  // 紫色小圆点
                }
                
                // 在终点添加标记 (绿色十字)
                if (!future_pixels.empty()) {
                    cv::drawMarker(img, future_pixels.back(), cv::Scalar(0, 255, 0), 
                                  cv::MARKER_TILTED_CROSS, 10, 2);
                }
            }
            
            // 显示历史轨迹（实际运动轨迹）
            if (point3D_history.count(id) && point3D_history[id].size() > 1) {
                const auto& hist = point3D_history[id];
                std::vector<cv::Point> history_pixels;
                
                // 转换历史点到2D像素坐标
                for (const auto& pt : hist) {
                    float point3d[3] = {pt.x, pt.y, pt.z};
                    float pixel[2];
                    rs2_project_point_to_pixel(pixel, &intrin, point3d);
                    if (pixel[0] > 0 && pixel[0] < width && pixel[1] > 0 && pixel[1] < height) {
                        history_pixels.emplace_back(static_cast<int>(pixel[0]), static_cast<int>(pixel[1]));
                    }
                }
                
                // 绘制历史轨迹（实线）
                if (history_pixels.size() > 1) {
                    for (size_t j = 1; j < history_pixels.size(); ++j) {
                        // cv::line(img, history_pixels[j-1], history_pixels[j], 
                        //         color, 2, cv::LINE_AA);
                        cv::line(img, history_pixels[j-1], history_pixels[j], 
                            cv::Scalar(0, 165, 255), 2, cv::LINE_AA);
                   
                    }
                    
                    // 在当前位置绘制目标图标
                    if (!history_pixels.empty()) {
                        //cv::circle(img, history_pixels.back(), 5, color, -1);
                        cv::circle(img, history_pixels.back(), 5, cv::Scalar(0, 165, 255), -1);

                    }
                }
            }
        }

        std::ostringstream filename;
        filename << output_dir << "/" << std::setw(6) << std::setfill('0') << g_frame_end_id << ".jpg";
        cv::imwrite(filename.str(), img);
        g_frame_end_id++;
        frame_count++;
        
        auto end_all = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count();
        if (elapsed > 1000.0f) {
            NN_LOG_INFO("FPS: %.2f", frame_count / (elapsed / 1000.0f));
            frame_count = 0;
            start_all = std::chrono::high_resolution_clock::now();
        }
    }
    g_pool->stopAll();
    NN_LOG_INFO("Result thread exited.");
}

// ------------------- RealSense数据流读取线程函数 -------------------
void read_stream() {
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    try {
        rs2::pipeline_profile profile = pipe.start(cfg);
        rs2::align align_to_color(RS2_STREAM_COLOR);
        
        // 获取深度传感器配置
        auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
        if (depth_sensor.supports(RS2_OPTION_EMITTER_ENABLED)) {
            depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 1.f);
        }
        
        while (!g_end) {
            rs2::frameset frameset;
            if (!pipe.try_wait_for_frames(&frameset, 1000)) continue;
            frameset = align_to_color.process(frameset);
            rs2::frame color_frame = frameset.get_color_frame();
            rs2::depth_frame depth_frame = frameset.get_depth_frame();
            if (!color_frame || !depth_frame) continue;
            
            rs2_intrinsics intrin = color_frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
            cv::Mat img(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
            
            g_pool->SetDepthFrame(depth_frame, intrin);
            auto submit_ret = g_pool->submitTask(img.clone(), g_frame_start_id);
            
            if (submit_ret != NN_SUCCESS) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            } else {
                g_frame_start_id++;
            }
        }
        pipe.stop();
    } catch (const rs2::error & e) {
        NN_LOG_ERROR("RealSense camera could not be started: %s", e.what()); 
        g_end = true;
    }
}

// ------------------- 程序主函数 -------------------
int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <model_file> [num_threads]\n", argv[0]);
        return 1;
    }
    std::string model_file = argv[1];
    const int num_threads = (argc > 2) ? std::stoi(argv[2]) : 4;
    signal(SIGINT, signal_handler);
    g_pool = new Yolov8ThreadPool();
    
    if (g_pool->setUp(model_file, num_threads) != NN_SUCCESS) {
        NN_LOG_ERROR("Failed to initialize YOLOv8 thread pool");
        delete g_pool;
        return -1;
    }
    
    std::thread reader(read_stream);
    std::thread writer(get_results, 640, 480, 30);
    
    reader.join();
    writer.join();
    
    delete g_pool;
    return 0;
} */



 //完整的代码第一版
/// 引入OpenCV主头文件，用于图像处理
#include <opencv2/opencv.hpp>
// 引入OpenCV的视频跟踪功能头文件
#include <opencv2/video/tracking.hpp>
// 引入Intel RealSense SDK的主头文件，用于与深度相机交互
#include <librealsense2/rs.hpp>
// 引入C++17的文件系统库，用于创建目录等操作
#include <filesystem>
// 引入C++的IO流控制库，用于格式化输出
#include <iomanip>
// 引入C++的字符串流库，用于在内存中构建字符串
#include <sstream>
// 引入C++的多线程库
#include <thread>
// 引入C语言的信号处理库，用于捕获Ctrl+C等信号
#include <csignal>
// 引入C++的map, deque, set, vector, numeric, iostream等标准库
#include <map>
#include <deque>
#include <set>
#include <vector>
#include <numeric>
#include <iostream>
#include <chrono>

// 引入SORT算法的核心组件
#include "utils/KalmanTracker.h"
#include "utils/Hungarian.h"
#include "utils/KalmanFilter3D.h"  // 使用头文件中定义的KalmanFilter3D

// 引入YOLOv8相关的自定义模块
#include "task/yolov8_custom.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"
#include "task/yolov8_thread_pool.h"

// 为 std::filesystem 创建一个简短的命名空间别名 fs
namespace fs = std::filesystem;

// ------------------- 全局变量定义 -------------------
static int g_frame_start_id = 0;
static int g_frame_end_id = 0;
static Yolov8ThreadPool* g_pool = nullptr;
static volatile bool g_end = false;

// 跟踪器、历史记录等相关数据结构
std::map<int, KalmanTracker> trackers;
std::map<int, KalmanFilter3D> trackers_3d;  // 使用头文件中的KalmanFilter3D
std::map<int, std::deque<cv::Point3f>> point3D_history;
std::map<int, std::deque<float>> diameter_history;
std::map<int, std::chrono::steady_clock::time_point> last_seen;
std::map<int, cv::Rect> last_known_boxes;
std::map<int, std::string> tracker_class_names;  // 存储每个跟踪目标的类别名称
std::map<int, float> tracker_confidences;        // 存储每个跟踪目标的最新置信度
const int MAX_HISTORY = 30;
const int MAX_DIAMETER_HISTORY = 1;  // 只使用最新直径值

// ------------------- 信号处理函数 -------------------
void signal_handler(int) {
    g_end = true;
    NN_LOG_INFO("Received Ctrl+C signal, stopping...");
}

// 辅助函数
// cv::Scalar get_track_color(int id) { 
//     cv::RNG rng(id); 
//     return cv::Scalar(rng.uniform(80,255), rng.uniform(80,255), rng.uniform(80,255)); 
// }
cv::Scalar get_track_color(int id) {
    return cv::Scalar(255, 0, 0);  // BGR: 黄色
}


double calculate_iou(const cv::Rect& box1, const cv::Rect& box2) { 
    cv::Rect intersection = box1 & box2; 
    double intersection_area = intersection.area(); 
    double union_area = box1.area() + box2.area() - intersection_area; 
    if (union_area < 1e-6) return 0; 
    return intersection_area / union_area; 
}

// 清理旧跟踪器
void cleanup_old_tracks() {
    const auto now = std::chrono::steady_clock::now();
    std::vector<int> ids_to_remove;
    
    for (const auto& [id, last_time] : last_seen) {
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - last_time).count();
        if (duration > 5) { // 超过5秒未更新则移除
            ids_to_remove.push_back(id);
        }
    }
    
    for (int id : ids_to_remove) {
        trackers.erase(id);
        trackers_3d.erase(id);
        point3D_history.erase(id);
        diameter_history.erase(id);
        last_seen.erase(id);
        last_known_boxes.erase(id);
        tracker_class_names.erase(id); // 清理类别名称
        tracker_confidences.erase(id); // 清理置信度
    }
}

// 3D边缘点计算
cv::Point3f get_3d_edge_point(const cv::Rect& box, bool is_left_edge, const rs2::depth_frame& depth, 
                             const rs2_intrinsics& intrin, int width, int height) {
    int x = is_left_edge ? box.x : box.x + box.width - 1;
    int y = box.y + box.height / 2;
    
    if (x < 0 || x >= width || y < 0 || y >= height) 
        return cv::Point3f(0, 0, 0);
    
    float d = depth.get_distance(x, y);
    if (d <= 0.1f || d >= 10.0f) 
        return cv::Point3f(0, 0, 0);
    
    float point3d[3], pixel[2] = {(float)x, (float)y};
    rs2_deproject_pixel_to_point(point3d, &intrin, pixel, d);
    return cv::Point3f(point3d[0], point3d[1], point3d[2]);
}

// 3D质心计算
cv::Point3f compute_3d_centroid(const cv::Rect& box, const rs2::depth_frame& depth, 
                               const rs2_intrinsics& intrin, int width, int height) {
    int valid_points = 0;
    cv::Point3f sum(0, 0, 0);
    
    for (int y = box.y; y < box.y + box.height; y += 4) {
        for (int x = box.x; x < box.x + box.width; x += 4) {
            if (x < 0 || x >= width || y < 0 || y >= height) 
                continue;
                
            float d = depth.get_distance(x, y);
            if (d >= 0.15f && d <= 8.0f) {
                float point3d[3];
                float pixel[2] = {(float)x, (float)y};
                rs2_deproject_pixel_to_point(point3d, &intrin, pixel, d);
                sum += cv::Point3f(point3d[0], point3d[1], point3d[2]);
                valid_points++;
            }
        }
    }
    
    if (valid_points > 10) {
        return sum * (1.0f / valid_points);
    }
    return cv::Point3f(0, 0, 0);
}

// ------------------- 跟踪与可视化主流程线程函数 (加粗字体+轨迹预测) -------------------
void get_results(int width = 640, int height = 480, int fps = 30) {
    auto start_all = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    static int next_track_id = 0;
    const double iou_threshold = 0.3;
    std::string output_dir = "output";
    fs::create_directories(output_dir);
    HungarianAlgorithm hungarian_solver;

    const float model_w = 640.0f;
    const float model_h = 640.0f;
    float scale = std::min(model_w / (float)width, model_h / (float)height);
    int pad_x = (model_w - (float)width * scale) / 2;
    int pad_y = (model_h - (float)height * scale) / 2;

    // 定义字体参数（全局使用）
    const float font_scale = 0.4;  // 字体大小
    const int font_thickness = 1.5;  // 加粗字体
    const int line_spacing = 10;   // 行间距

    // 速度计算相关变量
    std::map<int, cv::Point3f> last_positions;
    std::map<int, float> current_speeds; // 当前帧计算的速度
    const float fps_float = static_cast<float>(fps); // 帧率用于速度计算
    auto last_time = std::chrono::steady_clock::now();

    while (!g_end) {
        cv::Mat img;
        auto ret_img = g_pool->getTargetImgResult(img, g_frame_end_id);
        if (ret_img != NN_SUCCESS) { 
            if (g_end) break; 
            std::this_thread::sleep_for(std::chrono::milliseconds(5)); 
            continue; 
        }

        std::vector<Detection> dets_from_pool;
        auto ret_det = g_pool->getTargetResult(dets_from_pool, g_frame_end_id);

        std::vector<Detection> restored_dets;
        if (ret_det == NN_SUCCESS) {
            for (const auto& det : dets_from_pool) {
                Detection restored_det = det;
                restored_det.box.x = (det.box.x - pad_x) / scale;
                restored_det.box.y = (det.box.y - pad_y) / scale;
                restored_det.box.width = det.box.width / scale;
                restored_det.box.height = det.box.height / scale;
                if (restored_det.confidence < 0.5) continue;
                restored_dets.push_back(restored_det);
            }
        }

        auto depth = g_pool->GetDepth();
        rs2_intrinsics intrin = g_pool->GetIntrinsics();

        std::vector<int> track_ids;
        std::vector<cv::Rect> predicted_boxes;
        for (auto& [id, tracker] : trackers) {
            cv::Rect2f pred = tracker.predict();
            track_ids.push_back(id);
            predicted_boxes.push_back(pred);
        }

        std::vector<std::vector<double>> cost_matrix(track_ids.size(), std::vector<double>(restored_dets.size(), 1.0));
        for (size_t i = 0; i < track_ids.size(); ++i)
            for (size_t j = 0; j < restored_dets.size(); ++j) {
                double iou = calculate_iou(predicted_boxes[i], restored_dets[j].box);
                if (iou > iou_threshold) cost_matrix[i][j] = 1.0 - iou;
            }
        std::vector<int> assignment;
        if (!track_ids.empty() && !restored_dets.empty())
            hungarian_solver.Solve(cost_matrix, assignment);

        std::set<int> matched_det_indices;
        for (size_t i = 0; i < assignment.size(); ++i) {
            int det_idx = assignment[i];
            if (det_idx >= 0 && cost_matrix[i][det_idx] < 1.0 - iou_threshold) {
                int id = track_ids[i];
                trackers[id].update(restored_dets[det_idx].box);
                
                // 更新类别名称和置信度
                if (det_idx < restored_dets.size()) {
                    tracker_class_names[id] = restored_dets[det_idx].className;
                    tracker_confidences[id] = restored_dets[det_idx].confidence;
                }
                
                cv::Rect smoothed_box = trackers[id].get_state();

                cv::Rect analysis_box = smoothed_box;
                int shrink_x = smoothed_box.width * 0.15;
                int shrink_y = smoothed_box.height * 0.15;
                analysis_box.x += shrink_x;
                analysis_box.y += shrink_y;
                analysis_box.width -= 2 * shrink_x;
                analysis_box.height -= 2 * shrink_y;
                
                cv::Point3f noisy_centroid(0,0,0);
                if (analysis_box.width > 0 && analysis_box.height > 0) {
                    noisy_centroid = compute_3d_centroid(analysis_box, depth, intrin, width, height);
                }

                if (noisy_centroid.x == 0 && noisy_centroid.y == 0 && noisy_centroid.z == 0) {
                    int cx = smoothed_box.x + smoothed_box.width / 2;
                    int cy = smoothed_box.y + smoothed_box.height / 2;
                    if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
                        float d = depth.get_distance(cx, cy);
                        if (d > 0.1f && d < 10.0f) {
                            float point3d[3], pixel[2] = {(float)cx, (float)cy};
                            rs2_deproject_pixel_to_point(point3d, &intrin, pixel, d);
                            noisy_centroid = cv::Point3f(point3d[0], point3d[1], point3d[2]);
                        }
                    }
                }

                if (noisy_centroid.x != 0 || noisy_centroid.y != 0 || noisy_centroid.z != 0) {
                    cv::Point3f smoothed_centroid = trackers_3d[id].update(noisy_centroid);
                    point3D_history[id].push_back(smoothed_centroid);
                    if (point3D_history[id].size() > MAX_HISTORY) 
                        point3D_history[id].pop_front();
                    
                    // 计算速度
                    if (last_positions.find(id) != last_positions.end()) {
                        auto current_time = std::chrono::steady_clock::now();
                        float time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
                            current_time - last_time).count() / 1000.0f;
                        
                        if (time_diff > 0) {
                            cv::Point3f displacement = smoothed_centroid - last_positions[id];
                            float distance = std::sqrt(displacement.x * displacement.x + 
                                                     displacement.y * displacement.y + 
                                                     displacement.z * displacement.z);
                            current_speeds[id] = distance / time_diff; // m/s
                        }
                    }
                    last_positions[id] = smoothed_centroid;
                }
                
                last_seen[id] = std::chrono::steady_clock::now();
                last_known_boxes[id] = restored_dets[det_idx].box;
                matched_det_indices.insert(det_idx);
            }
        }
        
        for (size_t j = 0; j < restored_dets.size(); ++j) {
            if (matched_det_indices.find(j) == matched_det_indices.end()) {
                int new_id = next_track_id++;
                trackers[new_id] = KalmanTracker(restored_dets[j].box);
                last_known_boxes[new_id] = restored_dets[j].box;
                last_seen[new_id] = std::chrono::steady_clock::now();
                
                // 保存类别名称和置信度
                tracker_class_names[new_id] = restored_dets[j].className;
                tracker_confidences[new_id] = restored_dets[j].confidence;
                
                cv::Point3f initial_centroid = compute_3d_centroid(restored_dets[j].box, depth, intrin, width, height);
                if (initial_centroid.x != 0 || initial_centroid.y != 0 || initial_centroid.z != 0) {
                    trackers_3d[new_id].init(initial_centroid);
                    point3D_history[new_id].push_back(initial_centroid);
                    last_positions[new_id] = initial_centroid;
                }
            }
        }

        last_time = std::chrono::steady_clock::now();
        cleanup_old_tracks();

        for (auto const& [id, tracker] : trackers) {
            if (tracker.m_time_since_update > 2 && tracker.m_hit_streak < 3) continue;

            cv::Rect box = tracker.get_state();
            cv::Scalar color = get_track_color(id);
            cv::rectangle(img, box, color, 2);
            
            // 获取类别名称和置信度
            std::string class_name = "object";
            float confidence = 0.0f;
            
            if (tracker_class_names.find(id) != tracker_class_names.end()) {
                class_name = tracker_class_names[id];
            }
            
            if (tracker_confidences.find(id) != tracker_confidences.end()) {
                confidence = tracker_confidences[id];
            }
            
            // 显示主要信息（框上方）- 格式: id:6 ball 0.799
            std::ostringstream main_label;
            main_label << "id:" << id << " " << class_name << " " 
                       << std::fixed << std::setprecision(3) << confidence;
            // cv::putText(img, main_label.str(), 
            //            cv::Point(box.x, box.y - 5), 
            //            cv::FONT_HERSHEY_SIMPLEX, 
            //            font_scale, 
            //            color, 
            //            font_thickness);
            cv::putText(img, main_label.str(), 
                        cv::Point(box.x, box.y - 5), 
                        cv::FONT_HERSHEY_SIMPLEX, 
                        font_scale, 
                        cv::Scalar(255, 255, 255),  // 白色
                        font_thickness);


            
            // 显示3D坐标（框下方第一行）- 格式: xè=0.07 Yè=0.06 Zs0.30 m
            if (point3D_history.count(id) && !point3D_history[id].empty()) {
                const auto& pt = point3D_history[id].back();
                std::ostringstream pos_label;
                pos_label << "    x=" << std::fixed << std::setprecision(2) << pt.x 
                         << " Y=" << std::fixed << std::setprecision(2) << pt.y 
                         << " Z=" << std::fixed << std::setprecision(2) << pt.z << " m";
                // cv::putText(img, pos_label.str(), 
                //            cv::Point(box.x, box.y + box.height + line_spacing), 
                //            cv::FONT_HERSHEY_SIMPLEX, 
                //            font_scale, 
                //            color, 
                //            font_thickness);
                //cv::putText(img, pos_label.str(), ..., cv::Scalar(0, 255, 255), font_thickness);  // 黄色
                cv::putText(img, pos_label.str(), 
                            cv::Point(box.x, box.y + box.height + line_spacing), 
                            cv::FONT_HERSHEY_SIMPLEX, 
                            font_scale, 
                            cv::Scalar(0, 255, 255),  // 浅灰色
                            font_thickness);


            }
            
            // 显示直径（框下方第二行）- 格式: Dia: 38.1 mm
            cv::Point3f left_3d = get_3d_edge_point(box, true, depth, intrin, width, height);
            cv::Point3f right_3d = get_3d_edge_point(box, false, depth, intrin, width, height);
            if (left_3d.x != 0 && right_3d.x != 0) {
                float current_diameter_mm = cv::norm(left_3d - right_3d) * 1000.0f;
                diameter_history[id].push_back(current_diameter_mm);
                if (diameter_history[id].size() > MAX_DIAMETER_HISTORY) 
                    diameter_history[id].pop_front();
                
                if (!diameter_history[id].empty()) {
                    std::ostringstream dia_label;
                    dia_label << "    Dia: " << std::fixed << std::setprecision(1) 
                             << diameter_history[id].back() << " mm";
                    // cv::putText(img, dia_label.str(), 
                    //            cv::Point(box.x, box.y + box.height + line_spacing * 2), 
                    //            cv::FONT_HERSHEY_SIMPLEX, 
                    //            font_scale, 
                    //            color, 
                    //            font_thickness);
                    //cv::putText(img, dia_label.str(), ..., cv::Scalar(0, 0, 255), font_thickness);  // 红色
                    cv::putText(img, dia_label.str(), 
                            cv::Point(box.x, box.y + box.height + line_spacing * 2), 
                            cv::FONT_HERSHEY_SIMPLEX, 
                            font_scale, 
                            cv::Scalar(20, 0, 255),  // 浅灰色
                            font_thickness);


                }
            }

            // 显示速度（框下方第三行）- 格式: Speed: 0.110 m/s
            if (current_speeds.find(id) != current_speeds.end()) {
                std::ostringstream speed_label;
                speed_label << "    Speed: " << std::fixed << std::setprecision(3) 
                           << current_speeds[id] << " m/s";
                // cv::putText(img, speed_label.str(), 
                //            cv::Point(box.x, box.y + box.height + line_spacing * 3), 
                //            cv::FONT_HERSHEY_SIMPLEX, 
                //            font_scale, 
                //            color, 
                //            font_thickness);
                    cv::putText(img, speed_label.str(), 
                                cv::Point(box.x, box.y + box.height + line_spacing * 3), 
                                cv::FONT_HERSHEY_SIMPLEX, 
                                font_scale, 
                                cv::Scalar(200, 200, 200),  // 浅灰色 (BGR)
                                font_thickness);


            }

            // ============== 轨迹预测功能 ==============
            if (trackers_3d.count(id)) {
                // 预测未来轨迹 (20步)
                std::vector<cv::Point3f> future_points = trackers_3d[id].predict_future(40,10);
                std::vector<cv::Point> future_pixels;
                
                // 转换3D点到2D像素坐标
                for (const auto& pt : future_points) {
                    float point3d[3] = {pt.x, pt.y, pt.z};
                    float pixel[2];
                    rs2_project_point_to_pixel(pixel, &intrin, point3d);
                    if (pixel[0] > 0 && pixel[0] < width && pixel[1] > 0 && pixel[1] < height) {
                        future_pixels.emplace_back(static_cast<int>(pixel[0]), static_cast<int>(pixel[1]));
                    }
                }
                
                // 绘制预测轨迹点 (绿色小圆点)
                for (const auto& px : future_pixels) {
                    cv::circle(img, px, 4, cv::Scalar(255, 0, 255), -1);  // 紫色小圆点
                }
                for (size_t i = 1; i < future_pixels.size(); ++i) {
                    cv::line(img, future_pixels[i - 1], future_pixels[i], 
                             cv::Scalar(255, 0, 255), 1, cv::LINE_AA);  // 细粉线
                }
                
                // 在终点添加标记 (绿色十字)
                // if (!future_pixels.empty()) {
                //     cv::drawMarker(img, future_pixels.back(), cv::Scalar(0, 255, 0), 
                //                   cv::MARKER_TILTED_CROSS, 10, 2);
                // }
            }
            
            // 显示历史轨迹（实际运动轨迹）
            if (point3D_history.count(id) && point3D_history[id].size() > 1) {
                const auto& hist = point3D_history[id];
                std::vector<cv::Point> history_pixels;
                
                // 转换历史点到2D像素坐标
                for (const auto& pt : hist) {
                    float point3d[3] = {pt.x, pt.y, pt.z};
                    float pixel[2];
                    rs2_project_point_to_pixel(pixel, &intrin, point3d);
                    if (pixel[0] > 0 && pixel[0] < width && pixel[1] > 0 && pixel[1] < height) {
                        history_pixels.emplace_back(static_cast<int>(pixel[0]), static_cast<int>(pixel[1]));
                    }
                }
                
                // 绘制历史轨迹（实线）
                if (history_pixels.size() > 1) {
                    for (size_t j = 1; j < history_pixels.size(); ++j) {
                        // cv::line(img, history_pixels[j-1], history_pixels[j], 
                        //         color, 2, cv::LINE_AA);
                        cv::line(img, history_pixels[j-1], history_pixels[j], 
                            cv::Scalar(0, 165, 255), 1, cv::LINE_AA);
                   
                    }
                    
                    // 在当前位置绘制目标图标
                    if (!history_pixels.empty()) {
                        //cv::circle(img, history_pixels.back(), 5, color, -1);
                        cv::circle(img, history_pixels.back(), 5, cv::Scalar(0, 165, 255), -1);

                    }
                }
            }
        }

        std::ostringstream filename;
        filename << output_dir << "/" << std::setw(6) << std::setfill('0') << g_frame_end_id << ".jpg";
        cv::imwrite(filename.str(), img);
        g_frame_end_id++;
        frame_count++;
        
        auto end_all = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count();
        if (elapsed > 1000.0f) {
            NN_LOG_INFO("FPS: %.2f", frame_count / (elapsed / 1000.0f));
            frame_count = 0;
            start_all = std::chrono::high_resolution_clock::now();
        }
    }
    g_pool->stopAll();
    NN_LOG_INFO("Result thread exited.");
}

// ------------------- RealSense数据流读取线程函数 -------------------
void read_stream() {
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    try {
        rs2::pipeline_profile profile = pipe.start(cfg);
        rs2::align align_to_color(RS2_STREAM_COLOR);
        
        // 获取深度传感器配置
        auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
        if (depth_sensor.supports(RS2_OPTION_EMITTER_ENABLED)) {
            depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 1.f);
        }
        
        while (!g_end) {
            rs2::frameset frameset;
            if (!pipe.try_wait_for_frames(&frameset, 1000)) continue;
            frameset = align_to_color.process(frameset);
            rs2::frame color_frame = frameset.get_color_frame();
            rs2::depth_frame depth_frame = frameset.get_depth_frame();
            if (!color_frame || !depth_frame) continue;
            
            rs2_intrinsics intrin = color_frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
            cv::Mat img(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
            
            g_pool->SetDepthFrame(depth_frame, intrin);
            auto submit_ret = g_pool->submitTask(img.clone(), g_frame_start_id);
            
            if (submit_ret != NN_SUCCESS) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            } else {
                g_frame_start_id++;
            }
        }
        pipe.stop();
    } catch (const rs2::error & e) {
        NN_LOG_ERROR("RealSense camera could not be started: %s", e.what()); 
        g_end = true;
    }
}

// ------------------- 程序主函数 -------------------
int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <model_file> [num_threads]\n", argv[0]);
        return 1;
    }
    std::string model_file = argv[1];
    const int num_threads = (argc > 2) ? std::stoi(argv[2]) : 4;
    signal(SIGINT, signal_handler);
    g_pool = new Yolov8ThreadPool();
    
    if (g_pool->setUp(model_file, num_threads) != NN_SUCCESS) {
        NN_LOG_ERROR("Failed to initialize YOLOv8 thread pool");
        delete g_pool;
        return -1;
    }
    
    std::thread reader(read_stream);
    std::thread writer(get_results, 640, 480, 30);
    
    reader.join();
    writer.join();
    
    delete g_pool;
    return 0;
} 

 


 