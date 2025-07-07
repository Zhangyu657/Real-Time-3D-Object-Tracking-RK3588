# 设置最低版本号
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

# 设置项目名称
project(rk3588-demo VERSION 0.0.1 LANGUAGES CXX)

# 输出系统信息
message(STATUS "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")

# 设置编译器标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置库架构和设备名
set(LIB_ARCH "aarch64")
set(DEVICE_NAME "RK3588")

# rknn_api 文件夹路径和头/库路径
set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/librknn_api)
set(RKNN_API_INCLUDE_PATH ${RKNN_API_PATH}/include)
set(RKNN_API_LIB_PATH ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)

# 第三方路径
set(3RDPARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
set(RGA_DIR ${3RDPARTY_PATH}/rga/${DEVICE_NAME})
set(RGA_LIB ${RGA_DIR}/lib/Linux/${LIB_ARCH}/librga.so)

# RealSense SDK 路径（动态库已确认在 /usr/local/lib）
set(REALSENSE2_INCLUDE_DIR /usr/local/include)
set(REALSENSE2_LIBRARY_DIR /usr/local/lib)

# 查找 RealSense SDK
find_package(realsense2 REQUIRED)

# 查找 OpenCV
# set(OpenCV_DIR ${3RDPARTY_PATH}/opencv/opencv-linux-${LIB_ARCH}/share/OpenCV)
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")

# 头文件搜索路径
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${RKNN_API_INCLUDE_PATH}
    ${RGA_DIR}/include
    ${REALSENSE2_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 可选：动态库搜索路径（不加也能正常链接）
link_directories(
    ${REALSENSE2_LIBRARY_DIR}
)

# 构建预处理和后处理库
add_library(nn_process SHARED
    src/process/preprocess.cpp
    src/process/postprocess.cpp
)
target_link_libraries(nn_process
    ${OpenCV_LIBS}
    ${RGA_LIB}
)

# 封装 RKNN 推理引擎
add_library(rknn_engine SHARED src/engine/rknn_engine.cpp)
target_link_libraries(rknn_engine
    ${RKNN_API_LIB_PATH}
)

# yolov8 封装逻辑
add_library(yolov8_lib SHARED src/task/yolov8_custom.cpp)
target_link_libraries(yolov8_lib
    rknn_engine
    nn_process
)

# OpenCV 绘图模块
add_library(draw_lib SHARED src/draw/cv_draw.cpp)
target_link_libraries(draw_lib
    ${OpenCV_LIBS}
)





# yolov8 多线程推理 + RealSense 实时输入（核心程序）
add_executable(yolov8_thread_pool
    src/yolov8_thread_pool.cpp
    src/task/yolov8_thread_pool.cpp
    src/utils/Hungarian.cpp
    src/utils/KalmanTracker.cpp
    src/utils/KalmanFilter3D.cpp # <<<--- 在这里加上新创建的源文件

)
target_link_libraries(yolov8_thread_pool
    draw_lib
    yolov8_lib
    realsense2
)
