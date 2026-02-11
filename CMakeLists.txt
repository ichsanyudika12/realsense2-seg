cmake_minimum_required(VERSION 3.1)
project(realsense_cam)

find_package(OpenCV 4.11 REQUIRED)
find_package(realsense2 REQUIRED)

add_executable(main main/main.cpp)
target_link_libraries(main ${OpenCV_LIBS} realsense2)
