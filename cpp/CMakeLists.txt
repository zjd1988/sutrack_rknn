cmake_minimum_required(VERSION 3.10)
project(video_track_rknn CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

find_package(OpenCV REQUIRED)

add_executable(video_track_rknn video_track_rknn.cpp)
target_include_directories(video_track_rknn PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../runtime)
target_link_libraries(video_track_rknn ${OpenCV_LIBS} rknnrt)
