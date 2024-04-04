cmake_minimum_required(VERSION 3.0)
project(BlinkDetection)

find_package(OpenCV REQUIRED)

# Add executable
add_executable(BlinkDetection main.cpp)

target_link_libraries(BlinkDetection ${OpenCV_LIBS})
