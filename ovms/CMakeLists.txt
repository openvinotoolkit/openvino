cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenVINO REQUIRED)

add_executable(test_ov src/ov.cpp)

target_link_libraries(test_ov PRIVATE openvino::runtime)