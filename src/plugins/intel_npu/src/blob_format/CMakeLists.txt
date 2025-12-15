cmake_minimum_required(VERSION 3.12)

project("blob_format")

# find_package(OpenVINO REQUIRED)

set(TARGET_NAME "blob_format")

add_executable(${TARGET_NAME} main.cpp)
target_compile_features(${TARGET_NAME}  PRIVATE cxx_std_20)

# target_link_libraries(${TARGET_NAME}
#     PRIVATE openvino::runtime)