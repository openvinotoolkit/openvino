cmake_minimum_required(VERSION 3.13)

project(openvino_tssn_extension)

find_package(OpenVINO REQUIRED)

set(CMAKE_CXX_STANDARD 14)

add_library(openvino_tssn_extension SHARED 
    tssn_extension.cpp 
    tssn_op.cpp 
    composite_tssn.cpp 
    btl/btl_function_library.cpp
)

target_include_directories(openvino_tssn_extension PRIVATE "${CMAKE_SOURCE_DIR}/../../thirdparty/json/nlohmann_json/single_include")
target_link_libraries(openvino_tssn_extension PRIVATE openvino::runtime)

# Enable AVX512 if supported, otherwise AVX2
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(openvino_tssn_extension PRIVATE -mavx2)
elseif(MSVC)
    target_compile_options(openvino_tssn_extension PRIVATE /arch:AVX2)
endif()

add_executable(apply_incision_tool apply_incision_tool.cpp composite_tssn.cpp btl/btl_function_library.cpp)
target_link_libraries(apply_incision_tool PRIVATE openvino::runtime)
target_include_directories(apply_incision_tool PRIVATE "${CMAKE_SOURCE_DIR}/../../thirdparty/json/nlohmann_json/single_include")
if(MSVC)
    target_compile_options(apply_incision_tool PRIVATE /arch:AVX2)
endif()
