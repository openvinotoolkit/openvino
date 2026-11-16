cmake_minimum_required(VERSION 3.16)

set(TARGET_NAME Gemma4POC)

project(${TARGET_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT TARGET OpenCL::OpenCL)
    find_package(OpenCL REQUIRED)
endif()

add_library(${TARGET_NAME} SHARED
    megakernelImpl.cpp
    impl/gemma4POCRuntime.cpp)
add_library(megakernel::${TARGET_NAME} ALIAS ${TARGET_NAME})

include(GenerateExportHeader)
generate_export_header(${TARGET_NAME}
    EXPORT_FILE_NAME exportApi.h
    EXPORT_MACRO_NAME EXPORT_API)

target_compile_definitions(${TARGET_NAME} PRIVATE CL_TARGET_OPENCL_VERSION=300)

target_include_directories(${TARGET_NAME}
    PUBLIC
        "${CMAKE_CURRENT_LIST_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}"
    PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/impl")

target_link_libraries(${TARGET_NAME} PUBLIC OpenCL::OpenCL)

set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
