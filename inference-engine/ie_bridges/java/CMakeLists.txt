# Copyright (C) 2020 Intel Corporation

project(inference_engine_java_api)

# Find Java

find_package(Java REQUIRED)
include(UseJava)

set(JAVA_AWT_INCLUDE_PATH NotNeeded)

find_package(JNI REQUIRED)

# Build native part

file(GLOB_RECURSE sources ${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cpp)

add_library(${PROJECT_NAME} SHARED ${sources})

target_link_libraries(${PROJECT_NAME} PRIVATE inference_engine)
target_include_directories(${PROJECT_NAME} PRIVATE ${JNI_INCLUDE_DIRS})

# Build java part

file(GLOB_RECURSE java_source ${CMAKE_CURRENT_SOURCE_DIR}/org/intel/openvino/*.java)
add_jar(inference_engine_jar ${java_source}
        OUTPUT_NAME ${PROJECT_NAME} 
        OUTPUT_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

if(ENABLE_SAMPLES) 
    add_subdirectory(samples)
endif()
