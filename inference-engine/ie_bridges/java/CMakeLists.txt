# Copyright (C) 2020 Intel Corporation

find_package(Java REQUIRED)
include(UseJava)

set(JAVA_AWT_INCLUDE_PATH NotNeeded)

project(inference_engine_java_api)

set(InferenceEngine_LIBRARIES inference_engine)

find_package(JNI REQUIRED)

file(GLOB_RECURSE sources cpp/*.cpp)
add_library(${PROJECT_NAME} SHARED ${sources})

file(GLOB_RECURSE java_source org/intel/openvino/*.java)
add_jar(inference_engine_jar ${java_source} OUTPUT_NAME ${PROJECT_NAME} OUTPUT_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

target_link_libraries(${PROJECT_NAME} PRIVATE ${InferenceEngine_LIBRARIES})
target_include_directories(${PROJECT_NAME} PRIVATE ${JNI_INCLUDE_DIRS})

if(ENABLE_TESTS)
    file(DOWNLOAD https://search.maven.org/remotecontent?filepath=junit/junit/4.13/junit-4.13.jar ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/junit-4.13.jar)
    file(DOWNLOAD https://search.maven.org/remotecontent?filepath=org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/hamcrest-core-1.3.jar)

    file(GLOB_RECURSE java_tests tests/*.java)
    add_jar(java_tests_jar ${java_tests} OUTPUT_NAME openvino_tests OUTPUT_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} INCLUDE_JARS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/*)
endif()
