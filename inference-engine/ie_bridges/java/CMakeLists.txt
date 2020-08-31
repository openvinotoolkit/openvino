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
add_jar(inference_engine_jar ${java_source} 
        OUTPUT_NAME ${PROJECT_NAME} 
        OUTPUT_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

target_link_libraries(${PROJECT_NAME} PRIVATE inference_engine)
target_include_directories(${PROJECT_NAME} PRIVATE ${JNI_INCLUDE_DIRS})

if(ENABLE_TESTS)
    file(DOWNLOAD https://search.maven.org/remotecontent?filepath=junit/junit/4.13/junit-4.13.jar 
                  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/junit-4.13.jar)
    file(DOWNLOAD https://search.maven.org/remotecontent?filepath=org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar 
                  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/hamcrest-core-1.3.jar)

    file(GLOB_RECURSE java_tests_src tests/*.java)
    add_jar(ie_java_api_tests_jar
            SOURCES ${java_tests_src} samples/ArgumentParser.java
            OUTPUT_NAME ie_java_api_tests 
            OUTPUT_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} 
            INCLUDE_JARS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/*)
endif()

if(ENABLE_SAMPLES) 
    find_package(OpenCV) 
    if(OpenCV_FOUND)
        if(EXISTS "${OpenCV_INSTALL_PATH}/share/java")
            file(GLOB_RECURSE java_opencv_src ${OpenCV_INSTALL_PATH}/share/java/*.jar)
        elseif(EXISTS "${OpenCV_INSTALL_PATH}/bin")
            file(GLOB java_opencv_src ${OpenCV_INSTALL_PATH}/bin/*.jar)
        endif()
        if(EXISTS "${java_opencv_src}")
            file(GLOB java_samples samples/*)
            foreach(sample ${java_samples})
                if(IS_DIRECTORY ${sample})
                    get_filename_component(sample_name "${sample}" NAME)
                    file(GLOB_RECURSE sample_src ${sample}/*.java)
                    add_jar("${sample_name}_jar"
                            SOURCES ${sample_src} samples/ArgumentParser.java 
                            OUTPUT_NAME ${sample_name} 
                            OUTPUT_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} 
                            INCLUDE_JARS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/* ${java_opencv_src})
                endif()
            endforeach()
        else()
            message(WARNING ".jar file wasn't found for OpenCV. Java samples won't be build.")
        endif()
    endif()
endif()
