# Copyright (C) 2020 Intel Corporation

find_package(OpenCV) 
if(NOT OpenCV_FOUND)
    message(WARNING ".jar file wasn't found for OpenCV. Java samples won't be build.")
    return()
endif()

if(EXISTS "${OpenCV_INSTALL_PATH}/share/java")
    file(GLOB_RECURSE java_opencv_src ${OpenCV_INSTALL_PATH}/share/java/*.jar)
elseif(EXISTS "${OpenCV_INSTALL_PATH}/bin")
    file(GLOB java_opencv_src ${OpenCV_INSTALL_PATH}/bin/*.jar)
endif()

if(EXISTS "${java_opencv_src}")
    file(GLOB java_samples ${CMAKE_CURRENT_SOURCE_DIR}/samples/*)
    foreach(sample IN LISTS java_samples)
        if(IS_DIRECTORY "${sample}")
            get_filename_component(sample_name "${sample}" NAME)
            file(GLOB_RECURSE sample_src ${sample}/*.java)
            add_jar("${sample_name}_jar"
                    SOURCES ${sample_src}
                            ${PROJECT_SOURCE_DIR}/samples/ArgumentParser.java
                    OUTPUT_NAME ${sample_name} 
                    OUTPUT_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} 
                    INCLUDE_JARS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/* ${java_opencv_src})
        endif()
    endforeach()
endif()
