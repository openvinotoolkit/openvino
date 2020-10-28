# Copyright (C) 2020 Intel Corporation

file(DOWNLOAD https://search.maven.org/remotecontent?filepath=junit/junit/4.13/junit-4.13.jar 
                ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/junit-4.13.jar)
file(DOWNLOAD https://search.maven.org/remotecontent?filepath=org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar 
                ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/hamcrest-core-1.3.jar)

file(GLOB_RECURSE java_tests_src ${CMAKE_CURRENT_SOURCE_DIR}/*.java)
add_jar(ie_java_api_tests_jar
        SOURCES ${java_tests_src} ${PROJECT_SOURCE_DIR}/samples/ArgumentParser.java
        OUTPUT_NAME ie_java_api_tests 
        OUTPUT_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} 
        INCLUDE_JARS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/*)

add_dependencies(ie_java_api_tests_jar inference_engine_jar)
