# Copyright (c) 2018 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required(VERSION 2.8)

set (TARGET_NAME "calibration_tool")

file (GLOB MAIN_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../validation_app/pugixml/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../validation_app/ClassificationProcessor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../validation_app/classification_set_generator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../validation_app/image_decoder.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../validation_app/ObjectDetectionProcessor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../validation_app/Processor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../validation_app/VOCAnnotationParser.cpp
        )

file (GLOB MAIN_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pugixml/*.hpp
        )

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${MAIN_SRC})
source_group("include" FILES ${MAIN_HEADERS})

# opencv include folders
find_package(OpenCV QUIET COMPONENTS core imgproc highgui imgcodecs)
if(NOT(OpenCV_FOUND))
    find_package(OpenCV QUIET COMPONENTS world)
    if(NOT(OpenCV_FOUND))
        message(WARNING "No suitable OpenCV version detected, " ${TARGET_NAME} " skipped")
        return()
    endif()
endif()

# Properties->C/C++->General->Additional Include Directories
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/../classification_sample/core
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/os/windows
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/../validation_app)

link_directories(${LIB_FOLDER})

# Create library file from sources.
add_executable(${TARGET_NAME} ${MAIN_SRC} ${MAIN_HEADERS})

set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE" 
COMPILE_PDB_NAME ${TARGET_NAME})
target_link_libraries(${TARGET_NAME} gflags IE::ie_cpu_extension ${InferenceEngine_LIBRARIES} ${OpenCV_LIBRARIES})
if (UNIX)
    target_link_libraries(${TARGET_NAME} dl)
endif()

