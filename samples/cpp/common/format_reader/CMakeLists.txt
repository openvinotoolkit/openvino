# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "format_reader")

file (GLOB MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file (GLOB LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)
file (GLOB LIBRARY_PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${LIBRARY_SRC} ${LIBRARY_HEADERS})
source_group("include" FILES ${LIBRARY_PUBLIC_HEADERS})

# Create library file from sources.
add_library(${TARGET_NAME} STATIC ${MAIN_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS})

# Find OpenCV components if exist
find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)
if(NOT OpenCV_FOUND OR NOT OpenCV_VERSION VERSION_GREATER_EQUAL 3)
    message(WARNING "OpenCV ver. 3.0+ is not found, ${TARGET_NAME} will be built without OpenCV support")
else()
    target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCV_LIBRARIES} ie_samples_utils)
    if(UNIX AND NOT APPLE)
        # Workaround issue that rpath-link is missing for PRIVATE dependencies
        # Fixed in cmake 3.16.0 https://gitlab.kitware.com/cmake/cmake/issues/19556
        target_link_libraries(${TARGET_NAME} INTERFACE "-Wl,-rpath-link,${OpenCV_INSTALL_PATH}/lib")
    endif()
    target_compile_definitions(${TARGET_NAME} PRIVATE USE_OPENCV)
endif()

target_include_directories(${TARGET_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                          PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

set_target_properties(${TARGET_NAME} PROPERTIES FOLDER cpp_samples)

if(COMMAND ov_add_clang_format_target)
    ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
endif()
