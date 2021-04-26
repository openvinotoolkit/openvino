# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "format_reader")

file (GLOB MAIN_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

file (GLOB LIBRARY_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        )

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})

# Create library file from sources.
add_library(${TARGET_NAME} SHARED ${MAIN_SRC} ${LIBRARY_HEADERS})

# Find OpenCV components if exist
find_package(OpenCV COMPONENTS core imgproc imgcodecs QUIET)
if(NOT OpenCV_FOUND)
    message(WARNING "OPENCV is disabled or not found, ${TARGET_NAME} will be built without OPENCV support")
else()
    target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCV_LIBRARIES} ie_samples_utils)
    if(UNIX AND NOT APPLE)
        # Workaround issue that rpath-link is missing for PRIVATE dependencies
        # Fixed in cmake 3.16.0 https://gitlab.kitware.com/cmake/cmake/issues/19556
        target_link_libraries(${TARGET_NAME} INTERFACE "-Wl,-rpath-link,${OpenCV_INSTALL_PATH}/lib")
    endif()
    target_compile_definitions(${TARGET_NAME} PRIVATE USE_OPENCV)
endif()

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_FORMAT_READER)

target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
                                                 "${CMAKE_CURRENT_SOURCE_DIR}/..")

set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME}
                                                FOLDER cpp_samples)

if(COMMAND add_clang_format_target)
    add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
endif()