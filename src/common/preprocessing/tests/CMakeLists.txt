# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET fluid_preproc_tests)

ov_disable_deprecated_warnings()

find_package(OpenCV QUIET COMPONENTS gapi core imgproc)
if(NOT OpenCV_FOUND OR NOT OpenCV_VERSION VERSION_GREATER_EQUAL 4)
    message(WARNING "OpenCV ver. 4.0+ is not found, ${TARGET} skipped")
    return()
endif()

add_subdirectory(fluid_test_computations)

file(GLOB SOURCES *.cpp common/*.cpp cpu/*.cpp)
file(GLOB HEADERS *.hpp common/*.hpp cpu/*.hpp)

add_executable(${TARGET} ${SOURCES} ${HEADERS})

target_include_directories(${TARGET} PRIVATE
          "${CMAKE_CURRENT_SOURCE_DIR}/common"
          "${CMAKE_CURRENT_SOURCE_DIR}/cpu"
          $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
          $<TARGET_PROPERTY:openvino_gapi_preproc,INTERFACE_INCLUDE_DIRECTORIES>)

target_compile_definitions(${TARGET} PRIVATE
        $<TARGET_PROPERTY:openvino_gapi_preproc,INTERFACE_COMPILE_DEFINITIONS>)

target_link_libraries(${TARGET} PRIVATE opencv_core opencv_imgproc openvino::util
    inference_engine fluid_test_computations gtest gtest_main)

if(GAPI_TEST_PERF)
    target_compile_definitions(${TARGET} PRIVATE -DPERF_TEST=1)
else()
    target_compile_definitions(${TARGET} PRIVATE -DPERF_TEST=0)
endif()

add_test(NAME ${TARGET} COMMAND ${TARGET})
set_property(TEST ${TARGET} PROPERTY LABELS IE PREPROC)

install(TARGETS ${TARGET}
        RUNTIME DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL)
