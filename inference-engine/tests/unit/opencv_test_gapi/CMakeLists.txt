# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT ENABLE_GAPI_TESTS)
    message(WARNING "Skipping GAPI unit tests")
    return()
endif()

find_package(OpenCV COMPONENTS gapi)
if(NOT(OpenCV_FOUND))
    message(WARNING "No suitable OpenCV version detected, " ${TARGET_NAME} " skipped")
    return()
endif()

file(GLOB SOURCES *.cpp common/*.cpp cpu/*.cpp)
file(GLOB HEADERS *.hpp common/*.hpp cpu/*.hpp)

set(TARGET opencv_test_gapi)
add_executable(${TARGET} ${SOURCES} ${HEADERS})

target_include_directories(${TARGET}
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/common"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/cpu"
  PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/fluid/modules/gapi/include/")

target_link_libraries(${TARGET} ${OpenCV_LIBS} inference_engine gtest gtest_main)

if(GAPI_TEST_PERF)
  target_compile_definitions(${TARGET} PRIVATE -DPERF_TEST=1)
else()
  target_compile_definitions(${TARGET} PRIVATE -DPERF_TEST=0)
endif()
