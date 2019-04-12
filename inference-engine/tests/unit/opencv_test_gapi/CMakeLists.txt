#
# Copyright (C) 2018-2019 Intel Corporation.
#
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you (End User License Agreement for the Intel(R) Software
# Development Products (Version May 2017)). Unless the License provides
# otherwise, you may not use, modify, copy, publish, distribute, disclose or
# transmit this software or the related documents without Intel's prior
# written permission.
#
# This software and the related documents are provided as is, with no
# express or implied warranties, other than those that are expressly
# stated in the License.
#

if(NOT ENABLE_GAPI_TESTS)
    message(WARNING "Skipping GAPI unit tests")
    return()
endif()

find_package(OpenCV COMPONENTS gapi)
if(NOT OpenCV_FOUND)
    message(WARNING "No suitable OpenCV version detected, " ${TARGET_NAME} " skipped")
    return()
endif()

add_subdirectory(fluid_test_computations)

file(GLOB SOURCES *.cpp common/*.cpp cpu/*.cpp)
file(GLOB HEADERS *.hpp common/*.hpp cpu/*.hpp)

set(TARGET opencv_test_gapi)
add_executable(${TARGET} ${SOURCES} ${HEADERS})

target_include_directories(${TARGET} PRIVATE
          "${CMAKE_CURRENT_SOURCE_DIR}"
          "${CMAKE_CURRENT_SOURCE_DIR}/common"
          "${CMAKE_CURRENT_SOURCE_DIR}/cpu")

target_link_libraries(${TARGET} PRIVATE ${OpenCV_LIBS} inference_engine_s fluid_test_computations gtest gtest_main)

if(GAPI_TEST_PERF)
  target_compile_definitions(${TARGET} PRIVATE -DPERF_TEST=1)
else()
  target_compile_definitions(${TARGET} PRIVATE -DPERF_TEST=0)
endif()
