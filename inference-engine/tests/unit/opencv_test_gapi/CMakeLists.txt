# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT ENABLE_GAPI_TESTS OR WIN32)
    message(WARNING "Skipping GAPI unit tests")
    return()
endif()

find_package(OpenCV COMPONENTS gapi)
if(NOT OpenCV_FOUND)
    message(WARNING "No suitable OpenCV version detected, " ${TARGET_NAME} " skipped")
    return()
endif()

include_directories(${IE_MAIN_SOURCE_DIR}/src/inference_engine)

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
