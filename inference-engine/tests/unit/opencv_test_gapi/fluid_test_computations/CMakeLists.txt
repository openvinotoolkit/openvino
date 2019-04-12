#
# Copyright 2019 Intel Corporation.
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

file(GLOB SRC *.cpp)
file(GLOB HDR *.hpp)

add_library(fluid_test_computations SHARED ${SRC} ${HDR})

target_include_directories(fluid_test_computations PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(fluid_test_computations PRIVATE inference_engine_s
                                              PRIVATE fluid)
