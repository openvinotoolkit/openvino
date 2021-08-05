 # Copyright (C) 2018-2020 Intel Corporation
 # SPDX-License-Identifier: Apache-2.0
 #
 #set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
 #find_package(SYCL REQUIRED)

 #set(CMAKE_CXX_COMPILER ${SYCL_COMPILER})
 #set(CMAKE_CXX_STANDARD 17)

 #intel_config_flag_apply_settings(
 #CompilerOptions
 #CMAKE_CXX_FLAGS
 #ALL_PATTERN ""
 #SET_RAW
 #-stdlib=libstdc++
 #)

 #file (GLOB MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
 #file (GLOB MAIN_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

 file(GLOB MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/main_sycl.cpp ${CMAKE_CURRENT_SOURCE_DIR}/sycl_ext.cpp)
 file(GLOB MAIN_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/sycl_ext.hpp)

 ie_add_sample(NAME sycl_custom_layers
               SOURCES ${MAIN_SRC}
               HEADERS ${MAIN_HEADERS}
               DEPENDENCIES format_reader ie_samples_utils)

 find_package(ngraph REQUIRED)
 find_package(OpenCL)
 target_link_libraries(sycl_custom_layers PRIVATE ${NGRAPH_LIBRARIES} OpenCL)
