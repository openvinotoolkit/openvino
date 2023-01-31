# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_cpu_test_helpers)

add_library(mixed_affinity_rt_info STATIC $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/ngraph_transformations/rt_info/mixed_affinity_props.hpp
                                          $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/ngraph_transformations/rt_info/mixed_affinity_props.cpp)
target_link_libraries(mixed_affinity_rt_info PRIVATE openvino::runtime)

set(INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                                         $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src)
set(DEPENDENCIES openvino_intel_cpu_plugin ngraphFunctions)
set(LINK_LIBRARIES ngraphFunctions openvino::runtime::dev mixed_affinity_rt_info)

ov_add_target(
        NAME ${TARGET_NAME}
        TYPE STATIC
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES  ${INCLUDES}
        ADDITIONAL_SOURCE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
        DEPENDENCIES ${DEPENDENCIES}
        LINK_LIBRARIES ${LINK_LIBRARIES}
        ADD_CPPLINT
)

ie_faster_build(${TARGET_NAME} UNITY)
