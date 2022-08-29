# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME cpuFuncTests)

# cpuFuncTests is too big for debugging purpose, cpuDebugFuncTests
# is a specific version for debugging purpose, just set DEBUG_SRC_PATH
# to the test case to be debugged and debug using cpuDebugFuncTests
set(DEBUG_TARGET_NAME cpuDebugFuncTests)
set(DEBUG_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/subgraph_tests/src/conv_sum_broadcast.cpp)

add_library(cpuSpecificRtInfo STATIC $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/utils/rt_info/memory_formats_attribute.hpp
                                     $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/utils/rt_info/memory_formats_attribute.cpp)
target_link_libraries(cpuSpecificRtInfo PRIVATE openvino::runtime)

set(INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src)
set(DEPENDENCIES openvino_intel_cpu_plugin)
set(LINK_LIBRARIES funcSharedTests cpuSpecificRtInfo)
if (ENABLE_OV_ONNX_FRONTEND)
    list(APPEND DEFINES TEST_MODELS="${TEST_MODEL_ZOO}")
else()
    set(EXCLUDED_SOURCE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/extension ${CMAKE_CURRENT_SOURCE_DIR}/onnx)
endif()

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES ${INCLUDES}
        EXCLUDED_SOURCE_PATHS ${EXCLUDED_SOURCE_PATHS}
        DEFINES ${DEFINES}
        DEPENDENCIES ${DEPENDENCIES}
        LINK_LIBRARIES ${LINK_LIBRARIES}
        ADD_CPPLINT
        LABELS
            CPU
)


# remove all non-common files from debug
set(EXCLUDED_SOURCE_PATHS_FOR_DEBUG
    ${CMAKE_CURRENT_SOURCE_DIR}/behavior
    ${CMAKE_CURRENT_SOURCE_DIR}/bfloat16
    ${CMAKE_CURRENT_SOURCE_DIR}/blob
    ${CMAKE_CURRENT_SOURCE_DIR}/extension
    ${CMAKE_CURRENT_SOURCE_DIR}/onnx
    ${CMAKE_CURRENT_SOURCE_DIR}/single_layer_tests
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances
    ${CMAKE_CURRENT_SOURCE_DIR}/subgraph_tests/src)

# add the source file to debug
set(OBJECT_FILES_FOR_DEBUG
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/core_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/skip_tests_config.cpp
    ${DEBUG_SRC_PATH})

addIeTargetTest(
        NAME ${DEBUG_TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES ${INCLUDES}
        EXCLUDED_SOURCE_PATHS ${EXCLUDED_SOURCE_PATHS_FOR_DEBUG}
        OBJECT_FILES ${OBJECT_FILES_FOR_DEBUG}
        DEFINES ${DEFINES}
        DEPENDENCIES ${DEPENDENCIES}
        LINK_LIBRARIES ${LINK_LIBRARIES}
        ADD_CPPLINT
        LABELS
            CPU
)

set_ie_threading_interface_for(${TARGET_NAME})
set_ie_threading_interface_for(${DEBUG_TARGET_NAME})
