# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME cpuFuncTests)

add_library(cpuSpecificRtInfo STATIC ${IE_MAIN_SOURCE_DIR}/src/mkldnn_plugin/utils/rt_info/memory_formats_attribute.hpp
                                     ${IE_MAIN_SOURCE_DIR}/src/mkldnn_plugin/utils/rt_info/memory_formats_attribute.cpp)
target_link_libraries(cpuSpecificRtInfo PRIVATE ngraph)

set(INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} $<TARGET_PROPERTY:MKLDNNPlugin,SOURCE_DIR>)
set(DEPENDENCIES MKLDNNPlugin)
set(LINK_LIBRARIES funcSharedTests cpuSpecificRtInfo)
if (ENABLE_OV_ONNX_FRONTEND)
    list(APPEND INCLUDES "${OpenVINO_SOURCE_DIR}/docs/onnx_custom_op")
    list(APPEND LINK_LIBRARIES onnx_custom_op)
    list(APPEND DEPENDENCIES template_extension onnx_custom_op)
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

set_ie_threading_interface_for(${TARGET_NAME})
