# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME cpuFuncTests)

add_library(cpuSpecificRtInfo STATIC ${IE_MAIN_SOURCE_DIR}/src/mkldnn_plugin/utils/rt_info/memory_formats_attribute.hpp
                                     ${IE_MAIN_SOURCE_DIR}/src/mkldnn_plugin/utils/rt_info/memory_formats_attribute.cpp)
target_link_libraries(cpuSpecificRtInfo PRIVATE ${NGRAPH_LIBRARIES})

set(INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} ${IE_MAIN_SOURCE_DIR}/src/mkldnn_plugin)
set(DEPENDENCIES MKLDNNPlugin AutoPlugin)
set(LINK_LIBRARIES funcSharedTests cpuSpecificRtInfo)
if (NGRAPH_ONNX_IMPORT_ENABLE AND NOT NGRAPH_USE_PROTOBUF_LITE)
    list(APPEND INCLUDES "${OpenVINO_MAIN_SOURCE_DIR}/docs/onnx_custom_op")
    list(APPEND LINK_LIBRARIES onnx_custom_op)
    list(APPEND DEPENDENCIES template_extension onnx_custom_op)
else()
    set(EXCLUDED_SOURCE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/extension")
endif()

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES ${INCLUDES}
        EXCLUDED_SOURCE_PATHS ${EXCLUDED_SOURCE_PATHS}
        DEPENDENCIES ${DEPENDENCIES}
        LINK_LIBRARIES ${LINK_LIBRARIES}
        ADD_CPPLINT
        LABELS
            CPU
)
