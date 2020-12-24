# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME cpuFuncTests)

add_library(cpuSpecificRtInfo STATIC ${IE_MAIN_SOURCE_DIR}/src/mkldnn_plugin/utils/rt_info/memory_formats_attribute.hpp
                                     ${IE_MAIN_SOURCE_DIR}/src/mkldnn_plugin/utils/rt_info/memory_formats_attribute.cpp)
target_link_libraries(cpuSpecificRtInfo PRIVATE ${NGRAPH_LIBRARIES})

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}
                 ${IE_MAIN_SOURCE_DIR}/src/mkldnn_plugin
        DEPENDENCIES
            MKLDNNPlugin
        LINK_LIBRARIES
            funcSharedTests
            cpuSpecificRtInfo
        ADD_CPPLINT
        LABELS
            CPU
)
