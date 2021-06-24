# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME vpuUnitTests)

include(${XLINK_DIR}/XLink.cmake)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        ADDITIONAL_SOURCE_DIRS
            # because ngraphFunctions sources need to be compiled with LTO as well
            "${IE_TESTS_ROOT}/ngraph_helpers/ngraph_functions/src"
        INCLUDES
            "${IE_MAIN_SOURCE_DIR}/src/vpu/myriad_plugin"
            "${IE_MAIN_SOURCE_DIR}/thirdparty/movidius"
            "${IE_MAIN_SOURCE_DIR}/thirdparty/movidius/mvnc/include/watchdog"
            "${XLINK_INCLUDE}"
            "${XLINK_PLATFORM_INCLUDE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/base"
            $<TARGET_PROPERTY:ngraphFunctions,INTERFACE_INCLUDE_DIRECTORIES>
        LINK_LIBRARIES
            vpu_graph_transformer_test_static
            unitTestUtils
            mvnc
            ${NGRAPH_LIBRARIES}
            ngraph_backend
            interpreter_backend
            inference_engine_lp_transformations # for ngraphFunctions
        ADD_CPPLINT
        LABELS
            VPU
            MYRIAD
)
