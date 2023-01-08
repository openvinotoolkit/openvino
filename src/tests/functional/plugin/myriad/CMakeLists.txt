# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME myriadFuncTests)

disable_deprecated_warnings()

include(${XLINK_DIR}/XLink.cmake)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${OpenVINO_SOURCE_DIR}/src/tests_deprecated/behavior/vpu/myriad_tests/helpers
            ${XLINK_INCLUDE}
            ${XLINK_PLATFORM_INCLUDE}
        DEPENDENCIES
            openvino_intel_myriad_plugin
        LINK_LIBRARIES
            vpu_graph_transformer
            vpu_common_lib
            funcSharedTests
            mvnc
        ADD_CPPLINT
        DEFINES
            __PC__
        OBJECT_FILES
            ${OpenVINO_SOURCE_DIR}/src/tests_deprecated/behavior/vpu/myriad_tests/helpers/myriad_devices.hpp
            ${OpenVINO_SOURCE_DIR}/src/tests_deprecated/behavior/vpu/myriad_tests/helpers/myriad_devices.cpp
        LABELS
            VPU
            MYRIAD
)

# because vpu_common_lib and vpu_graph_transformer are compiled with LTO
if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0)
    set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
endif()
