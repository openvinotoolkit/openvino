# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME op_conformance_utils)

list(APPEND LIBRARIES
        openvino::runtime
        openvino::pugixml
        openvino::util
)

# add op_conformance_utils lib to get API
ov_add_target(
        NAME "${TARGET_NAME}"
        TYPE STATIC
        ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src"
        INCLUDES
            PUBLIC
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        LINK_LIBRARIES
            PUBLIC
                ${LIBRARIES}
        DEPENDENCIES
            ${LIBRARIES}
        ADD_CPPLINT
)

ov_build_target_faster(${TARGET_NAME} UNITY)
