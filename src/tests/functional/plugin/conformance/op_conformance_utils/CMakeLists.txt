# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME op_conformance_utils)

include(sources.cmake)

list(APPEND LIBRARIES
        openvino::runtime
        openvino::pugixml
        openvino::util
)

# add op_conformance_utils lib to get API
ov_add_target(
        NAME "${TARGET_NAME}"
        TYPE STATIC
        SOURCES ${OP_CONFORMANCE_UTILS_SOURCES}
        INCLUDES
            PUBLIC
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        LINK_LIBRARIES
            PUBLIC
                ${LIBRARIES}
        DEPENDENCIES
            ${LIBRARIES}
)

ov_build_target_faster(${TARGET_NAME} UNITY)
