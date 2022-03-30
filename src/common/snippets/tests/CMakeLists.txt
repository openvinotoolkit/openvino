# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_snippets_func_tests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            $<TARGET_PROPERTY:inference_engine_snippets,INTERFACE_INCLUDE_DIRECTORIES>
        LINK_LIBRARIES
            openvino::runtime::dev
            commonTestUtils
            snippetsNgraphFunctions
        ADD_CPPLINT
        LABELS
        IE SNIPPETS
)

ie_faster_build(${TARGET_NAME}
    UNITY
    PCH PRIVATE "src/precomp.hpp"
)