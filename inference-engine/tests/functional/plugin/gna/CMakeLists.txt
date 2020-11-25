# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME gnaFuncTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            GNAPlugin
        LINK_LIBRARIES
            funcSharedTests
        ADD_CPPLINT
        LABELS
            GNA
)
target_compile_definitions(${TARGET_NAME} PRIVATE BINARY_EXPORT_MODELS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/backward_compatibility/models/")
