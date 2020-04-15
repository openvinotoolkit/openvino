# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME unitTestUtils)

list(APPEND EXPORT_DEPENDENCIES
        commonTestUtils_s
        inference_engine_s
        inference_engine_preproc_s
        inference_engine_lp_transformations
        inference_engine_ir_readers
        gmock)

addIeTarget(
        NAME ${TARGET_NAME}
        TYPE STATIC
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        ADD_CPPLINT
        DEVELOPER_PACKAGE
        EXPORT_DEPENDENCIES
            ${EXPORT_DEPENDENCIES}
)

target_link_libraries(${TARGET_NAME}
        PUBLIC
        ${EXPORT_DEPENDENCIES})