# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_inference_unit_tests)

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            template_extension
        LINK_LIBRARIES
            unitTestUtils
        INCLUDES
        ADD_CLANG_FORMAT
        LABELS
            OV
)

check_cxx_compiler_flag(-Wno-suggest-override SUPPORT_WNO_SUGGEST_OVERRIDE)
if (SUPPORT_WNO_SUGGEST_OVERRIDE)
    ie_add_compiler_flags(-Wno-suggest-override)
endif()