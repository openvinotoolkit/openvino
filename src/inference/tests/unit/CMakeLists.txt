# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_inference_unit_tests)

if(SUGGEST_OVERRIDE_SUPPORTED)
    ov_add_compiler_flags(-Wno-suggest-override)
endif()

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            openvino_template_extension
        LINK_LIBRARIES
            unit_test_utils
        ADD_CLANG_FORMAT
        LABELS
            OV UNIT RUNTIME
)

# fix LTO add_symbol_to_partition_1 issue
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

ov_set_threading_interface_for(${TARGET_NAME})
