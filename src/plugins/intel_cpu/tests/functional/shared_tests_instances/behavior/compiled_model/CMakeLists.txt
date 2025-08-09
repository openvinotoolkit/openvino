# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(compiled_model_SRC
    core_integration.cpp
    cpu_reservation_test.cpp
    import_export.cpp
    properties.cpp
)

# Convert to full paths
foreach(src_file ${compiled_model_SRC})
    list(APPEND compiled_model_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${compiled_model_SRC_FULL})
