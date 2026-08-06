# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Unit tests: graph rewriter (no GPU) and device coordinator (needs 2 GPUs).
add_subdirectory(unit)

# Accuracy of the whole pipeline against a single GPU (needs 2 GPUs).
add_subdirectory(functional)

set(TARGET_NAME tp_test)

add_executable(${TARGET_NAME} tp_test.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)

set_target_properties(${TARGET_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${OpenVINO_SOURCE_DIR}/bin/intel64/$<CONFIG>")

