# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(itt)
add_subdirectory(conditional_compilation)
add_subdirectory(util)

if(CMAKE_COMPILER_IS_GNUCXX)
    ie_add_compiler_flags(-Wall)
    ie_add_compiler_flags(-Wmissing-declarations)
endif()

add_subdirectory(snippets)
add_subdirectory(legacy_api)
add_subdirectory(preprocessing)

add_dependencies(ov_runtime_libraries inference_engine_legacy
                                      openvino::preprocessing
                                      openvino::snippets)

openvino_developer_export_targets(COMPONENT openvino_common TARGETS openvino::itt openvino::conditional_compilation openvino::util)
