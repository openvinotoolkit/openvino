# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(itt)
add_subdirectory(conditional_compilation)
add_subdirectory(util)

add_subdirectory(snippets)
add_subdirectory(inference_engine_legacy_api)
add_subdirectory(preprocessing)
add_subdirectory(transformations)
add_subdirectory(low_precision_transformations)

add_dependencies(ov_runtime_libraries inference_engine_legacy
                                      inference_engine_transformations
                                      inference_engine_lp_transformations
                                      inference_engine_preproc
                                      inference_engine_preproc
                                      inference_engine_snippets)

openvino_developer_export_targets(COMPONENT openvino_common TARGETS openvino::itt openvino::conditional_compilation)
