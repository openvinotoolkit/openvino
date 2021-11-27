# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(itt)
add_subdirectory(conditional_compilation)
add_subdirectory(util)

add_subdirectory(inference_engine_legacy_api)

openvino_developer_export_targets(COMPONENT openvino_common TARGETS openvino::itt openvino::conditional_compilation)
