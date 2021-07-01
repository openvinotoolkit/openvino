# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(pp)
add_subdirectory(itt)
add_subdirectory(conditional_compilation)

openvino_developer_export_targets(COMPONENT openvino_common TARGETS openvino::pp openvino::itt openvino::conditional_compilation)
