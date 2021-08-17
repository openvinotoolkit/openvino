# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

project(layer_tests)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ DESTINATION tests/layer_tests_openvino COMPONENT tests EXCLUDE_FROM_ALL)
