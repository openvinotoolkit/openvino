# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

project(OpenVINO_JS_API)

ov_option(ENABLE_NODEJS "Enables OpenVINO Node.js API build" OFF)

if(ENABLE_JS)
  add_subdirectory(node)
endif()
