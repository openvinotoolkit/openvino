# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ie_add_compiler_flags(/wd4267)
endif()

ov_add_frontend(NAME tensorflow_lite
        LINKABLE_FRONTEND
        FILEDESCRIPTION "FrontEnd to load and convert TensorFlow Lite file format"
        LINK_LIBRARIES openvino::core::dev openvino::frontend::tensorflow_common)
