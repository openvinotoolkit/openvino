# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(${CMAKE_CURRENT_SOURCE_DIR}/sources.cmake)

ov_add_frontend(NAME tensorflow_lite
                LINKABLE_FRONTEND
                FILEDESCRIPTION "FrontEnd to load and convert TensorFlow Lite file format"
                LINK_LIBRARIES openvino::core::dev openvino::frontend::tensorflow_common
                SOURCES ${TENSORFLOW_LITE_FRONTEND_SRCS}
                HEADERS ${TENSORFLOW_LITE_FRONTEND_HEADERS}
                PUBLIC_HEADERS ${TENSORFLOW_LITE_FRONTEND_PUBLIC_HEADERS})

ov_build_target_faster(openvino_tensorflow_lite_frontend PCH)
