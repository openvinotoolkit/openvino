# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(${CMAKE_CURRENT_SOURCE_DIR}/sources.cmake)

ov_add_frontend(NAME paddle
                LINKABLE_FRONTEND
                PROTOBUF_REQUIRED
                PROTOBUF_LITE
                FILEDESCRIPTION "FrontEnd to load and convert PaddlePaddle file format"
                LINK_LIBRARIES openvino::util openvino::core::dev
                SOURCES ${PADDLE_FRONTEND_SRCS}
                HEADERS ${PADDLE_FRONTEND_HEADERS}
                PUBLIC_HEADERS ${PADDLE_FRONTEND_PUBLIC_HEADERS})

ov_build_target_faster(openvino_paddle_frontend PCH)

if(ENABLE_TESTS)
   add_subdirectory(tests)
endif()
