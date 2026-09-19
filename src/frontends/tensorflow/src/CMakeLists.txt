# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(${CMAKE_CURRENT_SOURCE_DIR}/sources.cmake)

ov_add_frontend(NAME tensorflow
                PROTOBUF_REQUIRED
                LINKABLE_FRONTEND
                FILEDESCRIPTION "FrontEnd to load and convert TensorFlow file format"
                LINK_LIBRARIES openvino::core::dev openvino::frontend::tensorflow_common
                SOURCES ${TENSORFLOW_FRONTEND_SRCS}
                HEADERS ${TENSORFLOW_FRONTEND_HEADERS}
                PUBLIC_HEADERS ${TENSORFLOW_FRONTEND_PUBLIC_HEADERS})

if(ENABLE_SNAPPY_COMPRESSION)
    target_link_libraries(openvino_tensorflow_frontend PRIVATE openvino::snappy)
    target_compile_definitions(openvino_tensorflow_frontend PRIVATE ENABLE_SNAPPY_COMPRESSION)
endif()

ov_build_target_faster(openvino_tensorflow_frontend PCH)
