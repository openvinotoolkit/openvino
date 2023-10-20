# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_frontend(NAME tensorflow
                LINKABLE_FRONTEND
                FILEDESCRIPTION "FrontEnd to load and convert TensorFlow file format"
                LINK_LIBRARIES openvino::core::dev openvino::frontend::tensorflow_common)

target_include_directories(openvino_tensorflow_frontend PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/ov_tensorflow")

if(ENABLE_SNAPPY_COMPRESSION)
    target_link_libraries(openvino_tensorflow_frontend PRIVATE openvino::snappy)
    target_compile_definitions(openvino_tensorflow_frontend PRIVATE ENABLE_SNAPPY_COMPRESSION)
endif()
