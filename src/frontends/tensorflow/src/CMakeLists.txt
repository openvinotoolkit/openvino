# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_frontend(NAME tensorflow
                LINKABLE_FRONTEND
                FILEDESCRIPTION "FrontEnd to load and convert TensorFlow file format"
                LINK_LIBRARIES openvino::util openvino::runtime::dev)

# add object library used in tests for private transformations

add_library(openvino_tensorflow_frontend_static_tests STATIC EXCLUDE_FROM_ALL
    helper_transforms/block_lstm_replacer.cpp
    helper_transforms/embedding_segments_feature_fusing.cpp
    helper_transforms/gru_block_cell_replacer.cpp
    pass/transpose_sinking.cpp
    exception.cpp
    openvino_conversions.cpp
    utils.cpp)

target_include_directories(openvino_tensorflow_frontend_static_tests
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:openvino::frontend::tensorflow,INTERFACE_INCLUDE_DIRECTORIES>)

add_library(openvino_tensorflow_frontend_common_logic STATIC EXCLUDE_FROM_ALL
        exception.cpp
        input_model.hpp
        input_model.cpp
        pass/transpose_sinking.hpp
        pass/transpose_sinking.cpp
        place.hpp
        place.cpp
        op_table.hpp
        op_table.cpp
        openvino_conversions.hpp
        openvino_conversions.cpp
        utils.hpp
        utils.cpp)

target_include_directories(openvino_tensorflow_frontend_common_logic
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        $<TARGET_PROPERTY:openvino::frontend::tensorflow,INTERFACE_INCLUDE_DIRECTORIES>
        PRIVATE
        $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>)
