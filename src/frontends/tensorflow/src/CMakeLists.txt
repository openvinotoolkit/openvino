# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT CPACK_GENERATOR STREQUAL "DEB")
    set(linkable_frontend LINKABLE_FRONTEND)
endif()

ov_add_frontend(NAME tensorflow
                ${linkable_frontend}
                FILEDESCRIPTION "FrontEnd to load and convert TensorFlow file format"
                LINK_LIBRARIES openvino::util openvino::runtime::dev)

# give a different name during installation to OpenVINO package
set(target_name "${FRONTEND_NAME_PREFIX}tensorflow${FRONTEND_NAME_SUFFIX}")
set_target_properties(${target_name} PROPERTIES OUTPUT_NAME openvino_tensorflow_fe)

if(NOT TARGET openvino::frontend::tensorflow)
    add_library(openvino::frontend::tensorflow ALIAS ${target_name})
endif()

# add object library used in tests for private transformations

add_library(openvino_tensorflow_frontend_static_tests STATIC EXCLUDE_FROM_ALL
    helper_transforms/block_lstm_replacer.cpp
    helper_transforms/embedding_segments_feature_fusing.cpp
    pass/transpose_sinking.cpp
    openvino_conversions.cpp
    utils.cpp)

target_include_directories(openvino_tensorflow_frontend_static_tests
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:openvino::frontend::tensorflow,INTERFACE_INCLUDE_DIRECTORIES>)
