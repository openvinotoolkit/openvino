# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_CORE_TESTS_PASS_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/constant_folding.cpp
    ${CMAKE_CURRENT_LIST_DIR}/visualize_tree.cpp
    ${CMAKE_CURRENT_LIST_DIR}/serialization/cleanup.cpp
    ${CMAKE_CURRENT_LIST_DIR}/serialization/const_compression.cpp
    ${CMAKE_CURRENT_LIST_DIR}/serialization/custom_ops.cpp
    ${CMAKE_CURRENT_LIST_DIR}/serialization/deterministicity.cpp
    ${CMAKE_CURRENT_LIST_DIR}/serialization/from_model.cpp
    ${CMAKE_CURRENT_LIST_DIR}/serialization/rt_info_serialization.cpp
    ${CMAKE_CURRENT_LIST_DIR}/serialization/serialize.cpp
    ${CMAKE_CURRENT_LIST_DIR}/serialization/tensor_iterator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/serialization/tensor_names.cpp
)
