# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_snippets_models)

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fake_quantize_helper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/function_helper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/precision_propagation_convertion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/precision_propagation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/snippets_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/subgraph_convert.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/subgraph_customizable.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/subgraph_fq.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/subgraph_gated_mlp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/subgraph_gelu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/subgraph_group_normalization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/subgraph_lowered.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/subgraph_matmul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/subgraph_mha.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/subgraph_mlp_seq.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/subgraph_reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/subgraph_roll_matmul_roll.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/subgraph_simple.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/subgraph_softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/subgraph_transpose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/two_binary_ops.cpp
)

set(PUBLIC_HEADERS
    ${PUBLIC_HEADERS_DIR}/fake_quantize_helper.hpp
    ${PUBLIC_HEADERS_DIR}/function_helper.hpp
    ${PUBLIC_HEADERS_DIR}/precision_propagation_convertion.hpp
    ${PUBLIC_HEADERS_DIR}/precision_propagation.hpp
    ${PUBLIC_HEADERS_DIR}/snippets_helpers.hpp
    ${PUBLIC_HEADERS_DIR}/subgraph_converts.hpp
    ${PUBLIC_HEADERS_DIR}/subgraph_customizable.hpp
    ${PUBLIC_HEADERS_DIR}/subgraph_fq.hpp
    ${PUBLIC_HEADERS_DIR}/subgraph_gated_mlp.hpp
    ${PUBLIC_HEADERS_DIR}/subgraph_gelu.hpp
    ${PUBLIC_HEADERS_DIR}/subgraph_group_normalization.hpp
    ${PUBLIC_HEADERS_DIR}/subgraph_lowered.hpp
    ${PUBLIC_HEADERS_DIR}/subgraph_matmul.hpp
    ${PUBLIC_HEADERS_DIR}/subgraph_mha.hpp
    ${PUBLIC_HEADERS_DIR}/subgraph_mlp_seq.hpp
    ${PUBLIC_HEADERS_DIR}/subgraph_reduce.hpp
    ${PUBLIC_HEADERS_DIR}/subgraph_roll_matmul_roll.hpp
    ${PUBLIC_HEADERS_DIR}/subgraph_simple.hpp
    ${PUBLIC_HEADERS_DIR}/subgraph_softmax.hpp
    ${PUBLIC_HEADERS_DIR}/subgraph_transpose.hpp
    ${PUBLIC_HEADERS_DIR}/two_binary_ops.hpp
)

set(PRIVATE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/precomp.hpp
)

source_group("src"     FILES ${SOURCES} ${PRIVATE_HEADERS})
source_group("include" FILES ${PUBLIC_HEADERS})

add_library(${TARGET_NAME} STATIC ${SOURCES} ${PUBLIC_HEADERS} ${PRIVATE_HEADERS})

target_include_directories(${TARGET_NAME}
    PUBLIC
        "$<BUILD_INTERFACE:${PUBLIC_HEADERS_DIR}>"
        "$<BUILD_INTERFACE:${COMMON_TEST_UTILS_INCLUDES}>"
)

target_link_libraries(${TARGET_NAME}
    PUBLIC
        openvino::runtime::dev
        common_test_utils
        openvino::snippets
        ov_lpt_models
)

ov_build_target_faster(${TARGET_NAME}
    UNITY
    PCH_HEADER "src/precomp.hpp"
)

# install & export

ov_developer_package_export_targets(TARGET ${TARGET_NAME}
                                    INSTALL_INCLUDE_DIRECTORIES "${PUBLIC_HEADERS_DIR}/")
