# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Experimental core operators and passes (only built when ENABLE_OV_EXPERIMENTAL is ON).
# The op implementations are compiled into openvino_core_obj; their public headers are exposed
# through openvino_core_dev so ABI/experimental-aware consumers (tests, plugins) can use them.

set(OV_EXP_INCLUDE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(OV_EXP_SHAPE_INFER_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/shape_inference/include)
set(OV_EXP_REF_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/reference/include)

target_sources(openvino_core_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/block_sparse_attention.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/draft_block_mask.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/opset/experimental_opset1.cpp)

# Define OPENVINO_EXPERIMENTAL while compiling the experimental sources (and the opset.cpp that
# wires their registration hooks into openvino_core_obj) so their `#if defined(OPENVINO_EXPERIMENTAL)`
# registration blocks are active. Scoped to the sources that need it rather than the whole target.
set(OV_CORE_EXP_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/block_sparse_attention.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/draft_block_mask.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/opset/experimental_opset1.cpp
)
set_property(SOURCE ${OV_CORE_EXP_SRCS} ${OpenVINO_SOURCE_DIR}/src/core/src/opsets/opset.cpp
    TARGET_DIRECTORY openvino_core_obj
    APPEND PROPERTY COMPILE_DEFINITIONS "OPENVINO_EXPERIMENTAL"
)

# The experimental sources must not rely on undefined preprocessor identifiers;
# promote -Wundef to an error so any `#if defined(OPENVINO_EXPERIMENTAL)` drift is
# caught at compile time rather than the combinatorial #if/#ifdef subtlety.
if(NOT MSVC)
    set_property(SOURCE ${OV_CORE_EXP_SRCS}
        TARGET_DIRECTORY openvino_core_obj
        APPEND PROPERTY COMPILE_OPTIONS "-Wundef;-Werror=undef"
    )
endif()

# Internal implementation headers (shape_inference + reference) are PRIVATE to openvino_core_obj.
# The public experimental op headers are exposed PUBLIC here (and again via openvino_core_dev below)
# so that consumers linking the core dev API can construct and introspect these ops.
target_include_directories(openvino_core_obj
    PUBLIC $<BUILD_INTERFACE:${OV_EXP_INCLUDE_ROOT}>
    PRIVATE
        ${OV_EXP_SHAPE_INFER_INCLUDE}
        ${OV_EXP_REF_INCLUDE})

# Expose the experimental op public headers through the core dev INTERFACE target.
target_include_directories(openvino_core_dev INTERFACE
    $<BUILD_INTERFACE:${OV_EXP_INCLUDE_ROOT}>)

# Install the experimental op public headers alongside the core dev headers.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${OV_CPACK_INCLUDEDIR}
        COMPONENT ${OV_CPACK_COMP_CORE_DEV} ${OV_CPACK_COMP_CORE_DEV_EXCLUDE_ALL}
        FILES_MATCHING PATTERN "*.hpp")

# Export a flag that consumers can use to detect an experimental-enabled build.
if(ENABLE_OV_EXPERIMENTAL)
    file(WRITE "${CMAKE_BINARY_DIR}/OpenVINOExperimental.cmake" "set(OV_EXPERIMENTAL_ENABLED ON)\n")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/OpenVINOExperimental.cmake" "set(OV_EXPERIMENTAL_ENABLED ON)\n")
    file(APPEND "${CMAKE_BINARY_DIR}/OpenVINOExperimental.cmake" "set(OV_EXPERIMENTAL_OPSETS experimental_opset1)\n")
    file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/OpenVINOExperimental.cmake" "set(OV_EXPERIMENTAL_OPSETS experimental_opset1)\n")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/OpenVINOExperimental.cmake"
            DESTINATION ${OV_CPACK_OPENVINO_CMAKEDIR}
            COMPONENT ${OV_CPACK_COMP_CORE})
endif()
