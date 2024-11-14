# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT ENABLE_INTEL_CPU)
    return()
endif()

set(TARGET_NAME "openvino_intel_cpu_plugin")

if((CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG) AND CMAKE_CXX_STANDARD GREATER_EQUAL 20)
    set(CMAKE_CXX_FLAGS "-Wno-error=deprecated ${CMAKE_CXX_FLAGS}")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # C4267, 4244 issues from oneDNN headers conversion from 'XXX' to 'YYY', possible loss of data
    ov_add_compiler_flags(/wd4018)
    ov_add_compiler_flags(/wd4267)
    ov_add_compiler_flags(/wd4244)
    # mkldnn headers: '<<': result of 32-bit shift implicitly converted to 64 bits
    ov_add_compiler_flags(/wd4334)
    # oneDNN arm64: unary minus operator applied to unsigned type, result still unsigned
    ov_add_compiler_flags(/wd4146)
elseif(OV_COMPILER_IS_CLANG)
    # -Wno-delete-non-abstract-non-virtual-dtor is support > clang 9.0
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0)
        ov_add_compiler_flags(-Wno-delete-non-abstract-non-virtual-dtor)
    else()
        ov_add_compiler_flags(-Wno-delete-non-virtual-dtor)
    endif()
elseif(CMAKE_COMPILER_IS_GNUCXX)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
        ov_add_compiler_flags(-Wno-array-bounds)
    endif()
elseif (OV_COMPILER_IS_INTEL_LLVM AND WIN32)
    ov_add_compiler_flags("/Wno-microsoft-include")
endif()
if(NOT BUILD_SHARED_LIBS)
    # Symbols are located in both src and include folders
    file(GLOB_RECURSE onednn_files
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onednn/include/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onednn/include/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onednn/include/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onednn/include/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onednn/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onednn/src/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onednn/src/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onednn/src/*.hpp")

    # parse API symbols
    foreach(onednn_file IN LISTS onednn_files)
        # symbols in form:
        # dnnl_status_t DNNL_API dnnl_engine_get_kind
        file(STRINGS "${onednn_file}" onednn_symbols_defined_on_single_line
             REGEX "DNNL_API[ \*]*dnnl[a-zA-Z0-9_]*")
        # symbols in form (cmake has issue with symbols defined on multiple lines and we have to use new pattern):
        # dnnl_status_t DNNL_API
        # dnnl_engine_get_kind
        file(STRINGS "${onednn_file}" onednn_symbols_defined_on_multiple_lines
             REGEX "^dnnl[a-zA-Z0-9_]*\\(")
        # symbols in form:
        # typedef struct dnnl_graph_graph *dnnl_graph_graph_t;
        file(STRINGS "${onednn_file}" onednn_symbols_typedef
             REGEX "^typedef struct dnnl_.*")

        if(onednn_symbols_defined_on_single_line OR
           onednn_symbols_defined_on_multiple_lines OR
           onednn_symbols_typedef)
            # parse concrete symbols from read line
            string(REGEX MATCHALL "dnnl[a-zA-Z0-9_]+" onednn_parsed_symbols
                ${onednn_symbols_defined_on_single_line}
                ${onednn_symbols_defined_on_multiple_lines}
                ${onednn_symbols_typedef})
            list(APPEND onednn_symbols ${onednn_parsed_symbols})
        endif()
    endforeach()

    # remove all duplicates
    list(REMOVE_DUPLICATES onednn_symbols)

    # also override namespaces
    list(APPEND onednn_symbols dnnl oneapi)

    # redefine all collected symbols
    foreach(onednn_symbol IN LISTS onednn_symbols)
        if(NOT onednn_symbol MATCHES "^#.+")
            add_compile_definitions(${onednn_symbol}=ov_cpu_${onednn_symbol})
        endif()
    endforeach()
endif()

if (AARCH64 AND NOT APPLE AND CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10.2)
    # according to https://github.com/ARM-software/ComputeLibrary/issues/1053#issuecomment-1846903707 comment
    # the 'multi_isa=1' below enables FP32, FP16 and SVE / SVE2 kernels
    # But: arm_sve.h header is not available on gcc older 10.2 (let's test it), so we have to check it
    set(OV_CPU_AARCH64_USE_MULTI_ISA_DEFAULT ON)
else()
    set(OV_CPU_AARCH64_USE_MULTI_ISA_DEFAULT OFF)
endif()
set(OV_CPU_AARCH64_USE_MULTI_ISA ${OV_CPU_AARCH64_USE_MULTI_ISA_DEFAULT} CACHE BOOL "Build multi-ISA ACL")

set(OV_CPU_ARM_TARGET_GENERIC_ARCHS armv8a
                                    armv8.2-a
                                    armv8.6-a armv8.6-a-sve armv8.6-a-sve2 armv8.6-a-sve2-sme2
                                    armv8r64 # the same as armv8.4-a
)
if(ARM)
    set(OV_CPU_ARM_TARGET_ARCH_DEFAULT armv7a)
    set(OV_CPU_ARM_TARGET_ARCHS armv7a armv7a-hf
                                # requires estate=32
                                ${OV_CPU_ARM_TARGET_GENERIC_ARCHS})
elseif(AARCH64)
    if(APPLE)
        set(OV_CPU_ARM_TARGET_ARCH_DEFAULT arm64-v8.2-a)
    else()
        if(OV_CPU_AARCH64_USE_MULTI_ISA)
            # set v8a even we want fp16 kernels, because
            # we use multi_isa=1 in ACLConfig.cmake to enable both fp16 and fp32 kernels
            # actual kernel is selected in runtime based on runtime capabilities
            set(OV_CPU_ARM_TARGET_ARCH_DEFAULT arm64-v8a)
        else()
            set(OV_CPU_ARM_TARGET_ARCH_DEFAULT arm64-v8.2-a)
        endif()
    endif()
    set(OV_CPU_ARM_TARGET_ARCHS arm64-v8a
                                arm64-v8.2-a arm64-v8.2-a-sve arm64-v8.2-a-sve2
                                # used with estate=64
                                ${OV_CPU_ARM_TARGET_GENERIC_ARCHS})
endif()
set(OV_CPU_ARM_TARGET_ARCH ${OV_CPU_ARM_TARGET_ARCH_DEFAULT} CACHE STRING "Architecture for ARM ComputeLibrary")
set_property(CACHE OV_CPU_ARM_TARGET_ARCH PROPERTY STRINGS ${OV_CPU_ARM_TARGET_ARCHS})

if(X86 OR X86_64 OR AARCH64)
    # disable mlas with webassembly and intel compiler on windows
    if(EMSCRIPTEN OR (WIN32 AND AARCH64) OR MINGW OR
       (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7) OR (OV_COMPILER_IS_INTEL_LLVM AND WIN32))
        set(ENABLE_MLAS_FOR_CPU_DEFAULT OFF)
    else()
        set(ENABLE_MLAS_FOR_CPU_DEFAULT ON)
    endif()
else()
    set(ENABLE_MLAS_FOR_CPU_DEFAULT OFF)
endif()
ov_option(ENABLE_MLAS_FOR_CPU "Enable MLAS for OpenVINO CPU Plugin" ${ENABLE_MLAS_FOR_CPU_DEFAULT})

if(RISCV64_XUANTIE)
    set(ENABLE_SHL_FOR_CPU_DEFAULT ON)
else()
    set(ENABLE_SHL_FOR_CPU_DEFAULT OFF)
endif()
ov_dependent_option(ENABLE_SHL_FOR_CPU "Enable SHL for OpenVINO CPU Plugin" ${ENABLE_SHL_FOR_CPU_DEFAULT} "RISCV64" OFF)

add_subdirectory(thirdparty)

if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")
endif()

if(ENABLE_CPU_DEBUG_CAPS)
    add_definitions(-DCPU_DEBUG_CAPS)
endif()

if (ENABLE_SNIPPETS_LIBXSMM_TPP)
    # Note: LIBXSMM_DEFAULT_CONFIG needed so libxsmm_config can be included without issues
    add_definitions(-DSNIPPETS_LIBXSMM_TPP -DLIBXSMM_DEFAULT_CONFIG)
endif()

set(OV_CPU_WITH_DNNL ON)
if(OV_CPU_WITH_DNNL)
    add_definitions(-DOV_CPU_WITH_DNNL)
endif()

if(DNNL_USE_ACL)
    add_definitions(-DOV_CPU_WITH_ACL)
    set(OV_CPU_WITH_ACL ON)
endif()

if(OV_CPU_WITH_ACL)
    set(CMAKE_CXX_STANDARD 14)
endif()

if (ENABLE_SHL_FOR_CPU)
    add_definitions(-DOV_CPU_WITH_SHL)
    set(OV_CPU_WITH_SHL ON)
endif()

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)

if(NOT OV_CPU_WITH_ACL)
    list(APPEND EXCLUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/executors/acl/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/kernels/acl/*)
endif()

if(NOT OV_CPU_WITH_SHL)
    list(APPEND EXCLUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/executors/shl/*)
endif()

if(NOT X86_64)
    list(APPEND EXCLUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/executors/x64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/kernels/x64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/emitters/plugin/x64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/emitters/snippets/x64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/transformations/cpu_opset/x64/*)
endif()

if (AARCH64)
    # this directory is reused on RISCV64
    list(APPEND EXCLUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/transformations/snippets/x64/*)
endif()

if(NOT (AARCH64 OR ARM))
    list(APPEND EXCLUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/transformations/cpu_opset/arm/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/emitters/plugin/aarch64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/executors/aarch64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/kernels/aarch64/*)
endif()

if(NOT AARCH64)
    list(APPEND EXCLUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/transformations/snippets/aarch64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/emitters/snippets/aarch64/*)
endif()

if (NOT ENABLE_MLAS_FOR_CPU)
    list(APPEND EXCLUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/executors/mlas/*)
    list(APPEND EXCLUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/mlas/*)
endif()

if (NOT ENABLE_SNIPPETS_LIBXSMM_TPP)
    list(APPEND EXCLUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/emitters/tpp/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/transformations/tpp/*)
endif ()

file(GLOB_RECURSE FILES_TO_REMOVE ${EXCLUDE_PATHS})
list(REMOVE_ITEM SOURCES ${FILES_TO_REMOVE})
list(REMOVE_ITEM HEADERS ${FILES_TO_REMOVE})

# create plugin

ov_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "CPU"
              AS_EXTENSION
              VERSION_DEFINES_FOR src/plugin.cpp
              SOURCES ${SOURCES} ${HEADERS})

# give a different file name depending on target platform architecture
if(ARM OR AARCH64)
    set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME "openvino_arm_cpu_plugin")
elseif(RISCV64)
    set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME "openvino_riscv_cpu_plugin")
endif()

ov_mark_target_as_cc(${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PRIVATE dnnl
                                             openvino::shape_inference
                                             openvino::snippets)

target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
if (ENABLE_MLAS_FOR_CPU)
    target_link_libraries(${TARGET_NAME} PRIVATE mlas)
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE $<TARGET_PROPERTY:mlas,INCLUDE_DIRECTORIES>)
    add_definitions(-DOV_CPU_WITH_MLAS)
endif()
if (ENABLE_SNIPPETS_LIBXSMM_TPP)
    target_compile_definitions(xsmm PRIVATE __BLAS=0)
    target_link_libraries(${TARGET_NAME} PRIVATE xsmm)
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE $<TARGET_PROPERTY:xsmm,INCLUDE_DIRECTORIES>)
endif ()
if(ENABLE_SHL_FOR_CPU)
    target_link_libraries(${TARGET_NAME} PRIVATE shl)
endif()
target_include_directories(${TARGET_NAME} SYSTEM PRIVATE $<TARGET_PROPERTY:dnnl,INCLUDE_DIRECTORIES>)

# Temporal solution to use template reference implementations in cases where optimizied implementation
# is not (yet) needed.
target_include_directories(${TARGET_NAME} PRIVATE $<TARGET_PROPERTY:openvino::reference,INTERFACE_INCLUDE_DIRECTORIES>)

# Cross compiled function
# TODO: The same for proposal, proposalONNX, topk
cross_compiled_file(${TARGET_NAME}
        ARCH AVX2 ANY
                    src/nodes/proposal_imp.cpp
        API         src/nodes/proposal_imp.hpp
        NAME        proposal_exec
        NAMESPACE   ov::Extensions::Cpu::XARCH
)
cross_compiled_file(${TARGET_NAME}
        ARCH AVX512F AVX2 NEON_FP16 ANY
                    src/nodes/kernels/scaled_attn/softmax.cpp
        API         src/nodes/kernels/scaled_attn/softmax.hpp
        NAME        attn_softmax
        NAMESPACE   ov::Extensions::Cpu::XARCH
)
cross_compiled_file(${TARGET_NAME}
        ARCH AVX512F AVX2 NEON_FP16 ANY
                    src/nodes/kernels/scaled_attn/mha_single_token.cpp
        API         src/nodes/kernels/scaled_attn/mha_single_token.hpp
        NAME        mha_single_token
        NAMESPACE   ov::Extensions::Cpu::XARCH
)
cross_compiled_file(${TARGET_NAME}
        ARCH AVX512F AVX2 ANY
                    src/nodes/kernels/scaled_attn/executor_pa.cpp
        API         src/nodes/kernels/scaled_attn/executor_pa.hpp
        NAME        make_pa_executor
        NAMESPACE   ov::Extensions::Cpu::XARCH
)
cross_compiled_file(${TARGET_NAME}
        ARCH AVX512F AVX2 ANY
                    src/nodes/kernels/scaled_attn/attn_memcpy.cpp
        API         src/nodes/kernels/scaled_attn/attn_memcpy.hpp
        NAME        attn_memcpy paged_attn_memcpy attn_memcpy2d_kernel
        NAMESPACE   ov::Extensions::Cpu::XARCH
)
cross_compiled_file(${TARGET_NAME}
        ARCH AVX512F AVX2 ANY
                    src/nodes/kernels/scaled_attn/attn_quant.cpp
        API         src/nodes/kernels/scaled_attn/attn_quant.hpp
        NAME        attn_quantkv paged_attn_quantkv attn_quant_u8 attn_dequant_u8
        NAMESPACE   ov::Extensions::Cpu::XARCH
)

cross_compiled_file(${TARGET_NAME}
        ARCH AVX512F ANY
                    src/nodes/kernels/x64/mlp_utils.cpp
        API         src/nodes/kernels/x64/mlp_utils.hpp
        NAME        llm_mlp_transpose_epi32_16x16  llm_mlp_quantize_bf16_i8 llm_mlp_quantize_f16_i8 llm_mlp_dequantize_i32_f32
        NAMESPACE   ov::Extensions::Cpu::XARCH
)

# system dependencies must go last
target_link_libraries(${TARGET_NAME} PRIVATE openvino::pugixml)
ov_set_threading_interface_for(${TARGET_NAME})

# must be called after all target_link_libraries
ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

# LTO
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

#
# add test object library
#

if(BUILD_SHARED_LIBS)
    add_library(${TARGET_NAME}_obj OBJECT ${SOURCES} ${HEADERS})
    ov_link_system_libraries(${TARGET_NAME}_obj PUBLIC dnnl openvino::pugixml)

    ov_add_version_defines(src/plugin.cpp ${TARGET_NAME}_obj)

    target_include_directories(${TARGET_NAME}_obj
        PRIVATE
            $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:openvino::itt,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:openvino::shape_inference,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:openvino::snippets,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:openvino::reference,INTERFACE_INCLUDE_DIRECTORIES>
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            $<TARGET_PROPERTY:openvino::conditional_compilation,INTERFACE_INCLUDE_DIRECTORIES>)

    target_include_directories(${TARGET_NAME}_obj SYSTEM PUBLIC $<TARGET_PROPERTY:dnnl,INCLUDE_DIRECTORIES>)
    if(ENABLE_SNIPPETS_LIBXSMM_TPP)
        target_include_directories(${TARGET_NAME}_obj SYSTEM PUBLIC $<TARGET_PROPERTY:xsmm,INCLUDE_DIRECTORIES>)
    endif()

    if(ENABLE_MLAS_FOR_CPU)
        target_include_directories(${TARGET_NAME}_obj SYSTEM PUBLIC $<TARGET_PROPERTY:mlas,INCLUDE_DIRECTORIES>)
    endif()

    if(ENABLE_SHL_FOR_CPU)
        target_include_directories(${TARGET_NAME}_obj SYSTEM PUBLIC $<TARGET_PROPERTY:shl,INTERFACE_INCLUDE_DIRECTORIES>)
    endif()

    ov_set_threading_interface_for(${TARGET_NAME}_obj)

    target_compile_definitions(${TARGET_NAME}_obj PRIVATE USE_STATIC_IE)

    set_target_properties(${TARGET_NAME}_obj PROPERTIES EXCLUDE_FROM_ALL ON)

    # LTO
    set_target_properties(${TARGET_NAME}_obj PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
endif()

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
