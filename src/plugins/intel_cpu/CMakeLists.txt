# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT ENABLE_INTEL_CPU)
    return()
endif()

set(TARGET_NAME "openvino_intel_cpu_plugin")

# Warning suppressions strategy:
# - Thirdparty code warnings are suppressed in thirdparty/CMakeLists.txt
# - intel_cpu/src subdirectories warnings will be suppressed per-file via set_source_files_properties
# - intel_cpu/src root-level files warnings are visible for developer attention

if(OV_COMPILER_IS_INTEL_LLVM AND WIN32)
    ov_add_compiler_flags("/Wno-microsoft-include")
endif()
if(NOT BUILD_SHARED_LIBS)
    # Symbols are located in both src and include folders
    file(GLOB_RECURSE onednn_files
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onednn/include/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onednn/include/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onednn/include/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onednn/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onednn/src/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onednn/src/*.h")

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

if(AARCH64)
    set(ENABLE_KLEIDIAI_FOR_CPU_DEFAULT ON)
else()
    set(ENABLE_KLEIDIAI_FOR_CPU_DEFAULT OFF)
endif()
ov_dependent_option(ENABLE_KLEIDIAI_FOR_CPU "Enable KleidiAI for OpenVINO CPU Plugin" ${ENABLE_KLEIDIAI_FOR_CPU_DEFAULT} "AARCH64" OFF)

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

if(ENABLE_KLEIDIAI_FOR_CPU)
    add_definitions(-DOV_CPU_WITH_KLEIDIAI)
    set(OV_CPU_WITH_KLEIDIAI ON)
endif()

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)

if(NOT OV_CPU_WITH_ACL)
    list(APPEND EXCLUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/executors/acl/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/kernels/acl/*)
endif()

if(NOT X86_64)
    list(APPEND EXCLUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/executors/x64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/kernels/x64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/emitters/plugin/x64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/emitters/snippets/x64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/emitters/tpp/x64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/transformations/snippets/x64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/transformations/cpu_opset/x64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/transformations/tpp/x64/*)
endif()

if(NOT (AARCH64 OR ARM))
    list(APPEND EXCLUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/transformations/cpu_opset/arm/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/transformations/tpp/aarch64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/emitters/plugin/aarch64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/emitters/tpp/aarch64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/executors/aarch64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/kernels/aarch64/*)
endif()

if(NOT AARCH64)
    list(APPEND EXCLUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/transformations/snippets/aarch64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/emitters/snippets/aarch64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/executors/kleidiai/*)
endif()

if (NOT RISCV64)
    list(APPEND EXCLUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/emitters/plugin/riscv64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/kernels/riscv64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/emitters/snippets/riscv64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/executors/riscv64/*)
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
              SOURCES ${SOURCES} ${HEADERS}
              ADD_CLANG_FORMAT
              ADD_CLANG_TIDY)

# Suppress warnings for specific src subdirectories only (not root-level src files)
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Define subdirectories where warnings should be suppressed
    set(SUPPRESS_WARNING_SUBDIRS
        cache
        emitters
        memory_desc
        mlas
        nodes
        onednn
        shape_inference
        transformations
        utils)

    # Collect files from these subdirectories
    set(FILES_TO_SUPPRESS_WARNINGS)
    foreach(subdir ${SUPPRESS_WARNING_SUBDIRS})
        file(GLOB_RECURSE subdir_files
            "${CMAKE_CURRENT_SOURCE_DIR}/src/${subdir}/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/${subdir}/*.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/${subdir}/*.h")
        list(APPEND FILES_TO_SUPPRESS_WARNINGS ${subdir_files})
    endforeach()

    # Apply warning suppressions to these files
    if(FILES_TO_SUPPRESS_WARNINGS)
        set_source_files_properties(${FILES_TO_SUPPRESS_WARNINGS} PROPERTIES
            COMPILE_FLAGS "/wd4018 /wd4267 /wd4244 /wd4334 /wd4146 /wd5002")
    endif()
endif()

# Suppress warnings from thirdparty headers in version target
# (flags are exported from thirdparty/CMakeLists.txt)
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND OV_CPU_THIRDPARTY_MSVC_SUPPRESS_FLAGS)
    target_compile_options(${TARGET_NAME}_version PRIVATE ${OV_CPU_THIRDPARTY_MSVC_SUPPRESS_FLAGS})
endif()

# give a different file name depending on target platform architecture
if(ARM OR AARCH64)
    set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME "openvino_arm_cpu_plugin")
elseif(RISCV64)
    set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME "openvino_riscv_cpu_plugin")
endif()

ov_mark_target_as_cc(${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PRIVATE dnnl
                                             openvino::shape_inference
                                             openvino::snippets
                                             openvino_xml_util)

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
if(ENABLE_KLEIDIAI_FOR_CPU)
    target_link_libraries(${TARGET_NAME} PRIVATE kleidiai)
endif()

if(RISCV64)
    # Set `XBYAK_RISCV_V=1` to compile Xbyak-code for RVV-related instructions
    target_compile_definitions(xbyak_riscv INTERFACE XBYAK_RISCV_V=1)
    target_link_libraries(${TARGET_NAME} PRIVATE xbyak_riscv)
    target_include_directories(${TARGET_NAME} SYSTEM INTERFACE $<TARGET_PROPERTY:xbyak_riscv::xbyak_riscv,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

# TODO: XBYAK64 should be used only for x64. Have to enable it for all the architectures for now
# since many files include x64 related headers unconditionally.
add_definitions(-DXBYAK64)

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
        ARCH AVX512F AVX2 SVE NEON_FP16 ANY
                    src/nodes/kernels/scaled_attn/softmax.cpp
        API         src/nodes/kernels/scaled_attn/softmax.hpp
        NAME        attn_softmax
        NAMESPACE   ov::Extensions::Cpu::XARCH
)
cross_compiled_file(${TARGET_NAME}
        ARCH AVX512F AVX2 SVE NEON_FP16 ANY
                    src/nodes/kernels/scaled_attn/mha_single_token.cpp
        API         src/nodes/kernels/scaled_attn/mha_single_token.hpp
        NAME        mha_single_token
        NAMESPACE   ov::Extensions::Cpu::XARCH
)
cross_compiled_file(${TARGET_NAME}
        ARCH AVX512F AVX2 SVE ANY
                    src/nodes/kernels/scaled_attn/executor_pa.cpp
        API         src/nodes/kernels/scaled_attn/executor_pa.hpp
        NAME        make_pa_executor
        NAMESPACE   ov::Extensions::Cpu::XARCH
)
cross_compiled_file(${TARGET_NAME}
        ARCH AVX512F AVX2 SVE ANY
                    src/nodes/kernels/scaled_attn/attn_memcpy.cpp
        API         src/nodes/kernels/scaled_attn/attn_memcpy.hpp
        NAME        attn_memcpy paged_attn_memcpy attn_memcpy2d attn_memcpy2d_kernel
        NAMESPACE   ov::Extensions::Cpu::XARCH
)
cross_compiled_file(${TARGET_NAME}
        ARCH AVX512F AVX2 SVE NEON_FP16 ANY
                    src/nodes/kernels/scaled_attn/attn_quant.cpp
        API         src/nodes/kernels/scaled_attn/attn_quant.hpp
        NAME        paged_attn_quantkv attn_quant_u8 attn_dequant_u8 attn_quant_by_token attn_quant_by_channel attn_quant_by_channel_u8 attn_dequant_by_channel_u8
        NAMESPACE   ov::Extensions::Cpu::XARCH
)

cross_compiled_file(${TARGET_NAME}
        ARCH AVX512F ANY
                    src/nodes/kernels/x64/mlp_utils.cpp
        API         src/nodes/kernels/x64/mlp_utils.hpp
        NAME        llm_mlp_transpose_epi32_16x16  llm_mlp_quantize_bf16_i8 llm_mlp_quantize_f16_i8 llm_mlp_dequantize_i32_f32
        NAMESPACE   ov::Extensions::Cpu::XARCH
)

cross_compiled_file(${TARGET_NAME}
        ARCH AVX512F AVX2 ANY
                    src/nodes/kernels/linear_attn/recurrent_linear_attn.cpp
        API         src/nodes/kernels/linear_attn/recurrent_linear_attn.hpp
        NAME        recurrent_linear_attn
        NAMESPACE   ov::Extensions::Cpu::XARCH
)

# Suppress warnings for cross-compiled files (generated in build directory)
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # List all cross-compiled source files based on the cross_compiled_file() calls above
    set(CROSS_COMPILED_SOURCES
        # proposal_imp: AVX2 ANY
        cross-compiled/AVX2/proposal_imp.cpp
        cross-compiled/ANY/proposal_imp.cpp
        # softmax: AVX512F AVX2 SVE NEON_FP16 ANY
        cross-compiled/AVX512F/softmax.cpp
        cross-compiled/AVX2/softmax.cpp
        cross-compiled/ANY/softmax.cpp
        # mha_single_token: AVX512F AVX2 SVE NEON_FP16 ANY
        cross-compiled/AVX512F/mha_single_token.cpp
        cross-compiled/AVX2/mha_single_token.cpp
        cross-compiled/ANY/mha_single_token.cpp
        # executor_pa: AVX512F AVX2 SVE ANY
        cross-compiled/AVX512F/executor_pa.cpp
        cross-compiled/AVX2/executor_pa.cpp
        cross-compiled/ANY/executor_pa.cpp
        # attn_memcpy: AVX512F AVX2 SVE ANY
        cross-compiled/AVX512F/attn_memcpy.cpp
        cross-compiled/AVX2/attn_memcpy.cpp
        cross-compiled/ANY/attn_memcpy.cpp
        # attn_quant: AVX512F AVX2 SVE NEON_FP16 ANY
        cross-compiled/AVX512F/attn_quant.cpp
        cross-compiled/AVX2/attn_quant.cpp
        cross-compiled/ANY/attn_quant.cpp
        # mlp_utils: AVX512F ANY
        cross-compiled/AVX512F/mlp_utils.cpp
        cross-compiled/ANY/mlp_utils.cpp
        # recurrent_linear_attn: AVX512F AVX2 ANY
        cross-compiled/AVX512F/recurrent_linear_attn.cpp
        cross-compiled/AVX2/recurrent_linear_attn.cpp
        cross-compiled/ANY/recurrent_linear_attn.cpp
    )

    # Apply warning suppression to each file
    # Use separate set_property calls for each flag to avoid MSVC D8021 error
    foreach(src_file ${CROSS_COMPILED_SOURCES})
        set_property(SOURCE ${src_file} APPEND_STRING PROPERTY COMPILE_OPTIONS " /wd4018")
        set_property(SOURCE ${src_file} APPEND_STRING PROPERTY COMPILE_OPTIONS " /wd4267")
        set_property(SOURCE ${src_file} APPEND_STRING PROPERTY COMPILE_OPTIONS " /wd4244")
        set_property(SOURCE ${src_file} APPEND_STRING PROPERTY COMPILE_OPTIONS " /wd4334")
        set_property(SOURCE ${src_file} APPEND_STRING PROPERTY COMPILE_OPTIONS " /wd4146")
        set_property(SOURCE ${src_file} APPEND_STRING PROPERTY COMPILE_OPTIONS " /wd5002")
    endforeach()
endif()

# system dependencies must go last
target_link_libraries(${TARGET_NAME} PRIVATE openvino::pugixml)
ov_set_threading_interface_for(${TARGET_NAME})

# must be called after all target_link_libraries
ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

# LTO
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

set(PCH_EXCLUDE_FILES
    src/emitters/plugin/x64/debug_capabilities.cpp
    src/emitters/plugin/aarch64/debug_capabilities.cpp
    src/nodes/executors/printers.cpp
    src/emitters/snippets/x64/jit_perf_count_rdtsc_emitters.cpp
    src/emitters/snippets/x64/jit_segfault_detector_emitter.cpp
    src/emitters/snippets/x64/verbose.cpp
    src/emitters/snippets/aarch64/jit_segfault_detector_emitter.cpp
    src/emitters/snippets/aarch64/verbose.cpp
    src/transformations/snippets/x64/op/perf_count_rdtsc.cpp)

ov_build_target_faster(${TARGET_NAME} PCH PCH_EXCLUDE ${PCH_EXCLUDE_FILES})

#
# add test object library
#

if(BUILD_SHARED_LIBS)
    add_library(${TARGET_NAME}_obj OBJECT ${SOURCES} ${HEADERS})
    target_include_directories(${TARGET_NAME}_obj SYSTEM PUBLIC $<TARGET_PROPERTY:dnnl,INCLUDE_DIRECTORIES>)

    set(CPU_OBJ_LINK_SYSTEM LIBRARIES dnnl openvino::pugixml)
    if(RISCV64)
        list(APPEND CPU_OBJ_LINK_SYSTEM xbyak_riscv::xbyak_riscv)
    endif()

    ov_link_system_libraries(${TARGET_NAME}_obj PUBLIC ${CPU_OBJ_LINK_SYSTEM})

    ov_add_version_defines(src/plugin.cpp ${TARGET_NAME}_obj)

    # Suppress warnings for specific src subdirectories in object library too
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND FILES_TO_SUPPRESS_WARNINGS)
        set_source_files_properties(${FILES_TO_SUPPRESS_WARNINGS} TARGET_DIRECTORY ${TARGET_NAME}_obj
            PROPERTIES COMPILE_FLAGS "/wd4018 /wd4267 /wd4244 /wd4334 /wd4146 /wd5002")
    endif()

    # Suppress warnings from thirdparty headers in version targets
    # (flags are exported from thirdparty/CMakeLists.txt)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND OV_CPU_THIRDPARTY_MSVC_SUPPRESS_FLAGS)
        target_compile_options(${TARGET_NAME}_obj_version PRIVATE ${OV_CPU_THIRDPARTY_MSVC_SUPPRESS_FLAGS})
    endif()

    target_include_directories(${TARGET_NAME}_obj
        PRIVATE
            $<TARGET_PROPERTY:openvino::itt,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:openvino::shape_inference,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:openvino::snippets,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:openvino::reference,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:openvino::xml_util,INTERFACE_INCLUDE_DIRECTORIES>
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            $<TARGET_PROPERTY:openvino::conditional_compilation,INTERFACE_INCLUDE_DIRECTORIES>)

    target_include_directories(${TARGET_NAME}_obj SYSTEM PUBLIC $<TARGET_PROPERTY:dnnl,INCLUDE_DIRECTORIES>)
    if(ENABLE_SNIPPETS_LIBXSMM_TPP)
        target_include_directories(${TARGET_NAME}_obj SYSTEM PUBLIC $<TARGET_PROPERTY:xsmm,INCLUDE_DIRECTORIES>)
    endif()

    target_include_directories(${TARGET_NAME}_obj SYSTEM PRIVATE $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>)
    if(ENABLE_MLAS_FOR_CPU)
        target_include_directories(${TARGET_NAME}_obj SYSTEM PUBLIC $<TARGET_PROPERTY:mlas,INCLUDE_DIRECTORIES>)
    endif()

    if(ENABLE_KLEIDIAI_FOR_CPU)
        target_include_directories(${TARGET_NAME}_obj SYSTEM PUBLIC $<TARGET_PROPERTY:kleidiai,INTERFACE_INCLUDE_DIRECTORIES>)
    endif()
    if(RISCV64)
        target_include_directories(${TARGET_NAME}_obj SYSTEM PUBLIC $<TARGET_PROPERTY:xbyak_riscv::xbyak_riscv,INTERFACE_INCLUDE_DIRECTORIES>)
    endif()

    ov_set_threading_interface_for(${TARGET_NAME}_obj)

    target_compile_definitions(${TARGET_NAME}_obj PRIVATE USE_STATIC_IE)

    set_target_properties(${TARGET_NAME}_obj PROPERTIES EXCLUDE_FROM_ALL ON)

    # LTO
    set_target_properties(${TARGET_NAME}_obj PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

    ov_build_target_faster(${TARGET_NAME}_obj PCH PCH_EXCLUDE ${PCH_EXCLUDE_FILES})
endif()

if(OV_CPU_WITH_ACL)
    target_include_directories(${TARGET_NAME} PRIVATE
        ${OpenVINO_SOURCE_DIR}/src/inference/include)

    if(BUILD_SHARED_LIBS)
        target_include_directories(${TARGET_NAME}_obj PRIVATE
            ${OpenVINO_SOURCE_DIR}/src/inference/include)
    endif()
endif()

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
