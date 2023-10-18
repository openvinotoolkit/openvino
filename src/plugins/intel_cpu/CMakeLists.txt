# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT ENABLE_INTEL_CPU)
    return()
endif()

set(TARGET_NAME "openvino_intel_cpu_plugin")

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
    ov_add_compiler_flags(-Wno-delete-non-abstract-non-virtual-dtor)
endif()

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
        # Apple M1 / M2 is assumed
        set(OV_CPU_ARM_TARGET_ARCH_DEFAULT arm64-v8.2-a)
    else()
        set(OV_CPU_ARM_TARGET_ARCH_DEFAULT arm64-v8a)
    endif()
    set(OV_CPU_ARM_TARGET_ARCHS arm64-v8a
                                arm64-v8.2-a arm64-v8.2-a-sve arm64-v8.2-a-sve2
                                # used with estate=64
                                ${OV_CPU_ARM_TARGET_GENERIC_ARCHS})
endif()
set(OV_CPU_ARM_TARGET_ARCH ${OV_CPU_ARM_TARGET_ARCH_DEFAULT} CACHE STRING "Architecture for ARM ComputeLibrary")
set_property(CACHE OV_CPU_ARM_TARGET_ARCH PROPERTY STRINGS ${OV_CPU_ARM_TARGET_ARCHS})
if(OV_CPU_ARM_TARGET_ARCH MATCHES "(armv|arm64-v)[8-9]\\.")
    add_definitions(-DOV_CPU_ARM_ENABLE_FP16)
endif()

if(X86 OR X86_64 OR AARCH64)
    # disable mlas with webassembly
    if(EMSCRIPTEN OR (WIN32 AND AARCH64) OR MINGW OR (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7))
        set(ENABLE_MLAS_FOR_CPU_DEFAULT OFF)
    else()
        set(ENABLE_MLAS_FOR_CPU_DEFAULT ON)
    endif()
else()
    set(ENABLE_MLAS_FOR_CPU_DEFAULT OFF)
endif()
ov_option(ENABLE_MLAS_FOR_CPU "Enable MLAS for OpenVINO CPU Plugin" ${ENABLE_MLAS_FOR_CPU_DEFAULT})

add_subdirectory(thirdparty)

if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")
endif()

if(ENABLE_CPU_DEBUG_CAPS)
    add_definitions(-DCPU_DEBUG_CAPS)
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

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)

if(NOT OV_CPU_WITH_ACL)
    list(APPEND EXCLUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/executors/acl/*)
endif()

if(NOT X86_64)
    list(APPEND EXCLUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/executors/x64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/kernels/x64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/emitters/x64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/transformations/cpu_opset/x64/*
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/transformations/snippets/x64/*)
endif()

if(NOT (AARCH64 OR ARM))
    list(APPEND EXCLUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/transformations/cpu_opset/arm/*)
endif()

if (NOT ENABLE_MLAS_FOR_CPU)
    list(APPEND EXCLUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/executors/mlas/*)
    list(APPEND EXCLUDE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/mlas/*)
endif()

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

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_EXTENSION_API)
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
if (ENABLE_MLAS_FOR_CPU)
    target_link_libraries(${TARGET_NAME} PRIVATE mlas)
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE $<TARGET_PROPERTY:mlas,INCLUDE_DIRECTORIES>)
    add_definitions(-DOV_CPU_WITH_MLAS)
endif()
target_include_directories(${TARGET_NAME} SYSTEM PRIVATE $<TARGET_PROPERTY:dnnl,INCLUDE_DIRECTORIES>)
# Cross compiled function
# TODO: The same for proposal, proposalONNX, topk
cross_compiled_file(${TARGET_NAME}
        ARCH AVX2 ANY
                    src/nodes/proposal_imp.cpp
        API         src/nodes/proposal_imp.hpp
        NAME        proposal_exec
        NAMESPACE   InferenceEngine::Extensions::Cpu::XARCH
)
cross_compiled_file(${TARGET_NAME}
        ARCH AVX512F AVX2 ANY
                    src/nodes/scaled_attn_softmax.cpp
        API         src/nodes/scaled_attn_softmax.hpp
        NAME        attn_softmax
        NAMESPACE   InferenceEngine::Extensions::Cpu::XARCH
)
cross_compiled_file(${TARGET_NAME}
        ARCH AVX512F AVX2 ANY
                    src/nodes/scaled_attn_dot_product.cpp
        API         src/nodes/scaled_attn_dot_product.hpp
        NAME        attn_dot_product
        NAMESPACE   InferenceEngine::Extensions::Cpu::XARCH
)
cross_compiled_file(${TARGET_NAME}
        ARCH AVX512F AVX2 ANY
                    src/nodes/scaled_attn_acc_value.cpp
        API         src/nodes/scaled_attn_acc_value.hpp
        NAME        attn_acc_value
        NAMESPACE   InferenceEngine::Extensions::Cpu::XARCH
)
cross_compiled_file(${TARGET_NAME}
        ARCH AVX512F AVX2 ANY
                    src/nodes/scaled_attn_reduce.cpp
        API         src/nodes/scaled_attn_reduce.hpp
        NAME        attn_reduce
        NAMESPACE   InferenceEngine::Extensions::Cpu::XARCH
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
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            $<TARGET_PROPERTY:openvino::conditional_compilation,INTERFACE_INCLUDE_DIRECTORIES>)

    target_include_directories(${TARGET_NAME}_obj SYSTEM PUBLIC $<TARGET_PROPERTY:dnnl,INCLUDE_DIRECTORIES>)
    
    if(ENABLE_MLAS_FOR_CPU)
        target_include_directories(${TARGET_NAME}_obj SYSTEM PUBLIC $<TARGET_PROPERTY:mlas,INCLUDE_DIRECTORIES>)
    endif()

    ov_set_threading_interface_for(${TARGET_NAME}_obj)

    target_compile_definitions(${TARGET_NAME}_obj PRIVATE
        USE_STATIC_IE IMPLEMENT_INFERENCE_ENGINE_PLUGIN IMPLEMENT_INFERENCE_EXTENSION_API
        $<TARGET_PROPERTY:ngraph,INTERFACE_COMPILE_DEFINITIONS>
        $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_COMPILE_DEFINITIONS>)

    set_target_properties(${TARGET_NAME}_obj PROPERTIES EXCLUDE_FROM_ALL ON)

    # LTO
    set_target_properties(${TARGET_NAME}_obj PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
endif()

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
