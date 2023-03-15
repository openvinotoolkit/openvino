# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT ENABLE_INTEL_CPU)
    return()
endif()

set(TARGET_NAME "openvino_intel_cpu_plugin")

if(CMAKE_COMPILER_IS_GNUCXX)
    ie_add_compiler_flags(-Wno-sign-compare)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # C4267, 4244 issues from mkl-dnn headers conversion from 'XXX' to 'YYY', possible loss of data
    ie_add_compiler_flags(/wd4267)
    ie_add_compiler_flags(/wd4244)
    # mkldnn headers: '<<': result of 32-bit shift implicitly converted to 64 bits
    ie_add_compiler_flags(/wd4334)
endif()

if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")
endif()

if (ENABLE_CPU_DEBUG_CAPS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCPU_DEBUG_CAPS")
endif()

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)

addVersionDefines(${CMAKE_CURRENT_SOURCE_DIR}/src/plugin.cpp CI_BUILD_NUMBER)

add_subdirectory(thirdparty)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

# create plugin

ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "CPU"
              AS_EXTENSION
              SOURCES ${SOURCES} ${HEADERS})

set_ie_threading_interface_for(${TARGET_NAME})

ie_mark_target_as_cc(${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PRIVATE dnnl
                                             ov_shape_inference
                                             openvino::pugixml
                                             inference_engine_snippets)

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_EXTENSION_API)
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
    $<TARGET_PROPERTY:dnnl,INCLUDE_DIRECTORIES>)

# Cross compiled function
# TODO: The same for proposal, proposalONNX, topk
cross_compiled_file(${TARGET_NAME}
        ARCH AVX2 ANY
                    src/nodes/proposal_imp.cpp
        API         src/nodes/proposal_imp.hpp
        NAME        proposal_exec
        NAMESPACE   InferenceEngine::Extensions::Cpu::XARCH
)

# must be called after all target_link_libraries
ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

#  add test object library

if(BUILD_SHARED_LIBS)
    add_library(${TARGET_NAME}_obj OBJECT ${SOURCES} ${HEADERS})
    link_system_libraries(${TARGET_NAME}_obj PUBLIC dnnl openvino::pugixml)

    target_include_directories(${TARGET_NAME}_obj
        PRIVATE
            $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:openvino::itt,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:ov_shape_inference,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:inference_engine_snippets,INTERFACE_INCLUDE_DIRECTORIES>
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            $<TARGET_PROPERTY:openvino::conditional_compilation,INTERFACE_INCLUDE_DIRECTORIES>)

    target_include_directories(${TARGET_NAME}_obj SYSTEM PUBLIC $<TARGET_PROPERTY:dnnl,INCLUDE_DIRECTORIES>)

    set_ie_threading_interface_for(${TARGET_NAME}_obj)

    target_compile_definitions(${TARGET_NAME}_obj PRIVATE
        USE_STATIC_IE IMPLEMENT_INFERENCE_ENGINE_PLUGIN IMPLEMENT_INFERENCE_EXTENSION_API
        $<TARGET_PROPERTY:ngraph,INTERFACE_COMPILE_DEFINITIONS>
        $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_COMPILE_DEFINITIONS>)

    set_target_properties(${TARGET_NAME}_obj PROPERTIES EXCLUDE_FROM_ALL ON)

    # LTO
    set_target_properties(${TARGET_NAME}_obj PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
endif()

# LTO
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
