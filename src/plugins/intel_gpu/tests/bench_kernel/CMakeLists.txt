# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "ov_gpu_bench_kernel")

if(SUGGEST_OVERRIDE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
endif()

file(GLOB_RECURSE SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

# Required plugin sources to resolve linker dependencies from openvino_intel_gpu_graph
set(PLUGIN_SRC_DIR "${CMAKE_HOME_DIRECTORY}/src/plugins/intel_gpu/src/plugin")
list(APPEND SOURCES
    "${PLUGIN_SRC_DIR}/variable_state.cpp"
    "${PLUGIN_SRC_DIR}/multi_tensor_variable_state.cpp"
    "${PLUGIN_SRC_DIR}/remote_context.cpp"
    "${PLUGIN_SRC_DIR}/remote_tensor.cpp"
    "${PLUGIN_SRC_DIR}/usm_host_tensor.cpp"
    "${PLUGIN_SRC_DIR}/common_utils.cpp"
    "${PLUGIN_SRC_DIR}/simple_math.cpp"
)

file(GLOB_RECURSE PLUGIN_TRANSFORMATIONS
    "${PLUGIN_SRC_DIR}/transformations/*.hpp"
    "${PLUGIN_SRC_DIR}/transformations/*.cpp"
)
list(APPEND SOURCES ${PLUGIN_TRANSFORMATIONS})

add_executable(${TARGET_NAME} ${SOURCES})

target_compile_definitions(${TARGET_NAME} PRIVATE CI_BUILD_NUMBER="")

ov_set_threading_interface_for(${TARGET_NAME})
ov_gpu_set_runtime_interface_for(${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

target_link_libraries(${TARGET_NAME} PRIVATE openvino_intel_gpu_graph
                                             gflags
                                             openvino::reference)

target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                                  $<TARGET_PROPERTY:openvino_intel_gpu_kernels,INTERFACE_INCLUDE_DIRECTORIES>
                                                  $<TARGET_PROPERTY:openvino_intel_gpu_runtime,INTERFACE_INCLUDE_DIRECTORIES>
                                                  ${CMAKE_HOME_DIRECTORY}/src/core/reference/include/)

if(WIN32)
    target_link_libraries(${TARGET_NAME} PRIVATE setupapi)
elseif((NOT ANDROID) AND (UNIX))
    target_link_libraries(${TARGET_NAME} PRIVATE pthread)
endif()

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL)
