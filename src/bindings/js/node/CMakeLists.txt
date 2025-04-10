# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CMAKE_VERSION VERSION_LESS 3.14)
    message(WARNING "JS API is not available with CMake version less than 3.14, skipping")
    return()
endif()

if(CMAKE_COMPILER_IS_GNUCXX AND LINUX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
    message(WARNING "JS API is not support gcc compiler version less than 7, skipping")
    return()
endif()

if(WIN32)
    set(CMAKE_SHARED_LINKER_FLAGS /DELAYLOAD:NODE.EXE)
    set(CMAKE_JS_LIB ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/node.lib)
    set(CMAKE_JS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/win_delay_load_hook.cc)

    set(CMAKE_JS_NODELIB_DEF ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/node-lib.def)
    set(CMAKE_JS_NODELIB_TARGET ${CMAKE_JS_LIB})
    set(DELAYIMP_LIB delayimp.lib)
endif()

cmake_minimum_required(VERSION 3.14)

project(ov_node_addon)

set(CMAKE_CXX_STANDARD 17)
add_definitions(-DNAPI_VERSION=8)

include(FetchContent)

FetchContent_Declare(
    node-api-headers
    URL      https://github.com/nodejs/node-api-headers/archive/refs/tags/v1.1.0.tar.gz
    URL_HASH SHA256=70608bc1e6dddce280285f3462f18a106f687c0720a4b90893e1ecd86e5a8bbf
)
FetchContent_MakeAvailable(node-api-headers)

FetchContent_Declare(
    node-addon-api
    URL      https://github.com/nodejs/node-addon-api/archive/refs/tags/v8.0.0.tar.gz
    URL_HASH SHA256=42424c5206b9d67b41af4fcff5d6e3cb22074168035a03b8467852938a281d47
)
FetchContent_MakeAvailable(node-addon-api)

add_library(${PROJECT_NAME} SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/node_output.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/async_reader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/preprocess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/pre_post_process_wrap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/preprocess_steps.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/input_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/output_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/input_tensor_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/output_tensor_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/input_model_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/resize_algorithm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/errors.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/helper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/type_validation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/infer_request.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/compiled_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core_wrap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model_wrap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/addon.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/element_type.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/partial_shape_wrap.cpp

    ${CMAKE_JS_SRC}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    "${node-api-headers_SOURCE_DIR}/include"
    "${node-addon-api_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

target_link_libraries(${PROJECT_NAME} PRIVATE openvino::core::dev openvino::runtime openvino::util ${DELAYIMP_LIB} ${CMAKE_JS_LIB})

if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
    # Generate node.lib
    execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
    ov_add_compiler_flags(-Wno-missing-declarations)
endif()

if(APPLE)
    target_link_options(${PROJECT_NAME} PRIVATE -Wl,-undefined,suppress,-flat_namespace)
elseif(AARCH64 OR ARM)
    target_link_options(${PROJECT_NAME} PRIVATE -Wl,--unresolved-symbols=ignore-all)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".node"
)

ov_set_install_rpath(${PROJECT_NAME}
    ${OV_CPACK_RUNTIMEDIR} ${OV_CPACK_RUNTIMEDIR})

ov_add_clang_format_target(${PROJECT_NAME}_clang FOR_TARGETS ${PROJECT_NAME})

ov_cpack_add_component(${OV_CPACK_COMP_NPM} HIDDEN)

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${PROJECT_NAME} ${OV_CPACK_COMP_NPM_EXCLUDE_ALL}
    RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${PROJECT_NAME} ${OV_CPACK_COMP_NPM_EXCLUDE_ALL}
)
