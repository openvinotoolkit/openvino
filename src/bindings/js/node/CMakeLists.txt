# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

project(ov_node_addon)

set(CMAKE_CXX_STANDARD 17)
add_definitions(-DNAPI_VERSION=6)

include(FetchContent)

FetchContent_Declare(
    node-api-headers
    URL      https://github.com/nodejs/node-api-headers/archive/refs/tags/v1.1.0.tar.gz
    URL_HASH SHA256=70608bc1e6dddce280285f3462f18a106f687c0720a4b90893e1ecd86e5a8bbf
)
FetchContent_MakeAvailable(node-api-headers)

FetchContent_Declare(
    node-addon-api
    URL      https://github.com/nodejs/node-addon-api/archive/refs/tags/v7.0.0.tar.gz
    URL_HASH SHA256=e0cf2ef9bed9965ac8f498b4a7007b9f0002fa022c1ff665f256ab54260d6305
)
FetchContent_MakeAvailable(node-addon-api)

add_library(${PROJECT_NAME} SHARED
    src/node_output.cpp
    src/async_reader.cpp
    src/preprocess/preprocess.cpp
    src/preprocess/pre_post_process_wrap.cpp
    src/preprocess/preprocess_steps.cpp
    src/preprocess/input_info.cpp
    src/preprocess/output_info.cpp
    src/preprocess/input_tensor_info.cpp
    src/preprocess/output_tensor_info.cpp
    src/preprocess/input_model_info.cpp
    src/preprocess/resize_algorithm.cpp
    src/errors.cpp
    src/helper.cpp
    src/tensor.cpp
    src/infer_request.cpp
    src/compiled_model.cpp
    src/core_wrap.cpp
    src/model_wrap.cpp
    src/addon.cpp
    src/element_type.cpp
    src/partial_shape_wrap.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE 
    "${node-api-headers_SOURCE_DIR}/include"
    "${node-addon-api_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(${PROJECT_NAME} PRIVATE openvino::runtime)

set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".node"
)

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION openvinojs-node/bin/ COMPONENT ${PROJECT_NAME} EXCLUDE_FROM_ALL
)

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION openvinojs-node/bin/ COMPONENT ${PROJECT_NAME} EXCLUDE_FROM_ALL
    RUNTIME DESTINATION openvinojs-node/bin/ COMPONENT ${PROJECT_NAME} EXCLUDE_FROM_ALL
)
