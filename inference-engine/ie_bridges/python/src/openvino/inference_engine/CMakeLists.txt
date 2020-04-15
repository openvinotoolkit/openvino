# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(TARGET_NAME "ie_api")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/inference_engine)

file(GLOB SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/*.pyx
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

set_source_files_properties(${SOURCE} PROPERTIES CYTHON_IS_CXX ON)

# create target

cython_add_module(${TARGET_NAME} ${SOURCE})

function(python_disable_deprecated_warnings)
    disable_deprecated_warnings()
    set(pyx_file "${CMAKE_CURRENT_BINARY_DIR}/ie_api.cxx")
    set_source_files_properties(${pyx_file} PROPERTIES COMPILE_FLAGS ${ie_c_cxx_deprecated})
endfunction()

python_disable_deprecated_warnings()

target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(${TARGET_NAME} PRIVATE ${InferenceEngine_LIBRARIES})

# Compatibility with python 2.7 which has deprecated "register" specifier
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${TARGET_NAME} PRIVATE "-Wno-error=register")
endif()

# perform copy
add_custom_command(TARGET ${TARGET_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_BRIDGE_SRC_ROOT}/src/openvino/inference_engine/__init__.py ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py
    COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_BRIDGE_SRC_ROOT}/requirements.txt ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../../requirements.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_BRIDGE_SRC_ROOT}/src/openvino/__init__.py ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../__init__.py
)

# creates a folder in openvino directory and a symlink to benchmark
# inside bin directory for developers for running python benchmark_app
if(UNIX)
    add_custom_command(TARGET ${TARGET_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../tools
)
    file(COPY ${OpenVINO_MAIN_SOURCE_DIR}/tools/benchmark DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../tools/)
endif()

# install

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION python/${PYTHON_VERSION}/openvino/inference_engine COMPONENT ${PYTHON_VERSION}
        ARCHIVE DESTINATION python/${PYTHON_VERSION}/openvino/inference_engine COMPONENT ${PYTHON_VERSION}
        LIBRARY DESTINATION python/${PYTHON_VERSION}/openvino/inference_engine COMPONENT ${PYTHON_VERSION})

install(PROGRAMS __init__.py
        DESTINATION python/${PYTHON_VERSION}/openvino/inference_engine
        COMPONENT ${PYTHON_VERSION})
