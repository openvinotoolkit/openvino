# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(${CMAKE_CURRENT_SOURCE_DIR}/sources.cmake)

ov_add_frontend(NAME tensorflow
                PROTOBUF_REQUIRED
                LINKABLE_FRONTEND
                FILEDESCRIPTION "FrontEnd to load and convert TensorFlow file format"
                LINK_LIBRARIES openvino::core::dev openvino::frontend::tensorflow_common
                SOURCES ${TENSORFLOW_FRONTEND_SRCS}
                HEADERS ${TENSORFLOW_FRONTEND_HEADERS}
                PUBLIC_HEADERS ${TENSORFLOW_FRONTEND_PUBLIC_HEADERS})

if(ENABLE_SNAPPY_COMPRESSION)
    target_link_libraries(openvino_tensorflow_frontend PRIVATE openvino::snappy)
    target_compile_definitions(openvino_tensorflow_frontend PRIVATE ENABLE_SNAPPY_COMPRESSION)
endif()

ov_build_target_faster(openvino_tensorflow_frontend PCH)

if(Python3_Interpreter_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/docs/check_supported_ops.py
        ${CMAKE_CURRENT_SOURCE_DIR}/src/op_table.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/docs/supported_ops.md
        RESULT_VARIABLE SCRIPT_RESULT
        OUTPUT_VARIABLE SCRIPT_OUTPUT
        ERROR_VARIABLE SCRIPT_ERROR
    )

    if(NOT SCRIPT_RESULT EQUAL 0)
        message(FATAL_ERROR "Python script failed with return code ${SCRIPT_RESULT}\nOutput: ${SCRIPT_OUTPUT}\nError: ${SCRIPT_ERROR}")
    endif()
endif()

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
