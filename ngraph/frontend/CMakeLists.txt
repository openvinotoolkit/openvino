# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT WIN32)
    message(${CMAKE_CURRENT_SOURCE_DIR}/cmake_static_protobuf)
    message(BINARY ${CMAKE_CURRENT_BINARY_DIR})

    # There seems no suitable other way to identify exact output binary name for libprotobuf
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        # Use 'protobufd' directly as it is done in the same way in protobuf cmake files
        set(PROTOBUF_STATIC_LIB_OUTPUT
                ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${CMAKE_STATIC_LIBRARY_PREFIX}protobufd${CMAKE_STATIC_LIBRARY_SUFFIX})
    else(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(PROTOBUF_STATIC_LIB_OUTPUT ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${CMAKE_STATIC_LIBRARY_PREFIX}protobuf${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif()

    message("Static protobuf lib: ${PROTOBUF_STATIC_LIB_OUTPUT}")

    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/libprotobuf_static)
    add_custom_command(
            OUTPUT
            ${PROTOBUF_STATIC_LIB_OUTPUT}
            COMMAND ${CMAKE_COMMAND} ${CMAKE_CURRENT_SOURCE_DIR}/cmake_static_protobuf
                    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
                    -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}
                    -DCMAKE_COMPILE_PDB_OUTPUT_DIRECTORY=${CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY}
                    -DCMAKE_PDB_OUTPUT_DIRECTORY=${CMAKE_PDB_OUTPUT_DIRECTORY}
                    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                    -DCMAKE_CXX_VISIBILITY_PRESET=${CMAKE_CXX_VISIBILITY_PRESET}
                    -DNGRAPH_INSTALL_LIB=${NGRAPH_INSTALL_LIB}
                    -DSYSTEM_PROTOC=${SYSTEM_PROTOC}
                     ${NGRAPH_FORWARD_CMAKE_ARGS}
            COMMAND ${CMAKE_COMMAND} --build . --target libprotobuf
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/libprotobuf_static
            COMMENT "Build Protobuf Static Library"
    )

    add_custom_target(libprotobuf_static
            DEPENDS ${PROTOBUF_STATIC_LIB_OUTPUT})
endif()

add_subdirectory(frontend_manager)

if (NGRAPH_ONNX_IMPORT_ENABLE)
    add_subdirectory(onnx_common)
    add_subdirectory(onnx_import)
endif()

if (NGRAPH_ONNX_EDITOR_ENABLE)
    add_subdirectory(onnx_editor)
endif()

if (NGRAPH_PDPD_FRONTEND_ENABLE)
    add_subdirectory(paddlepaddle)
endif()
