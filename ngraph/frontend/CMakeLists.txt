# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT WIN32)
    # DEBUG - print all variables
    # get_cmake_property(_variableNames VARIABLES)
    # set(ALL_VARS "")
    # foreach (_variableName ${_variableNames})
    #     set(ALL_VARS ${ALL_VARS} -D${_variableName}=${${_variableName}}\ )
    # endforeach()
    # message(---------------------ALL VARS: ${ALL_VARS}-------)

    message(${CMAKE_CURRENT_SOURCE_DIR}/cmake_static_protobuf)
    message(BINARY ${CMAKE_CURRENT_BINARY_DIR})

    # There seems no suitable other way to identify exact output binary name for libprotobuf
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(PROTOBUF_STATIC_LIB_OUTPUT ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/libprotobufd.a)
    else(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(PROTOBUF_STATIC_LIB_OUTPUT ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/libprotobuf.a)
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
                     ${NGRAPH_FORWARD_CMAKE_ARGS}
            COMMAND ${CMAKE_COMMAND} --build . --target libprotobuf
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/libprotobuf_static
            COMMENT "Build Protobuf Static Library"
            # TODO: add DEPENDS - not clear what shall be the input for this command
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
