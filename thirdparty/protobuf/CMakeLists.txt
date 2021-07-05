# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#------------------------------------------------------------------------------
# Configure and install Google Protobuf ...
#------------------------------------------------------------------------------

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)

if (MSVC)
    set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "")
endif()

if(CMAKE_CROSSCOMPILING)
    find_program(SYSTEM_PROTOC protoc PATHS ENV PATH)

    if(SYSTEM_PROTOC)
        execute_process(
            COMMAND ${SYSTEM_PROTOC} --version
            OUTPUT_VARIABLE PROTOC_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        string(REPLACE " " ";" PROTOC_VERSION ${PROTOC_VERSION})
        list(GET PROTOC_VERSION -1 PROTOC_VERSION)

        message("Detected system protoc version: ${PROTOC_VERSION}")
    else()
        message(FATAL_ERROR "System Protobuf is needed while cross-compiling")
    endif()

    set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "Build libprotoc and protoc compiler" FORCE)
endif()

if (CMAKE_GENERATOR STREQUAL "Ninja")
    set(MAKE_UTIL make)
else()
    set(MAKE_UTIL $(MAKE))
endif()

set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build tests")
set(protobuf_WITH_ZLIB OFF CACHE BOOL "Build with zlib support")

if (NOT BUILD_STANDALONE_STATIC)
    add_subdirectory(protobuf/cmake EXCLUDE_FROM_ALL)
    get_directory_property(protobuf_VERSION DIRECTORY protobuf/cmake DEFINITION protobuf_VERSION)
endif()
if (USE_STATIC_PROTOBUF)
    include(FetchContent)
    FetchContent_Declare(
            ext_protobuf_static
            URL ${CMAKE_CURRENT_SOURCE_DIR}/protobuf
    )
    FetchContent_GetProperties(ext_protobuf_static)
    if((NOT ext_protobuf_static_POPULATED) AND BUILD_STANDALONE_STATIC)
        FetchContent_Populate(ext_protobuf_static)
        add_subdirectory(${ext_protobuf_static_SOURCE_DIR}/cmake ${ext_protobuf_static_BINARY_DIR} EXCLUDE_FROM_ALL)
        get_directory_property(protobuf_VERSION DIRECTORY ${ext_protobuf_static_SOURCE_DIR}/cmake DEFINITION protobuf_VERSION)
    endif()
endif()

if (BUILD_STANDALONE_STATIC)
    set(Protobuf_INCLUDE_DIRS ${ext_protobuf_static_SOURCE_DIR}/src)
else()
    set(Protobuf_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/src)
endif()

if(NGRAPH_USE_PROTOBUF_LITE)
    set(Protobuf_LIBRARIES libprotobuf-lite)
else()
    set(Protobuf_LIBRARIES libprotobuf)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    set(_proto_libs ${Protobuf_LIBRARIES})
    if(TARGET libprotoc)
        list(APPEND _proto_libs libprotoc)
            target_compile_options(libprotoc PRIVATE -Wno-all -Wno-unused-variable)
    endif()
    set_target_properties(${_proto_libs} PROPERTIES
        CXX_VISIBILITY_PRESET default
        C_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN OFF)
        foreach(target libprotobuf libprotobuf-lite)
            target_compile_options(${target}
                PRIVATE -Wno-all -Wno-unused-variable -Wno-inconsistent-missing-override
                PUBLIC -Wno-undef)
        endforeach()
endif()

if(NGRAPH_USE_PROTOBUF_LITE)
    # if only libprotobuf-lite is used, both libprotobuf and libprotobuf-lite are built
    # libprotoc target needs symbols from libprotobuf, even in libprotobuf-lite configuration
    set_target_properties(libprotobuf PROPERTIES
        CXX_VISIBILITY_PRESET default
        C_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN OFF)
endif()

if(protobuf_VERSION VERSION_LESS "3.9" AND NGRAPH_USE_PROTOBUF_LITE)
    message(FATAL_ERROR "Minimum supported version of protobuf-lite library is 3.9.0")
endif()

if(ENABLE_LTO AND protobuf_VERSION VERSION_GREATER_EQUAL "3.8")
    message(WARNING "Protobuf in version 3.8.0+ can throw runtime exceptions if LTO is enabled.")
endif()

if(CMAKE_CROSSCOMPILING AND NOT PROTOC_VERSION VERSION_EQUAL protobuf_VERSION)
    message(WARNING "system protobuf version does not match with the compiled one, please update system protobuf or submodule")
endif()

if (NOT BUILD_STANDALONE_STATIC)
    message("NGRAPH_INSTALL_LIB = ${NGRAPH_INSTALL_LIB}")
    install(TARGETS ${Protobuf_LIBRARIES}
        RUNTIME DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        ARCHIVE DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph)
    export(TARGETS ${Protobuf_LIBRARIES} NAMESPACE ngraph:: APPEND FILE "${NGRAPH_TARGETS_FILE}")
endif()

# forward variables used in the other places
set(SYSTEM_PROTOC ${SYSTEM_PROTOC} PARENT_SCOPE)
set(Protobuf_LIBRARIES ${Protobuf_LIBRARIES} PARENT_SCOPE)
set(Protobuf_INCLUDE_DIRS ${Protobuf_INCLUDE_DIRS} PARENT_SCOPE)
