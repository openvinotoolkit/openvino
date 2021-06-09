# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(FetchContent)

#------------------------------------------------------------------------------
# Download and install Google Protobuf ...
#------------------------------------------------------------------------------

set(PUSH_CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE "${CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE}")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)

if (MSVC)
    set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "")
endif()

# This version of PROTOBUF is required by Microsoft ONNX Runtime.
set(NGRAPH_PROTOBUF_GIT_REPO_URL "https://github.com/protocolbuffers/protobuf")

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

        if(${PROTOC_VERSION} VERSION_EQUAL "3.0.0")
            message(WARNING "Protobuf 3.0.0 detected switching to 3.0.2 due to bug in gmock url")
            set(PROTOC_VERSION "3.0.2")
        endif()
    else()
        message(FATAL_ERROR "System Protobuf is needed while cross-compiling")
    endif()

    set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "Build libprotoc and protoc compiler" FORCE)
elseif(NGRAPH_USE_PROTOBUF_LITE)
    set(PROTOC_VERSION "3.9.2")
    if(ENABLE_LTO)
        message(WARNING "Protobuf in version 3.8.0+ can throw runtime exceptions if LTO is enabled.")
    endif()
else()
    set(PROTOC_VERSION "3.7.1")
endif()

set(NGRAPH_PROTOBUF_GIT_TAG "v${PROTOC_VERSION}")


if (CMAKE_GENERATOR STREQUAL "Ninja")
    set(MAKE_UTIL make)
else()
    set(MAKE_UTIL $(MAKE))
endif()

if(PROTOC_VERSION VERSION_LESS "3.9" AND NGRAPH_USE_PROTOBUF_LITE)
    message(FATAL_ERROR "Minimum supported version of protobuf-lite library is 3.9.0")
else()
    if(PROTOC_VERSION VERSION_GREATER_EQUAL "3.0")
        FetchContent_Declare(
            ext_protobuf
            GIT_REPOSITORY ${NGRAPH_PROTOBUF_GIT_REPO_URL}
            GIT_TAG ${NGRAPH_PROTOBUF_GIT_TAG}
            GIT_SHALLOW TRUE
        )

        FetchContent_GetProperties(ext_protobuf)
        if(NOT ext_protobuf_POPULATED)
            FetchContent_Populate(ext_protobuf)
            set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build tests")
            set(protobuf_WITH_ZLIB OFF CACHE BOOL "Build with zlib support")
            add_subdirectory(${ext_protobuf_SOURCE_DIR}/cmake ${ext_protobuf_BINARY_DIR} EXCLUDE_FROM_ALL)
        endif()
    else()
        message(FATAL_ERROR "Minimum supported version of protobuf library is 3.0.0")
    endif()

    set(Protobuf_INCLUDE_DIRS ${ext_protobuf_SOURCE_DIR}/src)
    if(NGRAPH_USE_PROTOBUF_LITE)
        set(Protobuf_LIBRARIES libprotobuf-lite)
    else()
        set(Protobuf_LIBRARIES libprotobuf)
    endif()

    if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
        set(_proto_libs ${Protobuf_LIBRARIES})
        if(TARGET libprotoc)
            list(APPEND _proto_libs libprotoc)
            set_target_properties(libprotoc PROPERTIES
                COMPILE_FLAGS "-Wno-all -Wno-unused-variable")
        endif()
        set_target_properties(${_proto_libs} PROPERTIES
            CXX_VISIBILITY_PRESET default
            C_VISIBILITY_PRESET default
            VISIBILITY_INLINES_HIDDEN OFF)
        set_target_properties(libprotobuf libprotobuf-lite PROPERTIES
            COMPILE_FLAGS "-Wno-all -Wno-unused-variable -Wno-inconsistent-missing-override")
    endif()

    if(NGRAPH_USE_PROTOBUF_LITE)
        # if only libprotobuf-lite is used, both libprotobuf and libprotobuf-lite are built
        # libprotoc target needs symbols from libprotobuf, even in libprotobuf-lite configuration
        set_target_properties(libprotobuf PROPERTIES
            CXX_VISIBILITY_PRESET default
            C_VISIBILITY_PRESET default
            VISIBILITY_INLINES_HIDDEN OFF)
    endif()
endif()

# Now make sure we restore the original flags
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE "${PUSH_CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE}")

install(TARGETS ${Protobuf_LIBRARIES}
    RUNTIME DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
    ARCHIVE DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
    LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph)
if (NGRAPH_EXPORT_TARGETS_ENABLE)
    export(TARGETS ${Protobuf_LIBRARIES} NAMESPACE ngraph:: APPEND FILE "${NGRAPH_TARGETS_FILE}")
endif()
