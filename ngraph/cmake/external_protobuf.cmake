# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

include(FetchContent)

#------------------------------------------------------------------------------
# Download and install Google Protobuf ...
#------------------------------------------------------------------------------

# Since this file is going to be modifying CMAKE_*_FLAGS we need to preserve
# it so we won't overwrite the caller's CMAKE_*_FLAGS
set(PUSH_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(PUSH_CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
set(PUSH_CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
set(PUSH_CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
set(PUSH_CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
set(PUSH_CMAKE_MODULE_LINKER_FLAGS_RELEASE "${CMAKE_MODULE_LINKER_FLAGS_RELEASE}")

set(CMAKE_CXX_FLAGS ${CMAKE_ORIGINAL_CXX_FLAGS})
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_ORIGINAL_CXX_FLAGS_RELEASE}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_ORIGINAL_C_FLAGS_RELEASE}")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_ORIGINAL_EXE_LINKER_FLAGS_RELEASE}")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_ORIGINAL_SHARED_LINKER_FLAGS_RELEASE}")
set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "${CMAKE_ORIGINAL_MODULE_LINKER_FLAGS_RELEASE}")

if (MSVC)
    string(REPLACE "/W3" "/W0" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error -fno-lto")
endif()

# This version of PROTOBUF is required by Microsoft ONNX Runtime.
set(NGRAPH_PROTOBUF_GIT_REPO_URL "https://github.com/protocolbuffers/protobuf")

if(CMAKE_CROSSCOMPILING)
    find_program(SYSTEM_PROTOC protoc PATHS ENV PATH)

    if(SYSTEM_PROTOC)
        execute_process(COMMAND ${SYSTEM_PROTOC} --version OUTPUT_VARIABLE PROTOC_VERSION)
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


if ("${CMAKE_GENERATOR}" STREQUAL "Ninja")
    set(MAKE_UTIL make)
else()
    set(MAKE_UTIL $(MAKE))
endif()

if(PROTOC_VERSION VERSION_LESS "3.9" AND NGRAPH_USE_PROTOBUF_LITE)
    message(FATAL_ERROR "Minimum supported version of protobuf-lite library is 3.9.0")
else()
    if(CMAKE_CXX_COMPILER_ID MATCHES ".*[Cc]lang")
        include(ExternalProject)
        set(Protobuf_INSTALL_PREFIX ${EXTERNAL_PROJECTS_ROOT}/protobuf)
        set(Protobuf_PROTOC_EXECUTABLE ${Protobuf_INSTALL_PREFIX}/bin/protoc)
        set(Protobuf_INCLUDE_DIRS ${Protobuf_INSTALL_PREFIX}/include)
        set(Protobuf_LIBRARY ${Protobuf_INSTALL_PREFIX}/lib/libprotobuf.a)
        # Don't manually set compiler on macos since it causes compile error on macos >= 10.14
        ExternalProject_Add(
            ext_protobuf
            PREFIX protobuf
            GIT_REPOSITORY ${NGRAPH_PROTOBUF_GIT_REPO_URL}
            GIT_TAG ${NGRAPH_PROTOBUF_GIT_TAG}
            UPDATE_COMMAND ""
            PATCH_COMMAND ""
            CONFIGURE_COMMAND ./autogen.sh COMMAND ./configure --prefix=${EXTERNAL_PROJECTS_ROOT}/protobuf --disable-shared
            BUILD_COMMAND ${MAKE_UTIL} "CXXFLAGS=-std=c++${NGRAPH_CXX_STANDARD} -fPIC"
            TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/tmp"
            STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/stamp"
            DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/download"
            SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/src"
            BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/src"
            INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf"
            EXCLUDE_FROM_ALL TRUE
            BUILD_BYPRODUCTS ${Protobuf_PROTOC_EXECUTABLE} ${Protobuf_LIBRARY}
        )

        # -----------------------------------------------------------------------------
        # Use the interface of FindProtobuf.cmake
        # -----------------------------------------------------------------------------
        if (NOT TARGET protobuf::libprotobuf)
        add_library(protobuf::libprotobuf UNKNOWN IMPORTED)
        set_target_properties(protobuf::libprotobuf PROPERTIES
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${Protobuf_INCLUDE_DIR}"
            IMPORTED_LOCATION "${Protobuf_LIBRARY}")
        add_dependencies(protobuf::libprotobuf ext_protobuf)
        endif()
        set(Protobuf_LIBRARIES protobuf::libprotobuf)

        if (NOT TARGET protobuf::protoc)
        add_executable(protobuf::protoc IMPORTED)
        set_target_properties(protobuf::protoc PROPERTIES
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${Protobuf_PROTOC_EXECUTABLE}"
            IMPORTED_LOCATION "${Protobuf_PROTOC_EXECUTABLE}")
        add_dependencies(protobuf::protoc ext_protobuf)
        endif()

        set(Protobuf_FOUND TRUE)
        set(PROTOBUF_FOUND TRUE)
        #add_dependencies(onnx ext_protobuf)
    else()
        if(PROTOC_VERSION VERSION_GREATER_EQUAL "3.0")
            FetchContent_Declare(
                ext_protobuf
                GIT_REPOSITORY ${NGRAPH_PROTOBUF_GIT_REPO_URL}
                GIT_TAG ${NGRAPH_PROTOBUF_GIT_TAG}
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

        set(Protobuf_INCLUDE_DIRS ${ext_protobuf_SOURCE_DIR})
        if(NGRAPH_USE_PROTOBUF_LITE)
            set(Protobuf_LIBRARIES libprotobuf-lite)
        else()
            set(Protobuf_LIBRARIES libprotobuf)
        endif()
    endif()
endif()

# Now make sure we restore the original CMAKE_*_FLAGS for the caller
set(CMAKE_CXX_FLAGS ${PUSH_CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS_RELEASE "${PUSH_CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_C_FLAGS_RELEASE "${PUSH_CMAKE_C_FLAGS_RELEASE}")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${PUSH_CMAKE_EXE_LINKER_FLAGS_RELEASE}")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${PUSH_CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "${PUSH_CMAKE_MODULE_LINKER_FLAGS_RELEASE}")
