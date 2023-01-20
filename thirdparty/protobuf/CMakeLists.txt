# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#------------------------------------------------------------------------------
# Configure Google Protobuf ...
#------------------------------------------------------------------------------

set(BUILD_SHARED_LIBS OFF)

if(SUGGEST_OVERRIDE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
endif()

if(OV_COMPILER_IS_CLANG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-inconsistent-missing-override")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # protobuf\src\google\protobuf\descriptor.cc(822) : error C4703: potentially uninitialized local pointer variable 'to_use' used
    add_definitions("/wd4703")
endif()

set(protobuf_VERBOSE ON)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)
set(protobuf_WITH_ZLIB OFF CACHE BOOL "Build with zlib support" FORCE)

# note: HOST_AARCH64 AND X86_64 are not handled for Apple explicitly, because it can work via Rosetta
if(CMAKE_CROSSCOMPILING OR
    (APPLE AND (HOST_X86_64 AND AARCH64)) OR
    (MSVC AND (HOST_X86_64 AND (AARCH64 OR ARM))))
    set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "Build protoc binaries" FORCE)
else()
    set(protobuf_BUILD_PROTOC_BINARIES ON CACHE BOOL "Build protoc binaries" FORCE)
endif()

# When we build dll libraries. These flags make sure onnx and protobuf build with /MD, not /MT.
# These two options can't be mixed, because they requires link two incompatible runtime.
if(NOT DEFINED protobuf_MSVC_STATIC_RUNTIME)
    set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "Link protobuf to static runtime libraries" FORCE)
endif()

add_subdirectory(protobuf/cmake EXCLUDE_FROM_ALL)
get_directory_property(protobuf_VERSION DIRECTORY protobuf/cmake DEFINITION protobuf_VERSION)

set(Protobuf_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/src)

# to hide libprotobuf warnings
target_include_directories(libprotobuf-lite SYSTEM PRIVATE "${Protobuf_INCLUDE_DIRS}")

if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG OR
   CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    if(protobuf_BUILD_PROTOC_BINARIES)
        list(APPEND _protoc_libs protoc libprotoc libprotobuf)
        set_target_properties(${_protoc_libs} PROPERTIES
            CXX_VISIBILITY_PRESET default
            C_VISIBILITY_PRESET default
            VISIBILITY_INLINES_HIDDEN OFF
            INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)
    endif()
    ov_disable_all_warnings(${_protoc_libs} libprotobuf-lite)
    set_target_properties(libprotobuf-lite PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
endif()

if(protobuf_VERSION VERSION_LESS "3.9")
    message(FATAL_ERROR "Minimum supported version of protobuf-lite library is 3.9.0 (provided ${protobuf_VERSION})")
endif()

# build protoc separatelly for build system processor

if(NOT protobuf_BUILD_PROTOC_BINARIES)
<<<<<<< HEAD
    set(HOST_PROTOC_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/protobuf/cmake")
    set(HOST_PROTOC_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/host_protoc_build")
    set(HOST_PROTOC_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/host_protoc_install")
    set(HOST_PROTOC_PREFIX_DIR "${CMAKE_CURRENT_BINARY_DIR}/host_protoc_root")

    ov_native_compile_external_project(
        TARGET_NAME host_protoc
        NATIVE_SOURCE_DIR "${HOST_PROTOC_SOURCE_DIR}"
        NATIVE_BUILD_DIR "${HOST_PROTOC_BUILD_DIR}"
        NATIVE_INSTALL_DIR "${HOST_PROTOC_INSTALL_DIR}"
        NATIVE_PREFIX_DIR "${HOST_PROTOC_PREFIX_DIR}"
        CMAKE_OPTIONS "-Dprotobuf_VERBOSE=${protobuf_VERBOSE}"
                      "-Dprotobuf_BUILD_TESTS=${protobuf_BUILD_TESTS}"
                      "-Dprotobuf_WITH_ZLIB=${protobuf_WITH_ZLIB}")
=======
    include(ExternalProject)

    if(YOCTO_AARCH64)
        # need to unset several variables which can set env to cross-environment
        foreach(var SDKTARGETSYSROOT CONFIG_SITE OECORE_NATIVE_SYSROOT OECORE_TARGET_SYSROOT
                    OECORE_ACLOCAL_OPTS OECORE_BASELIB OECORE_TARGET_ARCH OECORE_TARGET_OS CC CXX
                    CPP AS LD GDB STRIP RANLIB OBJCOPY OBJDUMP READELF AR NM M4 TARGET_PREFIX
                    CONFIGURE_FLAGS CFLAGS CXXFLAGS LDFLAGS CPPFLAGS KCFLAGS OECORE_DISTRO_VERSION
                    OECORE_SDK_VERSION ARCH CROSS_COMPILE OE_CMAKE_TOOLCHAIN_FILE OPENSSL_CONF
                    OE_CMAKE_FIND_LIBRARY_CUSTOM_LIB_SUFFIX PKG_CONFIG_SYSROOT_DIR PKG_CONFIG_PATH)
            if(DEFINED ENV{${var}})
                list(APPEND cmake_env --unset=${var})
            endif()
        endforeach()

        # filter out PATH from yocto locations
        string(REPLACE ":" ";" custom_path "$ENV{PATH}")
        foreach(path IN LISTS custom_path)
            if(NOT path MATCHES "^$ENV{OECORE_NATIVE_SYSROOT}")
                list(APPEND clean_path "${path}")
            endif()
        endforeach()

        find_host_program(HOST_CMAKE_COMMAND
            NAMES cmake
            PATHS ${clean_path}
            DOC "Host cmake"
            REQUIRED
            NO_DEFAULT_PATH)
    else()
        set(HOST_CMAKE_COMMAND "${CMAKE_COMMAND}")
    endif()

    # if env has CMAKE_TOOLCHAIN_FILE, we need to skip it
    if(DEFINED ENV{CMAKE_TOOLCHAIN_FILE})
        list(APPEND cmake_env --unset=CMAKE_TOOLCHAIN_FILE)
    endif()

    # compile flags
    if(CMAKE_COMPILER_IS_GNUCXX)
        set(compile_flags "-Wno-undef -Wno-error -Wno-deprecated-declarations")
    endif()

    set(HOST_PROTOC_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/build")
    set(HOST_PROTOC_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/install")
    set(HOST_PROTOC_CMAKE_COMMAND "${CMAKE_COMMAND}" -E env ${cmake_env} "${HOST_CMAKE_COMMAND}")

    ExternalProject_Add(host_protoc
        # Directory Options
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
        PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
        BINARY_DIR "${HOST_PROTOC_BINARY_DIR}"
        INSTALL_DIR "${HOST_PROTOC_INSTALL_DIR}"
        # Configure Step Options:
        CMAKE_COMMAND
            ${HOST_PROTOC_CMAKE_COMMAND}
        CMAKE_ARGS
            "-DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}"
            "-DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}"
            "-DCMAKE_CXX_LINKER_LAUNCHER=${CMAKE_CXX_LINKER_LAUNCHER}"
            "-DCMAKE_C_LINKER_LAUNCHER=${CMAKE_C_LINKER_LAUNCHER}"
            "-DCMAKE_CXX_FLAGS=${compile_flags}"
            "-DCMAKE_C_FLAGS=${compile_flags}"
            "-DCMAKE_POLICY_DEFAULT_CMP0069=NEW"
            "-DCMAKE_INSTALL_PREFIX=${HOST_PROTOC_INSTALL_DIR}"
            "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
            # protoc specific
            "-DTHREADS_PREFER_PTHREAD_FLAG=${THREADS_PREFER_PTHREAD_FLAG}"
            "-Dprotobuf_VERBOSE=${protobuf_VERBOSE}"
            "-Dprotobuf_BUILD_TESTS=${protobuf_BUILD_TESTS}"
            "-Dprotobuf_WITH_ZLIB=${protobuf_WITH_ZLIB}"
        CMAKE_GENERATOR "${CMAKE_GENERATOR}"
        SOURCE_SUBDIR
            "protobuf/cmake"
        # Build Step Options:
        BUILD_COMMAND
            ${HOST_PROTOC_CMAKE_COMMAND}
                --build "${HOST_PROTOC_BINARY_DIR}"
                --config Release
                -- protoc libprotobuf-lite
        # Test Step Options:
        TEST_EXCLUDE_FROM_MAIN ON
        # Target Options:
        EXCLUDE_FROM_ALL ON
    )
>>>>>>> upstream/master

    set(PROTOC_EXECUTABLE "${HOST_PROTOC_INSTALL_DIR}/bin/protoc")
    add_executable(protobuf::protoc IMPORTED GLOBAL)
    set_property(TARGET protobuf::protoc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
    set_target_properties(protobuf::protoc PROPERTIES
        IMPORTED_LOCATION_RELEASE "${PROTOC_EXECUTABLE}")
    set_target_properties(protobuf::protoc PROPERTIES
        MAP_IMPORTED_CONFIG_DEBUG Release
        MAP_IMPORTED_CONFIG_MINSIZEREL Release
        MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
    add_dependencies(protobuf::protoc host_protoc)

    set(PROTOC_DEPENDENCY host_protoc PARENT_SCOPE)
    set(PROTOC_EXECUTABLE "${PROTOC_EXECUTABLE}" PARENT_SCOPE)
else()
    set(PROTOC_EXECUTABLE $<TARGET_FILE:protoc> PARENT_SCOPE)
    set(PROTOC_DEPENDENCY protoc PARENT_SCOPE)
endif()

# set to parent scope

set(protobuf_VERSION ${protobuf_VERSION} PARENT_SCOPE)
set(Protobuf_LITE_LIBRARIES libprotobuf-lite PARENT_SCOPE)
set(Protobuf_LIBRARIES libprotobuf PARENT_SCOPE)
set(Protobuf_INCLUDE_DIRS ${Protobuf_INCLUDE_DIRS} PARENT_SCOPE)
