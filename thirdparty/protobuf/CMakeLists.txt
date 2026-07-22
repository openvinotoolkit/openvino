# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#------------------------------------------------------------------------------
# Configure Google Protobuf ...
#------------------------------------------------------------------------------

if(ENABLE_THREAD_SANITIZER AND OV_COMPILER_IS_CLANG)
    string(REPLACE "-ltsan" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
    string(REPLACE "-shared-libsan" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
endif()

set(BUILD_SHARED_LIBS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG OR (OV_COMPILER_IS_INTEL_LLVM AND UNIX))
    ov_add_compiler_flags(-Wno-all)
    if(CMAKE_COMPILER_IS_GNUCXX)
        # GCC may emit "non-constant condition for static assertion" in abseil's
        # ascii.cc because static_cast<char>(i) for i >= 128 is implementation-
        # defined when char is signed, and older GCC constexpr evaluators reject
        # it.  -fpermissive downgrades it to a warning on GCC >= 12.
        # -funsigned-char makes char unsigned so the cast is always in-range,
        # fixing all GCC versions.
        ov_add_compiler_flags(-fpermissive)
        ov_add_compiler_flags(-funsigned-char)
    endif()
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11 AND CMAKE_COMPILER_IS_GNUCXX)
        ov_add_compiler_flags(-Wno-stringop-overflow)
        ov_add_compiler_flags(-Wno-ignored-qualifiers)
        ov_add_compiler_flags(-Wno-deprecated-declarations)
        ov_add_compiler_flags(-Wno-attributes)
        ov_add_compiler_flags(-Wno-sign-conversion)
        ov_add_compiler_flags(-Wno-array-bounds)
    endif()
    if(OV_COMPILER_IS_CLANG)
        ov_add_compiler_flags(-Wno-sign-conversion)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # protobuf\src\google\protobuf\descriptor.cc(822) : error C4703: potentially uninitialized local pointer variable 'to_use' used
    ov_add_compiler_flags(/wd4703)
endif()

set(protobuf_VERBOSE ON)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)
set(protobuf_WITH_ZLIB OFF CACHE BOOL "Build with zlib support" FORCE)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "Abseil protogate CXX standard to dependent targets" FORCE)
# Disable protobuf's own install/export rules.  OpenVINO installs and exports
# libprotobuf/libprotobuf-lite via ov_install_static_lib → OpenVINOTargets.
# Keeping protobuf_INSTALL=ON would cause abseil-cpp.cmake to set
# ABSL_ENABLE_INSTALL=ON (plain local set, overrides any CACHE value), resulting
# in absl targets landing in both abslTargets and OpenVINOTargets — an error in
# CMake's export file generator.  Setting it OFF here (before add_subdirectory)
# prevents option() in protobuf's CMakeLists from re-enabling it.
set(protobuf_INSTALL OFF CACHE BOOL "Disable protobuf self-install (managed by OpenVINO)" FORCE)
# utf8_range.cmake sets utf8_range_ENABLE_INSTALL from protobuf_INSTALL but
# without FORCE, so a stale ON value in the cache survives even when
# protobuf_INSTALL=OFF.  Force it OFF explicitly here so that
# utf8_range-targets.cmake is never generated, keeping utf8_validity in a
# single export set (OpenVINOTargets) and avoiding a CMake multi-export error
# when ONNXTargets also references libprotobuf → utf8_validity.
set(utf8_range_ENABLE_INSTALL OFF CACHE BOOL "Disable utf8_range self-install (managed by OpenVINO)" FORCE)

# some projects define HAVE_ZLIB, which affects protobuf. Let's explicitly unset it
unset(HAVE_ZLIB CACHE)

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

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/protobuf/CMakeLists.txt")
    set(protobuf_dir protobuf)
else()
    set(protobuf_dir protobuf/cmake)
endif()

# Disable warning-as-error for the entire protobuf/abseil subtree.
# Abseil triggers -Warray-bounds on GCC 14 and -Wsign-conversion on
# Emscripten/Clang; these are false positives in third-party code that
# we cannot patch.
set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)

add_subdirectory(${protobuf_dir} EXCLUDE_FROM_ALL)
get_directory_property(protobuf_VERSION DIRECTORY ${protobuf_dir} DEFINITION protobuf_VERSION)

set(Protobuf_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/src)

include(GNUInstallDirs)

# protobuf's cmake/install.cmake normally replaces the bare-path
# INTERFACE_INCLUDE_DIRECTORIES with $<BUILD_INTERFACE:...>/$<INSTALL_INTERFACE:...>.
# Since we set protobuf_INSTALL=OFF that file is never included, so we apply
# the same fix here to avoid a "source-directory path" export error.
foreach(_proto_lib IN ITEMS libprotobuf libprotobuf-lite)
    if(TARGET "${_proto_lib}")
        set_property(TARGET "${_proto_lib}"
            PROPERTY INTERFACE_INCLUDE_DIRECTORIES
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/protobuf/src>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
    endif()
endforeach()
unset(_proto_lib)

# to hide libprotobuf warnings
target_include_directories(libprotobuf-lite SYSTEM PRIVATE "${Protobuf_INCLUDE_DIRS}")

if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG OR OV_COMPILER_IS_INTEL_LLVM)
    if(protobuf_BUILD_PROTOC_BINARIES)
        list(APPEND _protoc_libs protoc libprotoc libprotobuf)
        set_target_properties(${_protoc_libs} PROPERTIES
            CXX_VISIBILITY_PRESET default
            C_VISIBILITY_PRESET default
            VISIBILITY_INLINES_HIDDEN OFF
            INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)
    endif()
    # Disable thread sanitizer for all protobuf/abseil targets — protobuf
    # is incompatible with -fsanitize=thread (especially under clang).
    if(ENABLE_THREAD_SANITIZER)
        foreach(_lib protoc libprotoc libprotobuf libprotobuf-lite)
            if(TARGET ${_lib})
                target_compile_options(${_lib} PRIVATE -fno-sanitize=thread)
                target_link_options(${_lib} PRIVATE -fno-sanitize=thread)
            endif()
        endforeach()
    endif()
    ov_disable_all_warnings(${_protoc_libs} libprotobuf-lite)
    set_target_properties(libprotobuf-lite PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
endif()

if(protobuf_VERSION VERSION_LESS 3.9)
    message(FATAL_ERROR "Minimum supported version of protobuf-lite library is 3.9.0 (provided ${protobuf_VERSION})")
endif()

# build protoc separatelly for build system processor

if(NOT protobuf_BUILD_PROTOC_BINARIES)
    set(HOST_PROTOC_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/install")

    if(ENABLE_THREAD_SANITIZER)
        set(protoc_extra_flags "-fno-sanitize=thread")
    endif()

    if(CMAKE_COMPILER_IS_GNUCXX)
        # GCC < 9 may emit "overflow in constant expression" inside bundled
        # abseil-cpp (ascii.cc static_assert) and treats it as a hard error.
        # -fpermissive downgrades it to a warning, allowing the build to proceed.
        set(protoc_extra_flags "${protoc_extra_flags} -fpermissive")
    endif()

    ov_native_compile_external_project(
        TARGET_NAME host_protoc
        NATIVE_INSTALL_DIR "${HOST_PROTOC_INSTALL_DIR}"
        EXTRA_COMPILE_FLAGS "${protoc_extra_flags}"
        CMAKE_ARGS "-Dprotobuf_VERBOSE=${protobuf_VERBOSE}"
                   "-Dprotobuf_BUILD_TESTS=${protobuf_BUILD_TESTS}"
                   "-Dprotobuf_WITH_ZLIB=${protobuf_WITH_ZLIB}"
        NATIVE_SOURCE_SUBDIR "${protobuf_dir}"
        NATIVE_TARGETS protoc libprotobuf-lite)

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
