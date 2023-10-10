# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CMAKE_TOOLCHAIN_FILE MATCHES "vcpkg" OR DEFINED VCPKG_VERBOSE)
    set(OV_VCPKG_BUILD ON)
elseif(CMAKE_TOOLCHAIN_FILE MATCHES "conan_toolchain" OR DEFINED CONAN_EXPORTED)
    set(OV_CONAN_BUILD)
endif()

set(_old_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(_old_CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ${CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE})

find_package(PkgConfig QUIET)
# see https://cmake.org/cmake/help/latest/command/add_library.html#alias-libraries
# cmake older than 3.18 cannot create an alias for imported non-GLOBAL targets
# so, we have to use 'IMPORTED_GLOBAL' property
if(CMAKE_VERSION VERSION_LESS 3.18)
    set(OV_PkgConfig_VISILITY GLOBAL)
endif()

if(SUGGEST_OVERRIDE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
endif()

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

if(ENABLE_PROFILING_ITT)
    find_package(ittapi QUIET)
    if(ittapi_FOUND)
        if(TARGET ittapi::ittapi)
            # conan defines 'ittapi::ittapi' target
            set_target_properties(ittapi::ittapi PROPERTIES
                INTERFACE_COMPILE_DEFINITIONS ENABLE_PROFILING_ITT)
        elseif(TARGET ittapi::ittnotify)
            # official 'ittapi::ittnotify' target
            set_target_properties(ittapi::ittnotify PROPERTIES
                INTERFACE_COMPILE_DEFINITIONS ENABLE_PROFILING_ITT)
        endif()
    else()
        add_subdirectory(thirdparty/ittapi)
    endif()
    add_subdirectory(thirdparty/itt_collector EXCLUDE_FROM_ALL)
endif()

if(X86_64 OR X86 OR UNIVERSAL2)
    find_package(xbyak QUIET)
    if(xbyak_FOUND)
        # conan creates alias xbyak::xbyak, no extra steps are required
    else()
        add_subdirectory(thirdparty/xbyak EXCLUDE_FROM_ALL)
        # export and install xbyak
        openvino_developer_export_targets(COMPONENT openvino_common TARGETS xbyak::xbyak)
        ov_install_static_lib(xbyak ${OV_CPACK_COMP_CORE})
    endif()
endif()

#
# OpenCL
#

if(ENABLE_INTEL_GPU)
    if(ENABLE_SYSTEM_OPENCL)
        # try to find system OpenCL:
        # - 'apt-get install opencl-headers ocl-icd-opencl-dev'
        # - 'yum install ocl-icd-devel opencl-headers'
        # - 'conda install khronos-opencl-icd-loader -c conda-forge'
        # - 'vcpkg install opencl:<triplet>'
        # - 'conan install opencl-headers opencl-clhpp-headers opencl-icd-loader'
        # - 'brew install opencl-headers opencl-clhpp-headers opencl-icd-loader'
        find_package(OpenCL QUIET)
    endif()

    if(TARGET OpenCL::OpenCL)
        # try to find CL/opencl.hpp
        find_file(OpenCL_HPP
                  NAMES CL/opencl.hpp OpenCL/opencl.hpp
                  HINTS ${OpenCL_INCLUDE_DIRS} ${opencl_cpp_include_dirs}
                  DOC "Path to CL/opencl.hpp")

        # add definition to select proper header and suppress warnings
        if(OpenCL_HPP)
            set(opencl_interface_definitions OV_GPU_USE_OPENCL_HPP)

            # check whether CL/opencl.hpp contains C++ wrapper for property CL_DEVICE_UUID_KHR
            file(STRINGS "${OpenCL_HPP}" CL_DEVICE_UUID_KHR_CPP REGEX ".*CL_DEVICE_UUID_KHR.*")
            if(CL_DEVICE_UUID_KHR_CPP)
                list(APPEND opencl_interface_definitions OV_GPU_OPENCL_HPP_HAS_UUID)
            endif()

            set_target_properties(OpenCL::OpenCL PROPERTIES
                INTERFACE_COMPILE_DEFINITIONS "${opencl_interface_definitions}")
        endif()
    else()
        add_subdirectory(thirdparty/ocl)
    endif()

    # cmake cannot set properties for imported targets
    get_target_property(opencl_target OpenCL::OpenCL ALIASED_TARGET)
    if(NOT TARGET ${opencl_target})
        set(opencl_target OpenCL::OpenCL)
    endif()

    if(SUGGEST_OVERRIDE_SUPPORTED)
        set_target_properties(${opencl_target} PROPERTIES INTERFACE_COMPILE_OPTIONS
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-suggest-override>)
    endif()

    # used in tests
    add_library(opencl_new_headers INTERFACE)
    add_library(OpenCL::NewHeaders ALIAS opencl_new_headers)
    foreach(opencl_dir "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ocl/clhpp_headers/include"
                       "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ocl/cl_headers")
        if(EXISTS "${opencl_dir}")
            set_property(TARGET opencl_new_headers APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                $<BUILD_INTERFACE:${opencl_dir}>)
            set_target_properties(opencl_new_headers PROPERTIES
                INTERFACE_COMPILE_DEFINITIONS OV_GPU_USE_OPENCL_HPP)
        endif()
    endforeach()
endif()

#
# zlib
#

if(ENABLE_SAMPLES OR ENABLE_TESTS)
    find_package(ZLIB QUIET)
    if(ZLIB_FOUND)
        # FindZLIB module defines ZLIB::ZLIB, no extra steps are required
    endif()

    # cmake has failed to find zlib, let's try pkg-config
    if(NOT ZLIB_FOUND AND PkgConfig_FOUND)
        pkg_search_module(zlib QUIET
                          IMPORTED_TARGET
                          zlib)
        if(zlib_FOUND)
            add_library(ZLIB::ZLIB INTERFACE IMPORTED)
            set_target_properties(ZLIB::ZLIB PROPERTIES INTERFACE_LINK_LIBRARIES PkgConfig::zlib)
            message(STATUS "${PKG_CONFIG_EXECUTABLE}: zlib (${zlib_VERSION}) is found at ${zlib_PREFIX}")
        endif()
    endif()

    if(NOT (zlib_FOUND OR ZLIB_FOUND))
        add_subdirectory(thirdparty/zlib EXCLUDE_FROM_ALL)
    endif()
endif()

#
# cnpy
#

if(ENABLE_SAMPLES OR ENABLE_TESTS)
    add_subdirectory(thirdparty/cnpy EXCLUDE_FROM_ALL)
endif()

#
# Pugixml
#

if(ENABLE_SYSTEM_PUGIXML)
    # try system pugixml first
    # Note: we also specify 'pugixml' in NAMES because vcpkg
    find_package(PugiXML QUIET NAMES PugiXML pugixml)
    if(PugiXML_FOUND)
        # TODO: use static pugixml library in case of BUILD_SHARED_LIBS=OFF
        if(TARGET pugixml::shared)
            # example: cross-compilation on debian
            set(pugixml_target pugixml::shared)
        elseif(TARGET pugixml::pugixml)
            # or create an alias for pugixml::pugixml shared library
            # - 'brew install pugixml'
            # - 'conan install pugixml'
            set(pugixml_target pugixml::pugixml)
        elseif(TARGET pugixml)
            # or create an alias for pugixml shared library
            # - 'apt-get install libpugixml-dev'
            set(pugixml_target pugixml)
        elseif(TARGET pugixml::static)
            # sometimes pugixml::static target already exists, just need to create an alias
            # - 'conda install pugixml -c conda-forge'
            set(pugixml_target pugixml::static)
        else()
            message(FATAL_ERROR "Failed to detect pugixml library target name")
        endif()
        # to property generate OpenVINO Developer packages files
        set(PugiXML_FOUND ${PugiXML_FOUND} CACHE BOOL "" FORCE)
    elseif(PkgConfig_FOUND)
        # Ubuntu 18.04 case when cmake interface is not available
        pkg_search_module(pugixml QUIET
                          IMPORTED_TARGET
                          ${OV_PkgConfig_VISILITY}
                          pugixml)
        if(pugixml_FOUND)
            set(pugixml_target PkgConfig::pugixml)
            # PATCH: on Ubuntu 18.04 pugixml.pc contains incorrect include directories
            get_target_property(interface_include_dir ${pugixml_target} INTERFACE_INCLUDE_DIRECTORIES)
            if(interface_include_dir AND NOT EXISTS "${interface_include_dir}")
                set_target_properties(${pugixml_target} PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES "")
            endif()
            message(STATUS "${PKG_CONFIG_EXECUTABLE}: pugixml (${pugixml_VERSION}) is found at ${pugixml_PREFIX}")
        endif()
    endif()

    # debian 9 case: no cmake, no pkg-config files
    if(NOT TARGET ${pugixml_target})
        find_library(PUGIXML_LIBRARY NAMES pugixml DOC "Path to pugixml library")
        if(PUGIXML_LIBRARY)
            add_library(pugixml INTERFACE IMPORTED)
            set_target_properties(pugixml PROPERTIES INTERFACE_LINK_LIBRARIES "${PUGIXML_LIBRARY}")
            set(pugixml_target pugixml)
            set(PugiXML_FOUND ON)
            # because we don't need to have a dependency on specific cmake targets in this case
            # in file OpenVINOConfig.cmake static build case
            set(ENABLE_SYSTEM_PUGIXML OFF)
        endif()
    endif()

    if(TARGET ${pugixml_target})
        # we need to install dynamic library for wheel package
        get_target_property(target_type ${pugixml_target} TYPE)
        if(target_type STREQUAL "SHARED_LIBRARY")
            get_target_property(imported_configs ${pugixml_target} IMPORTED_CONFIGURATIONS)
            foreach(imported_config RELEASE RELWITHDEBINFO DEBUG NONE ${imported_configs})
                if(imported_config IN_LIST imported_configs)
                    get_target_property(pugixml_loc ${pugixml_target} IMPORTED_LOCATION_${imported_config})
                    break()
                endif()
            endforeach()
            get_filename_component(pugixml_dir "${pugixml_loc}" DIRECTORY)
            get_filename_component(name_we "${pugixml_loc}" NAME_WE)
            # grab all tbb files matching pattern
            file(GLOB pugixml_files "${pugixml_dir}/${name_we}.*")
            foreach(pugixml_file IN LISTS pugixml_files)
                ov_install_with_name("${pugixml_file}" pugixml)
            endforeach()
        elseif(target_type STREQUAL "INTERFACE_LIBRARY")
            get_target_property(pugixml_loc ${pugixml_target} INTERFACE_LINK_LIBRARIES)
            file(GLOB pugixml_libs "${pugixml_loc}.*")
            foreach(pugixml_lib IN LISTS pugixml_libs)
                ov_install_with_name("${pugixml_lib}" pugixml)
            endforeach()
        endif()

        # if dynamic libpugixml.so.1 and libpugixml.so.1.X are found
        if(NOT pugixml_INSTALLED AND CPACK_GENERATOR MATCHES "^(DEB|RPM)$")
            message(FATAL_ERROR "Debian | RPM package build requires shared Pugixml library")
        endif()

        if(OV_PkgConfig_VISILITY)
            # need to set GLOBAL visibility in order to create ALIAS for this target
            set_target_properties(${pugixml_target} PROPERTIES IMPORTED_GLOBAL ON)
        endif()
        # create an alias for real target which can be shared or static
        add_library(openvino::pugixml ALIAS ${pugixml_target})
    else()
        # reset to prevent improper code generation in OpenVINODeveloperPackage.cmake,
        # and OpenVINOConfig.cmake for static case
        set(ENABLE_SYSTEM_PUGIXML OFF)
    endif()
endif()

if(NOT TARGET openvino::pugixml)
    # use OpenVINO pugixml copy if system one is not found
    function(ov_build_pugixml)
        function(ov_build_pugixml_static)
            set(BUILD_SHARED_LIBS OFF)
            add_subdirectory(thirdparty/pugixml EXCLUDE_FROM_ALL)
        endfunction()
        ov_build_pugixml_static()
        set_property(TARGET pugixml-static PROPERTY EXPORT_NAME pugixml)
        add_library(openvino::pugixml ALIAS pugixml-static)
        openvino_developer_export_targets(COMPONENT openvino_common TARGETS openvino::pugixml)
        ov_install_static_lib(pugixml-static ${OV_CPACK_COMP_CORE})
    endfunction()

    ov_build_pugixml()
endif()

#
# Fluid, G-API, OpenCV HAL
#

if(ENABLE_GAPI_PREPROCESSING)
    add_library(ocv_hal INTERFACE)
    target_include_directories(ocv_hal INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ocv")

    # ade
    find_package(ade 0.1.0 QUIET)
    if(ade_FOUND)
        # conan and vcpkg create 'ade' target
        # we just need to remove non-numerical symbols from version,
        # because conan defines it as 0.1.2a, which is invalid in cmake
        string(REGEX REPLACE "[a-z]" "" ade_VERSION "${ade_VERSION}")
    else()
        add_subdirectory(thirdparty/ade EXCLUDE_FROM_ALL)

        set_target_properties(ade PROPERTIES FOLDER thirdparty)
        openvino_developer_export_targets(COMPONENT openvino_common TARGETS ade)

        ov_install_static_lib(ade ${OV_CPACK_COMP_CORE})
    endif()

    # fluid
    add_subdirectory(thirdparty/fluid/modules/gapi EXCLUDE_FROM_ALL)

    if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
        target_compile_options(fluid PRIVATE "-Wno-maybe-uninitialized")
    endif()
    if(UNUSED_BUT_SET_VARIABLE_SUPPORTED)
        target_compile_options(fluid PRIVATE "-Wno-unused-but-set-variable")
    endif()

    set_target_properties(fluid PROPERTIES FOLDER thirdparty)
    openvino_developer_export_targets(COMPONENT openvino_common TARGETS fluid)

    ov_install_static_lib(fluid ${OV_CPACK_COMP_CORE})
endif()

#
# Gflags
#

if(ENABLE_SAMPLES OR ENABLE_TESTS)
    if(OV_VCPKG_BUILD OR OV_CONAN_BUILD)
        # vcpkg contains only libs compiled with threads
        # conan case
        find_package(gflags QUIET)
    elseif(APPLE OR WIN32)
        # on Windows and macOS we don't use gflags, because will be dynamically linked
    elseif(CMAKE_HOST_LINUX AND LINUX)
        if(OV_OS_RHEL)
            set(gflag_component nothreads_shared)
        elseif(OV_OS_DEBIAN)
            set(gflag_component nothreads_static)
        endif()
        find_package(gflags QUIET OPTIONAL_COMPONENTS ${gflag_component})
    endif()

    if(gflags_FOUND)
        if(TARGET gflags)
            # no extra steps
        elseif(TARGET gflags_nothreads-static)
            # Debian 9: gflag_component is ignored
            set(gflags_target gflags_nothreads-static)
        elseif(TARGET gflags_nothreads-shared)
            # CentOS / RHEL / Fedora case
            set(gflags_target gflags_nothreads-shared)
        elseif(TARGET ${GFLAGS_TARGET})
            set(gflags_target ${GFLAGS_TARGET})
        else()
            message(FATAL_ERROR "Internal error: failed to find imported target 'gflags' using '${gflag_component}' component")
        endif()

        if(gflags_target)
            if(OV_PkgConfig_VISILITY)
                # need to set GLOBAL visibility in order to create ALIAS for this target
                set_target_properties(${gflags_target} PROPERTIES IMPORTED_GLOBAL ON)
            endif()
            add_library(gflags ALIAS ${gflags_target})
        endif()

        message(STATUS "gflags (${gflags_VERSION}) is found at ${gflags_DIR} using '${gflag_component}' component")
    endif()

    if(NOT TARGET gflags)
        add_subdirectory(thirdparty/gflags EXCLUDE_FROM_ALL)
        openvino_developer_export_targets(COMPONENT openvino_common TARGETS gflags)
    endif()
endif()

#
# Google Tests framework
#

if(ENABLE_TESTS)
    # TODO: migrate to official version of googltest
    # find_package(GTest QUIET)

    if(GTest_FOUND)
        foreach(gtest_target gtest gtest_main gmock gmock_main)
            if(OV_PkgConfig_VISILITY)
                # need to set GLOBAL visibility in order to create ALIAS for this target
                set_target_properties(GTest::${gtest_target} PROPERTIES IMPORTED_GLOBAL ON)
            endif()
            add_library(${gtest_target} ALIAS GTest::${gtest_target})
        endforeach()
    else()
        add_subdirectory(thirdparty/gtest EXCLUDE_FROM_ALL)
        openvino_developer_export_targets(COMPONENT tests
                                          TARGETS gmock gmock_main gtest gtest_main)
    endif()
endif()

#
# Protobuf
#

if(ENABLE_OV_PADDLE_FRONTEND OR ENABLE_OV_ONNX_FRONTEND OR ENABLE_OV_TF_FRONTEND)
    if(ENABLE_SYSTEM_PROTOBUF)
        # Note: Debian / Ubuntu / RHEL libprotobuf.a can only be used with -DBUILD_SHARED_LIBS=OFF
        # because they are compiled without -fPIC
        if(NOT DEFINED Protobuf_USE_STATIC_LIBS)
            set(Protobuf_USE_STATIC_LIBS ON)
        endif()
        if(CMAKE_VERBOSE_MAKEFILE)
            set(Protobuf_DEBUG ON)
        endif()
        if(OV_VCPKG_BUILD)
            set(protobuf_config CONFIG)
        endif()
        # try to find newer version first (major is changed)
        # see https://protobuf.dev/support/version-support/ and
        # https://github.com/protocolbuffers/protobuf/commit/d61f75ff6db36b4f9c0765f131f8edc2f86310fa
        find_package(Protobuf 4.22.0 QUIET ${protobuf_config})
        if(NOT Protobuf_FOUND)
            # otherwise, fallback to existing default
            find_package(Protobuf 3.20.3 REQUIRED ${protobuf_config})
        endif()
        set(PROTOC_EXECUTABLE protobuf::protoc)
    else()
        add_subdirectory(thirdparty/protobuf EXCLUDE_FROM_ALL)
    endif()

    # forward additional variables used in the other places
    set(Protobuf_IN_FRONTEND ON)

    # set public / interface compile options
    function(_ov_fix_protobuf_warnings target_name)
        set(link_type PUBLIC)
        if(ENABLE_SYSTEM_PROTOBUF)
            set(link_type INTERFACE)
        endif()
        if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
            get_target_property(original_name ${target_name} ALIASED_TARGET)
            if(TARGET ${original_name})
                # during build protobuf's cmake creates aliased targets
                set(target_name ${original_name})
            endif()
            target_compile_options(${target_name} ${link_type} -Wno-undef)
        endif()
    endfunction()

    _ov_fix_protobuf_warnings(protobuf::libprotobuf)
    if(TARGET protobuf::libprotobuf-lite)
        _ov_fix_protobuf_warnings(protobuf::libprotobuf-lite)
    endif()
endif()

#
# FlatBuffers
#

if(ENABLE_OV_TF_LITE_FRONTEND)
    if(ENABLE_SYSTEM_FLATBUFFERS)
        if(CMAKE_HOST_LINUX)
            set(_old_flat_CMAKE_LIBRARY_ARCHITECTURE ${CMAKE_LIBRARY_ARCHITECTURE})
            # without this WA cmake does not search in <triplet> subfolder
            # see https://cmake.org/cmake/help/latest/command/find_package.html#config-mode-search-procedure
            if(HOST_X86_64)
                set(CMAKE_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")
            elseif(HOST_AARCH64)
                set(CMAKE_LIBRARY_ARCHITECTURE "aarch64-linux-gnu")
            endif()
        endif()

        # on new Ubuntu versions like 23.04 we have config called FlatBuffersConfig.cmake
        # so, we need to provide alternative names
        find_host_package(Flatbuffers QUIET NAMES Flatbuffers FlatBuffers NO_CMAKE_FIND_ROOT_PATH)

        if(DEFINED _old_flat_CMAKE_LIBRARY_ARCHITECTURE)
            set(CMAKE_LIBRARY_ARCHITECTURE ${_old_flat_CMAKE_LIBRARY_ARCHITECTURE})
            unset(_old_flat_CMAKE_LIBRARY_ARCHITECTURE)
        endif()
    endif()

    if(Flatbuffers_FOUND)
        # we don't actually use library files (.so | .dylib | .a) itself, only headers
        if(TARGET flatbuffers::flatbuffers_shared)
            set(flatbuffers_LIBRARY flatbuffers::flatbuffers_shared)
        elseif(TARGET flatbuffers::flatbuffers)
            set(flatbuffers_LIBRARY flatbuffers::flatbuffers)
        else()
            message(FATAL_ERROR "Internal error: Failed to detect flatbuffers library target")
        endif()
        set(flatbuffers_COMPILER flatbuffers::flatc)
    else()
        add_subdirectory(thirdparty/flatbuffers EXCLUDE_FROM_ALL)
    endif()

    # set additional variables, used in other places of our cmake scripts
    set(flatbuffers_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:${flatbuffers_LIBRARY},INTERFACE_INCLUDE_DIRECTORIES>)
endif()

#
# Snappy Compression
#

if(ENABLE_SNAPPY_COMPRESSION)
    if(ENABLE_SYSTEM_SNAPPY)
        find_package(Snappy REQUIRED)

        set(ov_snappy_lib Snappy::snappy)
        if(NOT BUILD_SHARED_LIBS AND TARGET Snappy::snappy-static)
            # we can use static library only in static build, because in case od dynamic build
            # the libsnappy.a should be compiled with -fPIC, while Debian / Ubuntu / RHEL don't do it
            set(ov_snappy_lib Snappy::snappy-static)
        endif()

        if(OV_PkgConfig_VISILITY)
            # need to set GLOBAL visibility in order to create ALIAS for this target
            set_target_properties(${ov_snappy_lib} PROPERTIES IMPORTED_GLOBAL ON)
        endif()

        add_library(openvino::snappy ALIAS ${ov_snappy_lib})
    endif()

    if(NOT TARGET openvino::snappy)
        function(ov_build_snappy)
            set(BUILD_SHARED_LIBS OFF)
            set(SNAPPY_BUILD_BENCHMARKS OFF)
            set(SNAPPY_BUILD_TESTS OFF)
            set(INSTALL_GTEST OFF)
            set(CMAKE_CXX_STANDARD 14)
            if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
                # '<': signed/unsigned mismatch
                ov_add_compiler_flags(/wd4018)
                # conditional expression is constant
                ov_add_compiler_flags(/wd4127)
                # 'conversion' conversion from 'type1' to 'type2', possible loss of data
                ov_add_compiler_flags(/wd4244)
                # 'conversion' : conversion from 'type1' to 'type2', signed/unsigned mismatch
                ov_add_compiler_flags(/wd4245)
                # 'var' : conversion from 'size_t' to 'type', possible loss of data
                ov_add_compiler_flags(/wd4267)
            elseif(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
                # we need to pass -Wextra first, then -Wno-sign-compare
                # otherwise, snappy's CMakeLists.txt will do it for us
                ov_add_compiler_flags(-Wextra)
                ov_add_compiler_flags(-Wno-sign-compare)
            endif()

            add_subdirectory(thirdparty/snappy EXCLUDE_FROM_ALL)
            # need to create alias openvino::snappy
            add_library(openvino::snappy ALIAS snappy)

            # WA for emscripten build which currently requires -fexceptions
            if(EMSCRIPTEN)
                target_compile_options(snappy PRIVATE "-fexceptions")
            endif()
        endfunction()

        ov_build_snappy()
        ov_install_static_lib(snappy ${OV_CPACK_COMP_CORE})
    endif()
endif()

#
# ONNX
#

if(ENABLE_OV_ONNX_FRONTEND)
    find_package(ONNX 1.14.0 QUIET COMPONENTS onnx onnx_proto NO_MODULE)

    if(ONNX_FOUND)
        # conan and vcpkg create imported targets 'onnx' and 'onnx_proto'
    else()
        add_subdirectory(thirdparty/onnx)
    endif()
endif()

#
# nlohmann json
#

if(ENABLE_SAMPLES)
    # Note: NPU requires 3.9.0 version, because it contains 'nlohmann::ordered_json'
    find_package(nlohmann_json 3.9.0 QUIET)
    if(nlohmann_json_FOUND)
        # conan and vcpkg create imported target nlohmann_json::nlohmann_json
    else()
        add_subdirectory(thirdparty/json EXCLUDE_FROM_ALL)

        # this is required only because of NPU plugin reused this
        openvino_developer_export_targets(COMPONENT openvino_common TARGETS nlohmann_json)

        # for nlohmann library versions older than v3.0.0
        if(NOT TARGET nlohmann_json::nlohmann_json)
            add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
            set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
                INTERFACE_LINK_LIBRARIES nlohmann_json
                INTERFACE_COMPILE_DEFINITIONS JSON_HEADER)
        endif()
    endif()
endif()

#
# Install
#

if(CPACK_GENERATOR MATCHES "^(DEB|RPM|CONDA-FORGE|BREW|CONAN|VCPKG)$")
    # These libraries are dependencies for openvino-samples package
    if(ENABLE_SAMPLES OR ENABLE_TESTS)
        if(NOT gflags_FOUND AND CPACK_GENERATOR MATCHES "^(DEB|RPM)$")
            message(FATAL_ERROR "gflags must be used as a ${CPACK_GENERATOR} package. Install libgflags-dev / gflags-devel")
        endif()
        if(NOT (zlib_FOUND OR ZLIB_FOUND))
            message(FATAL_ERROR "zlib must be used as a ${CPACK_GENERATOR} package. Install zlib1g-dev / zlib-devel")
        endif()
    endif()

    if(NOT ENABLE_SYSTEM_PUGIXML)
        message(FATAL_ERROR "Pugixml must be used as a ${CPACK_GENERATOR} package. Install libpugixml-dev / pugixml-devel")
    endif()
elseif(APPLE OR WIN32)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/gflags
            DESTINATION ${OV_CPACK_SAMPLESDIR}/cpp/thirdparty
            COMPONENT ${OV_CPACK_COMP_CPP_SAMPLES}
            ${OV_CPACK_COMP_CPP_SAMPLES_EXCLUDE_ALL}
            PATTERN bazel EXCLUDE
            PATTERN doc EXCLUDE
            PATTERN .git EXCLUDE
            PATTERN appveyor.yml EXCLUDE
            PATTERN AUTHORS.txt EXCLUDE
            PATTERN BUILD EXCLUDE
            PATTERN ChangeLog.txt EXCLUDE
            PATTERN .gitattributes EXCLUDE
            PATTERN .gitignore EXCLUDE
            PATTERN .gitmodules EXCLUDE
            PATTERN test EXCLUDE
            PATTERN INSTALL.md EXCLUDE
            PATTERN README.md EXCLUDE
            PATTERN .travis.yml EXCLUDE
            PATTERN WORKSPACE EXCLUDE)

    file(GLOB zlib_sources ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/zlib/zlib/*.c
                           ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/zlib/zlib/*.h)
    install(FILES ${zlib_sources}
            DESTINATION ${OV_CPACK_SAMPLESDIR}/cpp/thirdparty/zlib/zlib
            COMPONENT ${OV_CPACK_COMP_CPP_SAMPLES}
            ${OV_CPACK_COMP_CPP_SAMPLES_EXCLUDE_ALL})
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/zlib/CMakeLists.txt
            DESTINATION ${OV_CPACK_SAMPLESDIR}/cpp/thirdparty/zlib
            COMPONENT ${OV_CPACK_COMP_CPP_SAMPLES}
            ${OV_CPACK_COMP_CPP_SAMPLES_EXCLUDE_ALL})

    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/json/nlohmann_json
            DESTINATION ${OV_CPACK_SAMPLESDIR}/cpp/thirdparty
            COMPONENT ${OV_CPACK_COMP_CPP_SAMPLES}
            ${OV_CPACK_COMP_CPP_SAMPLES_EXCLUDE_ALL}
            PATTERN ChangeLog.md EXCLUDE
            PATTERN CITATION.cff EXCLUDE
            PATTERN .clang-format EXCLUDE
            PATTERN .clang-tidy EXCLUDE
            PATTERN docs EXCLUDE
            PATTERN .git EXCLUDE
            PATTERN .github EXCLUDE
            PATTERN .gitignore EXCLUDE
            PATTERN .lgtm.yml EXCLUDE
            PATTERN Makefile EXCLUDE
            PATTERN meson.build EXCLUDE
            PATTERN README.md EXCLUDE
            PATTERN .reuse EXCLUDE
            PATTERN tests EXCLUDE
            PATTERN tools EXCLUDE
            PATTERN wsjcpp.yml EXCLUDE)
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cnpy
        DESTINATION ${OV_CPACK_SAMPLESDIR}/cpp/thirdparty
        COMPONENT ${OV_CPACK_COMP_CPP_SAMPLES}
        ${OV_CPACK_COMP_CPP_SAMPLES_EXCLUDE_ALL})

# restore state

set(CMAKE_CXX_FLAGS "${_old_CMAKE_CXX_FLAGS}")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ${_old_CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE})
