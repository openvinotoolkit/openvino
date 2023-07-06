# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

<<<<<<< HEAD:thirdparty/CMakeLists.txt
=======
if(CMAKE_TOOLCHAIN_FILE MATCHES "vcpkg" OR DEFINED VCPKG_VERBOSE)
    set(OV_VCPKG_BUILD ON)
elseif(CMAKE_TOOLCHAIN_FILE MATCHES "conan_toolchain" OR DEFINED CONAN_EXPORTED)
    set(OV_CONAN_BUILD)
endif()

set(_old_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(_old_CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ${CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE})

>>>>>>> 14292b8da5 (Allow to find protobuf via MODULE (#18395)):thirdparty/dependencies.cmake
# Android toolchain does not provide pkg-config file. So, cmake mistakenly uses
# build system pkg-config executable, which finds packages on build system. Such
# libraries cannot be linked into Android binaries.
if(NOT ANDROID)
    find_package(PkgConfig QUIET)
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
        # conan defines 'ittapi::ittapi' target
        # create more common alias 'ittapi::ittnotify'
        set_target_properties(ittapi::ittapi PROPERTIES
            IMPORTED_GLOBAL ON
            INTERFACE_COMPILE_DEFINITIONS ENABLE_PROFILING_ITT)
        add_library(ittapi::ittnotify ALIAS ittapi::ittapi)

        # set ittapi_FOUND to parent scope to properly generate OpenVINOConfig.cmake for static build
        set(ittapi_FOUND ${ittapi_FOUND} PARENT_SCOPE)
    else()
        add_subdirectory(ittapi)
    endif()
    add_subdirectory(itt_collector EXCLUDE_FROM_ALL)
endif()

if(ENABLE_SAMPLES OR ENABLE_TESTS)
    add_subdirectory(cnpy EXCLUDE_FROM_ALL)
endif()

if(X86_64 OR UNIVERSAL2)
    find_package(xbyak QUIET)
    if(xbyak_FOUND)
        # conan creates alias xbyak::xbyak, we only need to make it GLOBAL
        set_target_properties(xbyak::xbyak PROPERTIES IMPORTED_GLOBAL ON)
    else()
        add_subdirectory(xbyak EXCLUDE_FROM_ALL)
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
        # - 'brew install opencl-icd-loader'
        # - 'conan install opencl-icd-loader'
        find_package(OpenCLICDLoader QUIET)

        if(OpenCLICDLoader_FOUND)
            # examples:
            # - 'brew install opencl-headers'
            # - 'conan install opencl-headers'
            find_package(OpenCLHeaders QUIET)
            if(NOT OpenCLHeaders_FOUND)
                message(WARNING "OpenCLHeaders not found, but OpenCLICDLoader is installed. Please, install OpenCL headers")
            else()
                set_target_properties(OpenCL::Headers PROPERTIES IMPORTED_GLOBAL ON)
                set_property(TARGET OpenCL::OpenCL APPEND PROPERTY INTERFACE_LINK_LIBRARIES OpenCL::Headers)
            endif()

            # examples:
            # - 'brew install opencl-clhpp-headers'
            # - 'conan install opencl-clhpp-headers'
            find_package(OpenCLHeadersCpp QUIET)
            if(NOT OpenCLHeadersCpp_FOUND)
                message(WARNING "OpenCLHeadersCpp not found, but OpenCLICDLoader is installed. Please, install OpenCL C++ headers")
            else()
                set_target_properties(OpenCL::HeadersCpp PROPERTIES IMPORTED_GLOBAL ON)
                get_target_property(opencl_cpp_include_dirs OpenCL::HeadersCpp INTERFACE_INCLUDE_DIRECTORIES)
                set_property(TARGET OpenCL::OpenCL APPEND PROPERTY INTERFACE_LINK_LIBRARIES OpenCL::HeadersCpp)
            endif()

            # set OpenCLICDLoader_FOUND to parent scope to generate proper OpenVINOConfig.cmake for static libraries case
            set(OpenCLICDLoader_FOUND ON PARENT_SCOPE)
        else()
            # try to find system OpenCL:
            # - 'apt-get install opencl-headers ocl-icd-opencl-dev'
            # - 'yum install ocl-icd-devel opencl-headers'
            # - 'conda install khronos-opencl-icd-loader -c conda-forge'
            find_package(OpenCL QUIET)
        endif()
    endif()

    if(TARGET OpenCL::OpenCL)
        set_target_properties(OpenCL::OpenCL PROPERTIES IMPORTED_GLOBAL ON)

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

        # set variables for onednn_gpu
        if(OpenCLHeaders_FOUND)
            set(OpenCL_INCLUDE_DIR "$<TARGET_PROPERTY:OpenCL::Headers,INTERFACE_INCLUDE_DIRECTORIES>" PARENT_SCOPE)
            set(OpenCL_LIBRARY "$<TARGET_PROPERTY:OpenCL::OpenCL,IMPORTED_LOCATION_RELEASE>" PARENT_SCOPE)
        elseif(OpenCL_FOUND)
            # find_package(OpenCL) already defines OpenCL_INCLUDE_DIR and OpenCL_LIBRARY
            # see https://cmake.org/cmake/help/latest/module/FindOpenCL.html
            message(STATUS "Found OpenCL (ver. ${OpenCL_VERSION_STRING}, include dirs ${OpenCL_INCLUDE_DIRS})")
        else()
            message(FATAL_ERROR "Internal error: cannot find OpenCL headers")
        endif()
    else()
        add_subdirectory(ocl)

        set(OpenCL_INCLUDE_DIR "${OpenCL_INCLUDE_DIR}" PARENT_SCOPE)
        set(OpenCL_LIBRARY "${OpenCL_LIBRARY}" PARENT_SCOPE)
        set(opencl_root_hints "${opencl_root_hints}" PARENT_SCOPE)

        # system OpenCL is not found: set it explictly for consistent OpenVINOConfig.cmake generation
        set(ENABLE_SYSTEM_OPENCL OFF PARENT_SCOPE)
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
    foreach(opencl_dir "${CMAKE_CURRENT_SOURCE_DIR}/ocl/clhpp_headers/include" "${CMAKE_CURRENT_SOURCE_DIR}/ocl/cl_headers")
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
    find_package(ZLIB QUIET NO_MODULE)
    if(ZLIB_FOUND)
        # need to make it global to use outside of the current sub-directory
        set_target_properties(ZLIB::ZLIB PROPERTIES IMPORTED_GLOBAL ON)
    endif()

    # cmake has failed to find zlib, let's try pkg-config
    if(NOT ZLIB_FOUND AND PkgConfig_FOUND)
        pkg_search_module(zlib QUIET
                          IMPORTED_TARGET GLOBAL
                          zlib)
        if(zlib_FOUND)
            add_library(ZLIB::ZLIB ALIAS PkgConfig::zlib)
            message(STATUS "${PKG_CONFIG_EXECUTABLE}: zlib (${zlib_VERSION}) is found at ${zlib_PREFIX}")
        endif()
    endif()

    if(NOT (zlib_FOUND OR ZLIB_FOUND))
        add_subdirectory(zlib EXCLUDE_FROM_ALL)
    endif()
endif()

#
# Pugixml
#

if(ENABLE_SYSTEM_PUGIXML)
    # try system pugixml first
    find_package(PugiXML QUIET)
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
            # sometimes pugixml::static target already exists, just need to make it global
            # - 'conda install pugixml -c conda-forge'
            set(pugixml_target pugixml::static)
        else()
            message(FATAL_ERROR "Failed to detect pugixml library target name")
        endif()
        # to property generate OpenVINO Developer packages files
        set(PugiXML_FOUND ${PugiXML_FOUND} CACHE BOOL "" FORCE)
    elseif(PkgConfig_FOUND)
        # U18 case when cmake interface is not available
        pkg_search_module(pugixml QUIET
                          IMPORTED_TARGET
                          GLOBAL
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
            add_library(pugixml INTERFACE IMPORTED GLOBAL)
            set_target_properties(pugixml PROPERTIES INTERFACE_LINK_LIBRARIES "${PUGIXML_LIBRARY}")
            set(pugixml_target pugixml)
            set(PugiXML_FOUND ON)
        endif()
    endif()

    if(TARGET ${pugixml_target})
        # we need to install dynamic library for wheel package
        get_target_property(target_type ${pugixml_target} TYPE)
        if(target_type STREQUAL "SHARED_LIBRARY")
            get_target_property(imported_config ${pugixml_target} IMPORTED_CONFIGURATIONS)
            get_target_property(pugixml_loc ${pugixml_target} IMPORTED_LOCATION_${imported_config})
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

        set_target_properties(${pugixml_target} PROPERTIES IMPORTED_GLOBAL ON)
        # create an alias for real target which can be shared or static
        add_library(openvino::pugixml ALIAS ${pugixml_target})
    else()
        # reset to prevent improper code generation in OpenVINODeveloperPackage.cmake,
        # and OpenVINOConfig.cmake for static case
        set(ENABLE_SYSTEM_PUGIXML OFF PARENT_SCOPE)
    endif()
endif()

if(NOT TARGET openvino::pugixml)
    # use OpenVINO pugixml copy if system one is not found
    function(ov_build_pugixml)
        function(ov_build_pugixml_static)
            set(BUILD_SHARED_LIBS OFF)
            add_subdirectory(pugixml EXCLUDE_FROM_ALL)
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
    target_include_directories(ocv_hal INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/ocv")

    # ade
    find_package(ade 0.1.2 QUIET)
    if(ade_FOUND)
        # conan creates 'ade' target

        # set ade_FOUND to parent scope to properly generate OpenVINOConfig.cmake for static build
        set(ade_FOUND ${ade_FOUND} PARENT_SCOPE)
    else()
        add_subdirectory(ade EXCLUDE_FROM_ALL)

        set_target_properties(ade PROPERTIES FOLDER thirdparty)
        openvino_developer_export_targets(COMPONENT openvino_common TARGETS ade)

        ov_install_static_lib(ade ${OV_CPACK_COMP_CORE})
    endif()

    # fluid
    add_subdirectory(fluid/modules/gapi EXCLUDE_FROM_ALL)

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

if(ENABLE_SAMPLES OR ENABLE_COMPILE_TOOL OR ENABLE_TESTS)
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
    else()
        # conan case
        find_package(gflags QUIET)
    endif()

    if(gflags_FOUND)
        if(TARGET gflags)
            set_target_properties(gflags PROPERTIES IMPORTED_GLOBAL ON)
        elseif(TARGET gflags_nothreads-static)
            # Debian 9: gflag_component is ignored
            set_target_properties(gflags_nothreads-static PROPERTIES IMPORTED_GLOBAL ON)
            add_library(gflags ALIAS gflags_nothreads-static)
        elseif(TARGET gflags_nothreads-shared)
            # CentOS / RHEL / Fedora case
            set_target_properties(gflags_nothreads-shared PROPERTIES IMPORTED_GLOBAL ON)
            add_library(gflags ALIAS gflags_nothreads-shared)
        elseif(TARGET ${GFLAGS_TARGET})
            set_target_properties(${GFLAGS_TARGET} PROPERTIES IMPORTED_GLOBAL ON)
            add_library(gflags ALIAS ${GFLAGS_TARGET})
        else()
            message(FATAL_ERROR "Internal error: failed to find imported target 'gflags' using '${gflag_component}' component")
        endif()

        message(STATUS "gflags (${gflags_VERSION}) is found at ${gflags_DIR} using '${gflag_component}' component")
    endif()

    if(NOT TARGET gflags)
        add_subdirectory(gflags EXCLUDE_FROM_ALL)
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
            set_target_properties(GTest::${gtest_target} PROPERTIES IMPORTED_GLOBAL ON)
            add_library(${gtest_target} ALIAS GTest::${gtest_target})
        endforeach()
    else()
        add_subdirectory(gtest EXCLUDE_FROM_ALL)
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
        set(Protobuf_USE_STATIC_LIBS ON)
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

        # in case of system protobuf, we have to add version to OpenVINOConfig.cmake for static build
        # to ensure no mismatch between versions of protoc and libprotobuf, we need to use exactly
        # the same versions
        set(Protobuf_VERSION ${Protobuf_VERSION} PARENT_SCOPE)

        foreach(target ${PROTOC_EXECUTABLE} protobuf::libprotobuf protobuf::libprotobuf-lite)
            set_target_properties(${target} PROPERTIES IMPORTED_GLOBAL ON)
        endforeach()
    else()
        add_subdirectory(protobuf EXCLUDE_FROM_ALL)
    endif()

    # forward variables used in the other places
    set(PROTOC_DEPENDENCY ${PROTOC_DEPENDENCY} PARENT_SCOPE)
    set(PROTOC_EXECUTABLE ${PROTOC_EXECUTABLE} PARENT_SCOPE)
    set(Protobuf_IN_FRONTEND ON PARENT_SCOPE)

    # set public / interface compile options
    foreach(target_name protobuf::libprotobuf protobuf::libprotobuf-lite)
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
    endforeach()
endif()

#
# FlatBuffers
#

if(ENABLE_OV_TF_LITE_FRONTEND)
    if(ENABLE_SYSTEM_FLATBUFFERS)
        if(CMAKE_HOST_LINUX AND EMSCRIPTEN)
            set(_old_flat_CMAKE_LIBRARY_ARCHITECTURE ${CMAKE_LIBRARY_ARCHITECTURE})
            # without this WA cmake does not search in <triplet> subfolder
            # see https://cmake.org/cmake/help/latest/command/find_package.html#config-mode-search-procedure
            if(HOST_X86_64)
                set(CMAKE_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")
            elseif(HOST_AARCH64)
                set(CMAKE_LIBRARY_ARCHITECTURE "aarch64-linux-gnu")
            endif()
        endif()

        find_host_package(Flatbuffers QUIET NO_CMAKE_FIND_ROOT_PATH)

        if(DEFINED _old_flat_CMAKE_LIBRARY_ARCHITECTURE)
            set(CMAKE_LIBRARY_ARCHITECTURE ${_old_flat_CMAKE_LIBRARY_ARCHITECTURE})
            unset(_old_flat_CMAKE_LIBRARY_ARCHITECTURE)
        endif()
    endif()

    if(Flatbuffers_FOUND)
        # we don't actually use library files (.so | .dylib | .a) itself, only headers
        set(flatbuffers_LIBRARY flatbuffers::flatbuffers)
        set(flatbuffers_COMPILER flatbuffers::flatc)

        foreach(target IN LISTS flatbuffers_LIBRARY flatbuffers_COMPILER)
            set_property(TARGET ${target} PROPERTY IMPORTED_GLOBAL ON)
        endforeach()
    else()
        add_subdirectory(flatbuffers EXCLUDE_FROM_ALL)
    endif()

    set(flatbuffers_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:${flatbuffers_LIBRARY},INTERFACE_INCLUDE_DIRECTORIES> PARENT_SCOPE)
    set(flatbuffers_COMPILER ${flatbuffers_COMPILER} PARENT_SCOPE)
    set(flatbuffers_DEPENDENCY ${flatbuffers_DEPENDENCY} PARENT_SCOPE)
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

        set_target_properties(${ov_snappy_lib} PROPERTIES IMPORTED_GLOBAL ON)
        add_library(openvino::snappy ALIAS ${ov_snappy_lib})

        # set Snappy_VERSION to parent scope for consistent OpenVINOConfig.cmake generation
        # in case of static build with system dependencies
        set(Snappy_VERSION ${Snappy_VERSION} PARENT_SCOPE)
        set(ov_snappy_lib ${ov_snappy_lib} PARENT_SCOPE)
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
                ie_add_compiler_flags(/wd4018)
                # conditional expression is constant
                ie_add_compiler_flags(/wd4127)
                # 'conversion' conversion from 'type1' to 'type2', possible loss of data
                ie_add_compiler_flags(/wd4244)
                # 'conversion' : conversion from 'type1' to 'type2', signed/unsigned mismatch
                ie_add_compiler_flags(/wd4245)
                # 'var' : conversion from 'size_t' to 'type', possible loss of data
                ie_add_compiler_flags(/wd4267)
            elseif(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
                # we need to pass -Wextra first, then -Wno-sign-compare
                # otherwise, snappy's CMakeLists.txt will do it for us
                ie_add_compiler_flags(-Wextra)
                ie_add_compiler_flags(-Wno-sign-compare)
            endif()

            add_subdirectory(snappy EXCLUDE_FROM_ALL)
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
    find_package(ONNX 1.13.1 EXACT QUIET COMPONENTS onnx onnx_proto NO_MODULE)

    if(ONNX_FOUND)
        # conan creates imported targets 'onnx' and 'onnx_proto'
        set_target_properties(onnx onnx_proto PROPERTIES IMPORTED_GLOBAL ON)

        # set ONNX_VERSION to parent scope for correct OpenVINOConfig.cmake generation
        # in case of static libraries
        set(ONNX_VERSION ${ONNX_VERSION} PARENT_SCOPE)
        set(ENABLE_SYSTEM_ONNX ON PARENT_SCOPE)
    else()
        add_subdirectory(onnx)
    endif()
endif()

#
# nlohmann json
#

if(ENABLE_SAMPLES)
    # Note: VPUX requires 3.9.0 version, because it contains 'nlohmann::ordered_json'
    find_package(nlohmann_json 3.9.0 QUIET)
    if(nlohmann_json_FOUND)
        # conan creates imported target nlohmann_json::nlohmann_json
        # no needs to make the target global, because samples call find_package(nlohmann_json) as well
        # but we need to set nlohmann_json_FOUND to parent scope to properly generate InferenceEngineDeveloperPackageConfig.cmake
        set(nlohmann_json_FOUND ${nlohmann_json_FOUND} PARENT_SCOPE)
    else()
        add_subdirectory(json)

        # this is required only because of VPUX plugin reused this
        openvino_developer_export_targets(COMPONENT openvino_common TARGETS nlohmann_json)

        # for nlohmann library versions older than v3.0.0
        if(NOT TARGET nlohmann_json::nlohmann_json)
            add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED GLOBAL)
            set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
                INTERFACE_LINK_LIBRARIES nlohmann_json
                INTERFACE_COMPILE_DEFINITIONS JSON_HEADER)
        endif()
    endif()
endif()

#
# Install
#

if(CPACK_GENERATOR MATCHES "^(DEB|RPM|CONDA-FORGE|BREW|CONAN)$")
    # These libraries are dependencies for openvino-samples package

    if(ENABLE_SAMPLES OR ENABLE_COMPILE_TOOL OR ENABLE_TESTS)
        if(NOT gflags_FOUND)
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
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/gflags
            DESTINATION ${OV_CPACK_SAMPLESDIR}/cpp/thirdparty
            COMPONENT ${OV_CPACK_COMP_CPP_SAMPLES}
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

    file(GLOB zlib_sources ${CMAKE_CURRENT_SOURCE_DIR}/zlib/zlib/*.c
                           ${CMAKE_CURRENT_SOURCE_DIR}/zlib/zlib/*.h)
    install(FILES ${zlib_sources}
            DESTINATION ${OV_CPACK_SAMPLESDIR}/cpp/thirdparty/zlib/zlib
            COMPONENT ${OV_CPACK_COMP_CPP_SAMPLES})
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/zlib/CMakeLists.txt
            DESTINATION ${OV_CPACK_SAMPLESDIR}/cpp/thirdparty/zlib
            COMPONENT ${OV_CPACK_COMP_CPP_SAMPLES})

    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/json/nlohmann_json
            DESTINATION ${OV_CPACK_SAMPLESDIR}/cpp/thirdparty
            COMPONENT ${OV_CPACK_COMP_CPP_SAMPLES}
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

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cnpy
        DESTINATION ${OV_CPACK_SAMPLESDIR}/cpp/thirdparty
        COMPONENT ${OV_CPACK_COMP_CPP_SAMPLES})
