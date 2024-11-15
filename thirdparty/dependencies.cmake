# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CMAKE_TOOLCHAIN_FILE MATCHES "vcpkg" OR DEFINED VCPKG_VERBOSE)
    set(OV_VCPKG_BUILD ON)
elseif(CMAKE_TOOLCHAIN_FILE MATCHES "conan_toolchain" OR DEFINED CONAN_EXPORTED)
    set(OV_CONAN_BUILD)
endif()

set(_old_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(_old_CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
set(_old_CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ${CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE})
set(_old_CMAKE_COMPILE_WARNING_AS_ERROR ${CMAKE_COMPILE_WARNING_AS_ERROR})

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

# temporarily remove CMAKE_COMPILE_WARNING_AS_ERROR for thirdparty
if(CMAKE_COMPILE_WARNING_AS_ERROR AND WIN32)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND CMAKE_VERSION VERSION_LESS 3.24)
        ov_add_compiler_flags(/WX-)
    endif()
    set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)
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
    endif()
endif()

#
# LevelZero
#

if(ENABLE_INTEL_NPU)
    add_subdirectory(thirdparty/level_zero EXCLUDE_FROM_ALL)

    add_library(LevelZero::LevelZero ALIAS ze_loader)
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
    add_subdirectory(thirdparty/zlib EXCLUDE_FROM_ALL)
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
            set(PUGIXML_INSTALL OFF CACHE BOOL "" FORCE)
            add_subdirectory(thirdparty/pugixml EXCLUDE_FROM_ALL)
        endfunction()
        ov_build_pugixml_static()
        set_property(TARGET pugixml-static PROPERTY EXPORT_NAME pugixml)
        add_library(openvino::pugixml ALIAS pugixml-static)
        ov_developer_package_export_targets(TARGET openvino::pugixml)
        ov_install_static_lib(pugixml-static ${OV_CPACK_COMP_CORE})
    endfunction()

    ov_build_pugixml()
endif()

#
# Gflags
#

if(ENABLE_SAMPLES OR ENABLE_TESTS OR ENABLE_INTEL_NPU_INTERNAL)
    add_subdirectory(thirdparty/gflags EXCLUDE_FROM_ALL)
    ov_developer_package_export_targets(TARGET gflags)
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
        # install & export
        set(googletest_root "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/gtest/gtest")
        ov_developer_package_export_targets(TARGET gtest_main
                                            INSTALL_INCLUDE_DIRECTORIES "${googletest_root}/googletest/include/")
        ov_developer_package_export_targets(TARGET gtest
                                            INSTALL_INCLUDE_DIRECTORIES "${googletest_root}/googletest/include/")
        ov_developer_package_export_targets(TARGET gmock
                                            INSTALL_INCLUDE_DIRECTORIES "${googletest_root}/googlemock/include/")
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
        # try to find newer version first (major is changed)
        # see https://protobuf.dev/support/version-support/ and
        # https://github.com/protocolbuffers/protobuf/commit/d61f75ff6db36b4f9c0765f131f8edc2f86310fa
        find_package(Protobuf 5.26.0 QUIET CONFIG)
        if(NOT Protobuf_FOUND)
            find_package(Protobuf 4.22.0 QUIET CONFIG)
        endif()
        if(Protobuf_FOUND)
            # protobuf was found via CONFIG mode, let's save it for later usage in OpenVINOConfig.cmake static build
            set(protobuf_config CONFIG)
        else()
            if(OV_VCPKG_BUILD)
                set(protobuf_config CONFIG)
            endif()
            # otherwise, fallback to existing default
            find_package(Protobuf 3.20.3 REQUIRED ${protobuf_config})
        endif()

        # with newer protobuf versions (4.22 and newer), we use CONFIG first
        # so, the Protobuf_PROTOC_EXECUTABLE variable must be checked explicitly,
        # because it's not used in this case (oppositely to MODULE case)
        if(Protobuf_VERSION VERSION_GREATER_EQUAL 22 AND DEFINED Protobuf_PROTOC_EXECUTABLE)
            set(PROTOC_EXECUTABLE ${Protobuf_PROTOC_EXECUTABLE})
        else()
            set(PROTOC_EXECUTABLE protobuf::protoc)
        endif()
    else()
        add_subdirectory(thirdparty/protobuf EXCLUDE_FROM_ALL)
        # protobuf fails to build with -fsanitize=thread by clang
        if(ENABLE_THREAD_SANITIZER AND OV_COMPILER_IS_CLANG)
            foreach(proto_target protoc libprotobuf libprotobuf-lite)
                if(TARGET ${proto_target})
                    target_compile_options(${proto_target} PUBLIC -fno-sanitize=thread)
                    target_link_options(${proto_target} PUBLIC -fno-sanitize=thread)
                endif()
            endforeach()
        endif()
    endif()

    # forward additional variables used in the other places
    set(Protobuf_IN_FRONTEND ON)

    # set public / interface compile options
    function(_ov_fix_protobuf_warnings target_name)
        set(link_type PUBLIC)
        if(ENABLE_SYSTEM_PROTOBUF)
            set(link_type INTERFACE)
        endif()
        if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG OR (OV_COMPILER_IS_INTEL_LLVM AND UNIX))
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
        ov_cross_compile_define_debian_arch()

        # on new Ubuntu versions like 23.04 we have config called FlatBuffersConfig.cmake
        # so, we need to provide alternative names
        find_host_package(Flatbuffers QUIET NAMES Flatbuffers FlatBuffers NO_CMAKE_FIND_ROOT_PATH)

        ov_cross_compile_define_debian_arch_reset()
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

        # used by NPU repo
        set(flatc_COMMAND flatc)
        set(flatc_TARGET flatc)
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
            elseif(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG OR (OV_COMPILER_IS_INTEL_LLVM AND UNIX))
                # we need to pass -Wextra first, then -Wno-sign-compare
                # otherwise, snappy's CMakeLists.txt will do it for us
                ov_add_compiler_flags(-Wextra)
                ov_add_compiler_flags(-Wno-sign-compare)
            elseif(OV_COMPILER_IS_INTEL_LLVM AND WIN32)
                ov_add_compiler_flags(/WX-)
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
    find_package(ONNX 1.16.2 QUIET COMPONENTS onnx onnx_proto NO_MODULE)

    if(ONNX_FOUND)
        # conan and vcpkg create imported targets 'onnx' and 'onnx_proto'
        # newer versions of ONNX in vcpkg has ONNX:: prefix, let's create aliases
        if(TARGET ONNX::onnx)
            add_library(onnx ALIAS ONNX::onnx)
        endif()
        if(TARGET ONNX::onnx_proto)
            add_library(onnx_proto ALIAS ONNX::onnx_proto)
        endif()
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

        # this is required only because of NPU plugin reused this: export & install
        ov_developer_package_export_targets(TARGET nlohmann_json
                                            INSTALL_INCLUDE_DIRECTORIES "${OpenVINO_SOURCE_DIR}/thirdparty/json/nlohmann_json/include")

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
        PATTERN src/gflags_completions.sh EXCLUDE
        PATTERN WORKSPACE EXCLUDE)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/json/nlohmann_json
        DESTINATION ${OV_CPACK_SAMPLESDIR}/cpp/thirdparty
        COMPONENT ${OV_CPACK_COMP_CPP_SAMPLES}
        ${OV_CPACK_COMP_CPP_SAMPLES_EXCLUDE_ALL}
        PATTERN BUILD.bazel EXCLUDE
        PATTERN ChangeLog.md EXCLUDE
        PATTERN CITATION.cff EXCLUDE
        PATTERN .cirrus.yml EXCLUDE
        PATTERN .clang-format EXCLUDE
        PATTERN .clang-tidy EXCLUDE
        PATTERN docs EXCLUDE
        PATTERN .git EXCLUDE
        PATTERN .github EXCLUDE
        PATTERN .gitignore EXCLUDE
        PATTERN .lgtm.yml EXCLUDE
        PATTERN Makefile EXCLUDE
        PATTERN meson.build EXCLUDE
        PATTERN Package.swift EXCLUDE
        PATTERN README.md EXCLUDE
        PATTERN .reuse EXCLUDE
        PATTERN tests EXCLUDE
        PATTERN tools EXCLUDE
        PATTERN WORKSPACE.bazel EXCLUDE
        PATTERN wsjcpp.yml EXCLUDE)

# restore state

set(CMAKE_CXX_FLAGS "${_old_CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${_old_CMAKE_C_FLAGS}")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ${_old_CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE})
set(CMAKE_COMPILE_WARNING_AS_ERROR ${_old_CMAKE_COMPILE_WARNING_AS_ERROR})
