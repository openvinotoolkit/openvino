# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

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
    add_subdirectory(ittapi)
    add_subdirectory(itt_collector EXCLUDE_FROM_ALL)
endif()

if(ENABLE_SAMPLES OR ENABLE_TESTS)
    add_subdirectory(cnpy EXCLUDE_FROM_ALL)
endif()

add_subdirectory(xbyak EXCLUDE_FROM_ALL)
openvino_developer_export_targets(COMPONENT openvino_common TARGETS xbyak)
ov_install_static_lib(xbyak ${OV_CPACK_COMP_CORE})

#
# OpenCL
#

if(ENABLE_INTEL_GPU)
    if(ENABLE_SYSTEM_OPENCL)
        # try to find system OpenCL (installed via 'brew install opencl-icd-loader')
        find_package(OpenCLICDLoader QUIET)

        if(OpenCLICDLoader_FOUND)
            # 'brew install opencl-headers'
            find_package(OpenCLHeaders QUIET)
            if(NOT OpenCLHeaders_FOUND)
                message(FATAL_ERROR "OpenCLHeaders not found, but OpenCLICDLoader is installed. Please, install OpenCL headers")
            else()
                set_target_properties(OpenCL::Headers PROPERTIES IMPORTED_GLOBAL ON)
            endif()
        else()
            # try to find system OpenCL:
            # - 'sudo -E apt-get install opencl-headers ocl-icd-opencl-dev'
            # - 'sudo -E yum install ocl-icd-devel opencl-headers'
            # - 'conda install khronos-opencl-icd-loader -c conda-forge'
            find_package(OpenCL QUIET)
        endif()
    endif()

    if(TARGET OpenCL::OpenCL)
        set_target_properties(OpenCL::OpenCL PROPERTIES IMPORTED_GLOBAL ON)

        # try to find CL/opencl.hpp
        find_file(OpenCL_HPP
                  NAMES CL/opencl.hpp OpenCL/opencl.hpp
                  HINTS ${OpenCL_INCLUDE_DIRS}
                  DOC "Path to CL/opencl.hpp")

        # add definition to select proper header and suppress warnings
        if(OpenCL_HPP)
            set_target_properties(OpenCL::OpenCL PROPERTIES
                INTERFACE_COMPILE_DEFINITIONS OV_GPU_USE_OPENCL_HPP)

            # check whether CL/opencl.hpp contains C++ wrapper for property CL_DEVICE_UUID_KHR
            file(STRINGS "${OpenCL_HPP}" CL_DEVICE_UUID_KHR_CPP REGEX ".*CL_DEVICE_UUID_KHR.*")
            if(CL_DEVICE_UUID_KHR_CPP)
                set_target_properties(OpenCL::OpenCL PROPERTIES
                    INTERFACE_COMPILE_DEFINITIONS OV_GPU_OPENCL_HPP_HAS_UUID)
            endif()
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
    if(PkgConfig_FOUND)
        pkg_search_module(zlib QUIET
                          IMPORTED_TARGET GLOBAL
                          zlib)
        if(zlib_FOUND)
            add_library(zlib::zlib ALIAS PkgConfig::zlib)
            message(STATUS "${PKG_CONFIG_EXECUTABLE}: zlib (${zlib_VERSION}) is found at ${zlib_PREFIX}")
        endif()
    endif()

    if(NOT zlib_FOUND)
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
        if(TARGET pugixml::shared)
            # example: cross-compilation on debian
            set(pugixml_target pugixml::shared)
        elseif(TARGET pugixml::pugixml)
            # or create an alias for pugixml::pugixml shared library
            # example: pugixml brew package
            set(pugixml_target pugixml::pugixml)
        elseif(TARGET pugixml)
            # or create an alias for pugixml shared library
            # example: libpugixml-dev debian package
            set(pugixml_target pugixml)
        elseif(TARGET pugixml::static)
            # sometimes pugixml::static target already exists, just need to make it global
            # example: building using conda environment
            set(pugixml_target pugixml::static)
        else()
            message(FATAL_ERROR "Failed to detect pugixml library target name")
        endif()
        # to property generate OpenVINO Developer packages files
        set(PugiXML_FOUND ${PugiXML_FOUND} CACHE BOOL "" FORCE)
    elseif(PkgConfig_FOUND)
        # U18 case when cmake interface is not available
        pkg_search_module(pugixml QUIET
                          IMPORTED_TARGET GLOBAL
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
        # reset to prevent improper code generation in OpenVINODeveloperPackage
        set(ENABLE_SYSTEM_PUGIXML OFF CACHE BOOL "" FORCE)
    endif()
endif()

if(NOT TARGET openvino::pugixml)
    # use OpenVINO pugixml copy if system one is not found
    function(ie_build_pugixml)
        function(ie_build_pugixml_static)
            set(BUILD_SHARED_LIBS OFF)
            add_subdirectory(pugixml EXCLUDE_FROM_ALL)
        endfunction()
        ie_build_pugixml_static()
        set_property(TARGET pugixml-static PROPERTY EXPORT_NAME pugixml)
        add_library(openvino::pugixml ALIAS pugixml-static)
        openvino_developer_export_targets(COMPONENT openvino_common TARGETS openvino::pugixml)
        ov_install_static_lib(pugixml-static ${OV_CPACK_COMP_CORE})
    endfunction()

    ie_build_pugixml()
endif()

#
# Fluid, G-API, OpenCV HAL
#

if(ENABLE_GAPI_PREPROCESSING)
    add_library(ocv_hal INTERFACE)
    target_include_directories(ocv_hal INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/ocv")

    add_subdirectory(ade EXCLUDE_FROM_ALL)
    add_subdirectory(fluid/modules/gapi EXCLUDE_FROM_ALL)

    if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
        target_compile_options(fluid PRIVATE "-Wno-maybe-uninitialized")
    endif()
    if(UNUSED_BUT_SET_VARIABLE_SUPPORTED)
        target_compile_options(fluid PRIVATE "-Wno-unused-but-set-variable")
    endif()

    set_target_properties(ade fluid PROPERTIES FOLDER thirdparty)
    openvino_developer_export_targets(COMPONENT openvino_common TARGETS ade fluid)

    ov_install_static_lib(ade ${OV_CPACK_COMP_CORE})
    ov_install_static_lib(fluid ${OV_CPACK_COMP_CORE})
endif()

#
# Gflags
#

if(ENABLE_SAMPLES OR ENABLE_COMPILE_TOOL OR ENABLE_TESTS)
    # on Windows and macOS we don't use gflags, because will be dynamically linked
    if(CMAKE_HOST_LINUX AND LINUX)
        if(OV_OS_RHEL)
            set(gflag_component nothreads_shared)
        elseif(OV_OS_DEBIAN)
            set(gflag_component nothreads_static)
        endif()
        find_package(gflags QUIET OPTIONAL_COMPONENTS ${gflag_component})
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
        set(Protobuf_USE_STATIC_LIBS ON)
        if(CMAKE_VERBOSE_MAKEFILE)
            set(Protobuf_DEBUG ON)
        endif()
        find_package(Protobuf 3.18.2 REQUIRED)
        set(Protobuf_LITE_LIBRARIES protobuf::libprotobuf-lite)
        set(Protobuf_LIBRARIES protobuf::libprotobuf)
        set(PROTOC_EXECUTABLE protobuf::protoc)

        foreach(target IN LISTS PROTOC_EXECUTABLE Protobuf_LIBRARIES Protobuf_LITE_LIBRARIES)
            set_property(TARGET ${target} PROPERTY IMPORTED_GLOBAL ON)
        endforeach()
    else()
        add_subdirectory(protobuf EXCLUDE_FROM_ALL)
    endif()

    # forward variables used in the other places
    set(PROTOC_DEPENDENCY ${PROTOC_DEPENDENCY} PARENT_SCOPE)
    set(PROTOC_EXECUTABLE ${PROTOC_EXECUTABLE} PARENT_SCOPE)
    set(Protobuf_LIBRARIES ${Protobuf_LIBRARIES} PARENT_SCOPE)
    set(Protobuf_LITE_LIBRARIES ${Protobuf_LITE_LIBRARIES} PARENT_SCOPE)
    set(Protobuf_INCLUDE_DIRS ${Protobuf_INCLUDE_DIRS} PARENT_SCOPE)
    set(Protobuf_IN_FRONTEND ON PARENT_SCOPE)

    # set public / interface compile options
    foreach(target IN LISTS Protobuf_LITE_LIBRARIES Protobuf_LIBRARIES)
        set(link_type PUBLIC)
        if(ENABLE_SYSTEM_PROTOBUF)
            set(link_type INTERFACE)
        endif()
        if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
            target_compile_options(${target} ${link_type} -Wno-undef)
        endif()
    endforeach()
endif()

#
# Flat Buffers
#

if(ENABLE_OV_TF_LITE_FRONTEND)
    if(ENABLE_SYSTEM_FLATBUFFERS)
        if(CMAKE_HOST_LINUX AND EMSCRIPTEN AND HOST_X86_64)
            set(_old_flat_CMAKE_LIBRARY_ARCHITECTURE ${CMAKE_LIBRARY_ARCHITECTURE})
            # without this WA cmake does not search in <triplet> subfolder
            # see https://cmake.org/cmake/help/latest/command/find_package.html#config-mode-search-procedure
            set(CMAKE_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")
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
# ONNX
#

if(ENABLE_OV_ONNX_FRONTEND)
    add_subdirectory(onnx)
endif()

#
# nlohmann json
#

if(ENABLE_SAMPLES)
    add_subdirectory(json)

    # this is required only because of VPUX plugin reused this
    openvino_developer_export_targets(COMPONENT openvino_common
                                      TARGETS nlohmann_json)

    # for nlohmann library version v3.0.0 and later
    if(NOT TARGET nlohmann_json::nlohmann_json)
        add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED GLOBAL)
        set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
            INTERFACE_LINK_LIBRARIES nlohmann_json
            INTERFACE_COMPILE_DEFINITIONS JSON_HEADER)
    endif()
endif()

#
# Install
#

if(CPACK_GENERATOR MATCHES "^(DEB|RPM|CONDA-FORGE|BREW)$")
    # These libraries are dependencies for openvino-samples package

    if(ENABLE_SAMPLES OR ENABLE_COMPILE_TOOL OR ENABLE_TESTS)
        if(NOT gflags_FOUND)
            message(FATAL_ERROR "gflags must be used as a ${CPACK_GENERATOR} package. Install libgflags-dev / gflags-devel")
        endif()
        if(NOT zlib_FOUND)
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
