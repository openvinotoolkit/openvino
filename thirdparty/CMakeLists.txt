# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(SUGGEST_OVERRIDE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
endif()

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

add_subdirectory(ittapi)
add_subdirectory(itt_collector EXCLUDE_FROM_ALL)
add_subdirectory(cnpy EXCLUDE_FROM_ALL)
if(ENABLE_INTEL_GPU)
    add_subdirectory(ocl)
endif()

add_subdirectory(xbyak EXCLUDE_FROM_ALL)
openvino_developer_export_targets(COMPONENT openvino_common TARGETS xbyak)
ov_install_static_lib(xbyak ${OV_CPACK_COMP_CORE})

#
# zlib
#

if(NOT ANDROID)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_search_module(zlib QUIET
                          IMPORTED_TARGET GLOBAL
                          zlib)
        if(zlib_FOUND)
            add_library(zlib::zlib ALIAS PkgConfig::zlib)
            message(STATUS "${PKG_CONFIG_EXECUTABLE}: zlib (${zlib_VERSION}) is found at ${zlib_PREFIX}")
        endif()
    endif()
endif()

if(NOT zlib_FOUND)
    add_subdirectory(zlib EXCLUDE_FROM_ALL)
endif()

#
# Pugixml
#

if(ENABLE_SYSTEM_PUGIXML)
    find_package(PkgConfig QUIET)
    # try system pugixml first
    find_package(PugiXML QUIET)
    if(PugiXML_FOUND)
        if(TARGET pugixml::shared)
            # example: cross-compilation on debian
            set(pugixml_target pugixml::shared)
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
    elseif(PkgConfig_FOUND AND NOT ANDROID)
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
                # buggy pugixml.pc does not allow to work with openvino.pc well
                set(pugixml_buggy_pkgconfig ON CACHE INTERNAL "" FORCE)
                message(WARNING "openvino.pc pkg-config file generation has turned off because it should have to depend on pugixml.pc")
            endif()
            message(STATUS "${PKG_CONFIG_EXECUTABLE}: pugixml (${pugixml_VERSION}) is found at ${pugixml_PREFIX}")
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
                if(pugixml_file MATCHES "^.*\.${CMAKE_SHARED_LIBRARY_SUFFIX}(\.[0-9]+)*$")
                    list(APPEND pugixml_libs "${pugixml_file}")
                endif()
            endforeach()
        elseif(target_type STREQUAL "INTERFACE_LIBRARY")
            get_target_property(pugixml_loc ${pugixml_target} INTERFACE_LINK_LIBRARIES)
            file(GLOB pugixml_libs_all "${pugixml_loc}.*")
            foreach(pugixml_lib IN LISTS pugixml_libs_all)
                if(pugixml_lib MATCHES "^${pugixml_loc}(\.[0-9]+)+$")
                    list(APPEND pugixml_libs "${pugixml_lib}")
                endif()
            endforeach()
        endif()

        # if dynamic libpugixml.so.1 and libpugixml.so.1.X are found
        if(pugixml_libs)
            ie_cpack_add_component(pugixml HIDDEN)
            install(FILES ${pugixml_libs}
                    DESTINATION runtime/3rdparty/pugixml/lib
                    COMPONENT pugixml
                    EXCLUDE_FROM_ALL)
        elseif(CPACK_GENERATOR STREQUAL "DEB")
            message(FATAL_ERROR "Debian package build requires shared Pugixml library")
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

add_library(ocv_hal INTERFACE)
target_include_directories(ocv_hal INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/ocv")

add_subdirectory(ade EXCLUDE_FROM_ALL)
add_subdirectory(fluid/modules/gapi EXCLUDE_FROM_ALL)

if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11 AND NOT OV_COMPILER_IS_CLANG)
    target_compile_options(fluid PRIVATE "-Wno-maybe-uninitialized")
endif()

set_target_properties(ade fluid PROPERTIES FOLDER thirdparty)
openvino_developer_export_targets(COMPONENT openvino_common TARGETS ade fluid)

ov_install_static_lib(ade ${OV_CPACK_COMP_CORE})
ov_install_static_lib(fluid ${OV_CPACK_COMP_CORE})

#
# Gflags
#

if(NOT APPLE AND NOT DEFINED ENV{CONDA_PREFIX})
    # - conda-forge contains only dynamic libraries
    # - on Apple only dynamic libraries are available
    #   also, we can easily mix arm64 and x86_64 binaries when cross-compile for Intel CPU
    find_package(gflags QUIET COMPONENTS nothreads_static)
endif()

if(gflags_FOUND)
    if(TARGET gflags_nothreads-static)
        set_target_properties(gflags_nothreads-static PROPERTIES IMPORTED_GLOBAL ON)
        add_library(gflags ALIAS gflags_nothreads-static)
    elseif(TARGET gflags)
        set_target_properties(gflags PROPERTIES IMPORTED_GLOBAL ON)
    else()
        message(FATAL_ERROR "Internal error: failed to find imported target 'gflags'")
    endif()
    message(STATUS "gflags (${gflags_VERSION}) is found at ${gflags_DIR}")
else()
    add_subdirectory(gflags EXCLUDE_FROM_ALL)
    openvino_developer_export_targets(COMPONENT openvino_common TARGETS gflags)
endif()

#
# Google Tests framework
#

if(ENABLE_TESTS)
    add_subdirectory(gtest EXCLUDE_FROM_ALL)

    openvino_developer_export_targets(COMPONENT tests
                                      TARGETS gmock gmock_main gtest gtest_main)
endif()

#
# Protobuf
#

if(ENABLE_OV_PADDLE_FRONTEND OR ENABLE_OV_ONNX_FRONTEND OR ENABLE_OV_TF_FRONTEND)
    if(ENABLE_SYSTEM_PROTOBUF)
        set(Protobuf_USE_STATIC_LIBS ON)
        if(VERBOSE_BUILD)
            set(Protobuf_DEBUG ON)
        endif()
        find_package(Protobuf 3.18.2 REQUIRED)
        set(Protobuf_LITE_LIBRARIES protobuf::libprotobuf-lite)
        set(Protobuf_LIBRARIES protobuf::libprotobuf)
        set(PROTOC_EXECUTABLE protobuf::protoc)

        foreach(target ${PROTOC_EXECUTABLE} ${Protobuf_LIBRARIES} ${Protobuf_LITE_LIBRARIES})
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
# ONNX
#

if(ENABLE_OV_ONNX_FRONTEND)
    add_subdirectory(onnx)
endif()

#
# nlohmann json
#

add_subdirectory(json)

openvino_developer_export_targets(COMPONENT openvino_common
                                  TARGETS nlohmann_json nlohmann_json_schema_validator)

ov_install_static_lib(nlohmann_json ${OV_CPACK_COMP_CORE})
ov_install_static_lib(nlohmann_json_schema_validator ${OV_CPACK_COMP_CORE})

#
# Install
#

if(CPACK_GENERATOR STREQUAL "DEB")
    # These libraries are dependencies for openvino-samples-2022.1 package

    if(NOT gflags_FOUND)
        message(FATAL_ERROR "gflags must be used as a debian package. Install libgflags-dev")
    endif()
    if(NOT zlib_FOUND)
        message(FATAL_ERROR "zlib must be used as a debian package. Install zlib1g-dev")
    endif()
    if(NOT ENABLE_SYSTEM_PUGIXML)
        message(FATAL_ERROR "Pugixml must be used as a debian package. Install libpugixml-dev")
    endif()
else()
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
            PATTERN appveyor.yml EXCLUDE
            PATTERN benchmarks EXCLUDE
            PATTERN ChangeLog.md EXCLUDE
            PATTERN CITATION.cff EXCLUDE
            PATTERN .clang-format EXCLUDE
            PATTERN .clang-tidy EXCLUDE
            PATTERN CODE_OF_CONDUCT.md EXCLUDE
            PATTERN doc EXCLUDE
            PATTERN .git EXCLUDE
            PATTERN .drone.yml EXCLUDE
            PATTERN .github EXCLUDE
            PATTERN .gitignore EXCLUDE
            PATTERN Makefile EXCLUDE
            PATTERN meson.build EXCLUDE
            PATTERN nlohmann_json.natvis EXCLUDE
            PATTERN README.md EXCLUDE
            PATTERN test EXCLUDE
            PATTERN third_party EXCLUDE
            PATTERN .travis.yml EXCLUDE
            PATTERN wsjcpp.yml EXCLUDE)
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cnpy
        DESTINATION ${OV_CPACK_SAMPLESDIR}/cpp/thirdparty
        COMPONENT ${OV_CPACK_COMP_CPP_SAMPLES})
