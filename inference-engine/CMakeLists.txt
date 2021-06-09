# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

project(InferenceEngine)

if(CMAKE_SOURCE_DIR STREQUAL "${InferenceEngine_SOURCE_DIR}")
    message(FATAL_ERROR "Cmake source directory must point to <openvino_root>")
endif()

include(cmake/features.cmake)

# resolving dependencies for the project
include(cmake/dependencies.cmake)

function(ie_developer_export_targets)
    openvino_developer_export_targets(COMPONENT inference_engine TARGETS ${ARGN})
endfunction()

function(ie_developer_export)
    set(all_dev_targets gflags ie_libraries)
    foreach(component IN LISTS openvino_export_components)
        export(TARGETS ${${component}} NAMESPACE IE::
               APPEND FILE "${CMAKE_BINARY_DIR}/${component}_dev_targets.cmake")
        list(APPEND all_dev_targets ${${component}})
    endforeach()

    add_custom_target(ie_dev_targets ALL DEPENDS ${all_dev_targets})
endfunction()

add_subdirectory(thirdparty)

add_subdirectory(src)

add_subdirectory(ie_bridges/c)

if(ENABLE_TESTS)
    add_subdirectory(tests_deprecated)
    add_subdirectory(tests)
endif()

add_subdirectory(tools)

function(ie_build_samples)
    # samples should be build with the same flags as from OpenVINO package,
    # so unset all flags
    foreach(var CMAKE_CXX_FLAGS CMAKE_C_FLAGS CMAKE_CXX_STANDARD
                CMAKE_EXE_LINKER_FLAGS CMAKE_POLICY_DEFAULT_CMP0063
                CMAKE_CXX_VISIBILITY_PRESET CMAKE_C_VISIBILITY_PRESET
                CMAKE_VISIBILITY_INLINES_HIDDEN CMAKE_POSITION_INDEPENDENT_CODE
                THREADS_PREFER_PTHREAD_FLAG X86_64 X86 ARM AARCH64 LINUX
                MINGW64 CMAKE_BUILD_TYPE CMAKE_MACOSX_RPATH)
        unset(${var})
    endforeach()
    include("${IEDevScripts_DIR}/compile_flags/sanitizer.cmake")
    add_subdirectory(samples)
endfunction()

# gflags and format_reader targets are kept inside of samples directory and
# they must be built even if samples build is disabled (required for tests and tools).
ie_build_samples()

if(ENABLE_PYTHON)
    add_subdirectory(ie_bridges/python)
endif()

#
# Install
#

# install C++ samples

ie_cpack_add_component(cpp_samples DEPENDS core)

install(DIRECTORY ../thirdparty/zlib
        DESTINATION ${IE_CPACK_IE_DIR}/samples/cpp/thirdparty
        COMPONENT cpp_samples
        USE_SOURCE_PERMISSIONS
        PATTERN .clang-format EXCLUDE)

install(DIRECTORY ../thirdparty/cnpy
        DESTINATION ${IE_CPACK_IE_DIR}/samples/cpp/thirdparty
        COMPONENT cpp_samples
        USE_SOURCE_PERMISSIONS
        PATTERN .clang-format EXCLUDE)

if(UNIX)
    install(DIRECTORY samples/
            DESTINATION ${IE_CPACK_IE_DIR}/samples/cpp
            COMPONENT cpp_samples
            USE_SOURCE_PERMISSIONS
            PATTERN *.bat EXCLUDE
            PATTERN speech_libs_and_demos EXCLUDE
            PATTERN .clang-format EXCLUDE)
elseif(WIN32)
    install(DIRECTORY samples/
            DESTINATION ${IE_CPACK_IE_DIR}/samples/cpp
            COMPONENT cpp_samples
            USE_SOURCE_PERMISSIONS
            PATTERN *.sh EXCLUDE
            PATTERN speech_libs_and_demos EXCLUDE
            PATTERN .clang-format EXCLUDE)
endif()

# install C samples

ie_cpack_add_component(c_samples DEPENDS core_c)

if(UNIX)
    install(PROGRAMS samples/build_samples.sh
            DESTINATION ${IE_CPACK_IE_DIR}/samples/c
            COMPONENT c_samples)
elseif(WIN32)
    install(PROGRAMS samples/build_samples_msvc.bat
            DESTINATION ${IE_CPACK_IE_DIR}/samples/c
            COMPONENT c_samples)
endif()

install(DIRECTORY ie_bridges/c/samples/
        DESTINATION ${IE_CPACK_IE_DIR}/samples/c
        COMPONENT c_samples
        PATTERN ie_bridges/c/samples/CMakeLists.txt EXCLUDE
        PATTERN ie_bridges/c/samples/.clang-format EXCLUDE)

install(FILES samples/CMakeLists.txt
        DESTINATION ${IE_CPACK_IE_DIR}/samples/c
        COMPONENT c_samples)

# install speech demo files

if(SPEECH_LIBS_AND_DEMOS)
    ie_cpack_add_component(speech_demo_files)

    install(DIRECTORY ${TEMP}/deployment_tools
                      ${TEMP}/data_processing
            DESTINATION .
            USE_SOURCE_PERMISSIONS
            COMPONENT speech_demo_files)
endif()

#
# Developer package
#

openvino_developer_export_targets(COMPONENT openvino_common TARGETS format_reader gflags ie_samples_utils)

# for Template plugin
if(NGRAPH_INTERPRETER_ENABLE)
    openvino_developer_export_targets(COMPONENT ngraph TARGETS ngraph_backend interpreter_backend)
endif()

function(ie_generate_dev_package_config)
    # dummy check that OpenCV is here
    find_package(OpenCV QUIET)

    ie_developer_export()

    configure_package_config_file("${InferenceEngine_SOURCE_DIR}/cmake/templates/InferenceEngineDeveloperPackageConfig.cmake.in"
                                "${CMAKE_BINARY_DIR}/InferenceEngineDeveloperPackageConfig.cmake"
                                INSTALL_DESTINATION share # not used
                                PATH_VARS "OpenVINO_MAIN_SOURCE_DIR;IE_MAIN_SOURCE_DIR;gflags_BINARY_DIR"
                                NO_CHECK_REQUIRED_COMPONENTS_MACRO)

    configure_file("${IE_MAIN_SOURCE_DIR}/cmake/templates/InferenceEngineConfig-version.cmake.in"
                "${CMAKE_BINARY_DIR}/InferenceEngineDeveloperPackageConfig-version.cmake"
                @ONLY)
endfunction()

ie_generate_dev_package_config()

#
# Coverage
#

if(ENABLE_COVERAGE)
    include(cmake/coverage.cmake)
endif()

#
# Add extra modules
#

function(register_extra_modules)
    # post export
    ie_developer_export_targets(inference_engine)
    openvino_developer_export_targets(COMPONENT ngraph TARGETS ${NGRAPH_LIBRARIES})

    set(InferenceEngineDeveloperPackage_DIR "${CMAKE_CURRENT_BINARY_DIR}/build-modules")

    function(generate_fake_dev_package)
        set(iedevconfig_file "${InferenceEngineDeveloperPackage_DIR}/InferenceEngineDeveloperPackageConfig.cmake")
        file(REMOVE "${iedevconfig_file}")

        file(WRITE "${iedevconfig_file}" "\# !! AUTOGENERATED: DON'T EDIT !!\n\n")
        file(APPEND "${iedevconfig_file}" "ie_deprecated_no_errors()\n")

        foreach(target IN LISTS ${openvino_export_components})
            if(target)
                file(APPEND "${iedevconfig_file}" "add_library(IE::${target} ALIAS ${target})\n")
            endif()
        endforeach()
    endfunction()

    generate_fake_dev_package()

    # automatically import plugins from the 'plugins' folder
    file(GLOB local_extra_modules "plugins/*")
    if(NGRAPH_INTERPRETER_ENABLE)
        list(APPEND local_extra_modules "${OpenVINO_MAIN_SOURCE_DIR}/docs/template_plugin")
    endif()

    # detect where IE_EXTRA_MODULES contains folders with CMakeLists.txt
    # other folders are supposed to have sub-folders with CMakeLists.txt
    foreach(module_path IN LISTS IE_EXTRA_MODULES)
        if(EXISTS "${module_path}/CMakeLists.txt")
            list(APPEND extra_modules "${module_path}")
        elseif(module_path)
            file(GLOB extra_modules ${extra_modules} "${module_path}/*")
        endif()
    endforeach()

    # add each extra module
    foreach(module_path IN LISTS extra_modules local_extra_modules)
        if(module_path)
            get_filename_component(module_name "${module_path}" NAME)
            set(build_module ON)
            if(NOT EXISTS "${module_path}/CMakeLists.txt") # if module is built not using cmake
                set(build_module OFF)
            endif()
            if(NOT DEFINED BUILD_${module_name})
                set(BUILD_${module_name} ${build_module} CACHE BOOL "Build ${module_name} extra module" FORCE)
            endif()
            if(BUILD_${module_name})
                message(STATUS "Register ${module_name} to be built in build-modules/${module_name}")
                add_subdirectory("${module_path}" "build-modules/${module_name}")
            endif()
        endif()
    endforeach()
endfunction()

register_extra_modules()
