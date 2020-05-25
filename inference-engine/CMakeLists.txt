# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
project(InferenceEngine)

set(CMAKE_MODULE_PATH "${IE_MAIN_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

include(features_ie)

# include developer package
include(developer_package_ie)

# These options are shared with 3rdparty plugins
# by means of developer package
include(check_features_ie)

# resolving dependencies for the project
include(dependencies)


# Fuzz tests also building without ENABLE_FUZZING
include(fuzzing)

if (ENABLE_FUZZING)
    enable_fuzzing()
endif()

find_package(Threads REQUIRED)

unset(IEDeveloperPackageTargets CACHE)
function(ie_developer_export_targets)
    set(IEDeveloperPackageTargets "${IEDeveloperPackageTargets};${ARGV}")

    # to allow exporting of aliased targets with the original names
    foreach(target_name ${IEDeveloperPackageTargets})
        if(TARGET "${target_name}")
            get_target_property(original_name ${target_name} ALIASED_TARGET)
            if(TARGET "${original_name}")
                message(STATUS "The name ${target_name} is an ALIAS for ${original_name}. "
                        "It will be exported to the InferenceEngineDeveloperPackage with the original name.")
                list(REMOVE_ITEM IEDeveloperPackageTargets ${target_name})
                list(APPEND IEDeveloperPackageTargets ${original_name})
            endif()
        endif()
    endforeach()

    list(REMOVE_DUPLICATES IEDeveloperPackageTargets)
    set(IEDeveloperPackageTargets "${IEDeveloperPackageTargets}" CACHE INTERNAL
        "Paths to extra Inference Engine plugins" FORCE)
endfunction()

function(ie_developer_export)
    export(TARGETS ${IEDeveloperPackageTargets} NAMESPACE IE::
            APPEND FILE "${CMAKE_BINARY_DIR}/targets_developer.cmake")

    # Custom target to build only Inference Engine Developer Package targets
    add_custom_target(ie_dev_targets ALL DEPENDS ${IEDeveloperPackageTargets} gflags)
endfunction()

add_subdirectory(thirdparty)

add_subdirectory(src)

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

    add_subdirectory(samples)
endfunction()

# gflags and format_reader targets are kept inside of samples directory and
# they must be built even if samples build is disabled (required for tests and tools).
ie_build_samples()

file(GLOB_RECURSE SAMPLES_SOURCES samples/*.cpp samples/*.hpp samples/*.h)
add_cpplint_target(sample_cpplint
    FOR_SOURCES ${SAMPLES_SOURCES}
    EXCLUDE_PATTERNS "thirdparty/*" "pugixml/*")

if (ENABLE_PYTHON)
    add_subdirectory(ie_bridges/python)
endif()

if (ENABLE_C)
    add_subdirectory(ie_bridges/c)
endif()

add_cpplint_report_target()

#
# Install
#

# install C++ samples

ie_cpack_add_component(cpp_samples REQUIRED DEPENDS core)

if(UNIX)
    install(DIRECTORY samples/
            DESTINATION ${IE_CPACK_IE_DIR}/samples/cpp
            COMPONENT cpp_samples
            USE_SOURCE_PERMISSIONS
            PATTERN *.bat EXCLUDE
            PATTERN speech_libs_and_demos EXCLUDE)
elseif(WIN32)
    install(DIRECTORY samples/
            DESTINATION ${IE_CPACK_IE_DIR}/samples/cpp
            COMPONENT cpp_samples
            USE_SOURCE_PERMISSIONS
            PATTERN *.sh EXCLUDE
            PATTERN speech_libs_and_demos EXCLUDE)
endif()

# install C samples

ie_cpack_add_component(c_samples REQUIRED DEPENDS core)

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
        PATTERN ie_bridges/c/samples/CMakeLists.txt EXCLUDE)

install(FILES samples/CMakeLists.txt
        DESTINATION ${IE_CPACK_IE_DIR}/samples/c
        COMPONENT c_samples)

# install Python samples

if(ENABLE_PYTHON)
    ie_cpack_add_component(python_samples REQUIRED DEPENDS core)

    install(DIRECTORY ${ie_python_api_SOURCE_DIR}/sample/
            DESTINATION ${IE_CPACK_IE_DIR}/samples/python
            COMPONENT python_samples)
endif()

# install speech demo files

if(SPEECH_LIBS_AND_DEMOS)
    ie_cpack_add_component(speech_demo_files REQUIRED)

    install(DIRECTORY ${TEMP}/deployment_tools
                      ${TEMP}/data_processing
            DESTINATION .
            COMPONENT speech_demo_files)
endif()

#
# Developer package
#

ie_developer_export_targets(inference_engine_reader_api)
ie_developer_export_targets(format_reader)
ie_developer_export_targets(${NGRAPH_LIBRARIES})

ie_developer_export()

configure_file(
    "${IE_MAIN_SOURCE_DIR}/cmake/developer_package_config.cmake.in"
    "${CMAKE_BINARY_DIR}/InferenceEngineDeveloperPackageConfig.cmake"
    @ONLY)

configure_file(
    "${IE_MAIN_SOURCE_DIR}/cmake/share/InferenceEngineConfig-version.cmake.in"
    "${CMAKE_BINARY_DIR}/InferenceEngineDeveloperPackageConfig-version.cmake"
    COPYONLY)

#
# Coverage
#

if(ENABLE_COVERAGE)
    include(coverage_ie)
endif()

#
# Add plugins
#

function(register_extra_plugins)
    set(InferenceEngineDeveloperPackage_DIR "${CMAKE_CURRENT_BINARY_DIR}/build-plugins")
    set(iedevconfig_file "${InferenceEngineDeveloperPackage_DIR}/InferenceEngineDeveloperPackageConfig.cmake")
    file(REMOVE "${iedevconfig_file}")

    file(WRITE "${iedevconfig_file}" "\# !! AUTOGENERATED: DON'T EDIT !!\n\n")
    file(APPEND "${iedevconfig_file}" "ie_deprecated_no_errors()\n")

    foreach(target IN LISTS IEDeveloperPackageTargets)
        if(target)
            file(APPEND "${iedevconfig_file}" "add_library(IE::${target} ALIAS ${target})\n")
        endif()
    endforeach()

    # automatically import plugins from the 'plugins' folder
    file(GLOB local_extra_plugins "plugins/*")
    list(APPEND local_extra_plugins "${OpenVINO_MAIN_SOURCE_DIR}/docs/template_plugin")

    foreach(plugin_path IN LISTS IE_EXTRA_PLUGINS local_extra_plugins)
        get_filename_component(plugin_dir "${plugin_path}" NAME)
        message(STATUS "Register ${plugin_dir} to be built in build-plugins/${plugin_dir}")
        add_subdirectory("${plugin_path}" "build-plugins/${plugin_dir}")
    endforeach()
endfunction()

register_extra_plugins()
