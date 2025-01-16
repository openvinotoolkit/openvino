# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

macro(ov_search_api_validator)
    if(NOT ENABLE_API_VALIDATOR)
        return()
    endif()

    # CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION is only set when
    # Visual Studio generators are used, but we need it
    # when we use Ninja as well
    if(NOT DEFINED CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION)
        if(DEFINED ENV{WindowsSDKVersion})
            string(REPLACE "\\" "" WINDOWS_SDK_VERSION $ENV{WindowsSDKVersion})
            set(CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION ${WINDOWS_SDK_VERSION})
            message(STATUS "Use ${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION} Windows SDK version")
        else()
            message(FATAL_ERROR "WindowsSDKVersion environment variable is not set,\
can't find Windows SDK version. Try to use vcvarsall.bat script")
        endif()
    endif()

    set(PROGRAMFILES_ENV "ProgramFiles\(X86\)")

    # check that PROGRAMFILES_ENV is defined, because in case of cross-compilation for Windows
    # we don't have such variable
    if(DEFINED ENV{${PROGRAMFILES_ENV}})
        file(TO_CMAKE_PATH $ENV{${PROGRAMFILES_ENV}} PROGRAMFILES)

        set(WDK_PATHS "${PROGRAMFILES}/Windows Kits/10/bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/x64"
                      "${PROGRAMFILES}/Windows Kits/10/bin/x64")

        message(STATUS "Trying to find apivalidator in: ")
        foreach(wdk_path IN LISTS WDK_PATHS)
            message("    * ${wdk_path}")
        endforeach()

        find_host_program(ONECORE_API_VALIDATOR
                          NAMES apivalidator
                          PATHS ${WDK_PATHS}
                          DOC "ApiValidator for OneCore compliance")

        if(ONECORE_API_VALIDATOR)
            message(STATUS "Found apivalidator: ${ONECORE_API_VALIDATOR}")
        endif()
    endif()
endmacro()


if(ENABLE_API_VALIDATOR)
    ov_search_api_validator()
endif()

function(_ov_add_api_validator_post_build_step_recursive)
    cmake_parse_arguments(API_VALIDATOR "" "TARGET" "" ${ARGN})

    get_target_property(LIBRARY_TYPE ${API_VALIDATOR_TARGET} TYPE)
    if(LIBRARY_TYPE MATCHES "^(SHARED_LIBRARY|MODULE_LIBRARY|EXECUTABLE)$" AND
        NOT ${API_VALIDATOR_TARGET} IN_LIST API_VALIDATOR_TARGETS)
        list(APPEND API_VALIDATOR_TARGETS ${API_VALIDATOR_TARGET})
    endif()
    # keep checks target list to track cyclic dependencies, leading to infinite recursion
    list(APPEND checked_targets ${API_VALIDATOR_TARGET})

    if(NOT LIBRARY_TYPE STREQUAL "INTERFACE_LIBRARY")
        get_target_property(LINKED_LIBRARIES ${API_VALIDATOR_TARGET} LINK_LIBRARIES)
    else()
        set(LINKED_LIBRARIES)
    endif()
    get_target_property(INTERFACE_LINKED_LIBRARIES ${API_VALIDATOR_TARGET} INTERFACE_LINK_LIBRARIES)

    foreach(library IN LISTS LINKED_LIBRARIES INTERFACE_LINKED_LIBRARIES)
        if(TARGET "${library}")
            get_target_property(orig_library ${library} ALIASED_TARGET)
            if(orig_library IN_LIST checked_targets OR library IN_LIST checked_targets)
                # in case of cyclic dependencies, we need to skip current target
                continue()
            endif()
            if(TARGET "${orig_library}")
                _ov_add_api_validator_post_build_step_recursive(TARGET ${orig_library})
            else()
                _ov_add_api_validator_post_build_step_recursive(TARGET ${library})
            endif()
        endif()
    endforeach()

    set(API_VALIDATOR_TARGETS ${API_VALIDATOR_TARGETS} PARENT_SCOPE)
endfunction()

set(VALIDATED_TARGETS "" CACHE INTERNAL "")

function(_ov_add_api_validator_post_build_step)
    if((NOT ONECORE_API_VALIDATOR) OR (WINDOWS_STORE OR WINDOWS_PHONE))
        return()
    endif()

    # see https://learn.microsoft.com/en-us/windows-hardware/drivers/develop/validating-windows-drivers#known-apivalidator-issues
    # ApiValidator does not run on Arm64 because AitStatic does not work on Arm64
    if(HOST_AARCH64)
        return()
    endif()

    if(X86_64)
        set(wdk_platform "x64")
    elseif(X86)
        set(wdk_platform "x86")
    elseif(ARM)
        set(wdk_platform "arm")
    elseif(AARCH64)
        set(wdk_platform "arm64")
    else()
        message(FATAL_ERROR "Unknown configuration: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
    endif()

    find_file(ONECORE_API_VALIDATOR_APIS NAMES UniversalDDIs.xml
              PATHS "${PROGRAMFILES}/Windows Kits/10/build/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/universalDDIs/${wdk_platform}"
                    "${PROGRAMFILES}/Windows Kits/10/build/universalDDIs/${wdk_platform}"
              DOC "Path to UniversalDDIs.xml file")
    find_file(ONECORE_API_VALIDATOR_EXCLUSION NAMES BinaryExclusionlist.xml
              PATHS ${WDK_PATHS}
              DOC "Path to BinaryExclusionlist.xml file")

    if(NOT ONECORE_API_VALIDATOR_APIS)
        message(FATAL_ERROR "Internal error: apiValidator is found (${ONECORE_API_VALIDATOR}), but UniversalDDIs.xml file has not been found for ${wdk_platform} platform")
    endif()

    cmake_parse_arguments(API_VALIDATOR "" "TARGET" "EXTRA" "" ${ARGN})

    if(NOT API_VALIDATOR_TARGET)
        message(FATAL_ERROR "RunApiValidator requires TARGET to validate!")
    endif()

    if(NOT TARGET ${API_VALIDATOR_TARGET})
        message(FATAL_ERROR "${API_VALIDATOR_TARGET} is not a TARGET in the project tree.")
    endif()

    # collect targets
    _ov_add_api_validator_post_build_step_recursive(TARGET ${API_VALIDATOR_TARGET})
    if (API_VALIDATOR_EXTRA)
        foreach(target IN LISTS API_VALIDATOR_EXTRA)
            _ov_add_api_validator_post_build_step_recursive(TARGET ${target})
        endforeach()
    endif()

    # remove targets which were tested before
    foreach(item IN LISTS VALIDATED_TARGETS)
        list(REMOVE_ITEM API_VALIDATOR_TARGETS ${item})
    endforeach()

    if(NOT API_VALIDATOR_TARGETS)
        return()
    endif()

    # apply check
    macro(api_validator_get_target_name)
        get_target_property(is_imported ${target} IMPORTED)
        get_target_property(orig_target ${target} ALIASED_TARGET)
        if(is_imported)
            get_target_property(imported_configs ${target} IMPORTED_CONFIGURATIONS)
            foreach(imported_config RELEASE RELWITHDEBINFO DEBUG)
                if(imported_config IN_LIST imported_configs)
                    get_target_property(target_location ${target} IMPORTED_LOCATION_${imported_config})
                    get_filename_component(target_name "${target_location}" NAME_WE)
                    break()
                endif()
            endforeach()
            unset(imported_configs)
        elseif(TARGET "${orig_target}")
            set(target_name ${orig_target})
            set(target_location $<TARGET_FILE:${orig_target}>)
        else()
            set(target_name ${target})
            set(target_location $<TARGET_FILE:${target}>)
        endif()

        unset(orig_target)
        unset(is_imported)
    endmacro()

    foreach(target IN LISTS API_VALIDATOR_TARGETS)
        api_validator_get_target_name()
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20 AND OV_GENERATOR_MULTI_CONFIG)
            set(output_file "${OpenVINO_BINARY_DIR}/api_validator/$<CONFIG>/${target_name}.txt")
        else()
            set(output_file "${OpenVINO_BINARY_DIR}/api_validator/${target_name}.txt")
        endif()

        list(APPEND post_build_commands
             ${CMAKE_COMMAND} --config $<CONFIG>
                -D ONECORE_API_VALIDATOR=${ONECORE_API_VALIDATOR}
                -D ONECORE_API_VALIDATOR_TARGET=${target_location}
                -D ONECORE_API_VALIDATOR_APIS=${ONECORE_API_VALIDATOR_APIS}
                -D ONECORE_API_VALIDATOR_EXCLUSION=${ONECORE_API_VALIDATOR_EXCLUSION}
                -D ONECORE_API_VALIDATOR_OUTPUT=${output_file}
                -D CMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
                -P "${OpenVINODeveloperScripts_DIR}/api_validator/api_validator_run.cmake")
        list(APPEND byproducts_files ${output_file})

        unset(target_name)
        unset(target_location)
    endforeach()

    add_custom_command(TARGET ${API_VALIDATOR_TARGET} POST_BUILD
        COMMAND ${post_build_commands}
        BYPRODUCTS ${byproducts_files}
        COMMENT "[apiValidator] Check ${API_VALIDATOR_TARGET} and dependencies for OneCore compliance"
        VERBATIM)

    # update list of validated libraries

    list(APPEND VALIDATED_TARGETS ${API_VALIDATOR_TARGETS})
    set(VALIDATED_TARGETS "${VALIDATED_TARGETS}" CACHE INTERNAL "" FORCE)
endfunction()

#
# ov_add_api_validator_post_build_step(TARGET <name>)
#
function(ov_add_api_validator_post_build_step)
    _ov_add_api_validator_post_build_step(${ARGN})
endfunction()
