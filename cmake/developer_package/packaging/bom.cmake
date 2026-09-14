# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# BOM (Bill of Materials) generation for CPack components.
#
# For every CPack component this creates a dedicated 'ov_bom_<component>' target which,
# once built, re-installs *only* that component into an isolated staging directory and
# records the resulting relative file layout into a small JSON manifest. Manifests are
# written per OS (linux / windows / macos), because the same component may legitimately
# contain a different set of files depending on the platform it is built for.
#
# These manifests are meant to be compared against "golden" files checked into the
# repository under OV_BOM_GOLDEN_DIR: whenever a change to CMake / install() rules alters
# the file layout of a public CPack component, the corresponding golden BOM file must be
# updated by its owner as an explicit acknowledgement that the change is intentional and
# safe to ship in the public package. That comparison itself is wired up separately
# (e.g. by a CI job); this file is only responsible for producing the manifests.
#
# BOM generation is part of the default build. It can also be invoked explicitly:
#   cmake --build <build-dir> --target bom
#   cmake --build <build-dir> --target ov_bom_core
#

ov_option(ENABLE_BOM_GENERATION "Generate CPack component BOM (bill of materials) files describing installed file layout" ON)

# remember this file's directory now, since ov_add_bom_targets() below can be called
# later from a different directory scope (the top-level CMakeLists.txt)
set(_OV_BOM_SCRIPT_DIR "${CMAKE_CURRENT_LIST_DIR}")

# directory with BOM files actually produced during this build
set(OV_BOM_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bom" CACHE PATH "Output directory for generated CPack component BOM files")

# directory with 'golden' BOM files, checked into the repository, used for future comparison in CI
set(OV_BOM_GOLDEN_DIR "${CMAKE_SOURCE_DIR}/bom" CACHE PATH "Directory with golden (reference) CPack component BOM files")

# normalize current OS name; the file layout of a component can legitimately differ across platforms
if(WIN32)
    set(_ov_bom_platform "windows")
elseif(APPLE)
    set(_ov_bom_platform "macos")
elseif(LINUX OR UNIX)
    set(_ov_bom_platform "linux")
else()
    string(TOLOWER "${CMAKE_SYSTEM_NAME}" _ov_bom_platform)
endif()
set(OV_BOM_PLATFORM "${_ov_bom_platform}" CACHE STRING "Platform sub-folder / suffix used for BOM files" FORCE)
unset(_ov_bom_platform)

#
# ov_add_bom_targets(<list of components>)
#
# Registers one 'ov_bom_<component>' custom target per CPack component (plus an aggregate
# 'bom' target which depends on all of them). Building a target (re)generates the BOM
# manifest for the corresponding component under
#   ${OV_BOM_OUTPUT_DIR}/${OV_BOM_PLATFORM}/<component>.json
#
function(_ov_collect_bom_build_targets directory output)
    get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    get_property(subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)

    foreach(subdirectory IN LISTS subdirectories)
        _ov_collect_bom_build_targets("${subdirectory}" subdirectory_targets)
        list(APPEND targets ${subdirectory_targets})
    endforeach()

    set(build_targets)
    foreach(target IN LISTS targets)
        get_target_property(type "${target}" TYPE)
        get_target_property(imported "${target}" IMPORTED)
        get_target_property(exclude_from_all "${target}" EXCLUDE_FROM_ALL)
        if(NOT imported AND NOT exclude_from_all AND NOT type STREQUAL "INTERFACE_LIBRARY")
            list(APPEND build_targets "${target}")
        endif()
    endforeach()

    set(${output} "${build_targets}" PARENT_SCOPE)
endfunction()

function(ov_add_bom_targets components)
    if(NOT ENABLE_BOM_GENERATION OR NOT components)
        return()
    endif()

    if(TARGET bom)
        # ov_add_bom_targets() has already been called once (e.g. ov_cpack() invoked
        # multiple times); nothing else to do
        return()
    endif()

    _ov_collect_bom_build_targets("${CMAKE_SOURCE_DIR}" build_targets)
    add_custom_target(ov_bom_build_dependencies)
    if(build_targets)
        add_dependencies(ov_bom_build_dependencies ${build_targets})
    endif()

    set(bom_targets)

    foreach(component IN LISTS components)
        if(TARGET ov_bom_${component})
            continue()
        endif()

        set(output_file "${OV_BOM_OUTPUT_DIR}/${OV_BOM_PLATFORM}/${component}.json")
        set(staging_dir "${CMAKE_BINARY_DIR}/bom_staging/${component}")

        add_custom_target(ov_bom_${component}
            COMMAND "${CMAKE_COMMAND}"
                    "-DOV_BOM_COMPONENT=${component}"
                    "-DOV_BOM_BUILD_DIR=${CMAKE_BINARY_DIR}"
                    "-DOV_BOM_STAGING_DIR=${staging_dir}"
                    "-DOV_BOM_OUTPUT_FILE=${output_file}"
                    "-DOV_BOM_CONFIG=$<CONFIG>"
                    -P "${_OV_BOM_SCRIPT_DIR}/bom_generate.cmake"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            COMMENT "[BOM] Generating manifest for CPack component '${component}' (${OV_BOM_PLATFORM})"
            VERBATIM)
        add_dependencies(ov_bom_${component} ov_bom_build_dependencies)

        list(APPEND bom_targets ov_bom_${component})
    endforeach()

    if(bom_targets)
        add_custom_target(bom ALL
            COMMENT "[BOM] Regenerating manifests for all CPack components (${OV_BOM_PLATFORM})")
        add_dependencies(bom ${bom_targets})
    endif()
endfunction()
