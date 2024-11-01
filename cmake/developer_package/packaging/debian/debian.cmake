# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(GNUInstallDirs)

#
# ov_debian_cpack_set_dirs()
#
# Set directories for cpack
#
macro(ov_debian_cpack_set_dirs)
    # override default locations for Debian
    set(OV_CPACK_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR})
    set(OV_CPACK_RUNTIMEDIR ${CMAKE_INSTALL_LIBDIR})
    if(CMAKE_CROSSCOMPILING)
        if(ARM)
            set(OV_CPACK_RUNTIMEDIR "${OV_CPACK_RUNTIMEDIR}/arm-linux-gnueabihf")
        elseif(AARCH64)
            set(OV_CPACK_RUNTIMEDIR "${OV_CPACK_RUNTIMEDIR}/aarch64-linux-gnu")
        elseif(X86)
            set(OV_CPACK_RUNTIMEDIR "${OV_CPACK_RUNTIMEDIR}/i386-linux-gnu")
        elseif(X86_64)
            set(OV_CPACK_RUNTIMEDIR "${OV_CPACK_RUNTIMEDIR}/x86_64-linux-gnu")
        elseif(RISCV64)
            set(OV_CPACK_RUNTIMEDIR "${OV_CPACK_RUNTIMEDIR}/riscv64-linux-gnu")
        endif()
    endif()
    set(OV_CPACK_LIBRARYDIR ${OV_CPACK_RUNTIMEDIR})
    set(OV_CPACK_ARCHIVEDIR ${OV_CPACK_RUNTIMEDIR})
    set(OV_CPACK_PLUGINSDIR ${OV_CPACK_RUNTIMEDIR}/openvino-${OpenVINO_VERSION})
    set(OV_CPACK_OPENVINO_CMAKEDIR ${OV_CPACK_RUNTIMEDIR}/cmake/openvino${OpenVINO_VERSION})
    set(OV_CPACK_DOCDIR ${CMAKE_INSTALL_DATADIR}/doc/openvino-${OpenVINO_VERSION})
    set(OV_CPACK_LICENSESDIR ${OV_CPACK_DOCDIR}/licenses)
    set(OV_CPACK_PYTHONDIR lib/python3/dist-packages)

    # non-native stuff
    set(OV_CPACK_SHAREDIR ${CMAKE_INSTALL_DATADIR}/openvino) # internal
    set(OV_CPACK_SAMPLESDIR ${OV_CPACK_SHAREDIR}/samples)
    set(OV_CPACK_DEVREQDIR ${OV_CPACK_SHAREDIR})
    unset(OV_CPACK_SHAREDIR)

    # skipped during debian packaging
    set(OV_CPACK_WHEELSDIR "wheels")
endmacro()

ov_debian_cpack_set_dirs()

#
# Override CPack components name for Debian generator
# This is needed to change the granularity, i.e. merge several components
# into a single one
#

macro(ov_override_component_names)
    # merge C++ and C runtimes
    set(OV_CPACK_COMP_CORE_C "${OV_CPACK_COMP_CORE}")
    set(OV_CPACK_COMP_CORE_C_DEV "${OV_CPACK_COMP_CORE_DEV}")
    # merge all C / C++ samples as a single samples component
    set(OV_CPACK_COMP_CPP_SAMPLES "samples")
    set(OV_CPACK_COMP_C_SAMPLES "${OV_CPACK_COMP_CPP_SAMPLES}")
    # merge links and pkgconfig with dev component
    set(OV_CPACK_COMP_LINKS "${OV_CPACK_COMP_CORE_DEV}")
    set(OV_CPACK_COMP_PKG_CONFIG "${OV_CPACK_COMP_CORE_DEV}")
endmacro()

ov_override_component_names()

#
# Override include / exclude rules for components
# This is required to exclude some files from installation
# (e.g. debian packages don't require setupvars scripts)
#

macro(ov_define_component_include_rules)
    # core components
    unset(OV_CPACK_COMP_CORE_EXCLUDE_ALL)
    set(OV_CPACK_COMP_CORE_C_EXCLUDE_ALL ${OV_CPACK_COMP_CORE_EXCLUDE_ALL})
    unset(OV_CPACK_COMP_CORE_DEV_EXCLUDE_ALL)
    set(OV_CPACK_COMP_CORE_C_DEV_EXCLUDE_ALL ${OV_CPACK_COMP_CORE_DEV_EXCLUDE_ALL})
    # tbb
    set(OV_CPACK_COMP_TBB_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_TBB_DEV_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # openmp
    set(OV_CPACK_COMP_OPENMP_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # licensing
    set(OV_CPACK_COMP_LICENSING_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # samples
    unset(OV_CPACK_COMP_CPP_SAMPLES_EXCLUDE_ALL)
    set(OV_CPACK_COMP_C_SAMPLES_EXCLUDE_ALL ${OV_CPACK_COMP_CPP_SAMPLES_EXCLUDE_ALL})
    if(ENABLE_PYTHON_PACKAGING)
        unset(OV_CPACK_COMP_PYTHON_SAMPLES_EXCLUDE_ALL)
    else()
        set(OV_CPACK_COMP_PYTHON_SAMPLES_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    endif()
    # python
    if(ENABLE_PYTHON_PACKAGING)
        # pack artifacts of pip install
        unset(OV_CPACK_COMP_PYTHON_OPENVINO_PACKAGE_EXCLUDE_ALL)
    else()
        set(OV_CPACK_COMP_PYTHON_OPENVINO_PACKAGE_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    endif()
    # we don't pack python components itself, we pack artifacts of pip install
    set(OV_CPACK_COMP_PYTHON_OPENVINO_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_BENCHMARK_APP_EXCLUDE_ALL ${OV_CPACK_COMP_PYTHON_OPENVINO_EXCLUDE_ALL})
    set(OV_CPACK_COMP_OVC_EXCLUDE_ALL ${OV_CPACK_COMP_PYTHON_OPENVINO_EXCLUDE_ALL})
    # we don't need wheels in Debian packages
    set(OV_CPACK_COMP_PYTHON_WHEELS_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # because numpy is installed by apt
    set(OV_CPACK_COMP_OPENVINO_REQ_FILES_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # nodejs
    set(OV_CPACK_COMP_NPM_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # scripts
    set(OV_CPACK_COMP_INSTALL_DEPENDENCIES_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_SETUPVARS_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # pkgconfig
    set(OV_CPACK_COMP_PKG_CONFIG_EXCLUDE_ALL ${OV_CPACK_COMP_CORE_DEV_EXCLUDE_ALL})
    # symbolic links
    set(OV_CPACK_COMP_LINKS_EXCLUDE_ALL ${OV_CPACK_COMP_CORE_DEV_EXCLUDE_ALL})
    # npu internal tools
    set(OV_CPACK_COMP_NPU_INTERNAL_EXCLUDE_ALL EXCLUDE_FROM_ALL)
endmacro()

ov_define_component_include_rules()

#
# Common Debian specific settings
#

macro(ov_debian_specific_settings)
    # multiple packages are generated
    set(CPACK_DEB_COMPONENT_INSTALL ON)
    # automatically find dependencies for binaries
    set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
    # enable dependencies between components
    set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS ON)
    # control file permissions
    set(CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION OFF)
    # homepage
    set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://docs.openvino.ai/")
    # use lintian to check packages in post-build step
    set(CPACK_POST_BUILD_SCRIPTS "${OpenVINODeveloperScripts_DIR}/packaging/debian/post_build.cmake")
    # to make sure that lib/<multiarch-triplet> is created on Debian
    set(CMAKE_INSTALL_PREFIX "/usr" CACHE PATH "Cmake install prefix" FORCE)
    # enable for debug cpack run
    if(NOT DEFINED CPACK_DEBIAN_PACKAGE_DEBUG)
        set(CPACK_DEBIAN_PACKAGE_DEBUG OFF)
    endif()

    # WA: dpkg-shlibdeps requires folder with libraries
    # proper way is to use -l (path to libs) and -L (path to shlibs) for other already installed components
    # but it requires CMake source code changes
    # with current WA automatic deps detection via dpkg-shlibdeps for "our libraries"
    # is ignored; but dependencies between our components are here because of
    # CPACK_COMPONENT_<UCOMP>_DEPENDS variables

    if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        if(OV_GENERATOR_MULTI_CONFIG)
            # $<CONFIG> generator expression does not work in this place, have to add all possible configs
            foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
                list(APPEND CPACK_DEBIAN_PACKAGE_SHLIBDEPS_PRIVATE_DIRS "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${config}")
            endforeach()
        else()
            list(APPEND CPACK_DEBIAN_PACKAGE_SHLIBDEPS_PRIVATE_DIRS "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
        endif()
    else()
        message(FATAL_ERROR "CMAKE_LIBRARY_OUTPUT_DIRECTORY is empty")
    endif()

    # automatic dependencies discovering between openvino and user packages
    set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS ON)
    # OpenVINO does not have backward and forward compatibility
    set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS_POLICY "=")
    # naming convention for debian package files
    set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
    # need to update this version once we rebuild the same package with additional fixes
    # set(CPACK_DEBIAN_PACKAGE_RELEASE "1")
    # enable this if someday we change the version scheme
    # set(CPACK_DEBIAN_PACKAGE_EPOCH "2")

    # temporary WA for debian package architecture for cross-compilation
    # proper solution: to force cmake auto-detect this
    if(CMAKE_CROSSCOMPILING)
        if(AARCH64)
            set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE arm64)
        elseif(ARM)
            set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE armhf)
        elseif(x86)
            set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE i386)
        elseif(X86_64)
            set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE x86_64)
        elseif(RISCV64)
            set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE riscv64)
        endif()
    endif()

    # we don't need RPATHs, because libraries are search by standard paths
    set(CMAKE_SKIP_INSTALL_RPATH ON)
endmacro()

ov_debian_specific_settings()

# needed to override cmake auto generated files
set(def_postinst "${CMAKE_CURRENT_BINARY_DIR}/_CPack_Packages/postinst")
set(def_postrm "${CMAKE_CURRENT_BINARY_DIR}/_CPack_Packages/postrm")
set(def_triggers "${CMAKE_CURRENT_BINARY_DIR}/_CPack_Packages/triggers")

set(triggers_content "activate-noawait ldconfig\n\n")
set(post_content "#!/bin/sh\n\nset -e;\nset -e\n\n")

file(REMOVE ${def_postinst} ${def_postrm} ${def_triggers})
file(WRITE "${def_postinst}" "${post_content}")
file(WRITE "${def_postrm}" "${post_content}")
file(WRITE "${def_triggers}" "${triggers_content}")

#
# Functions helpful for packaging your modules with Debian cpack
#

#
# ov_debian_add_changelog_and_copyright(<comp name> <copyright_name>)
#
function(ov_debian_add_changelog_and_copyright comp copyright_name)
    string(TOUPPER "${comp}" ucomp)
    if(NOT DEFINED CPACK_DEBIAN_${ucomp}_PACKAGE_NAME)
        message(FATAL_ERROR "CPACK_DEBIAN_${ucomp}_PACKAGE_NAME is not defined")
    else()
        set(package_name "${CPACK_DEBIAN_${ucomp}_PACKAGE_NAME}")
    endif()
    set(package_name "${CPACK_DEBIAN_${ucomp}_PACKAGE_NAME}")

    # copyright

    install(FILES "${OpenVINO_SOURCE_DIR}/cmake/packaging/copyrights/${copyright_name}"
            DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/${package_name}/
            COMPONENT ${comp}
            RENAME "copyright")

    # create changelog.gz

    find_host_program(gzip_PROGRAM NAMES gzip DOC "Path to gzip tool")
    if(NOT gzip_PROGRAM)
        message(FATAL_ERROR "Failed to find gzip tool")
    endif()

    set(changelog_src "${OpenVINO_SOURCE_DIR}/cmake/developer_package/packaging/debian/changelog")
    set(package_bin_dir "${OpenVINO_BINARY_DIR}/_CPack_Packages/${package_name}")
    set(changelog_output "${package_bin_dir}/changelog")

    file(REMOVE "${changelog_output}")
    file(REMOVE "${changelog_output}.gz")

    file(MAKE_DIRECTORY "${package_bin_dir}")
    configure_file("${changelog_src}" "${changelog_output}" COPYONLY)

    execute_process(COMMAND gzip -n -9 "${changelog_output}"
        WORKING_DIRECTORY "${package_bin_dir}"
        OUTPUT_VARIABLE output_message
        ERROR_VARIABLE error_message
        RESULT_VARIABLE exit_code
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    # install changelog.gz

    install(FILES "${changelog_output}.gz"
            DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/${package_name}/
            COMPONENT ${comp})
endfunction()

#
# ov_debian_add_lintian_suppression(<comp name> <suppression1, suppression2, ...>)
#
function(ov_debian_add_lintian_suppression comp)
    set(lines ${ARGN})

    string(TOUPPER "${comp}" ucomp)
    if(NOT DEFINED CPACK_DEBIAN_${ucomp}_PACKAGE_NAME)
        message(FATAL_ERROR "CPACK_DEBIAN_${ucomp}_PACKAGE_NAME is not defined")
    else()
        set(package_name "${CPACK_DEBIAN_${ucomp}_PACKAGE_NAME}")
    endif()

    foreach(line IN LISTS lines)
        set(line "${package_name} binary: ${line}")
        if(content)
            set(content "${content}\n${line}")
        else()
            set(content "${line}")
        endif()
    endforeach()

    set(lintian_override_file "${OpenVINO_BINARY_DIR}/_CPack_Packages/lintian/${package_name}")
    file(REMOVE ${lintian_override_file})
    file(WRITE ${lintian_override_file} ${content})

    install(FILES ${lintian_override_file}
            DESTINATION ${CMAKE_INSTALL_DATADIR}/lintian/overrides/
            COMPONENT ${comp})
endfunction()

#
# ov_debian_generate_conflicts(<comp name>)
#
function(ov_debian_generate_conflicts comp)
    set(cpack_name_versions ${ARGN})
    string(TOUPPER "${comp}" ucomp)

    # sanity check
    if(NOT DEFINED CPACK_DEBIAN_${ucomp}_PACKAGE_NAME)
        message(FATAL_ERROR "CPACK_DEBIAN_${ucomp}_PACKAGE_NAME is not defined")
    else()
        if(NOT DEFINED cpack_name_ver)
            message(FATAL_ERROR "Internal variable 'cpack_name_ver' is not defined")
        endif()

        string(REPLACE "${cpack_name_ver}" "" package_name_base "${CPACK_DEBIAN_${ucomp}_PACKAGE_NAME}")
    endif()

    foreach(cpack_name_version IN LISTS cpack_name_versions)
        if(package_names)
            set(package_names "${package_names}, ${package_name_base}${cpack_name_version}")
        else()
            set(package_names "${package_name_base}${cpack_name_version}")
        endif()
    endforeach()

    set(CPACK_DEBIAN_${ucomp}_PACKAGE_CONFLICTS "${package_names}" PARENT_SCOPE)
endfunction()

#
# ov_debian_add_latest_component(<comp>)
#
# Adds latest component for `comp`, but without a version
# Description and other stuff (arch) is taken from the main component
#
macro(ov_debian_add_latest_component comp)
    string(TOUPPER "${comp}" ucomp)
    set(comp_name "${comp}_latest")
    set(upper_case "${ucomp}_LATEST")

    set(CPACK_COMPONENT_${upper_case}_DESCRIPTION "${CPACK_COMPONENT_${ucomp}_DESCRIPTION}")
    set(CPACK_DEBIAN_${upper_case}_PACKAGE_ARCHITECTURE "all")
    set(CPACK_COMPONENT_${upper_case}_DEPENDS "${comp}")
    set(${comp_name}_copyright "generic")

    # take package name
    if(DEFINED CPACK_DEBIAN_${ucomp}_PACKAGE_NAME)
        string(REPLACE "-${cpack_name_ver}" ""
            CPACK_DEBIAN_${upper_case}_PACKAGE_NAME
            "${CPACK_DEBIAN_${ucomp}_PACKAGE_NAME}")
    else()
        message(FATAL_ERROR "CPACK_DEBIAN_${ucomp}_PACKAGE_NAME is not defined")
    endif()

    # add latest to a list of debian packages
    list(APPEND CPACK_COMPONENTS_ALL ${comp_name})
endmacro()
