# Copyright (C) 2018-2022 Intel Corporation
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
    set(OV_CPACK_TOOLSDIR ${CMAKE_INSTALL_BINDIR}) # only C++ tools are here
    set(OV_CPACK_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR})
    set(OV_CPACK_LIBRARYDIR ${CMAKE_INSTALL_LIBDIR})
    set(OV_CPACK_RUNTIMEDIR ${CMAKE_INSTALL_LIBDIR})
    set(OV_WHEEL_RUNTIMEDIR ${OV_CPACK_RUNTIMEDIR})
    set(OV_CPACK_ARCHIVEDIR ${CMAKE_INSTALL_LIBDIR})
    set(OV_CPACK_PLUGINSDIR ${CMAKE_INSTALL_LIBDIR}/openvino-${OpenVINO_VERSION})
    set(OV_CPACK_IE_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/inferenceengine${OpenVINO_VERSION})
    set(OV_CPACK_NGRAPH_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/ngraph${OpenVINO_VERSION})
    set(OV_CPACK_OPENVINO_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/openvino${OpenVINO_VERSION})
    set(OV_CPACK_DOCDIR ${CMAKE_INSTALL_DATADIR}/doc/openvino-${OpenVINO_VERSION})

    # non-native stuff
    set(OV_CPACK_PYTHONDIR ${OV_CPACK_PLUGINSDIR})
    set(OV_CPACK_SHAREDIR ${CMAKE_INSTALL_DATADIR}/openvino) # internal
    set(OV_CPACK_SAMPLESDIR ${OV_CPACK_SHAREDIR}/samples)
    set(OV_CPACK_DEVREQDIR ${OV_CPACK_SHAREDIR})
    unset(OV_CPACK_SHAREDIR)

    # skipped during debian packaging
    set(OV_CPACK_WHEELSDIR "tools")

    # for BW compatibility
    set(IE_CPACK_LIBRARY_PATH ${OV_CPACK_LIBRARYDIR})
    set(IE_CPACK_RUNTIME_PATH ${OV_CPACK_RUNTIMEDIR})
    set(IE_CPACK_ARCHIVE_PATH ${OV_CPACK_ARCHIVEDIR})
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
    # merge all pythons into a single component
    set(OV_CPACK_COMP_PYTHON_OPENVINO "pyopenvino")
    set(OV_CPACK_COMP_PYTHON_IE_API "${OV_CPACK_COMP_PYTHON_OPENVINO}")
    set(OV_CPACK_COMP_PYTHON_NGRAPH "${OV_CPACK_COMP_PYTHON_OPENVINO}")
    # merge all C / C++ samples as a single samples component
    set(OV_CPACK_COMP_CPP_SAMPLES "samples")
    set(OV_CPACK_COMP_C_SAMPLES "${OV_CPACK_COMP_CPP_SAMPLES}")
    # move requirements.txt to core-dev
    set(OV_CPACK_COMP_DEV_REQ_FILES "${OV_CPACK_COMP_CORE_DEV}")
    # move core_tools to core-dev
    set(OV_CPACK_COMP_CORE_TOOLS "${OV_CPACK_COMP_CORE_DEV}")
endmacro()

ov_override_component_names()

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
        set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS_PRIVATE_DIRS "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
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
        endif()
    endif()
endmacro()

ov_debian_specific_settings()

# needed to override cmake auto generated files
set(def_postinst "${OpenVINO_BINARY_DIR}/_CPack_Packages/postinst")
set(def_postrm "${OpenVINO_BINARY_DIR}/_CPack_Packages/postrm")
set(def_triggers "${OpenVINO_BINARY_DIR}/_CPack_Packages/triggers")

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
# ov_debian_add_changelog_and_copyright(<comp name>)
#
function(ov_debian_add_changelog_and_copyright comp)
    string(TOUPPER "${comp}" ucomp)
    if(NOT DEFINED CPACK_DEBIAN_${ucomp}_PACKAGE_NAME)
        message(FATAL_ERROR "CPACK_DEBIAN_${ucomp}_PACKAGE_NAME is not defined")
    else()
        set(package_name "${CPACK_DEBIAN_${ucomp}_PACKAGE_NAME}")
    endif()
    set(package_name "${CPACK_DEBIAN_${ucomp}_PACKAGE_NAME}")

    # copyright

    install(FILES "${OpenVINO_SOURCE_DIR}/cmake/developer_package/packaging/copyright"
            DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/${package_name}/
            COMPONENT ${comp})

    # create changelog.gz

    find_host_program(gzip_PROGRAM NAMES gzip DOC "Path to gzip tool")
    if(NOT gzip_PROGRAM)
        message(FATAL_ERROR "Failed to find gzip tool")
    endif()

    set(changelog_src "${OpenVINO_SOURCE_DIR}/cmake/developer_package/packaging/changelog")
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
    set(CPACK_COMPONENT_${upper_case}_ARCHITECTURE "${CPACK_COMPONENT_${ucomp}_ARCHITECTURE}")
    set(CPACK_COMPONENT_${upper_case}_DEPENDS "${comp}")

    # take package name
    if(DEFINED CPACK_DEBIAN_${ucomp}_PACKAGE_NAME)
        string(REPLACE "-${cpack_name_ver}" ""
            CPACK_DEBIAN_${upper_case}_PACKAGE_NAME
            "${CPACK_DEBIAN_${ucomp}_PACKAGE_NAME}")
    else()
        message(FATAL_ERROR "CPACK_DEBIAN_${ucomp}_PACKAGE_NAME is not defined")
    endif()

    ov_debian_add_lintian_suppression(${comp_name}
        # it's umbrella package
        "empty-binary-package")

    # add latest to a list of debian packages
    list(APPEND CPACK_COMPONENTS_ALL ${comp_name})
endmacro()
