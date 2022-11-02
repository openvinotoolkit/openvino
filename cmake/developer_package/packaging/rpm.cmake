# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(GNUInstallDirs)

#
# ov_rpm_cpack_set_dirs()
#
# Set directories for cpack
#
macro(ov_rpm_cpack_set_dirs)
    # override default locations for RPM
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

    ov_get_pyversion(pyversion)
    if(pyversion)
        set(OV_CPACK_PYTHONDIR ${CMAKE_INSTALL_LIBDIR}/${pyversion}/site-packages)
    endif()

    # non-native stuff
    set(OV_CPACK_SHAREDIR ${CMAKE_INSTALL_DATADIR}/openvino-${OpenVINO_VERSION}) # internal
    set(OV_CPACK_SAMPLESDIR ${OV_CPACK_SHAREDIR}/samples)
    set(OV_CPACK_DEVREQDIR ${OV_CPACK_SHAREDIR})
    unset(OV_CPACK_SHAREDIR)

    # skipped during rpm packaging
    set(OV_CPACK_WHEELSDIR "tools")

    # for BW compatibility
    set(IE_CPACK_LIBRARY_PATH ${OV_CPACK_LIBRARYDIR})
    set(IE_CPACK_RUNTIME_PATH ${OV_CPACK_RUNTIMEDIR})
    set(IE_CPACK_ARCHIVE_PATH ${OV_CPACK_ARCHIVEDIR})
endmacro()

ov_rpm_cpack_set_dirs()

#
# Override CPack components name for RPM generator
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
    # set(OV_CPACK_COMP_PYTHON_SAMPLES "${OV_CPACK_COMP_CPP_SAMPLES}")
    # move requirements.txt to core-dev
    set(OV_CPACK_COMP_DEV_REQ_FILES "${OV_CPACK_COMP_CORE_DEV}")
    # move core_tools to core-dev
    set(OV_CPACK_COMP_CORE_TOOLS "${OV_CPACK_COMP_CORE_DEV}")
endmacro()

ov_override_component_names()

#
# Common RPM specific settings
#

macro(ov_rpm_specific_settings)
    # multiple packages are generated
    set(CPACK_RPM_COMPONENT_INSTALL ON)
    # automatically find dependencies for binaries
    set(CPACK_RPM_PACKAGE_AUTOREQPROV ON)
    # enable dependencies between components
    set(CPACK_RPM_ENABLE_COMPONENT_DEPENDS ON)
    # homepage
    set(CPACK_RPM_PACKAGE_URL "https://docs.openvino.ai/")
    # ASL 2.0 # Apache Software License 2.0
    set(CPACK_RPM_PACKAGE_LICENSE "ASL 2.0")
    # group
    set(CPACK_RPM_PACKAGE_GROUP "Development/Libraries")
    # changelog file
    # TODO: fix "error: bad date in %changelog"
    # set(CPACK_RPM_CHANGELOG_FILE "${OpenVINO_SOURCE_DIR}/cmake/developer_package/packaging/changelog-rpm")
    # use rpmlint to check packages in post-build step
    set(CPACK_POST_BUILD_SCRIPTS "${IEDevScripts_DIR}/packaging/rpm_post_build.cmake")
    # enable for debug cpack run
    if(NOT DEFINED CPACK_RPM_PACKAGE_DEBUG)
        set(CPACK_RPM_PACKAGE_DEBUG OFF)
    endif()

    # naming convention for rpm package files
    set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
    # need to update this version once we rebuild the same package with additional fixes
    # set(CPACK_RPM_PACKAGE_RELEASE "1")
    # enable this if someday we change the version scheme
    # set(CPACK_RPM_PACKAGE_EPOCH "2")
endmacro()

ov_rpm_specific_settings()

# needed to add triggers for packages with libraries
set(def_triggers "${OpenVINO_BINARY_DIR}/_CPack_Packages/triggers")
set(triggers_content "activate-noawait ldconfig\n\n")
file(WRITE "${def_triggers}" "${triggers_content}")

#
# Functions helpful for packaging your modules with RPM cpack
#

#
# ov_rpm_copyright(<comp name>)
#
function(ov_rpm_copyright comp)
    string(TOUPPER "${comp}" ucomp)
    if(NOT DEFINED CPACK_RPM_${ucomp}_PACKAGE_NAME)
        message(FATAL_ERROR "CPACK_RPM_${ucomp}_PACKAGE_NAME is not defined")
    else()
        set(package_name "${CPACK_RPM_${ucomp}_PACKAGE_NAME}")
    endif()
    set(package_name "${CPACK_RPM_${ucomp}_PACKAGE_NAME}")

    # copyright

    install(FILES "${OpenVINO_SOURCE_DIR}/cmake/developer_package/packaging/copyright"
            DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/${package_name}/
            COMPONENT ${comp})
endfunction()

#
# ov_rpm_add_rpmlint_suppression(<comp name> <suppression1, suppression2, ...>)
#
function(ov_rpm_add_rpmlint_suppression comp)
    set(lines ${ARGN})

    string(TOUPPER "${comp}" ucomp)
    if(NOT DEFINED CPACK_RPM_${ucomp}_PACKAGE_NAME)
        message(FATAL_ERROR "CPACK_RPM_${ucomp}_PACKAGE_NAME is not defined")
    else()
        set(package_name "${CPACK_RPM_${ucomp}_PACKAGE_NAME}")
    endif()

    foreach(line IN LISTS lines)
        set(line "addFilter(\"${line}\")")
        if(content)
            set(content "${content}\n${line}")
        else()
            set(content "${line}")
        endif()
    endforeach()

    if(DEFINED CPACK_RPM_${ucomp}_PACKAGE_ARCHITECTURE)
        set(arch "${CPACK_RPM_${ucomp}_PACKAGE_ARCHITECTURE}")
    elseif(X86_64)
        set(arch "x86_64")
    elseif(X86)
        set(arch "i686")
    else()
        message(FATAL_ERROR "RPM: Unsupported architecture ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    set(package_file_name "${package_name}-${OpenVINO_VERSION}-1.${arch}.rpm")
    set(rpmlint_override_file "${OpenVINO_BINARY_DIR}/_CPack_Packages/rpmlint/${package_file_name}.rpmlintrc")
    file(REMOVE ${rpmlint_override_file})
    file(WRITE ${rpmlint_override_file} ${content})
endfunction()

#
# ov_rpm_generate_conflicts(<comp name>)
#
function(ov_rpm_generate_conflicts comp)
    set(cpack_name_versions ${ARGN})
    string(TOUPPER "${comp}" ucomp)

    # sanity check
    if(NOT DEFINED CPACK_RPM_${ucomp}_PACKAGE_NAME)
        message(FATAL_ERROR "CPACK_RPM_${ucomp}_PACKAGE_NAME is not defined")
    else()
        if(NOT DEFINED cpack_name_ver)
            message(FATAL_ERROR "Internal variable 'cpack_name_ver' is not defined")
        endif()

        string(REPLACE "${cpack_name_ver}" "" package_name_base "${CPACK_RPM_${ucomp}_PACKAGE_NAME}")
    endif()

    foreach(cpack_name_version IN LISTS cpack_name_versions)
        if(package_names)
            set(package_names "${package_names}, ${package_name_base}${cpack_name_version}")
        else()
            set(package_names "${package_name_base}${cpack_name_version}")
        endif()
    endforeach()

    set(CPACK_RPM_${ucomp}_PACKAGE_CONFLICTS "${package_names}" PARENT_SCOPE)
endfunction()

#
# ov_rpm_add_latest_component(<comp>)
#
# Adds latest component for `comp`, but without a version
# Description and other stuff (arch) is taken from the main component
#
macro(ov_rpm_add_latest_component comp)
    string(TOUPPER "${comp}" ucomp)
    set(comp_name "${comp}_latest")
    set(upper_case "${ucomp}_LATEST")

    set(CPACK_COMPONENT_${upper_case}_DESCRIPTION "${CPACK_COMPONENT_${ucomp}_DESCRIPTION}")
    set(CPACK_COMPONENT_${upper_case}_DEPENDS "${comp}")
    set(CPACK_RPM_${upper_case}_PACKAGE_ARCHITECTURE "noarch")

    # take package name
    if(DEFINED CPACK_RPM_${ucomp}_PACKAGE_NAME)
        string(REPLACE "-${cpack_name_ver}" ""
            CPACK_RPM_${upper_case}_PACKAGE_NAME
            "${CPACK_RPM_${ucomp}_PACKAGE_NAME}")
    else()
        message(FATAL_ERROR "CPACK_RPM_${ucomp}_PACKAGE_NAME is not defined")
    endif()

    # add latest to a list of rpm packages
    list(APPEND CPACK_COMPONENTS_ALL ${comp_name})
endmacro()
