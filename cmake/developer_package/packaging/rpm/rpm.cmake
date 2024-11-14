# Copyright (C) 2018-2024 Intel Corporation
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
    set(OV_CPACK_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR})
    set(OV_CPACK_LIBRARYDIR ${CMAKE_INSTALL_LIBDIR})
    set(OV_CPACK_RUNTIMEDIR ${CMAKE_INSTALL_LIBDIR})
    set(OV_CPACK_ARCHIVEDIR ${CMAKE_INSTALL_LIBDIR})
    set(OV_CPACK_PLUGINSDIR ${CMAKE_INSTALL_LIBDIR}/openvino-${OpenVINO_VERSION})
    set(OV_CPACK_OPENVINO_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/openvino${OpenVINO_VERSION})
    set(OV_CPACK_DOCDIR ${CMAKE_INSTALL_DATADIR}/doc/openvino-${OpenVINO_VERSION})
    set(OV_CPACK_LICENSESDIR ${OV_CPACK_DOCDIR}/licenses)

    ov_get_pyversion(pyversion)
    if(pyversion)
        set(OV_CPACK_PYTHONDIR ${CMAKE_INSTALL_LIBDIR}/${pyversion}/site-packages)
    endif()

    # non-native stuff
    set(OV_CPACK_SHAREDIR ${CMAKE_INSTALL_DATADIR}/openvino) # internal
    set(OV_CPACK_SAMPLESDIR ${OV_CPACK_SHAREDIR}/samples)
    set(OV_CPACK_DEVREQDIR ${OV_CPACK_SHAREDIR})
    unset(OV_CPACK_SHAREDIR)

    # skipped during rpm packaging
    set(OV_CPACK_WHEELSDIR "wheels")
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
# (e.g. rpm packages don't require setupvars scripts or others)
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
    # we don't need wheels in RPM packages
    set(OV_CPACK_COMP_PYTHON_WHEELS_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # because numpy is installed by rpm
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
# Common RPM specific settings
#

macro(ov_rpm_specific_settings)
    # multiple packages are generated
    set(CPACK_RPM_COMPONENT_INSTALL ON)
    # automatically find dependencies for binaries
    set(CPACK_RPM_PACKAGE_AUTOREQPROV ON)
    # homepage
    set(CPACK_RPM_PACKAGE_URL "https://docs.openvino.ai/")
    # ASL 2.0 # Apache Software License 2.0
    set(CPACK_RPM_PACKAGE_LICENSE "ASL 2.0")
    # group
    set(CPACK_RPM_PACKAGE_GROUP "Development/Libraries")
    # changelog file
    set(CPACK_RPM_CHANGELOG_FILE "${OpenVINO_SOURCE_DIR}/cmake/developer_package/packaging/rpm/changelog")
    # use rpmlint to check packages in post-build step
    set(CPACK_POST_BUILD_SCRIPTS "${OpenVINODeveloperScripts_DIR}/packaging/rpm/post_build.cmake")
    # enable for debug cpack run
    ov_set_if_not_defined(CPACK_RPM_PACKAGE_DEBUG OFF)

    # naming convention for rpm package files
    set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
    # need to update this version once we rebuild the same package with additional fixes
    set(CPACK_RPM_PACKAGE_RELEASE "1")
    # enable this if someday we change the version scheme
    # set(CPACK_RPM_PACKAGE_EPOCH "2")

    # temporary WA for rpm package architecture for cross-compilation
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

    # we don't need RPATHs, because libraries are search by standard paths
    set(CMAKE_SKIP_INSTALL_RPATH ON)
endmacro()

ov_rpm_specific_settings()

# needed to add triggers for packages with libraries
set(def_triggers "${OpenVINO_BINARY_DIR}/_CPack_Packages/triggers")
set(triggers_content "# /bin/sh -p\n/sbin/ldconfig\n")
file(WRITE "${def_triggers}" "${triggers_content}")

#
# Functions helpful for packaging your modules with RPM cpack
#

#
# ov_rpm_copyright(<comp name> <copyright_name>)
#
function(ov_rpm_copyright comp copyright_name)
    string(TOUPPER "${comp}" ucomp)
    if(NOT DEFINED CPACK_RPM_${ucomp}_PACKAGE_NAME)
        message(FATAL_ERROR "CPACK_RPM_${ucomp}_PACKAGE_NAME is not defined")
    else()
        set(package_name "${CPACK_RPM_${ucomp}_PACKAGE_NAME}")
    endif()
    set(package_name "${CPACK_RPM_${ucomp}_PACKAGE_NAME}")

    # copyright

    install(FILES "${OpenVINO_SOURCE_DIR}/cmake/packaging/copyrights/${copyright_name}"
            DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/${package_name}/
            COMPONENT ${comp}
            RENAME "copyright")
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
    elseif(DEFINED CPACK_RPM_PACKAGE_ARCHITECTURE)
        set(arch "${CPACK_RPM_PACKAGE_ARCHITECTURE}")
    elseif(X86_64)
        set(arch "x86_64")
    elseif(X86)
        set(arch "i686")
    elseif(AARCH64)
        set(arch "aarch64")
    elseif(ARM)
        set(arch "armhf")
    else()
        message(FATAL_ERROR "RPM: Unsupported architecture ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    set(package_file_name "${package_name}-${CPACK_PACKAGE_VERSION}-1.${arch}.rpm")
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
    set(CPACK_RPM_${upper_case}_PACKAGE_REQUIRES "${CPACK_RPM_${ucomp}_PACKAGE_NAME} = ${cpack_full_ver}")
    set(CPACK_RPM_${upper_case}_PACKAGE_ARCHITECTURE "noarch")
    set(${comp_name}_copyright "generic")

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
