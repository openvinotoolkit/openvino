# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# ov_cpack_set_dirs()
#
# Set directories for cpack
#
macro(ov_debian_cpack_set_dirs)
    # override default locations for Debian
    set(OV_CPACK_TOOLSDIR ${CMAKE_INSTALL_BINDIR}) # only C++ tools are here
    set(OV_CPACK_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR})
    set(OV_CPACK_LIBRARYDIR ${CMAKE_INSTALL_LIBDIR})
    set(OV_CPACK_RUNTIMEDIR ${CMAKE_INSTALL_LIBDIR})
    set(OV_CPACK_ARCHIVEDIR ${CMAKE_INSTALL_LIBDIR})
    set(OV_CPACK_PLUGINSDIR ${CMAKE_INSTALL_LIBDIR}/openvino${OpenVINO_VERSION})
    set(OV_CPACK_IE_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/inferenceengine${OpenVINO_VERSION})
    set(OV_CPACK_NGRAPH_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/ngraph${OpenVINO_VERSION})
    set(OV_CPACK_OPENVINO_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/openvino${OpenVINO_VERSION})
    set(OV_CPACK_DOCDIR ${CMAKE_INSTALL_DATADIR}/doc/openvino${OpenVINO_VERSION})

    # non-native stuff
    set(OV_CPACK_PYTHONDIR ${OV_CPACK_PLUGINSDIR})
    set(OV_CPACK_SHAREDIR ${CMAKE_INSTALL_DATADIR}/openvino${OpenVINO_VERSION}) # internal
    set(OV_CPACK_SAMPLESDIR ${OV_CPACK_SHAREDIR}/samples)
    set(OV_CPACK_DEVREQDIR ${OV_CPACK_SHAREDIR})

    set(OV_CPACK_WHEELSDIR .) # TODO

    # for BW compatibility
    set(IE_CPACK_LIBRARY_PATH ${OV_CPACK_LIBRARYDIR})
    set(IE_CPACK_RUNTIME_PATH ${OV_CPACK_RUNTIMEDIR})
    set(IE_CPACK_ARCHIVE_PATH ${OV_CPACK_ARCHIVEDIR})
endmacro()

ov_debian_cpack_set_dirs()

#
# override CPack components name for Debian generator
#

# merge C++ and C runtimes
set(OV_CPACK_COMP_CORE_C "${OV_CPACK_COMP_CORE}")
set(OV_CPACK_COMP_CORE_C_DEV "${OV_CPACK_COMP_CORE_DEV}")
# merge all pythons into a single component
set(OV_CPACK_COMP_PYTHON_OPENVINO "python")
set(OV_CPACK_COMP_PYTHON_IE_API "${OV_CPACK_COMP_PYTHON_OPENVINO}")
set(OV_CPACK_COMP_PYTHON_NGRAPH "${OV_CPACK_COMP_PYTHON_OPENVINO}")
# merge all C / C++ samples as a single samples component
set(OV_CPACK_COMP_CPP_SAMPLES "samples")
set(OV_CPACK_COMP_C_SAMPLES "${OV_CPACK_COMP_CPP_SAMPLES}")
set(OV_CPACK_COMP_PYTHON_SAMPLES "${OV_CPACK_COMP_CPP_SAMPLES}")
# move requirements.txt to core-dev
set(OV_CPACK_COMP_DEV_REQ_FILES "${OV_CPACK_COMP_CORE_DEV}")
# move core_tools to core-dev
set(OV_CPACK_COMP_CORE_TOOLS "${OV_CPACK_COMP_CORE_DEV}")
# move licensing to core
set(OV_CPACK_COMP_LICENSING "${OV_CPACK_COMP_CORE}")

#
# Common settings
#

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
# but it require CMake source code changes
# with current WA automatic deps detection via dpkg-shlibdeps for "our libraries"
# is ignored; but dependencies between our components are here because of
# CPACK_COMPONENT_<UCOMP>_DEPENDS variables
# More proper WA is try to enable INSTALL_RPATH
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS_PRIVATE_DIRS "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

# automatic dependencies discovering
set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS ON)
# OpenVINO does not have backward and forward compatibility
set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS_POLICY "=")
# naming convention for debian package files
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")

# needed to override cmake auto generated files
set(def_postinst "${OpenVINO_BINARY_DIR}/_CPack_Packages/postinst")
set(def_postrm "${OpenVINO_BINARY_DIR}/_CPack_Packages/postrm")
set(def_triggers "${OpenVINO_BINARY_DIR}/_CPack_Packages/triggers/${package_name}/triggers")

set(triggers_content "activate-noawait ldconfig\n")
set(post_content "#!/bin/sh\n\nset -e\n\n")

file(WRITE "${def_postinst}" "${post_content}")
file(WRITE "${def_postrm}" "${post_content}")
file(WRITE "${def_triggers}" "${triggers_content}")

#
# Common functions
#

function(ov_add_lintian_suppression comp)
    set(lines ${ARGN})
    list(APPEND lines "copyright-file-contains-full-apache-2-license")
    list(APPEND lines "copyright-should-refer-to-common-license-file-for-apache-2")
    list(APPEND lines "copyright-without-copyright-notice")
    # TODO: fix
    list(APPEND lines "changelog-file-missing-in-native-package")
    # TODO: remove them
    list(APPEND lines "maintainer-script-empty postinst")
    list(APPEND lines "maintainer-script-empty postrm")

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

macro(ov_add_latest_component comp)
    string(TOUPPER "${comp}" ucomp)
    set(latest "${ucomp}_LATEST")

    set(CPACK_COMPONENT_${latest}_DESCRIPTION "${CPACK_COMPONENT_${ucomp}_DESCRIPTION}")
    set(CPACK_COMPONENT_${latest}_ARCHITECTURE "${CPACK_COMPONENT_${ucomp}_ARCHITECTURE}")
    set(CPACK_COMPONENT_${latest}_DEPENDS "${ucomp}")

    # take package name
    if(DEFINED CPACK_DEBIAN_${ucomp}_PACKAGE_NAME)
        string(REPLACE "${cpack_ver_mm}" ""
            CPACK_DEBIAN_${latest}_PACKAGE_NAME
            "${CPACK_DEBIAN_${ucomp}_PACKAGE_NAME}")
    else()
        message(FATAL_ERROR "CPACK_DEBIAN_${ucomp}_PACKAGE_NAME is not defined")
    endif()

    # add latest to a list of debian packages
    list(APPEND CPACK_COMPONENTS_ALL ${latest})
endmacro()

#
# OpenVINO Core components including frontends
#

macro(ov_debian_components)
    # fill a list of components which are part of debian
    set(cpack_components_all ${CPACK_COMPONENTS_ALL})
    unset(CPACK_COMPONENTS_ALL)
    foreach(item ${cpack_components_all})
        # don't provide python components and deployment_manager to end users
        if(# NOT ${item} MATCHES ".*(python).*" AND
           NOT ${item} MATCHES "^${OV_CPACK_COMP_DEPLOYMENT_MANAGER}$")
            list(APPEND CPACK_COMPONENTS_ALL ${item})
        endif()
    endforeach()
    list(REMOVE_DUPLICATES CPACK_COMPONENTS_ALL)

    # CPACK_PACKAGE_VERSION_MAJOR.CPACK_PACKAGE_VERSION_MINOR
    set(cpack_ver_mm "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}")

    # core
    set(CPACK_COMPONENT_CORE_DESCRIPTION "OpenVINO C / C++ Runtime libraries")
    set(CPACK_DEBIAN_CORE_PACKAGE_NAME "libopenvino${cpack_ver_mm}")
    set(CPACK_DEBIAN_CORE_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")

    ov_add_lintian_suppression(core
        # OpenVINO runtime library is named differently
        "package-name-doesnt-match-sonames")

    # core_dev
    set(CPACK_COMPONENT_CORE_DEV_DESCRIPTION "OpenVINO C / C++ Runtime development files")
    set(CPACK_COMPONENT_CORE_DEV_DEPENDS "core")
    set(CPACK_DEBIAN_CORE_DEV_PACKAGE_NAME "libopenvino${cpack_ver_mm}-dev")
    set(CPACK_DEBIAN_CORE_DEV_PACKAGE_CONFLICTS "libopenvino2021.3-dev, libopenvino2021.4-dev")
    ov_add_lintian_suppression(core_dev)

    #
    # Plugins
    #

    # hetero
    if(ENABLE_HETERO)
        set(CPACK_COMPONENT_HETERO_DESCRIPTION "OpenVINO Hetero plugin")
        set(CPACK_COMPONENT_HETERO_DEPENDS "core")
        set(CPACK_DEBIAN_HETERO_PACKAGE_NAME "libopenvino-hetero${cpack_ver_mm}")
        set(CPACK_DEBIAN_HETERO_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")
        list(APPEND installed_plugins "hetero")
    endif()

    # auto batch
    if(ENABLE_AUTO_BATCH)
        set(CPACK_COMPONENT_BATCH_DESCRIPTION "OpenVINO Auto Batch plugin")
        set(CPACK_COMPONENT_BATCH_DEPENDS "core")
        set(CPACK_DEBIAN_BATCH_PACKAGE_NAME "libopenvino-auto-batch${cpack_ver_mm}")
        set(CPACK_DEBIAN_BATCH_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")
        list(APPEND installed_plugins "batch")
    endif()

    # multi / auto plugins
    if(ENABLE_MULTI)
        if(ENABLE_AUTO)
            set(CPACK_COMPONENT_MULTI_DESCRIPTION "OpenVINO Auto / Multi plugin")
        else()
            set(CPACK_COMPONENT_MULTI_DESCRIPTION "OpenVINO Multi plugin")
        endif()
        set(CPACK_COMPONENT_MULTI_DEPENDS "core")
        set(CPACK_DEBIAN_MULTI_PACKAGE_NAME "libopenvino-auto${cpack_ver_mm}")
        set(CPACK_DEBIAN_MULTI_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")
        list(APPEND installed_plugins "multi")
    elseif(ENABLE_AUTO)
        set(CPACK_COMPONENT_AUTO_DESCRIPTION "OpenVINO Auto plugin")
        set(CPACK_COMPONENT_AUTO_DEPENDS "core")
        set(CPACK_DEBIAN_AUTO_PACKAGE_NAME "libopenvino-auto${cpack_ver_mm}")
        set(CPACK_DEBIAN_AUTO_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")
        list(APPEND installed_plugins "auto")
    endif()

    # intel-cpu
    if(ENABLE_INTEL_CPU)
        set(CPACK_COMPONENT_CPU_DESCRIPTION "OpenVINO Intel CPU plugin")
        set(CPACK_COMPONENT_CPU_DEPENDS "core")
        set(CPACK_DEBIAN_CPU_PACKAGE_NAME "libopenvino-intel-cpu${cpack_ver_mm}")
        set(CPACK_DEBIAN_CPU_PACKAGE_SUGGESTS "libopenvino-auto${cpack_ver_mm} (= ${CPACK_PACKAGE_VERSION}), libopenvino-hetero${cpack_ver_mm} (= ${CPACK_PACKAGE_VERSION})")
        set(CPACK_DEBIAN_CPU_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")
        list(APPEND installed_plugins "cpu")
    endif()

    # intel-gpu
    if(ENABLE_INTEL_GPU)
        set(CPACK_COMPONENT_GPU_DESCRIPTION "OpenVINO Intel GPU plugin")
        set(CPACK_COMPONENT_GPU_DEPENDS "core")
        set(CPACK_DEBIAN_GPU_PACKAGE_NAME "libopenvino-intel-gpu${cpack_ver_mm}")
        set(CPACK_DEBIAN_GPU_PACKAGE_SUGGESTS "libopenvino-auto${cpack_ver_mm} (= ${CPACK_PACKAGE_VERSION}), libopenvino-hetero${cpack_ver_mm} (= ${CPACK_PACKAGE_VERSION})")
        set(CPACK_DEBIAN_GPU_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")
        list(APPEND installed_plugins "gpu")
    endif()

    # intel-myriad
    if(ENABLE_INTEL_MYRIAD)
        set(CPACK_COMPONENT_MYRIAD_DESCRIPTION "OpenVINO Intel Myriad plugin")
        set(CPACK_COMPONENT_MYRIAD_DEPENDS "core")
        set(CPACK_DEBIAN_MYRIAD_PACKAGE_NAME "libopenvino-intel-myriad${cpack_ver_mm}")
        set(CPACK_DEBIAN_MYRIAD_PACKAGE_SUGGESTS "libopenvino-auto${cpack_ver_mm} (= ${CPACK_PACKAGE_VERSION}), libopenvino-hetero${cpack_ver_mm} (= ${CPACK_PACKAGE_VERSION})")
        set(CPACK_DEBIAN_MYRIAD_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")
        list(APPEND installed_plugins "myriad")
    endif()

    # intel-gna
    if(ENABLE_INTEL_GNA)
        set(CPACK_COMPONENT_GNA_DESCRIPTION "OpenVINO Intel GNA plugin")
        set(CPACK_COMPONENT_GNA_DEPENDS "core")
        set(CPACK_DEBIAN_GNA_PACKAGE_NAME "libopenvino-intel-gna${cpack_ver_mm}")
        set(CPACK_DEBIAN_GNA_PACKAGE_SUGGESTS "libopenvino-auto${cpack_ver_mm} (= ${CPACK_PACKAGE_VERSION}), libopenvino-hetero${cpack_ver_mm} (= ${CPACK_PACKAGE_VERSION})")
        set(CPACK_DEBIAN_GNA_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")
        list(APPEND installed_plugins "gna")
    endif()

    #
    # Python bindings
    #

    set(CPACK_COMPONENT_PYTHON_PYTHON3.8_DESCRIPTION "OpenVINO Python bindings")
    if(installed_plugins)
        set(CPACK_COMPONENT_PYTHON_PYTHON3.8_DEPENDS "${installed_plugins}")
    else()
        set(CPACK_COMPONENT_PYTHON_PYTHON3.8_DEPENDS "core")
    endif()
    set(CPACK_DEBIAN_PYTHON_PYTHON3.8_PACKAGE_NAME "libopenvino-python${cpack_ver_mm}")

    #
    # Samples
    #

    set(samples_build_deps "cmake, g++, gcc, libc6-dev, make")
    set(samples_build_deps_suggest "${samples_build_deps}, libopencv-core-dev, libopencv-imgproc-dev, libopencv-imgcodecs-dev")

    # c_samples / cpp_samples
    set(CPACK_COMPONENT_SAMPLES_DESCRIPTION "OpenVINO C / C++ samples")
    set(CPACK_COMPONENT_SAMPLES_DEPENDS "core_dev")
    set(CPACK_DEBIAN_SAMPLES_PACKAGE_NAME "libopenvino-samples${cpack_ver_mm}")
    set(CPACK_DEBIAN_SAMPLES_PACKAGE_SUGGESTS "${samples_build_deps_suggest}, libopenvino-hetero${cpack_ver_mm} (= ${CPACK_PACKAGE_VERSION})")
    set(CPACK_DEBIAN_SAMPLES_PACKAGE_DEPENDS "libgflags-dev, nlohmann-json3-dev, zlib1g-dev, ${samples_build_deps}")
    set(CPACK_DEBIAN_SAMPLES_PACKAGE_ARCHITECTURE "all")

    # python_samples
    set(CPACK_COMPONENT_PYTHON_SAMPLES_DESCRIPTION "OpenVINO Python samples")
    set(CPACK_COMPONENT_PYTHON_SAMPLES_DEPENDS "python_python3.8")
    set(CPACK_DEBIAN_PYTHON_SAMPLES_PACKAGE_NAME "libopenvino-samples-python${cpack_ver_mm}")
    set(CPACK_DEBIAN_PYTHON_SAMPLES_PACKAGE_ARCHITECTURE "all")

    #
    # Add virtual packages
    #

    # all libraries
    set(CPACK_COMPONENT_LIBRARIES_DESCRIPTION "OpenVINO all runtime libraries")
    if(installed_plugins)
        set(CPACK_COMPONENT_LIBRARIES_DEPENDS "${installed_plugins}")
    else()
        set(CPACK_COMPONENT_LIBRARIES_DEPENDS "core")
    endif()
    set(CPACK_DEBIAN_LIBRARIES_PACKAGE_NAME "libopenvino-libraries${cpack_ver_mm}")
    list(APPEND CPACK_COMPONENTS_ALL "libraries")

    # all libraries-dev
    set(CPACK_COMPONENT_LIBRARIES_DEV_DESCRIPTION "OpenVINO all runtime libraries and development files")
    set(CPACK_COMPONENT_LIBRARIES_DEV_DEPENDS "core_dev;${installed_plugins}")
    set(CPACK_DEBIAN_LIBRARIES_DEV_PACKAGE_NAME "libopenvino-libraries${cpack_ver_mm}-dev")
    list(APPEND CPACK_COMPONENTS_ALL "libraries_dev")

    #
    # install debian common files
    #

    foreach(comp IN LISTS CPACK_COMPONENTS_ALL)
        string(TOUPPER "${comp}" ucomp)
        set(package_name "${CPACK_DEBIAN_${ucomp}_PACKAGE_NAME}")

        # copyright
        # install(FILES "${OpenVINO_SOURCE_DIR}/LICENSE"
        #         DESTINATION share/doc/${package_name}/
        #         COMPONENT ${comp}
        #         RENAME copyright)

        # TODO: install changelog

        # install triggers
        set(triggers_content "activate-noawait ldconfig")
        set(triggers_file "${OpenVINO_BINARY_DIR}/_CPack_Packages/triggers/${package_name}/triggers")
        file(REMOVE ${triggers_file})
        file(WRITE ${triggers_file} ${triggers_content})
        install(FILES ${triggers_file}
                DESTINATION ../DEBIAN/
                COMPONENT ${comp})
    endforeach()

    #
    # Install latest symlink packages
    #

    # NOTE: we expicitly don't add runtime latest packages
    # since a user needs to depend on specific VERSIONED runtime package
    # with fixed SONAMEs, while latest package can be updated multiple times

    # ov_add_latest_component(core_dev)
    # ov_add_latest_component(samples)
    # ov_add_latest_component(libraries_dev)
endmacro()
