# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(CMakeParseArguments)
include(CPackComponent)
include(GNUInstallDirs)

#
# ie_cpack_set_library_dir()
#
# Set library directory for cpack
#
function(ie_cpack_set_library_dir)
    if(WIN32)
        set(IE_CPACK_LIBRARY_PATH lib/${ARCH_FOLDER}/$<CONFIG> PARENT_SCOPE)
        set(IE_CPACK_RUNTIME_PATH bin/${ARCH_FOLDER}/$<CONFIG> PARENT_SCOPE)
        set(IE_CPACK_ARCHIVE_PATH lib/${ARCH_FOLDER}/$<CONFIG> PARENT_SCOPE)
        set(IE_CPACK_PLUGIN_PATH lib/${ARCH_FOLDER}/$<CONFIG> PARENT_SCOPE)
    else()
        # if(CPACK_GENERATOR STREQUAL "DEB")
            set(IE_CPACK_LIBRARY_PATH ${CMAKE_INSTALL_LIBDIR}/${CMAKE_LIBRARY_ARCHITECTURE} PARENT_SCOPE)
            set(IE_CPACK_RUNTIME_PATH ${CMAKE_INSTALL_LIBDIR}/${CMAKE_LIBRARY_ARCHITECTURE} PARENT_SCOPE)
            set(IE_CPACK_ARCHIVE_PATH ${CMAKE_INSTALL_LIBDIR}/${CMAKE_LIBRARY_ARCHITECTURE} PARENT_SCOPE)
            set(IE_CPACK_PLUGIN_PATH
                ${CMAKE_INSTALL_LIBDIR}/${CMAKE_LIBRARY_ARCHITECTURE}/openvino${OpenVINO_VERSION}
                PARENT_SCOPE)
        # else()
        #     set(IE_CPACK_LIBRARY_PATH lib/${ARCH_FOLDER} PARENT_SCOPE)
        #     set(IE_CPACK_RUNTIME_PATH lib/${ARCH_FOLDER} PARENT_SCOPE)
        #     set(IE_CPACK_ARCHIVE_PATH lib/${ARCH_FOLDER} PARENT_SCOPE)
        #     set(IE_CPACK_PLUGIN_PATH lib/${ARCH_FOLDER} PARENT_SCOPE)
        # endif()
    endif()
endfunction()

ie_cpack_set_library_dir()

#
# ie_cpack_add_component(NAME ...)
#
# Wraps original `cpack_add_component` and adds component to internal IE list
#
unset(IE_CPACK_COMPONENTS_ALL CACHE)
macro(ie_cpack_add_component NAME)
    list(APPEND IE_CPACK_COMPONENTS_ALL ${NAME})
    set(IE_CPACK_COMPONENTS_ALL "${IE_CPACK_COMPONENTS_ALL}" CACHE STRING "" FORCE)

    cpack_add_component(${NAME} ${args})
endmacro()

# create test component
if(ENABLE_TESTS)
    cpack_add_component(tests DISABLED)
endif()

#
# List of public OpenVINO components
#

set(OV_COMP_CORE "core")
set(OV_COMP_CORE_C "core_c")
set(OV_COMP_CORE_DEV "core_dev")
set(OV_COMP_CORE_C_DEV "core_c_dev")
set(OV_COMP_CORE_TOOLS "core_tools")
set(OV_COMP_CPP_SAMPLES "cpp_samples")
set(OV_COMP_C_SAMPLES "c_samples")
set(OV_COMP_PYTHON_SAMPLES "python_samples")
set(OV_COMP_PYTHON_IE_API "pyie")
set(OV_COMP_PYTHON_NGRAPH "pyngraph")
set(OV_COMP_PYTHON_OPENVINO "pyopenvino")
set(OV_COMP_DEV_REQ_FILES "openvino_dev_req_files")
set(OV_COMP_LICENSING "licensing")
set(OV_COMP_INSTALL_DEPENDENCIES "install_dependencies")

# override cpack components name for DEB cpack generator
if(CPACK_GENERATOR STREQUAL "DEB")
    # merge C++ and C runtimes
    set(OV_COMP_CORE_C "${OV_COMP_CORE}")
    set(OV_COMP_CORE_C_DEV "${OV_COMP_CORE_DEV}")
    # merge all pythons into a single component
    set(OV_COMP_PYTHON_IE_API "${OV_COMP_PYTHON_OPENVINO}")
    set(OV_COMP_PYTHON_NGRAPH "${OV_COMP_PYTHON_OPENVINO}")
    set(OV_COMP_PYTHON_OPENVINO "${OV_COMP_PYTHON_OPENVINO}")
    # merge all C / C++ samples as a single samples component
    set(OV_COMP_CPP_SAMPLES "samples")
    set(OV_COMP_C_SAMPLES "samples")
    # move requirements.txt to core-dev
    set(OV_COMP_DEV_REQ_FILES "${OV_COMP_CORE_DEV}")
    # move core_tools to core-dev
    set(OV_COMP_CORE_TOOLS "${OV_COMP_CORE_DEV}")
    # move licensing to core
    set(OV_COMP_LICENSING "${OV_COMP_CORE}")
    # move install_dependencies to core as well
    set(OV_COMP_INSTALL_DEPENDENCIES "${OV_COMP_CORE}")
endif()

macro(ie_cpack)
    if(NOT DEFINED CPACK_GENERATOR)
        set(CPACK_GENERATOR "TGZ")
    endif()
    set(CPACK_SOURCE_GENERATOR "") # not used
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "OpenVINO™ Toolkit")
    set(CPACK_COMPONENT_UNSPECIFIED_REQUIRED OFF)
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
    set(CPACK_PACKAGE_VENDOR "Intel Corporation")
    set(CPACK_PACKAGE_CONTACT "OpenVINO Developers <openvino@intel.com>")
    set(CPACK_VERBATIM_VARIABLES ON)
    set(CPACK_COMPONENTS_ALL ${ARGN})

    # default permissions for directories creation
    set(CMAKE_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE)

    # archive operations can be run in parallels since CMake 3.20
    set(CPACK_THREADS 8)

    if (NOT DEFINED CPACK_STRIP_FILES)
        set(CPACK_STRIP_FILES ON)
    endif()

    string(REPLACE "/" "_" CPACK_PACKAGE_VERSION "${OpenVINO_VERSION}")
    if(WIN32)
        set(CPACK_PACKAGE_NAME openvino_${CMAKE_BUILD_TYPE})
    else()
        set(CPACK_PACKAGE_NAME openvino)
    endif()

    foreach(ver MAJOR MINOR PATCH)
        if(DEFINED OpenVINO_VERSION_${ver})
            set(CPACK_PACKAGE_VERSION_${ver} ${OpenVINO_VERSION_${ver}})
        endif()
    endforeach()

    # CPACK_PACKAGE_VERSION_MAJOR.CPACK_PACKAGE_VERSION_MINOR
    set(cpack_ver_mm "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}")

    if(OS_FOLDER)
        set(CPACK_SYSTEM_NAME "${OS_FOLDER}")
    endif()

    # generator specific variables
    if(CPACK_GENERATOR MATCHES "^(7Z|TBZ2|TGZ|TXZ|TZ|ZIP)$")
        # multiple packages are generated
        set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
    elseif(CPACK_GENERATOR STREQUAL "DEB")
        # fill a list of components
        unset(CPACK_COMPONENTS_ALL)
        foreach(item ${ARGN})
            # don't provide python components and deployment_manager to end users
            if(NOT ${item} MATCHES ".*(python).*" AND NOT ${item} MATCHES "^deployment_manager$")
                list(APPEND CPACK_COMPONENTS_ALL ${item})
            endif()
        endforeach()
        list(REMOVE_DUPLICATES CPACK_COMPONENTS_ALL)

        # per component configuration
        # list of components:
        # - core
        # - core_c
        # - core_dev
        # - core_c_dev
        # - c_samples
        # - cpp_samples
        # - python_samples
        # - deployment_manager
        # - [install_dependencies] probably should be removed
        # - licensing
        # - docs
        # - [python .*]
        # - (cpu|gpu|hetero)

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
        # is ignored; but dependnencies between our components are here because of
        # CPACK_COMPONENT_<UCOMP>_DEPENDS variables
        # More proper WA is try to enable INSTALL_RPATH
        set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS_PRIVATE_DIRS "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

        # automatic dependencies discovering
        set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS ON)
        # OpenVINO does not have backward and forward compatibility
        set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS_POLICY "=")
        # naming convention for debian package files
        set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")

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
                    DESTINATION share/lintian/overrides/
                    COMPONENT ${comp})
        endfunction()

        # needed to override cmake auto generated files
        set(def_postinst "${OpenVINO_BINARY_DIR}/_CPack_Packages/postinst")
        set(def_postrm "${OpenVINO_BINARY_DIR}/_CPack_Packages/postrm")
        set(triggers_content "activate-noawait ldconfig\n")
        set(def_triggers "${OpenVINO_BINARY_DIR}/_CPack_Packages/triggers/${package_name}/triggers")
        file(WRITE "${def_postinst}" "#!/bin/sh\n\nset -e\n\n")
        file(WRITE "${def_postrm}" "#!/bin/sh\n\nset -e\n\n")
        file(WRITE "${def_triggers}" "${triggers_content}")

        #
        # OpenVINO Core components including frontends
        #

        # core
        set(CPACK_COMPONENT_CORE_DESCRIPTION "OpenVINO C / C++ Runtime libraries")
        set(CPACK_DEBIAN_CORE_PACKAGE_NAME "libopenvino${cpack_ver_mm}")
        # TODO: build with system pugixml and depend on libpugixml1v5
        # set(CPACK_DEBIAN_CORE_PACKAGE_DEPENDS "libpugixml1v5")
        set(CPACK_DEBIAN_CORE_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")

        ov_add_lintian_suppression(core
            # OpenVINO runtime library is named differently
            "package-name-doesnt-match-sonames")

        # core_dev
        set(CPACK_COMPONENT_CORE_DEV_DESCRIPTION "OpenVINO C / C++ Runtime development files")
        set(CPACK_COMPONENT_CORE_DEV_DEPENDS "core")
        set(CPACK_DEBIAN_CORE_DEV_PACKAGE_NAME "libopenvino${cpack_ver_mm}-dev")
        # set(CPACK_DEBIAN_CORE_DEV_PACKAGE_CONFLICTS "")
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
        if(ENABLE_MKL_DNN)
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
        if(ENABLE_MYRIAD)
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
            set(CPACK_DEFIAN_GNA_PACKAGE_SHLIBDEPS OFF)
            set(CPACK_DEBIAN_GNA_PACKAGE_NAME "libopenvino-intel-gna${cpack_ver_mm}")
            set(CPACK_DEBIAN_GNA_PACKAGE_SUGGESTS "libopenvino-auto${cpack_ver_mm} (= ${CPACK_PACKAGE_VERSION}), libopenvino-hetero${cpack_ver_mm} (= ${CPACK_PACKAGE_VERSION})")
            set(CPACK_DEBIAN_GNA_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")
            list(APPEND installed_plugins "gna")
        endif()

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
        set(CPACK_COMPONENT_PYTHON_SAMPLES_DEPENDS "python3")
        set(CPACK_DEBIAN_PYTHON_SAMPLES_PACKAGE_NAME "libopenvino-samples-python${cpack_ver_mm}")
        set(CPACK_DEBIAN_PYTHON_SAMPLES_PACKAGE_ARCHITECTURE "all")

        #
        # Other components
        #

        # deployment manager for runtime
        set(CPACK_COMPONENT_DEPLOYMENT_MANAGER_DESCRIPTION "OpenVINO Deployment Manager")
        set(CPACK_COMPONENT_DEPLOYMENT_MANAGER_DEPENDS "core")
        set(CPACK_DEBIAN_DEPLOYMENT_MANAGER_PACKAGE_NAME "libopenvino-deployment-manager${cpack_ver_mm}")
        set(CPACK_DEBIAN_DEPLOYMENT_MANAGER_PACKAGE_DEPENDS "python3")
        set(CPACK_DEBIAN_DEPLOYMENT_MANAGER_PACKAGE_ARCHITECTURE "all")

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
        # Install latest packages
        #

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

        # NOTE: we expicitly don't add runtime latest packages
        # since a user needs to depend on specific VERSIONED runtime package
        # with fixed SONAMEs, while latest package can be updated multiple times

        ov_add_latest_component(core_dev)
        ov_add_latest_component(samples)
        ov_add_latest_component(libraries_dev)
    endif()

    include(CPack)
endmacro()
