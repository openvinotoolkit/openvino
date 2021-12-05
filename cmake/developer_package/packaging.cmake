# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(CMakeParseArguments)
include(CPackComponent)

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
    else()
        if(CPACK_GENERATOR STREQUAL "DEB")
            # TODO
            set(IE_CPACK_LIBRARY_PATH lib PARENT_SCOPE)
            set(IE_CPACK_RUNTIME_PATH lib PARENT_SCOPE)
            set(IE_CPACK_ARCHIVE_PATH lib PARENT_SCOPE)
        else()
            set(IE_CPACK_LIBRARY_PATH lib/${ARCH_FOLDER} PARENT_SCOPE)
            set(IE_CPACK_RUNTIME_PATH lib/${ARCH_FOLDER} PARENT_SCOPE)
            set(IE_CPACK_ARCHIVE_PATH lib/${ARCH_FOLDER} PARENT_SCOPE)
        endif()
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
    set(CMAKE_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE)

    # archive operations can be run in parallels since CMake 3.20
    set(CPACK_THREADS 8)

    if (NOT DEFINED CPACK_STRIP_FILES)
        set(CPACK_STRIP_FILES ON)
    endif()

    string(REPLACE "/" "_" CPACK_PACKAGE_VERSION "2022.1.1")
    if(WIN32)
        set(CPACK_PACKAGE_NAME openvino_${CMAKE_BUILD_TYPE})
    else()
        set(CPACK_PACKAGE_NAME openvino)
    endif()

    foreach(ver IN LISTS MAJOR MINOR PATCH)
        if(DEFINED IE_VERSION_${ver})
            set(CPACK_PACKAGE_VERSION_${ver} ${IE_VERSION_${ver}})
        endif()
    endforeach()

    if(OS_FOLDER)
        set(CPACK_SYSTEM_NAME "${OS_FOLDER}")
    endif()

    # generator specific variables
    if(CPACK_GENERATOR MATCHES "^(7Z|TBZ2|TGZ|TXZ|TZ|ZIP)$")
        # multiple packages are generated
        set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
    elseif(CPACK_GENERATOR STREQUAL "DEB")
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
        set(CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION ON)
        # homepage
        set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://docs.openvino.ai/")
        # enable for debug cpack run
        if(NOT DEFINED CPACK_DEBIAN_PACKAGE_DEBUG)
            set(CPACK_DEBIAN_PACKAGE_DEBUG OFF)
        endif()

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
            # add SOVERSION
            list(APPEND lines "package-must-activate-ldconfig-trigger")
            # add SOVERSION
            list(APPEND lines "shlib-without-versioned-soname")
            # add SOVERSION
            list(APPEND lines "sharedobject-in-library-directory-missing-soname")
            list(APPEND lines "maintscript-calls-ldconfig postinst")
            list(APPEND lines "maintscript-calls-ldconfig postrm")

            foreach(line IN LISTS lines)
                set(line "openvino-${comp} binary: ${line}")
                if(content)
                    set(content "${content}\n${line}")
                else()
                    set(content "${line}")
                endif()
            endforeach()

            string(TOUPPER "${comp}" ucomp)
            set(package_name "${CPACK_DEBIAN_${ucomp}_PACKAGE_NAME}")
            set(lintian_override_file "${OpenVINO_BINARY_DIR}/_CPack_Packages/lintian/${package_name}")
            file(WRITE ${lintian_override_file} ${content})
            install(FILES ${lintian_override_file}
                    DESTINATION share/lintian/overrides/
                    COMPONENT ${comp})
        endfunction()

        #
        # OpenVINO Core components including frontends
        #

        # TODO
        # add preinst script for core to ask about aptin

        # admin, devel, doc, see https://www.debian.org/doc/debian-policy/ch-archive.html#s-subsections
        # set(CPACK_DEBIAN_ngraph_PACKAGE_SECTION devel)
        # see required, important, standard, optional, extra
        # https://www.debian.org/doc/debian-policy/ch-archive.html#s-priorities
        # set(CPACK_DEBIAN_ngraph_PACKAGE_PRIORITY standard)

        # core
        set(CPACK_COMPONENT_CORE_DESCRIPTION "OpenVINO C++ Runtime libraries")
        set(CPACK_DEBIAN_CORE_PACKAGE_NAME "libopenvino-core")
        set(CPACK_COMPONENT_CORE_DEV_DEPENDS "install_dependencies;licensing")
        # TODO: should be discovered automatically
        # TODO: install build dependencies libpugixml-dev, libtbb-dev
        set(CPACK_DEBIAN_CORE_PACKAGE_DEPENDS "libpugixml1v5")
        ov_add_lintian_suppression(core
            "package-name-doesnt-match-sonames"
            "script-not-executable usr/share/samples/scripts/utils.sh"
            "non-standard-file-perm usr/share/samples/cpp/thirdparty/.*")

        # core_dev
        set(CPACK_COMPONENT_CORE_DEV_DESCRIPTION "OpenVINO C++ Runtime development files")
        set(CPACK_COMPONENT_CORE_DEV_DEPENDS "core")
        set(CPACK_DEBIAN_CORE_DEV_PACKAGE_NAME "libopenvino-core-dev")
        # Looks like it's arch dependent
        # set(CPACK_DEBIAN_CORE_DEV_PACKAGE_ARCHITECTURE "all")
        # set(CPACK_DEBIAN_CORE_PACKAGE_DEPENDS "libtbb-dev")
        # set(CPACK_DEBIAN_CORE_DEV_PACKAGE_CONFLICTS "!!!")
        ov_add_lintian_suppression(core_dev
            "bad-package-name"
            "package-name-doesnt-match-sonames"
            "script-not-executable usr/share/samples/scripts/utils.sh"
            "non-standard-file-perm usr/share/samples/cpp/thirdparty/.*")

        # core_c
        set(CPACK_COMPONENT_CORE_C_DESCRIPTION "OpenVINO C Runtime libraries")
        set(CPACK_COMPONENT_CORE_C_DEPENDS "core")
        set(CPACK_DEBIAN_CORE_C_PACKAGE_NAME "libopenvino-core-c")

        # core_c_dev
        set(CPACK_COMPONENT_CORE_C_DEV_DESCRIPTION "OpenVINO C Runtime development files")
        set(CPACK_COMPONENT_CORE_C_DEV_DEPENDS "core_c;core_dev")
        set(CPACK_DEBIAN_CORE_C_DEV_PACKAGE_NAME "libopenvino-core-c-dev")
        # Looks like it's arch dependent
        # set(CPACK_DEBIAN_CORE_C_DEV_PACKAGE_ARCHITECTURE "all")
        # set(CPACK_DEBIAN_CORE_C_DEV_PACKAGE_CONFLICTS "!!!")

        #
        # Plugins
        #

        # hetero
        if(ENABLE_HETERO)
            set(CPACK_COMPONENT_HETERO_DESCRIPTION "OpenVINO Hetero plugin")
            set(CPACK_COMPONENT_HETERO_DEPENDS "core")
            set(CPACK_DEBIAN_HETERO_PACKAGE_NAME "libopenvino-hetero")
        endif()

        # multi / auto plugins
        if(ENABLE_MULTI)
            set(CPACK_COMPONENT_MULTI_DESCRIPTION "OpenVINO Multi / Auto plugin")
            set(CPACK_COMPONENT_MULTI_DEPENDS "core")
            set(CPACK_DEBIAN_MULTI_PACKAGE_NAME "libopenvino-multi")
        endif()

        # cpu
        if(ENABLE_MKL_DNN)
            set(CPACK_COMPONENT_CPU_DESCRIPTION "OpenVINO Intel CPU plugin")
            set(CPACK_COMPONENT_CPU_DEPENDS "core")
            set(CPACK_DEBIAN_CPU_PACKAGE_NAME "libopenvino-intel-cpu")
            set(CPACK_DEBIAN_CPU_PACKAGE_SUGGESTS "openvino-multi (= ${CPACK_PACKAGE_VERSION}), openvino-hetero (= ${CPACK_PACKAGE_VERSION})")
        endif()

        # gpu
        if(ENABLE_INTEL_GPU)
            set(CPACK_COMPONENT_GPU_DESCRIPTION "OpenVINO Intel GPU plugin")
            set(CPACK_COMPONENT_GPU_DEPENDS "core")
            set(CPACK_DEBIAN_GPU_PACKAGE_NAME "libopenvino-intel-gpu")
            set(CPACK_DEBIAN_GPU_PACKAGE_SUGGESTS "openvino-multi (= ${CPACK_PACKAGE_VERSION}), openvino-hetero (= ${CPACK_PACKAGE_VERSION})")
        endif()

        # myriad
        if(ENABLE_MYRIAD)
            set(CPACK_COMPONENT_MYRIAD_DESCRIPTION "OpenVINO Intel Myriad plugin")
            set(CPACK_COMPONENT_MYRIAD_DEPENDS "core")
            set(CPACK_DEBIAN_MYRIAD_PACKAGE_NAME "libopenvino-intel-myriad")
            set(CPACK_DEBIAN_MYRIAD_PACKAGE_SUGGESTS "openvino-multi (= ${CPACK_PACKAGE_VERSION}), openvino-hetero (= ${CPACK_PACKAGE_VERSION})")
        endif()

        # gna
        if(ENABLE_GNA)
            set(CPACK_COMPONENT_GNA_DESCRIPTION "OpenVINO Intel GNA plugin")
            set(CPACK_COMPONENT_GNA_DEPENDS "core")
            set(CPACK_DEBIAN_GNA_PACKAGE_NAME "libopenvino-intel-gna")
            set(CPACK_DEBIAN_GNA_PACKAGE_SUGGESTS "openvino-multi (= ${CPACK_PACKAGE_VERSION}), openvino-hetero (= ${CPACK_PACKAGE_VERSION})")
        endif()

        #
        # Samples
        #

        set(samples_build_deps "cmake, g++, gcc, libc6-dev, make")

        # cpp_samples
        set(CPACK_COMPONENT_CPP_SAMPLES_DESCRIPTION "OpenVINO C++ samples")
        set(CPACK_COMPONENT_CPP_SAMPLES_DEPENDS "core_dev")
        set(CPACK_DEBIAN_CPP_SAMPLES_PACKAGE_NAME "libopenvino-samples-cpp")
        set(CPACK_DEBIAN_CPP_SAMPLES_PACKAGE_SUGGESTS "openvino-hetero (= ${CPACK_PACKAGE_VERSION})")
        set(CPACK_DEBIAN_CPP_SAMPLES_PACKAGE_DEPENDS "libgflags-dev, nlohmann-json3-dev, zlib1g-dev, ${samples_build_deps}")
        set(CPACK_DEBIAN_CPP_SAMPLES_PACKAGE_ARCHITECTURE "all")

        # c_samples
        set(CPACK_COMPONENT_C_SAMPLES_DESCRIPTION "OpenVINO C samples")
        set(CPACK_COMPONENT_C_SAMPLES_DEPENDS "core_c_dev")
        set(CPACK_DEBIAN_C_SAMPLES_PACKAGE_NAME "libopenvino-samples-c")
        set(CPACK_DEBIAN_C_SAMPLES_PACKAGE_DEPENDS "${samples_build_deps}")
        set(CPACK_DEBIAN_C_SAMPLES_PACKAGE_ARCHITECTURE "all")

        # python_samples
        set(CPACK_COMPONENT_PYTHON_SAMPLES_DESCRIPTION "OpenVINO Python samples")
        set(CPACK_COMPONENT_PYTHON_SAMPLES_DEPENDS "python3")
        set(CPACK_DEBIAN_PYTHON_SAMPLES_PACKAGE_NAME "libopenvino-samples-python")
        set(CPACK_DEBIAN_PYTHON_SAMPLES_PACKAGE_ARCHITECTURE "all")

        #
        # other
        #

        # deployment manager for runtime
        set(CPACK_COMPONENT_DEPLOYMENT_MANAGER_DESCRIPTION "OpenVINO Deployment Manager")
        set(CPACK_COMPONENT_DEPLOYMENT_MANAGER_DEPENDS "core")
        set(CPACK_DEBIAN_DEPLOYMENT_MANAGER_PACKAGE_NAME "libopenvino-deployment-manager")
        set(CPACK_DEBIAN_DEPLOYMENT_MANAGER_PACKAGE_DEPENDS "python3")
        set(CPACK_DEBIAN_DEPLOYMENT_MANAGER_PACKAGE_ARCHITECTURE "all")

        # install dependencies
        set(CPACK_COMPONENT_INSTALL_DEPENDENCIES_DESCRIPTION "OpenVINO install dependencies")
        set(CPACK_DEBIAN_INSTALL_DEPENDENCIES_PACKAGE_NAME "libopenvino-install-dependencies")
        set(CPACK_DEBIAN_INSTALL_DEPENDENCIES_PACKAGE_DEPENDS "python3, TODO")
        set(CPACK_DEBIAN_INSTALL_DEPENDENCIES_PACKAGE_ARCHITECTURE "all")

        # licensing
        set(CPACK_COMPONENT_LICENSING_DESCRIPTION "OpenVINO lincences")
        set(CPACK_DEBIAN_LICENSING_PACKAGE_NAME "libopenvino-licensing")
        set(CPACK_DEBIAN_LICENSING_PACKAGE_ARCHITECTURE "all")

        # install debian common files
        foreach(comp IN LISTS CPACK_COMPONENTS_ALL)
            string(TOUPPER "${comp}" ucomp)
            set(package_name "${CPACK_DEBIAN_${ucomp}_PACKAGE_NAME}")

            # copyright
            install(FILES "${OpenVINO_SOURCE_DIR}/LICENSE"
                    DESTINATION share/doc/${package_name}/
                    COMPONENT ${comp}
                    RENAME copyright)

            # TODO: changelog
        endforeach()
    endif()

    include(CPack)
endmacro()
