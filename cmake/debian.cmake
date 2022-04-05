# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# OpenVINO Core components including frontends, plugins, etc
#

macro(ov_debian_components)
    # fill a list of components which are part of debian
    set(cpack_components_all ${CPACK_COMPONENTS_ALL})
    unset(CPACK_COMPONENTS_ALL)
    foreach(item ${cpack_components_all})
        # filter out some components
        if(# NOT ${item} MATCHES ".*(python).*" AND
           # even for case of system TBB we have installation rules for wheels packages
           # so, need to skip this explicitly
           NOT item MATCHES "^tbb(_dev)?$" AND
           NOT item STREQUAL OV_CPACK_COMP_DEPLOYMENT_MANAGER)
            list(APPEND CPACK_COMPONENTS_ALL ${item})
        endif()
    endforeach()
    list(REMOVE_DUPLICATES CPACK_COMPONENTS_ALL)

    # CPACK_PACKAGE_VERSION_MAJOR.CPACK_PACKAGE_VERSION_MINOR
    set(cpack_ver_mm "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}")

    # core
    set(CPACK_COMPONENT_CORE_DESCRIPTION "OpenVINO C / C++ Runtime libraries")
    set(CPACK_DEBIAN_CORE_PACKAGE_NAME "libopenvino-${cpack_ver_mm}")
    set(CPACK_DEBIAN_CORE_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")

    ov_add_lintian_suppression(core
        # OpenVINO runtime library is named differently
        "package-name-doesnt-match-sonames")

    # core_dev
    set(CPACK_COMPONENT_CORE_DEV_DESCRIPTION "OpenVINO C / C++ Runtime development files")
    set(CPACK_COMPONENT_CORE_DEV_DEPENDS "core")
    set(CPACK_DEBIAN_CORE_DEV_PACKAGE_NAME "libopenvino-${cpack_ver_mm}-dev")
    set(CPACK_DEBIAN_CORE_DEV_PACKAGE_CONFLICTS "libopenvino2021.3-dev, libopenvino2021.4-dev")
    ov_add_lintian_suppression(core_dev)

    #
    # Plugins
    #

    # hetero
    if(ENABLE_HETERO)
        set(CPACK_COMPONENT_HETERO_DESCRIPTION "OpenVINO Hetero plugin")
        set(CPACK_COMPONENT_HETERO_DEPENDS "core")
        set(CPACK_DEBIAN_HETERO_PACKAGE_NAME "libopenvino-hetero-${cpack_ver_mm}")
        set(CPACK_DEBIAN_HETERO_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")
        list(APPEND installed_plugins "hetero")
    endif()

    # auto batch
    if(ENABLE_AUTO_BATCH)
        set(CPACK_COMPONENT_BATCH_DESCRIPTION "OpenVINO Auto Batch plugin")
        set(CPACK_COMPONENT_BATCH_DEPENDS "core")
        set(CPACK_DEBIAN_BATCH_PACKAGE_NAME "libopenvino-auto-batch-${cpack_ver_mm}")
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
        set(CPACK_DEBIAN_MULTI_PACKAGE_NAME "libopenvino-auto-${cpack_ver_mm}")
        set(CPACK_DEBIAN_MULTI_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")
        list(APPEND installed_plugins "multi")
    elseif(ENABLE_AUTO)
        set(CPACK_COMPONENT_AUTO_DESCRIPTION "OpenVINO Auto plugin")
        set(CPACK_COMPONENT_AUTO_DEPENDS "core")
        set(CPACK_DEBIAN_AUTO_PACKAGE_NAME "libopenvino-auto-${cpack_ver_mm}")
        set(CPACK_DEBIAN_AUTO_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")
        list(APPEND installed_plugins "auto")
    endif()

    # intel-cpu
    if(ENABLE_INTEL_CPU)
        set(CPACK_COMPONENT_CPU_DESCRIPTION "OpenVINO Intel CPU plugin")
        set(CPACK_COMPONENT_CPU_DEPENDS "core")
        set(CPACK_DEBIAN_CPU_PACKAGE_NAME "libopenvino-intel-cpu-${cpack_ver_mm}")
        set(CPACK_DEBIAN_CPU_PACKAGE_SUGGESTS "libopenvino-auto-${cpack_ver_mm} (= ${CPACK_PACKAGE_VERSION}), libopenvino-hetero-${cpack_ver_mm} (= ${CPACK_PACKAGE_VERSION})")
        set(CPACK_DEBIAN_CPU_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")
        list(APPEND installed_plugins "cpu")
    endif()

    # intel-gpu
    if(ENABLE_INTEL_GPU)
        set(CPACK_COMPONENT_GPU_DESCRIPTION "OpenVINO Intel GPU plugin")
        set(CPACK_COMPONENT_GPU_DEPENDS "core")
        set(CPACK_DEBIAN_GPU_PACKAGE_NAME "libopenvino-intel-gpu-${cpack_ver_mm}")
        set(CPACK_DEBIAN_GPU_PACKAGE_SUGGESTS "libopenvino-auto-${cpack_ver_mm} (= ${CPACK_PACKAGE_VERSION}), libopenvino-hetero-${cpack_ver_mm} (= ${CPACK_PACKAGE_VERSION})")
        set(CPACK_DEBIAN_GPU_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")
        list(APPEND installed_plugins "gpu")
    endif()

    # intel-myriad
    if(ENABLE_INTEL_MYRIAD)
        set(CPACK_COMPONENT_MYRIAD_DESCRIPTION "OpenVINO Intel Myriad plugin")
        set(CPACK_COMPONENT_MYRIAD_DEPENDS "core")
        set(CPACK_DEBIAN_MYRIAD_PACKAGE_NAME "libopenvino-intel-myriad-${cpack_ver_mm}")
        set(CPACK_DEBIAN_MYRIAD_PACKAGE_SUGGESTS "libopenvino-auto-${cpack_ver_mm} (= ${CPACK_PACKAGE_VERSION}), libopenvino-hetero-${cpack_ver_mm} (= ${CPACK_PACKAGE_VERSION})")
        set(CPACK_DEBIAN_MYRIAD_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")
        list(APPEND installed_plugins "myriad")
    endif()

    # intel-gna
    if(ENABLE_INTEL_GNA)
        set(CPACK_COMPONENT_GNA_DESCRIPTION "OpenVINO Intel GNA plugin")
        set(CPACK_COMPONENT_GNA_DEPENDS "core")
        set(CPACK_DEBIAN_GNA_PACKAGE_NAME "libopenvino-intel-gna-${cpack_ver_mm}")
        set(CPACK_DEBIAN_GNA_PACKAGE_SUGGESTS "libopenvino-auto-${cpack_ver_mm} (= ${CPACK_PACKAGE_VERSION}), libopenvino-hetero-${cpack_ver_mm} (= ${CPACK_PACKAGE_VERSION})")
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
    set(CPACK_DEBIAN_PYTHON_PYTHON3.8_PACKAGE_NAME "libopenvino-python-${cpack_ver_mm}")
    set(CPACK_DEBIAN_PYTHON_PYTHON3.8_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")

    #
    # Samples
    #

    set(samples_build_deps "cmake, g++, gcc, libc6-dev, make")
    set(samples_build_deps_suggest "${samples_build_deps}, libopencv-core-dev, libopencv-imgproc-dev, libopencv-imgcodecs-dev")

    # c_samples / cpp_samples
    set(CPACK_COMPONENT_SAMPLES_DESCRIPTION "OpenVINO C / C++ samples")
    set(CPACK_COMPONENT_SAMPLES_DEPENDS "core_dev")
    set(CPACK_DEBIAN_SAMPLES_PACKAGE_NAME "libopenvino-samples-${cpack_ver_mm}")
    set(CPACK_DEBIAN_SAMPLES_PACKAGE_SUGGESTS "${samples_build_deps_suggest}, libopenvino-hetero-${cpack_ver_mm} (= ${CPACK_PACKAGE_VERSION})")
    set(CPACK_DEBIAN_SAMPLES_PACKAGE_DEPENDS "libgflags-dev, nlohmann-json3-dev, zlib1g-dev, ${samples_build_deps}")
    set(CPACK_DEBIAN_SAMPLES_PACKAGE_ARCHITECTURE "all")

    # python_samples
    set(CPACK_COMPONENT_PYTHON_SAMPLES_DESCRIPTION "OpenVINO Python samples")
    set(CPACK_COMPONENT_PYTHON_SAMPLES_DEPENDS "python_python3.8")
    set(CPACK_DEBIAN_PYTHON_SAMPLES_PACKAGE_NAME "libopenvino-samples-python-${cpack_ver_mm}")
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
    set(CPACK_DEBIAN_LIBRARIES_PACKAGE_NAME "libopenvino-libraries-${cpack_ver_mm}")
    list(APPEND CPACK_COMPONENTS_ALL "libraries")

    # all libraries-dev
    set(CPACK_COMPONENT_LIBRARIES_DEV_DESCRIPTION "OpenVINO all runtime libraries and development files")
    set(CPACK_COMPONENT_LIBRARIES_DEV_DEPENDS "core_dev;${installed_plugins}")
    set(CPACK_DEBIAN_LIBRARIES_DEV_PACKAGE_NAME "libopenvino-libraries-${cpack_ver_mm}-dev")
    list(APPEND CPACK_COMPONENTS_ALL "libraries_dev")

    #
    # install debian common files
    #

    foreach(comp IN LISTS CPACK_COMPONENTS_ALL)
        string(TOUPPER "${comp}" ucomp)
        set(package_name "${CPACK_DEBIAN_${ucomp}_PACKAGE_NAME}")

        # copyright
        # install(FILES "${OpenVINO_SOURCE_DIR}/LICENSE"
        #         DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/${package_name}/
        #         COMPONENT ${comp}
        #         RENAME copyright)

        # TODO: install changelog

        # install triggers
        install(FILES ${def_triggers}
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
