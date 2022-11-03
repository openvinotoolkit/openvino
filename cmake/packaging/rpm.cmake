# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# OpenVINO Core components including frontends, plugins, etc
#

function(_ov_add_plugin comp is_pseudo)
    string(TOUPPER "${comp}" ucomp)
    if(NOT DEFINED CPACK_RPM_${ucomp}_PACKAGE_NAME)
        message(FATAL_ERROR "CPACK_RPM_${ucomp}_PACKAGE_NAME is not defined")
    else()
        set(package_name "${CPACK_RPM_${ucomp}_PACKAGE_NAME}")
    endif()

    if(NOT DEFINED cpack_full_ver)
        message(FATAL_ERROR "Internal variable 'cpack_full_ver' is not defined")
    endif()

    if(is_pseudo)
        if(pseudo_plugins_recommends)
            set(pseudo_plugins_recommends "${pseudo_plugins_recommends}, ${package_name} (= ${cpack_full_ver})")
        else()
            set(pseudo_plugins_recommends "${package_name} (= ${cpack_full_ver})")
        endif()
    endif()

    if(all_plugins_suggest)
        set(all_plugins_suggest "${all_plugins_suggest}, ${package_name} (= ${cpack_full_ver})")
    else()
        set(all_plugins_suggest "${package_name} (= ${cpack_full_ver})")
    endif()

    list(APPEND installed_plugins ${comp})

    set(pseudo_plugins_recommends "${pseudo_plugins_recommends}" PARENT_SCOPE)
    set(all_plugins_suggest "${all_plugins_suggest}" PARENT_SCOPE)
    set(installed_plugins "${installed_plugins}" PARENT_SCOPE)
endfunction()

macro(ov_cpack_settings)
    # fill a list of components which are part of rpm
    set(cpack_components_all ${CPACK_COMPONENTS_ALL})
    unset(CPACK_COMPONENTS_ALL)
    foreach(item IN LISTS cpack_components_all)
        # filter out some components, which are not needed to be wrapped to .deb package
        if(# skip OpenVINO Pyhon API and samples
           NOT item MATCHES "^${OV_CPACK_COMP_PYTHON_OPENVINO}_python.*" AND
           NOT item STREQUAL OV_CPACK_COMP_PYTHON_SAMPLES AND
           # python wheels are not needed to be wrapped by rpm packages
           NOT item STREQUAL OV_CPACK_COMP_PYTHON_WHEELS AND
           # see ticket # 82605
           NOT item STREQUAL "gna" AND
           # even for case of system TBB we have installation rules for wheels packages
           # so, need to skip this explicitly
           NOT item MATCHES "^tbb(_dev)?$" AND
           # the same for pugixml
           NOT item STREQUAL "pugixml" AND
           # TF component is not released
           NOT item STREQUAL "tensorflow" AND
           # we have copyright file for rpm package
           NOT item STREQUAL OV_CPACK_COMP_LICENSING AND
           # not appropriate components
           NOT item STREQUAL OV_CPACK_COMP_DEPLOYMENT_MANAGER AND
           NOT item STREQUAL OV_CPACK_COMP_INSTALL_DEPENDENCIES AND
           NOT item STREQUAL OV_CPACK_COMP_SETUPVARS)
            list(APPEND CPACK_COMPONENTS_ALL ${item})
        endif()
    endforeach()
    list(REMOVE_DUPLICATES CPACK_COMPONENTS_ALL)

    # version with 3 components
    set(cpack_name_ver "${OpenVINO_VERSION}")
    # full version with epoch and release components
    set(cpack_full_ver "${CPACK_PACKAGE_VERSION}")

    # take release version into account
    if(DEFINED CPACK_RPM_PACKAGE_RELEASE)
        set(cpack_full_ver "${cpack_full_ver}-${CPACK_RPM_PACKAGE_RELEASE}")
    endif()

    # take epoch version into account
    if(DEFINED CPACK_RPM_PACKAGE_EPOCH)
        set(cpack_full_ver "${CPACK_RPM_PACKAGE_EPOCH}:${cpack_full_ver}")
    endif()

    # a list of conflicting package versions
    set(conflicting_versions
        # 2022 release series
        # - 2022.1.0 is the last public release with rpm packages from Intel install team
        # - 2022.1.1, 2022.2 do not have rpm packages enabled, distributed only as archives
        # - 2022.3 is the first release where RPM updated packages are introduced
        2022.1.0)

    find_host_program(rpmlint_PROGRAM NAMES rpmlint DOC "Path to rpmlint")
    if(rpmlint_PROGRAM)
        execute_process(COMMAND "${rpmlint_PROGRAM}" --version
                        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
                        RESULT_VARIABLE rpmlint_code
                        OUTPUT_VARIABLE rpmlint_version)

        if(NOT rpmlint_code EQUAL 0)
            message(FATAL_ERROR "Internal error: Failed to determine rpmlint version")
        else()
            if(rpmlint_version MATCHES ".*version ([0-9]+\.[0-9]+) Copyright.*")
                set(rpmlint_version "${CMAKE_MATCH_1}")
            else()
                message(WARNING "Failed to extract rpmlint version from '${rpmlint_version}'")
            endif()
        endif()
    else()
        message(WARNING "Failed to find 'rpmlint' tool, use 'sudo yum / dnf install rpmlint' to install it")
    endif()

    #
    # core: base dependency for all components
    #

    set(CPACK_COMPONENT_CORE_DESCRIPTION "OpenVINO C / C++ Runtime libraries")
    set(CPACK_RPM_CORE_PACKAGE_NAME "libopenvino-${cpack_name_ver}")
    set(CPACK_RPM_CORE_POST_INSTALL_SCRIPT_FILE "${def_triggers}")
    set(CPACK_RPM_CORE_POST_UNINSTALL_SCRIPT_FILE "${def_triggers}")

    #
    # Plugins
    #

    # hetero
    if(ENABLE_HETERO)
        set(CPACK_COMPONENT_HETERO_DESCRIPTION "OpenVINO Hetero plugin")
        set(CPACK_COMPONENT_HETERO_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_RPM_HETERO_PACKAGE_NAME "libopenvino-hetero-plugin-${cpack_name_ver}")
        _ov_add_plugin(hetero ON)
    endif()

    # auto batch
    if(ENABLE_AUTO_BATCH)
        set(CPACK_COMPONENT_BATCH_DESCRIPTION "OpenVINO Automatic Batching plugin")
        set(CPACK_COMPONENT_BATCH_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_RPM_BATCH_PACKAGE_NAME "libopenvino-auto-batch-plugin-${cpack_name_ver}")
        _ov_add_plugin(batch ON)
    endif()

    # multi / auto plugins
    if(ENABLE_MULTI)
        if(ENABLE_AUTO)
            set(CPACK_COMPONENT_MULTI_DESCRIPTION "OpenVINO Auto / Multi plugin")
        else()
            set(CPACK_COMPONENT_MULTI_DESCRIPTION "OpenVINO Multi plugin")
        endif()
        set(CPACK_COMPONENT_MULTI_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_RPM_MULTI_PACKAGE_NAME "libopenvino-auto-plugin-${cpack_name_ver}")
        _ov_add_plugin(multi ON)
    elseif(ENABLE_AUTO)
        set(CPACK_COMPONENT_AUTO_DESCRIPTION "OpenVINO Auto plugin")
        set(CPACK_COMPONENT_AUTO_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_RPM_AUTO_PACKAGE_NAME "libopenvino-auto-plugin-${cpack_name_ver}")
        _ov_add_plugin(auto ON)
    endif()

    # intel-cpu
    if(ENABLE_INTEL_CPU OR DEFINED openvino_arm_cpu_plugin_SOURCE_DIR)
        if(ENABLE_INTEL_CPU)
            set(CPACK_COMPONENT_CPU_DESCRIPTION "Intel® CPU")
        else()
            set(CPACK_COMPONENT_CPU_DESCRIPTION "ARM CPU")
        endif()
        set(CPACK_COMPONENT_CPU_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_RPM_CPU_PACKAGE_NAME "libopenvino-intel-cpu-plugin-${cpack_name_ver}")
        _ov_add_plugin(cpu OFF)
    endif()

    # intel-gpu
    if(ENABLE_INTEL_GPU)
        set(CPACK_COMPONENT_GPU_DESCRIPTION "Intel® Processor Graphics")
        set(CPACK_COMPONENT_GPU_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_RPM_GPU_PACKAGE_NAME "libopenvino-intel-gpu-plugin-${cpack_name_ver}")
        # auto batch exhances GPU
        # set(CPACK_RPM_BATCH_PACKAGE_ENHANCES "${CPACK_RPM_GPU_PACKAGE_NAME} = (${cpack_full_ver})")
        _ov_add_plugin(gpu OFF)
    endif()

    # intel-myriad
    if(ENABLE_INTEL_MYRIAD)
        set(CPACK_COMPONENT_MYRIAD_DESCRIPTION "Intel® Movidius™ VPU")
        set(CPACK_COMPONENT_MYRIAD_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_RPM_MYRIAD_PACKAGE_NAME "libopenvino-intel-vpu-plugin-${cpack_name_ver}")
        _ov_add_plugin(myriad OFF)
    endif()

    # intel-gna
    if(ENABLE_INTEL_GNA AND "gna" IN_LIST CPACK_COMPONENTS_ALL)
        set(CPACK_COMPONENT_GNA_DESCRIPTION "Intel® Gaussian Neural Accelerator")
        set(CPACK_COMPONENT_GNA_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_RPM_GNA_PACKAGE_NAME "libopenvino-intel-gna-plugin-${cpack_name_ver}")
        _ov_add_plugin(gna OFF)
    endif()

    #
    # Frontends
    #

    if(ENABLE_OV_IR_FRONTEND)
        set(CPACK_COMPONENT_IR_DESCRIPTION "OpenVINO IR Frontend")
        set(CPACK_COMPONENT_IR_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_RPM_IR_PACKAGE_NAME "libopenvino-ir-frontend-${cpack_name_ver}")
        set(CPACK_RPM_IR_POST_INSTALL_SCRIPT_FILE "${def_triggers}")
        set(CPACK_RPM_IR_POST_UNINSTALL_SCRIPT_FILE "${def_triggers}")
        list(APPEND frontends ir)
    endif()

    if(ENABLE_OV_ONNX_FRONTEND)
        set(CPACK_COMPONENT_ONNX_DESCRIPTION "OpenVINO ONNX Frontend")
        set(CPACK_COMPONENT_ONNX_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_RPM_ONNX_PACKAGE_NAME "libopenvino-onnx-frontend-${cpack_name_ver}")
        set(CPACK_RPM_ONNX_POST_INSTALL_SCRIPT_FILE "${def_triggers}")
        set(CPACK_RPM_ONNX_POST_UNINSTALL_SCRIPT_FILE "${def_triggers}")
        list(APPEND frontends onnx)
    endif()

    if(ENABLE_OV_TF_FRONTEND AND "tensorflow" IN_LIST CPACK_COMPONENTS_ALL)
        set(CPACK_COMPONENT_TENSORFLOW_DESCRIPTION "OpenVINO TensorFlow Frontend")
        set(CPACK_COMPONENT_TENSORFLOW_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_RPM_TENSORFLOW_PACKAGE_NAME "libopenvino-tensorflow-frontend-${cpack_name_ver}")
        set(CPACK_RPM_TENSORFLOW_POST_INSTALL_SCRIPT_FILE "${def_triggers}")
        set(CPACK_RPM_TENSORFLOW_POST_UNINSTALL_SCRIPT_FILE "${def_triggers}")
        list(APPEND frontends tensorflow)
    endif()

    if(ENABLE_OV_PADDLE_FRONTEND)
        set(CPACK_COMPONENT_PADDLE_DESCRIPTION "OpenVINO Paddle Frontend")
        set(CPACK_COMPONENT_PADDLE_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_RPM_PADDLE_PACKAGE_NAME "libopenvino-paddle-frontend-${cpack_name_ver}")
        set(CPACK_RPM_PADDLE_POST_INSTALL_SCRIPT_FILE "${def_triggers}")
        set(CPACK_RPM_PADDLE_POST_UNINSTALL_SCRIPT_FILE "${def_triggers}")
        list(APPEND frontends paddle)
    endif()

    #
    # core_dev: depends on core and frontends (since frontends don't want to provide its own dev packages)
    #

    set(CPACK_COMPONENT_CORE_DEV_DESCRIPTION "Intel(R) Distribution of OpenVINO(TM) Toolkit C / C++ Development files")
    set(CPACK_COMPONENT_CORE_DEV_DEPENDS "${OV_CPACK_COMP_CORE};${frontends}")
    set(CPACK_RPM_CORE_DEV_PACKAGE_NAME "libopenvino-devel-${cpack_name_ver}")
    # TODO: update once compile_tool is excluded from the package
    # set(CPACK_RPM_CORE_DEV_PACKAGE_ARCHITECTURE "noarch")
    ov_rpm_generate_conflicts("${OV_CPACK_COMP_CORE_DEV}" ${conflicting_versions})

    ov_rpm_add_rpmlint_suppression("${OV_CPACK_COMP_CORE_DEV}"
        # contains samples source codes
        "devel-file-in-non-devel-package /usr/${OV_CPACK_INCLUDEDIR}/ngraph"
        "devel-file-in-non-devel-package /usr/${OV_CPACK_INCLUDEDIR}/ie"
        "devel-file-in-non-devel-package /usr/${OV_CPACK_INCLUDEDIR}/openvino"
        "devel-file-in-non-devel-package /usr/${OV_CPACK_RUNTIMEDIR}/libopenvino*"
        "devel-file-in-non-devel-package /usr/${OV_CPACK_RUNTIMEDIR}/pkgconfig/openvino.pc"
        # CVS-79409: create man page for compile_tool, remove together with compile_tool
        "no-manual-page-for-binary compile_tool")

    #
    # Python bindings
    #

    if(ENABLE_PYTHON)
        ov_get_pyversion(pyversion)
        set(python_component "${OV_CPACK_COMP_PYTHON_OPENVINO}_${pyversion}")
        string(TOUPPER "${pyversion}" pyversion)

        set(CPACK_COMPONENT_PYOPENVINO_${pyversion}_DESCRIPTION "OpenVINO Python bindings")
        set(CPACK_COMPONENT_PYOPENVINO_${pyversion}_DEPENDS "${OV_CPACK_COMP_CORE}")
        list(APPEND CPACK_COMPONENT_PYOPENVINO_${pyversion}_DEPENDS ${installed_plugins})
        list(APPEND CPACK_COMPONENT_PYOPENVINO_${pyversion}_DEPENDS ${frontends})
        set(CPACK_RPM_PYOPENVINO_${pyversion}_PACKAGE_REQUIRES "python3")

        set(CPACK_RPM_PYOPENVINO_${pyversion}_PACKAGE_NAME "libopenvino-python-${cpack_name_ver}")
    endif()

    #
    # Samples
    #

    # TODO: provide different packages in two lines below
    set(samples_build_deps "cmake3, gcc-c++, gcc, glibc-devel, make")
    set(samples_build_deps_suggest "opencv-devel (>= 3.0)")

    # c_samples / cpp_samples
    set(CPACK_COMPONENT_SAMPLES_DESCRIPTION "Intel(R) Distribution of OpenVINO(TM) Toolkit C / C++ Samples")
    set(CPACK_COMPONENT_SAMPLES_DEPENDS "core_dev")
    set(CPACK_RPM_SAMPLES_PACKAGE_NAME "openvino-samples-${cpack_name_ver}")
    # SUGGESTS may be unsupported, it's part of RPM 4.12.0 (Sep 16th 2014) only
    # see https://rpm.org/timeline.html
    set(CPACK_RPM_SAMPLES_PACKAGE_SUGGESTS "${samples_build_deps_suggest}, ${all_plugins_suggest}")
    set(CPACK_RPM_SAMPLES_PACKAGE_REQUIRES "gflags-devel, zlib-devel")
    set(CPACK_RPM_SAMPLES_PACKAGE_ARCHITECTURE "noarch")

    ov_rpm_add_rpmlint_suppression("${OV_CPACK_COMP_CPP_SAMPLES}"
        # contains samples source codes
        "devel-file-in-non-devel-package /usr/${OV_CPACK_SAMPLESDIR}/cpp/*"
        "devel-file-in-non-devel-package /usr/${OV_CPACK_SAMPLESDIR}/c/*"
        # depends on gflags-devel
        "devel-dependency gflags-devel")

    # python_samples
    if(ENABLE_PYTHON)
        set(CPACK_COMPONENT_PYTHON_SAMPLES_DESCRIPTION "Intel(R) Distribution of OpenVINO(TM) Toolkit Python Samples")
        set(CPACK_COMPONENT_PYTHON_SAMPLES_DEPENDS "${python_component}")
        set(CPACK_RPM_PYTHON_SAMPLES_PACKAGE_NAME "openvino-samples-python-${cpack_name_ver}")
        set(CPACK_RPM_PYTHON_SAMPLES_PACKAGE_REQUIRES "python3")
        set(CPACK_RPM_PYTHON_SAMPLES_PACKAGE_ARCHITECTURE "noarch")
    endif()

    #
    # Add umbrella packages
    #

    # all libraries
    set(CPACK_COMPONENT_LIBRARIES_DESCRIPTION "Intel(R) Distribution of OpenVINO(TM) Toolkit Libraries")
    if(installed_plugins)
        set(CPACK_COMPONENT_LIBRARIES_DEPENDS "${installed_plugins}")
    else()
        set(CPACK_COMPONENT_LIBRARIES_DEPENDS "${OV_CPACK_COMP_CORE}")
    endif()
    set(CPACK_RPM_LIBRARIES_PACKAGE_NAME "openvino-libraries-${cpack_name_ver}")
    set(CPACK_RPM_LIBRARIES_PACKAGE_ARCHITECTURE "noarch")

    # all libraries-dev
    set(CPACK_COMPONENT_LIBRARIES_DEV_DESCRIPTION "Intel(R) Distribution of OpenVINO(TM) Toolkit Libraries and Development files")
    set(CPACK_COMPONENT_LIBRARIES_DEV_DEPENDS "core_dev;libraries")
    set(CPACK_RPM_LIBRARIES_DEV_PACKAGE_NAME "openvino-libraries-devel-${cpack_name_ver}")
    set(CPACK_RPM_LIBRARIES_DEV_PACKAGE_ARCHITECTURE "noarch")
    ov_rpm_generate_conflicts(libraries_dev ${conflicting_versions})

    # all openvino
    set(CPACK_COMPONENT_OPENVINO_DESCRIPTION "Intel(R) Distribution of OpenVINO(TM) Toolkit Libraries and Development files")
    set(CPACK_COMPONENT_OPENVINO_DEPENDS "libraries_dev;samples")
    set(CPACK_RPM_OPENVINO_PACKAGE_NAME "openvino-${cpack_name_ver}")
    set(CPACK_RPM_OPENVINO_PACKAGE_ARCHITECTURE "noarch")
    ov_rpm_generate_conflicts(openvino ${conflicting_versions})

    list(APPEND CPACK_COMPONENTS_ALL "libraries;libraries_dev;openvino")

    #
    # Install latest symlink packages
    #

    # NOTE: we expicitly don't add runtime latest packages
    # since a user needs to depend on specific VERSIONED runtime package
    # with fixed SONAMEs, while latest package can be updated multiple times
    # ov_rpm_add_latest_component(libraries)

    ov_rpm_add_latest_component(libraries_dev)
    ov_rpm_add_latest_component(openvino)

    # users can manually install specific version of package
    # e.g. sudo yum install openvino=2022.1.0
    # even if we have latest package version 2022.2.0

    #
    # install common files
    #

    foreach(comp IN LISTS CPACK_COMPONENTS_ALL)
        ov_rpm_copyright("${comp}")
    endforeach()
endmacro()
