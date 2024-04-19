# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# OpenVINO Core components including frontends, plugins, etc
#

function(_ov_add_package package_names comp)
    string(TOUPPER "${comp}" ucomp)
    if(NOT DEFINED CPACK_RPM_${ucomp}_PACKAGE_NAME)
        message(FATAL_ERROR "CPACK_RPM_${ucomp}_PACKAGE_NAME is not defined")
    else()
        set(package_name "${CPACK_RPM_${ucomp}_PACKAGE_NAME}")
    endif()

    if(NOT DEFINED cpack_full_ver)
        message(FATAL_ERROR "Internal variable 'cpack_full_ver' is not defined")
    endif()

    if(${package_names})
        set("${package_names}" "${${package_names}}, ${package_name} = ${cpack_full_ver}" PARENT_SCOPE)
    else()
        set("${package_names}" "${package_name} = ${cpack_full_ver}" PARENT_SCOPE)
    endif()
endfunction()

macro(ov_cpack_settings)
    # fill a list of components which are part of rpm
    set(cpack_components_all ${CPACK_COMPONENTS_ALL})
    unset(CPACK_COMPONENTS_ALL)
    foreach(item IN LISTS cpack_components_all)
        string(TOUPPER ${item} UPPER_COMP)
        # filter out some components, which are not needed to be wrapped to .rpm package
        if(NOT OV_CPACK_COMP_${UPPER_COMP}_EXCLUDE_ALL AND
           # skip OpenVINO Python API (pattern in form of "pyopenvino_python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}")
           NOT item MATCHES "^${OV_CPACK_COMP_PYTHON_OPENVINO}_python.*" AND
           # because in case of .rpm package, pyopenvino_package_python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR} is installed
           (NOT item MATCHES "^${OV_CPACK_COMP_PYTHON_OPENVINO_PACKAGE}_python.*" OR ENABLE_PYTHON_PACKAGING) AND
           # temporary block nvidia
           NOT item STREQUAL "nvidia" AND
           # don't install node_addon
           NOT item MATCHES "node_addon" AND
           # temporary block npu
           NOT item STREQUAL "npu" AND
           # don't install Intel OpenMP
           NOT item STREQUAL "omp" AND
           # the same for pugixml
           NOT item STREQUAL "pugixml")
           list(APPEND CPACK_COMPONENTS_ALL ${item})
        endif()
    endforeach()
    unset(cpack_components_all)
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
        # - 2022.3 is the first release where RPM updated packages are introduced, others 2022.3.X are LTS
        2022.3.0 2022.3.1 2022.3.2 2022.3.3 2022.3.4 2022.3.5
        2023.0.0 2023.0.1 2023.0.2 2023.0.3
        2023.1.0
        2023.2.0
        2023.3.0 2023.3.1 2023.3.2 2023.3.3 2023.3.4 2023.3.5
        2024.0
        2024.1
        )

    find_host_program(rpmlint_PROGRAM NAMES rpmlint DOC "Path to rpmlint")
    if(rpmlint_PROGRAM)
        execute_process(COMMAND "${rpmlint_PROGRAM}" --version
                        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
                        RESULT_VARIABLE rpmlint_code
                        OUTPUT_VARIABLE rpmlint_version
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_STRIP_TRAILING_WHITESPACE)

        if(NOT rpmlint_code EQUAL 0)
            message(FATAL_ERROR "Internal error: Failed to determine rpmlint version")
        else()
            if(rpmlint_version MATCHES "([0-9]+\.[0-9]+)")
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
    set(core_package "${CPACK_RPM_CORE_PACKAGE_NAME} = ${cpack_full_ver}")
    set(CPACK_RPM_CORE_POST_INSTALL_SCRIPT_FILE "${def_triggers}")
    set(CPACK_RPM_CORE_POST_UNINSTALL_SCRIPT_FILE "${def_triggers}")
    set(${OV_CPACK_COMP_CORE}_copyright "generic")

    #
    # Plugins
    #

    # hetero
    if(ENABLE_HETERO)
        set(CPACK_COMPONENT_HETERO_DESCRIPTION "OpenVINO Hetero software plugin")
        set(CPACK_RPM_HETERO_PACKAGE_REQUIRES "${core_package}")
        set(CPACK_RPM_HETERO_PACKAGE_NAME "libopenvino-hetero-plugin-${cpack_name_ver}")
        _ov_add_package(plugin_packages hetero)
        set(hetero_copyright "generic")
    endif()

    # auto batch
    if(ENABLE_AUTO_BATCH)
        set(CPACK_COMPONENT_BATCH_DESCRIPTION "OpenVINO Automatic Batching software plugin")
        set(CPACK_RPM_BATCH_PACKAGE_REQUIRES "${core_package}")
        set(CPACK_RPM_BATCH_PACKAGE_NAME "libopenvino-auto-batch-plugin-${cpack_name_ver}")
        _ov_add_package(plugin_packages batch)
        set(batch_copyright "generic")
    endif()

    # multi / auto plugins
    if(ENABLE_MULTI)
        if(ENABLE_AUTO)
            set(CPACK_COMPONENT_MULTI_DESCRIPTION "OpenVINO Auto / Multi software plugin")
        else()
            set(CPACK_COMPONENT_MULTI_DESCRIPTION "OpenVINO Multi software plugin")
        endif()
        set(CPACK_RPM_MULTI_PACKAGE_REQUIRES "${core_package}")
        set(CPACK_RPM_MULTI_PACKAGE_NAME "libopenvino-auto-plugin-${cpack_name_ver}")
        _ov_add_package(plugin_packages multi)
        set(multi_copyright "generic")
    elseif(ENABLE_AUTO)
        set(CPACK_COMPONENT_AUTO_DESCRIPTION "OpenVINO Auto software plugin")
        set(CPACK_RPM_AUTO_PACKAGE_REQUIRES "${core_package}")
        set(CPACK_RPM_AUTO_PACKAGE_NAME "libopenvino-auto-plugin-${cpack_name_ver}")
        _ov_add_package(plugin_packages auto)
        set(auto_copyright "generic")
    endif()

    # cpu
    if(ENABLE_INTEL_CPU)
        if(ARM OR AARCH64)
            set(CPACK_RPM_CPU_PACKAGE_NAME "libopenvino-arm-cpu-plugin-${cpack_name_ver}")
            set(CPACK_COMPONENT_CPU_DESCRIPTION "ARM® CPU inference plugin")
            set(cpu_copyright "arm_cpu")
        elseif(X86 OR X86_64)
            set(CPACK_RPM_CPU_PACKAGE_NAME "libopenvino-intel-cpu-plugin-${cpack_name_ver}")
            set(CPACK_COMPONENT_CPU_DESCRIPTION "Intel® CPU inference plugin")
            set(cpu_copyright "generic")
        else()
            message(FATAL_ERROR "Unsupported CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
        endif()
        set(CPACK_RPM_CPU_PACKAGE_REQUIRES "${core_package}")
        _ov_add_package(plugin_packages cpu)
    endif()

    # intel-gpu
    if(ENABLE_INTEL_GPU)
        set(CPACK_COMPONENT_GPU_DESCRIPTION "Intel® Processor Graphics inference plugin")
        set(CPACK_RPM_GPU_PACKAGE_REQUIRES "${core_package}")
        set(CPACK_RPM_GPU_PACKAGE_NAME "libopenvino-intel-gpu-plugin-${cpack_name_ver}")
        _ov_add_package(plugin_packages gpu)
        set(gpu_copyright "generic")
    endif()

    # intel-npu
    if(ENABLE_INTEL_NPU AND "npu" IN_LIST CPACK_COMPONENTS_ALL)
        set(CPACK_COMPONENT_NPU_DESCRIPTION "Intel® Neural Processing Unit inference plugin")
        set(CPACK_RPM_NPU_PACKAGE_REQUIRES "${core_package}")
        set(CPACK_RPM_NPU_PACKAGE_NAME "libopenvino-intel-npu-plugin-${cpack_name_ver}")
        _ov_add_package(plugin_packages npu)
        set(npu_copyright "generic")
    endif()

    #
    # Frontends
    #

    if(ENABLE_OV_IR_FRONTEND)
        set(CPACK_COMPONENT_IR_DESCRIPTION "OpenVINO IR Frontend")
        set(CPACK_RPM_IR_PACKAGE_NAME "libopenvino-ir-frontend-${cpack_name_ver}")
        set(CPACK_RPM_IR_POST_INSTALL_SCRIPT_FILE "${def_triggers}")
        set(CPACK_RPM_IR_POST_UNINSTALL_SCRIPT_FILE "${def_triggers}")
        _ov_add_package(frontend_packages ir)
        set(ir_copyright "generic")
    endif()

    if(ENABLE_OV_ONNX_FRONTEND)
        set(CPACK_COMPONENT_ONNX_DESCRIPTION "OpenVINO ONNX Frontend")
        set(CPACK_RPM_ONNX_PACKAGE_NAME "libopenvino-onnx-frontend-${cpack_name_ver}")
        set(CPACK_RPM_ONNX_POST_INSTALL_SCRIPT_FILE "${def_triggers}")
        set(CPACK_RPM_ONNX_POST_UNINSTALL_SCRIPT_FILE "${def_triggers}")
        _ov_add_package(frontend_packages onnx)
        set(onnx_copyright "generic")
    endif()

    if(ENABLE_OV_TF_FRONTEND)
        set(CPACK_COMPONENT_TENSORFLOW_DESCRIPTION "OpenVINO TensorFlow Frontend")
        set(CPACK_RPM_TENSORFLOW_PACKAGE_NAME "libopenvino-tensorflow-frontend-${cpack_name_ver}")
        set(CPACK_RPM_TENSORFLOW_POST_INSTALL_SCRIPT_FILE "${def_triggers}")
        set(CPACK_RPM_TENSORFLOW_POST_UNINSTALL_SCRIPT_FILE "${def_triggers}")
        _ov_add_package(frontend_packages tensorflow)
        set(tensorflow_copyright "generic")
    endif()

    if(ENABLE_OV_PADDLE_FRONTEND)
        set(CPACK_COMPONENT_PADDLE_DESCRIPTION "OpenVINO Paddle Frontend")
        set(CPACK_RPM_PADDLE_PACKAGE_NAME "libopenvino-paddle-frontend-${cpack_name_ver}")
        set(CPACK_RPM_PADDLE_POST_INSTALL_SCRIPT_FILE "${def_triggers}")
        set(CPACK_RPM_PADDLE_POST_UNINSTALL_SCRIPT_FILE "${def_triggers}")
        _ov_add_package(frontend_packages paddle)
        set(paddle_copyright "generic")
    endif()

    if(ENABLE_OV_PYTORCH_FRONTEND)
        set(CPACK_COMPONENT_PYTORCH_DESCRIPTION "OpenVINO PyTorch Frontend")
        set(CPACK_RPM_PYTORCH_PACKAGE_NAME "libopenvino-pytorch-frontend-${cpack_name_ver}")
        set(CPACK_RPM_PYTORCH_POST_INSTALL_SCRIPT_FILE "${def_triggers}")
        set(CPACK_RPM_PYTORCH_POST_UNINSTALL_SCRIPT_FILE "${def_triggers}")
        _ov_add_package(frontend_packages pytorch)
        set(pytorch_copyright "generic")
    endif()

    if(ENABLE_OV_TF_LITE_FRONTEND)
        set(CPACK_COMPONENT_TENSORFLOW_LITE_DESCRIPTION "OpenVINO TensorFlow Lite Frontend")
        set(CPACK_RPM_TENSORFLOW_LITE_PACKAGE_NAME "libopenvino-tensorflow-lite-frontend-${cpack_name_ver}")
        set(CPACK_RPM_TENSORFLOW_LITE_POST_INSTALL_SCRIPT_FILE "${def_triggers}")
        set(CPACK_RPM_TENSORFLOW_LITE_POST_UNINSTALL_SCRIPT_FILE "${def_triggers}")
        _ov_add_package(frontend_packages tensorflow_lite)
        set(tensorflow_lite_copyright "generic")
    endif()

    #
    # core_dev: depends on core and frontends (since frontends don't want to provide its own dev packages)
    #

    set(CPACK_COMPONENT_CORE_DEV_DESCRIPTION "Intel(R) Distribution of OpenVINO(TM) Toolkit C / C++ Development files")
    set(CPACK_RPM_CORE_DEV_PACKAGE_REQUIRES "${core_package}, ${frontend_packages}")
    set(CPACK_RPM_CORE_DEV_PACKAGE_NAME "libopenvino-devel-${cpack_name_ver}")
    set(core_dev_package "${CPACK_RPM_CORE_DEV_PACKAGE_NAME} = ${cpack_full_ver}")
    ov_rpm_generate_conflicts("${OV_CPACK_COMP_CORE_DEV}" ${conflicting_versions})

    ov_rpm_add_rpmlint_suppression("${OV_CPACK_COMP_CORE_DEV}"
        # contains samples source codes
        "devel-file-in-non-devel-package /usr/${OV_CPACK_INCLUDEDIR}/ngraph"
        "devel-file-in-non-devel-package /usr/${OV_CPACK_INCLUDEDIR}/ie"
        "devel-file-in-non-devel-package /usr/${OV_CPACK_INCLUDEDIR}/openvino"
        "devel-file-in-non-devel-package /usr/${OV_CPACK_RUNTIMEDIR}/libopenvino*"
        "devel-file-in-non-devel-package /usr/${OV_CPACK_RUNTIMEDIR}/pkgconfig/openvino.pc")
    set(${OV_CPACK_COMP_CORE_DEV}_copyright "generic")

    #
    # Python bindings
    #

    if(ENABLE_PYTHON_PACKAGING)
        ov_get_pyversion(pyversion)
        set(python_component "${OV_CPACK_COMP_PYTHON_OPENVINO_PACKAGE}_${pyversion}")
        string(TOUPPER "${pyversion}" pyversion_upper)

        set(CPACK_COMPONENT_PYOPENVINO_PACKAGE_${pyversion_upper}_DESCRIPTION "OpenVINO Python API")
        set(CPACK_RPM_PYOPENVINO_PACKAGE_${pyversion_upper}_PACKAGE_REQUIRES
            "${core_package}, ${frontend_packages}, ${plugin_packages}, python3, python3-numpy")
        set(CPACK_RPM_PYOPENVINO_PACKAGE_${pyversion_upper}_PACKAGE_NAME "python3-openvino-${cpack_full_ver}")
        set(python_package "${CPACK_RPM_PYOPENVINO_PACKAGE_${pyversion_upper}_PACKAGE_NAME} = ${cpack_full_ver}")
        set(${python_component}_copyright "generic")

        # we can have a single python installed, so we need to generate conflicts for all other versions
        ov_rpm_generate_conflicts(${python_component} ${conflicting_versions})

        ov_rpm_add_rpmlint_suppression("${python_component}"
            # all directories
            "non-standard-dir-perm /usr/lib64/${pyversion}/site-packages/openvino/*"
            )
    endif()

    #
    # Samples
    #

    set(samples_build_deps "cmake3, gcc-c++, gcc, glibc-devel, make, pkgconf-pkg-config")
    set(samples_build_deps_suggest "opencv-devel >= 3.0")
    set(samples_opencl_deps_suggest "ocl-icd-devel, opencl-headers")

    # c_samples / cpp_samples
    set(CPACK_COMPONENT_SAMPLES_DESCRIPTION "Intel(R) Distribution of OpenVINO(TM) Toolkit C / C++ Samples")
    set(CPACK_RPM_SAMPLES_PACKAGE_NAME "openvino-samples-${cpack_name_ver}")
    set(samples_package "${CPACK_RPM_SAMPLES_PACKAGE_NAME} = ${cpack_full_ver}")
    # SUGGESTS may be unsupported, it's part of RPM 4.12.0 (Sep 16th 2014) only
    # see https://rpm.org/timeline.html
    set(CPACK_RPM_SAMPLES_PACKAGE_SUGGESTS "${samples_build_deps_suggest}, ${samples_opencl_deps_suggest}, ${plugin_packages}")
    set(CPACK_RPM_SAMPLES_PACKAGE_REQUIRES "${core_dev_package}, ${samples_build_deps}")
    set(CPACK_RPM_SAMPLES_PACKAGE_ARCHITECTURE "noarch")
    ov_rpm_generate_conflicts(${OV_CPACK_COMP_CPP_SAMPLES} ${conflicting_versions})

    ov_rpm_add_rpmlint_suppression("${OV_CPACK_COMP_CPP_SAMPLES}"
        # contains samples source codes
        "devel-file-in-non-devel-package /usr/${OV_CPACK_SAMPLESDIR}/cpp/*"
        "devel-file-in-non-devel-package /usr/${OV_CPACK_SAMPLESDIR}/c/*"
        # duplicated files are OK
        "files-duplicate /usr/${OV_CPACK_SAMPLESDIR}/cpp/CMakeLists.txt /usr/${OV_CPACK_SAMPLESDIR}/c/CMakeLists.txt"
        "files-duplicate /usr/${OV_CPACK_SAMPLESDIR}/cpp/build_samples.sh /usr/${OV_CPACK_SAMPLESDIR}/c/build_samples.sh"
        )
    set(samples_copyright "generic")

    # python_samples
    if(ENABLE_PYTHON_PACKAGING)
        set(CPACK_COMPONENT_PYTHON_SAMPLES_DESCRIPTION "Intel(R) Distribution of OpenVINO(TM) Toolkit Python Samples")
        set(CPACK_RPM_PYTHON_SAMPLES_PACKAGE_REQUIRES "${python_package}, python3")
        set(CPACK_RPM_PYTHON_SAMPLES_PACKAGE_NAME "openvino-samples-python-${cpack_name_ver}")
        set(python_samples_package "${CPACK_RPM_PYTHON_SAMPLES_PACKAGE_NAME} = ${cpack_full_ver}")
        set(CPACK_RPM_PYTHON_SAMPLES_PACKAGE_ARCHITECTURE "noarch")
        ov_rpm_generate_conflicts(${OV_CPACK_COMP_PYTHON_SAMPLES} ${conflicting_versions})
        set(python_samples_copyright "generic")

        ov_rpm_add_rpmlint_suppression(${OV_CPACK_COMP_PYTHON_SAMPLES}
            # all files
            "non-executable-script /usr/share/openvino/samples/python/*"
            # similar requirements.txt files
            "files-duplicate /usr/share/openvino/samples/python/*"
            )
    endif()

    #
    # Add umbrella packages
    #

    # all libraries
    set(CPACK_COMPONENT_LIBRARIES_DESCRIPTION "Intel(R) Distribution of OpenVINO(TM) Toolkit Libraries")
    if(plugin_packages)
        set(CPACK_RPM_LIBRARIES_PACKAGE_REQUIRES "${plugin_packages}")
    else()
        set(CPACK_RPM_LIBRARIES_PACKAGE_REQUIRES "${core_package}")
    endif()
    set(CPACK_RPM_LIBRARIES_PACKAGE_NAME "openvino-libraries-${cpack_name_ver}")
    set(libraries_package "${CPACK_RPM_LIBRARIES_PACKAGE_NAME} = ${cpack_full_ver}")
    set(CPACK_RPM_LIBRARIES_PACKAGE_ARCHITECTURE "noarch")
    set(libraries_copyright "generic")

    # all libraries-dev
    set(CPACK_COMPONENT_LIBRARIES_DEV_DESCRIPTION "Intel(R) Distribution of OpenVINO(TM) Toolkit Libraries and Development files")
    set(CPACK_RPM_LIBRARIES_DEV_PACKAGE_REQUIRES "${core_dev_package}, ${libraries_package}")
    set(CPACK_RPM_LIBRARIES_DEV_PACKAGE_NAME "openvino-libraries-devel-${cpack_name_ver}")
    set(libraries_dev_package "${CPACK_RPM_LIBRARIES_DEV_PACKAGE_NAME} = ${cpack_full_ver}")
    set(CPACK_RPM_LIBRARIES_DEV_PACKAGE_ARCHITECTURE "noarch")
    ov_rpm_generate_conflicts(libraries_dev ${conflicting_versions})
    set(libraries_dev_copyright "generic")

    # all openvino
    set(CPACK_COMPONENT_OPENVINO_DESCRIPTION "Intel(R) Distribution of OpenVINO(TM) Toolkit Libraries and Development files")
    set(CPACK_RPM_OPENVINO_PACKAGE_REQUIRES "${libraries_dev_package}, ${samples_package}")
    if(ENABLE_PYTHON_PACKAGING)
        set(CPACK_DEBIAN_OPENVINO_PACKAGE_DEPENDS "${CPACK_RPM_OPENVINO_PACKAGE_REQUIRES}, ${python_package}, ${python_samples_package}")
    endif()
    set(CPACK_RPM_OPENVINO_PACKAGE_NAME "openvino-${cpack_name_ver}")
    set(CPACK_RPM_OPENVINO_PACKAGE_ARCHITECTURE "noarch")
    ov_rpm_generate_conflicts(openvino ${conflicting_versions})
    set(openvino_copyright "generic")

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
        if(NOT DEFINED "${comp}_copyright")
            message(FATAL_ERROR "Copyright file name is not defined for ${comp}")
        endif()
        ov_rpm_copyright("${comp}" "${${comp}_copyright}")
    endforeach()
endmacro()
