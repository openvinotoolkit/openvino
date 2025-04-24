# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# OpenVINO Core components including frontends, plugins, etc
#

function(_ov_add_plugin comp is_pseudo)
    string(TOUPPER "${comp}" ucomp)
    if(NOT DEFINED CPACK_DEBIAN_${ucomp}_PACKAGE_NAME)
        message(FATAL_ERROR "CPACK_DEBIAN_${ucomp}_PACKAGE_NAME is not defined")
    else()
        set(package_name "${CPACK_DEBIAN_${ucomp}_PACKAGE_NAME}")
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
    # fill a list of components which are part of debian
    set(cpack_components_all ${CPACK_COMPONENTS_ALL})
    unset(CPACK_COMPONENTS_ALL)
    foreach(item IN LISTS cpack_components_all)
        string(TOUPPER ${item} UPPER_COMP)
        # filter out some components, which are not needed to be wrapped to .deb package
        if(NOT OV_CPACK_COMP_${UPPER_COMP}_EXCLUDE_ALL AND
           # skip OpenVINO Python API (pattern in form of "pyopenvino_python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}")
           NOT item MATCHES "^${OV_CPACK_COMP_PYTHON_OPENVINO}_python.*" AND
           # because in case of .deb package, pyopenvino_package_python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR} is installed
           (NOT item MATCHES "^${OV_CPACK_COMP_PYTHON_OPENVINO_PACKAGE}_python.*" OR ENABLE_PYTHON_PACKAGING) AND
           # temporary block nvidia
           NOT item STREQUAL "nvidia" AND
           # don't install node_addon
           NOT item MATCHES "node_addon" AND
           # don't install Intel OpenMP
           NOT item STREQUAL "omp" AND
           # the same for pugixml
           NOT item STREQUAL "pugixml" AND
           # It was decided not to distribute JAX as C++ component
           NOT item STREQUAL "jax")
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
    if(DEFINED CPACK_DEBIAN_PACKAGE_RELEASE)
        set(cpack_full_ver "${cpack_full_ver}-${CPACK_DEBIAN_PACKAGE_RELEASE}")
    endif()

    # take epoch version into account
    if(DEFINED CPACK_DEBIAN_PACKAGE_EPOCH)
        set(cpack_full_ver "${CPACK_DEBIAN_PACKAGE_EPOCH}:${cpack_full_ver}")
    endif()

    # a list of conflicting package versions
    set(conflicting_versions
        # 2022 release series
        # - 2022.1.0 is the last public release with debian packages from Intel install team
        # - 2022.1.1, 2022.2 do not have debian packages enabled, distributed only as archives
        # - 2022.3 is the first release where Debian updated packages are introduced, others 2022.3.X are LTS
        2022.3.0 2022.3.1 2022.3.2 2022.3.3 2022.3.4 2022.3.5
        2023.0.0 2023.0.1 2023.0.2 2023.0.3
        2023.1.0
        2023.2.0
        2023.3.0 2023.3.1 2023.3.2 2023.3.3 2023.3.4 2023.3.5
        2024.0.0
        2024.1.0
        2024.2.0
        2024.3.0
        2024.4.0
        2024.5.0 2024.5.1
        2024.6.0
        2025.0.0 2025.0.1
        2025.1.0
        )

    ov_check_conflicts_versions(conflicting_versions)

    #
    # core: base dependency for each component
    #

    # core
    set(CPACK_COMPONENT_CORE_DESCRIPTION "OpenVINO C / C++ Runtime libraries")
    set(CPACK_DEBIAN_CORE_PACKAGE_NAME "libopenvino-${cpack_name_ver}")
    # we need triggers to run ldconfig for openvino
    set(CPACK_DEBIAN_CORE_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")

    ov_debian_add_lintian_suppression("${OV_CPACK_COMP_CORE}"
        # package-name-doesnt-match-sonames libopenvino202230 libopenvino-c20223
        "package-name-doesnt-match-sonames")
    set(${OV_CPACK_COMP_CORE}_copyright "generic")

    #
    # Plugins
    #

    # hetero
    if(ENABLE_HETERO)
        set(CPACK_COMPONENT_HETERO_DESCRIPTION "OpenVINO Hetero software plugin")
        set(CPACK_COMPONENT_HETERO_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_DEBIAN_HETERO_PACKAGE_NAME "libopenvino-hetero-plugin-${cpack_name_ver}")
        set(CPACK_DEBIAN_HETERO_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm}")
        _ov_add_plugin(hetero ON)
        set(hetero_copyright "generic")
    endif()

    # auto batch
    if(ENABLE_AUTO_BATCH)
        set(CPACK_COMPONENT_BATCH_DESCRIPTION "OpenVINO Automatic Batching software plugin")
        set(CPACK_COMPONENT_BATCH_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_DEBIAN_BATCH_PACKAGE_NAME "libopenvino-auto-batch-plugin-${cpack_name_ver}")
        set(CPACK_DEBIAN_BATCH_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm}")
        _ov_add_plugin(batch ON)
        set(batch_copyright "generic")
    endif()

    # multi / auto plugins
    if(ENABLE_MULTI)
        if(ENABLE_AUTO)
            set(CPACK_COMPONENT_MULTI_DESCRIPTION "OpenVINO Auto / Multi software plugin")
        else()
            set(CPACK_COMPONENT_MULTI_DESCRIPTION "OpenVINO Multi software plugin")
        endif()
        set(CPACK_COMPONENT_MULTI_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_DEBIAN_MULTI_PACKAGE_NAME "libopenvino-auto-plugin-${cpack_name_ver}")
        set(CPACK_DEBIAN_MULTI_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm}")
        _ov_add_plugin(multi ON)
        set(multi_copyright "generic")
    elseif(ENABLE_AUTO)
        set(CPACK_COMPONENT_AUTO_DESCRIPTION "OpenVINO Auto software plugin")
        set(CPACK_COMPONENT_AUTO_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_DEBIAN_AUTO_PACKAGE_NAME "libopenvino-auto-plugin-${cpack_name_ver}")
        set(CPACK_DEBIAN_AUTO_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm}")
        _ov_add_plugin(auto ON)
        set(auto_copyright "generic")
    endif()

    # cpu
    if(ENABLE_INTEL_CPU)
        if(ARM OR AARCH64)
            set(CPACK_DEBIAN_CPU_PACKAGE_NAME "libopenvino-arm-cpu-plugin-${cpack_name_ver}")
            set(CPACK_COMPONENT_CPU_DESCRIPTION "ARM® CPU inference plugin")
            set(cpu_copyright "arm_cpu")
        elseif(X86 OR X86_64)
            set(CPACK_DEBIAN_CPU_PACKAGE_NAME "libopenvino-intel-cpu-plugin-${cpack_name_ver}")
            set(CPACK_COMPONENT_CPU_DESCRIPTION "Intel® CPU inference plugin")
            set(cpu_copyright "generic")
        else()
            message(FATAL_ERROR "Unsupported CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
        endif()
        set(CPACK_COMPONENT_CPU_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_DEBIAN_CPU_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm}")
        _ov_add_plugin(cpu OFF)
    endif()

    # intel-gpu
    if(ENABLE_INTEL_GPU)
        set(CPACK_COMPONENT_GPU_DESCRIPTION "Intel® Processor Graphics inference plugin")
        set(CPACK_COMPONENT_GPU_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_DEBIAN_GPU_PACKAGE_NAME "libopenvino-intel-gpu-plugin-${cpack_name_ver}")
        set(CPACK_DEBIAN_GPU_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm}")
        # auto batch exhances GPU
        # set(CPACK_DEBIAN_BATCH_PACKAGE_ENHANCES "${CPACK_DEBIAN_GPU_PACKAGE_NAME} (= ${cpack_full_ver})")
        _ov_add_plugin(gpu OFF)
        set(gpu_copyright "generic")
    endif()

    # intel-npu
    if(ENABLE_INTEL_NPU)
        set(CPACK_COMPONENT_NPU_DESCRIPTION "Intel® Neural Processing Unit inference plugin")
        set(CPACK_COMPONENT_NPU_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_DEBIAN_NPU_PACKAGE_NAME "libopenvino-intel-npu-plugin-${cpack_name_ver}")
        set(CPACK_DEBIAN_NPU_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm}")
        _ov_add_plugin(npu OFF)
        set(npu_copyright "generic")

        # NPU plugin also builds level-zero as thirdparty
        # let's add it to the list of dependency search directories to avoid missing dependncy on libze_loader.so.1
        if(OV_GENERATOR_MULTI_CONFIG)
            # $<CONFIG> generator expression does not work in this place, have to add all possible configs
            foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
                list(APPEND CPACK_DEBIAN_PACKAGE_SHLIBDEPS_PRIVATE_DIRS "${CMAKE_BINARY_DIR}/lib/${config}")
            endforeach()
        else()
            list(APPEND CPACK_DEBIAN_PACKAGE_SHLIBDEPS_PRIVATE_DIRS "${CMAKE_BINARY_DIR}/lib")
        endif()
    endif()

    # # add pseudo plugins are recommended to core component
    # if(pseudo_plugins_recommends)
    #     # see https://superuser.com/questions/70031/what-is-the-difference-between-recommended-and-suggested-packages-ubuntu.
    #     # we suppose that pseudo plugins are needed for core
    #     set(CPACK_DEBIAN_CORE_PACKAGE_RECOMMENDS "${pseudo_plugins_recommends}")
    # endif()

    #
    # Frontends
    #

    if(ENABLE_OV_IR_FRONTEND)
        set(CPACK_COMPONENT_IR_DESCRIPTION "OpenVINO IR Frontend")
        set(CPACK_COMPONENT_IR_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_DEBIAN_IR_PACKAGE_NAME "libopenvino-ir-frontend-${cpack_name_ver}")
        set(CPACK_DEBIAN_IR_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm}")
        ov_debian_add_lintian_suppression(ir
            # we have different package name strategy; it suggests libopenvino-ir-frontend202230
            "package-name-doesnt-match-sonames"
            # IR FE should not linked directly by end users
            "package-must-activate-ldconfig-trigger")
        list(APPEND frontends ir)
        set(ir_copyright "generic")
    endif()

    # It was decided not to distribute JAX as C++ component
    if(ENABLE_OV_JAX_FRONTEND AND OFF)
        set(CPACK_COMPONENT_JAX_DESCRIPTION "OpenVINO JAX Frontend")
        set(CPACK_COMPONENT_JAX_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_DEBIAN_JAX_PACKAGE_NAME "libopenvino-jax-frontend-${cpack_name_ver}")
        # since we JAX FE is linkable target, we need to call ldconfig (i.e. `def_triggers`)
        set(CPACK_DEBIAN_JAX_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")
        ov_debian_add_lintian_suppression(jax
            # we have different package name strategy; it suggests libopenvino-jax-frontend202230
            "package-name-doesnt-match-sonames")
        list(APPEND frontends jax)
        set(jax_copyright "generic")
    endif()

    if(ENABLE_OV_ONNX_FRONTEND)
        set(CPACK_COMPONENT_ONNX_DESCRIPTION "OpenVINO ONNX Frontend")
        set(CPACK_COMPONENT_ONNX_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_DEBIAN_ONNX_PACKAGE_NAME "libopenvino-onnx-frontend-${cpack_name_ver}")
        # since we ONNX FE is linkable target, we need to call ldconfig (i.e. `def_triggers`)
        set(CPACK_DEBIAN_ONNX_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")
        ov_debian_add_lintian_suppression(onnx
            # we have different package name strategy; it suggests libopenvino-onnx-frontend202230
            "package-name-doesnt-match-sonames")
        list(APPEND frontends onnx)
        set(onnx_copyright "generic")
    endif()

    if(ENABLE_OV_TF_FRONTEND)
        set(CPACK_COMPONENT_TENSORFLOW_DESCRIPTION "OpenVINO TensorFlow Frontend")
        set(CPACK_COMPONENT_TENSORFLOW_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_DEBIAN_TENSORFLOW_PACKAGE_NAME "libopenvino-tensorflow-frontend-${cpack_name_ver}")
        # since we TF FE is linkable target, we need to call ldconfig (i.e. `def_triggers`)
        set(CPACK_DEBIAN_TENSORFLOW_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")
        ov_debian_add_lintian_suppression(tensorflow
            # we have different package name strategy; it suggests libopenvino-tensorflow-frontend202230
            "package-name-doesnt-match-sonames")
        list(APPEND frontends tensorflow)
        set(tensorflow_copyright "generic")
    endif()

    if(ENABLE_OV_PADDLE_FRONTEND)
        set(CPACK_COMPONENT_PADDLE_DESCRIPTION "OpenVINO Paddle Frontend")
        set(CPACK_COMPONENT_PADDLE_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_DEBIAN_PADDLE_PACKAGE_NAME "libopenvino-paddle-frontend-${cpack_name_ver}")
        # since we PADDLE FE is linkable target, we need to call ldconfig (i.e. `def_triggers`)
        set(CPACK_DEBIAN_PADDLE_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")
        ov_debian_add_lintian_suppression(paddle
            # we have different package name strategy; it suggests libopenvino-paddle-frontend202230
            "package-name-doesnt-match-sonames")
        list(APPEND frontends paddle)
        set(paddle_copyright "generic")
    endif()

    if(ENABLE_OV_PYTORCH_FRONTEND)
        set(CPACK_COMPONENT_PYTORCH_DESCRIPTION "OpenVINO PyTorch Frontend")
        set(CPACK_COMPONENT_PYTORCH_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_DEBIAN_PYTORCH_PACKAGE_NAME "libopenvino-pytorch-frontend-${cpack_name_ver}")
        # since we PYTORCH FE is linkable target, we need to call ldconfig (i.e. `def_triggers`)
        set(CPACK_DEBIAN_PYTORCH_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")
        ov_debian_add_lintian_suppression(pytorch
            # we have different package name strategy; it suggests libopenvino-pytorch-frontend202230
            "package-name-doesnt-match-sonames")
        list(APPEND frontends pytorch)
        set(pytorch_copyright "generic")
    endif()

    if(ENABLE_OV_TF_LITE_FRONTEND)
        set(CPACK_COMPONENT_TENSORFLOW_LITE_DESCRIPTION "OpenVINO TensorFlow Lite Frontend")
        set(CPACK_COMPONENT_TENSORFLOW_LITE_DEPENDS "${OV_CPACK_COMP_CORE}")
        set(CPACK_DEBIAN_TENSORFLOW_LITE_PACKAGE_NAME "libopenvino-tensorflow-lite-frontend-${cpack_name_ver}")
        # since we TF Lite FE is linkable target, we need to call ldconfig (i.e. `def_triggers`)
        set(CPACK_DEBIAN_TENSORFLOW_LITE_PACKAGE_CONTROL_EXTRA "${def_postinst};${def_postrm};${def_triggers}")
        ov_debian_add_lintian_suppression(tensorflow_lite
            # we have different package name strategy; it suggests libopenvino-tensorflow-lite-frontend202230
            "package-name-doesnt-match-sonames")
        list(APPEND frontends tensorflow_lite)
        set(tensorflow_lite_copyright "generic")
    endif()

    #
    # core_dev: depends on core and frontends (since frontends don't want to provide its own dev packages)
    #

    set(CPACK_COMPONENT_CORE_DEV_DESCRIPTION "Intel(R) Distribution of OpenVINO(TM) Toolkit C / C++ Development files")
    set(CPACK_COMPONENT_CORE_DEV_DEPENDS "${OV_CPACK_COMP_CORE}")
    list(APPEND CPACK_COMPONENT_CORE_DEV_DEPENDS ${frontends})
    set(CPACK_DEBIAN_CORE_DEV_PACKAGE_NAME "libopenvino-dev-${cpack_name_ver}")
    ov_debian_generate_conflicts("${OV_CPACK_COMP_CORE_DEV}" ${conflicting_versions})
    set(${OV_CPACK_COMP_CORE_DEV}_copyright "generic")

    #
    # Python API
    #

    if(ENABLE_PYTHON_PACKAGING)
        ov_get_pyversion(pyversion)
        set(python_component "${OV_CPACK_COMP_PYTHON_OPENVINO_PACKAGE}_${pyversion}")
        string(TOUPPER "${pyversion}" pyversion)

        set(CPACK_COMPONENT_PYOPENVINO_PACKAGE_${pyversion}_DESCRIPTION "OpenVINO Python API")
        set(CPACK_COMPONENT_PYOPENVINO_PACKAGE_${pyversion}_DEPENDS "${OV_CPACK_COMP_CORE}")
        list(APPEND CPACK_COMPONENT_PYOPENVINO_PACKAGE_${pyversion}_DEPENDS ${installed_plugins})
        list(APPEND CPACK_COMPONENT_PYOPENVINO_PACKAGE_${pyversion}_DEPENDS ${frontends})

        set(CPACK_DEBIAN_PYOPENVINO_PACKAGE_${pyversion}_PACKAGE_NAME "python3-openvino-${cpack_name_ver}")
        set(python_package "${CPACK_DEBIAN_PYOPENVINO_PACKAGE_${pyversion}_PACKAGE_NAME} (= ${cpack_full_ver})")
        set(CPACK_DEBIAN_PYOPENVINO_PACKAGE_${pyversion}_PACKAGE_DEPENDS "python3, python3-numpy, python3-packaging")

        # we can have a single python installed, so we need to generate conflicts for all other versions
        ov_debian_generate_conflicts(${python_component} ${conflicting_versions})

        # TODO: fix all the warnings
        ov_debian_add_lintian_suppression(${python_component}
            # usr/lib/python3/dist-packages/requirements.txt
            "unknown-file-in-python-module-directory"
            # all directories
            "non-standard-dir-perm"
            # usr/bin/benchmark_app
            "binary-without-manpage"
            # usr/bin/benchmark_app
            "non-standard-executable-perm"
            # all python files
            "non-standard-file-perm")
        set(${python_component}_copyright "generic")
    endif()

    #
    # Samples
    #

    set(samples_build_deps "cmake, g++, gcc, libc6-dev, make, pkgconf")
    set(samples_build_deps_suggest "libopencv-core-dev, libopencv-imgproc-dev, libopencv-imgcodecs-dev")
    set(samples_opencl_suggest "ocl-icd-opencl-dev, opencl-headers")

    # c_samples / cpp_samples
    set(CPACK_COMPONENT_SAMPLES_DESCRIPTION "Intel(R) Distribution of OpenVINO(TM) Toolkit C / C++ Samples")
    set(CPACK_COMPONENT_SAMPLES_DEPENDS "${OV_CPACK_COMP_CORE_DEV}")
    set(CPACK_DEBIAN_SAMPLES_PACKAGE_NAME "openvino-samples-${cpack_name_ver}")
    set(CPACK_DEBIAN_SAMPLES_PACKAGE_SUGGESTS "${samples_build_deps_suggest}, ${samples_opencl_suggest}, ${all_plugins_suggest}")
    # can be skipped with --no-install-recommends
    set(CPACK_DEBIAN_SAMPLES_PACKAGE_RECOMMENDS "${samples_build_deps}")
    set(CPACK_DEBIAN_SAMPLES_PACKAGE_ARCHITECTURE "all")
    ov_debian_generate_conflicts(${OV_CPACK_COMP_CPP_SAMPLES} ${conflicting_versions})
    set(samples_copyright "generic")

    # python_samples
    if(ENABLE_PYTHON_PACKAGING)
        set(CPACK_COMPONENT_PYTHON_SAMPLES_DESCRIPTION "Intel(R) Distribution of OpenVINO(TM) Toolkit Python Samples")
        set(CPACK_COMPONENT_PYTHON_SAMPLES_DEPENDS "${python_component}")
        set(CPACK_DEBIAN_PYTHON_SAMPLES_PACKAGE_NAME "openvino-samples-python-${cpack_name_ver}")
        set(python_samples_package "${CPACK_DEBIAN_PYTHON_SAMPLES_PACKAGE_NAME} (= ${cpack_full_ver})")
        set(CPACK_DEBIAN_PYTHON_SAMPLES_PACKAGE_DEPENDS "python3, ${python_package}")
        set(CPACK_DEBIAN_PYTHON_SAMPLES_PACKAGE_ARCHITECTURE "all")
        ov_debian_generate_conflicts(${OV_CPACK_COMP_PYTHON_SAMPLES} ${conflicting_versions})
        set(python_samples_copyright "generic")
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
    set(CPACK_DEBIAN_LIBRARIES_PACKAGE_NAME "openvino-libraries-${cpack_name_ver}")
    set(CPACK_DEBIAN_LIBRARIES_PACKAGE_ARCHITECTURE "all")
    set(libraries_copyright "generic")

    # all libraries-dev
    set(CPACK_COMPONENT_LIBRARIES_DEV_DESCRIPTION "Intel(R) Distribution of OpenVINO(TM) Toolkit Libraries and Development files")
    set(CPACK_COMPONENT_LIBRARIES_DEV_DEPENDS "${OV_CPACK_COMP_CORE_DEV};libraries")
    set(CPACK_DEBIAN_LIBRARIES_DEV_PACKAGE_NAME "openvino-libraries-dev-${cpack_name_ver}")
    set(CPACK_DEBIAN_LIBRARIES_DEV_PACKAGE_ARCHITECTURE "all")
    ov_debian_generate_conflicts(libraries_dev ${conflicting_versions})
    set(libraries_dev_copyright "generic")

    # all openvino
    set(CPACK_COMPONENT_OPENVINO_DESCRIPTION "Intel(R) Distribution of OpenVINO(TM) Toolkit Libraries and Development files")
    set(CPACK_COMPONENT_OPENVINO_DEPENDS "libraries_dev;${OV_CPACK_COMP_CPP_SAMPLES}")
    if(ENABLE_PYTHON_PACKAGING)
        list(APPEND CPACK_DEBIAN_OPENVINO_PACKAGE_DEPENDS "${python_package}, ${python_samples_package}")
    endif()
    set(CPACK_DEBIAN_OPENVINO_PACKAGE_NAME "openvino-${cpack_name_ver}")
    set(CPACK_DEBIAN_OPENVINO_PACKAGE_ARCHITECTURE "all")
    ov_debian_generate_conflicts(openvino ${conflicting_versions})
    set(openvino_copyright "generic")

    list(APPEND CPACK_COMPONENTS_ALL "libraries;libraries_dev;openvino")

    #
    # Install latest symlink packages
    #

    # NOTE: we expicitly don't add runtime latest packages
    # since a user needs to depend on specific VERSIONED runtime package
    # with fixed SONAMEs, while latest package can be updated multiple times
    # ov_debian_add_latest_component(libraries)

    ov_debian_add_latest_component(libraries_dev)
    ov_debian_add_latest_component(openvino)
    ov_debian_add_lintian_suppression(openvino_latest
        # reproduced only on ubu18
        "description-starts-with-package-name")

    # users can manually install specific version of package
    # e.g. sudo apt-get install openvino=2022.1.0
    # even if we have latest package version 2022.2.0

    #
    # install debian common files
    #

    foreach(comp IN LISTS CPACK_COMPONENTS_ALL)
        if(NOT DEFINED "${comp}_copyright")
            message(FATAL_ERROR "Copyright file name is not defined for ${comp}")
        endif()
        ov_debian_add_changelog_and_copyright("${comp}" "${${comp}_copyright}")
    endforeach()
endmacro()
