# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(ie_generate_dev_package_config)
    # dummy check that OpenCV is here
    find_package(OpenCV QUIET)
    if(OpenCV_VERSION VERSION_LESS 3.0)
        set(OpenCV_FOUND OFF)
    endif()

    foreach(component IN LISTS openvino_export_components)
        # export all targets with prefix and use them during extra modules build
        export(TARGETS ${${component}} NAMESPACE IE::
               APPEND FILE "${CMAKE_BINARY_DIR}/${component}_dev_targets.cmake")
        list(APPEND all_dev_targets ${${component}})
    endforeach()
    add_custom_target(ie_dev_targets DEPENDS ${all_dev_targets})

    set(PATH_VARS "OpenVINO_SOURCE_DIR")
    if(ENABLE_SAMPLES OR ENABLE_TESTS)
        list(APPEND PATH_VARS "gflags_BINARY_DIR")
        # if we've found system gflags
        if(gflags_DIR)
            set(gflags_BINARY_DIR "${gflags_DIR}")
        endif()
    endif()

    configure_package_config_file("${OpenVINO_SOURCE_DIR}/cmake/templates/InferenceEngineDeveloperPackageConfig.cmake.in"
                                  "${CMAKE_BINARY_DIR}/InferenceEngineDeveloperPackageConfig.cmake"
                                  INSTALL_DESTINATION share # not used
                                  PATH_VARS ${PATH_VARS}
                                  NO_CHECK_REQUIRED_COMPONENTS_MACRO)

    configure_file("${OpenVINO_SOURCE_DIR}/cmake/templates/InferenceEngineConfig-version.cmake.in"
                   "${CMAKE_BINARY_DIR}/InferenceEngineDeveloperPackageConfig-version.cmake"
                   @ONLY)
endfunction()

function(ov_generate_dev_package_config)
    # dummy check that OpenCV is here
    find_package(OpenCV QUIET)
    if(OpenCV_VERSION VERSION_LESS 3.0)
        set(OpenCV_FOUND OFF)
    endif()

    foreach(component IN LISTS openvino_export_components)
        # filter out targets which are installed by OpenVINOConfig.cmake static build case
        set(exported_targets)
        foreach(target IN LISTS ${component})
            if(NOT target IN_LIST openvino_installed_targets)
                list(APPEND exported_targets ${target})
            endif()
        endforeach()
        # export all developer targets with prefix and use them during extra modules build
        export(TARGETS ${exported_targets} NAMESPACE openvino::
               APPEND FILE "${CMAKE_BINARY_DIR}/ov_${component}_dev_targets.cmake")
        list(APPEND all_dev_targets ${${component}})
    endforeach()
    add_custom_target(ov_dev_targets DEPENDS ${all_dev_targets})

    #
    # OpenVINODeveloperPackageConfig.cmake for build tree
    #

    set(PATH_VARS "OpenVINO_SOURCE_DIR")
    if(ENABLE_SAMPLES OR ENABLE_TESTS)
        list(APPEND PATH_VARS "gflags_BINARY_DIR")
        # if we've found system gflags
        if(gflags_DIR)
            set(gflags_BINARY_DIR "${gflags_DIR}")
        endif()
    endif()

    configure_package_config_file("${OpenVINO_SOURCE_DIR}/cmake/templates/OpenVINODeveloperPackageConfig.cmake.in"
                                  "${CMAKE_BINARY_DIR}/OpenVINODeveloperPackageConfig.cmake"
                                  INSTALL_DESTINATION share # not used
                                  PATH_VARS ${PATH_VARS}
                                  NO_CHECK_REQUIRED_COMPONENTS_MACRO)

    configure_file("${OpenVINO_SOURCE_DIR}/cmake/templates/OpenVINOConfig-version.cmake.in"
                   "${CMAKE_BINARY_DIR}/OpenVINODeveloperPackageConfig-version.cmake" 
                   @ONLY)

    #
    # OpenVINODeveloperPackageConfig.cmake for installation tree
    #

    set(DEV_PACKAGE_ROOT_DIR developer_package)
    set(DEV_PACKAGE_CMAKE_DIR ${DEV_PACKAGE_ROOT_DIR}/cmake)

    set(DEVELOPER_PACKAGE_COMPONENT developer_package)

    # create and install main developer package config files

    function(_ov_generate_relocatable_openvino_developer_package_config)
        # add a flag to denote developer package is relocable
        set(OV_RELOCATABLE_DEVELOPER_PACKAGE ON)

        # overwrite OpenVINO_SOURCE_DIR to make it empty - it's not available for relocatable developer package
        set(openvino_developer_package_config_template
            "${OpenVINO_SOURCE_DIR}/cmake/templates/OpenVINODeveloperPackageConfig.cmake.in")
        set(OpenVINO_SOURCE_DIR "")

        configure_package_config_file("${openvino_developer_package_config_template}"
                                      "${OpenVINO_BINARY_DIR}/share/OpenVINODeveloperPackageConfig.cmake"
                                      INSTALL_DESTINATION ${DEV_PACKAGE_CMAKE_DIR}
                                      PATH_VARS ${PATH_VARS}
                                      NO_CHECK_REQUIRED_COMPONENTS_MACRO)
    endfunction()

    _ov_generate_relocatable_openvino_developer_package_config()

    configure_file("${OpenVINO_SOURCE_DIR}/cmake/templates/OpenVINOConfig-version.cmake.in"
                   "${OpenVINO_BINARY_DIR}/share/OpenVINODeveloperPackageConfig-version.cmake" 
                   @ONLY)

    install(FILES "${OpenVINO_BINARY_DIR}/share/OpenVINODeveloperPackageConfig.cmake"
                  "${OpenVINO_BINARY_DIR}/share/OpenVINODeveloperPackageConfig-version.cmake"
            DESTINATION ${DEV_PACKAGE_CMAKE_DIR}
            COMPONENT ${DEVELOPER_PACKAGE_COMPONENT})

    # Install whole 'cmake/developer_package' folder

    install(DIRECTORY "${OpenVINODeveloperScripts_DIR}/"
            DESTINATION "${DEV_PACKAGE_CMAKE_DIR}/"
            COMPONENT ${DEVELOPER_PACKAGE_COMPONENT})

    # Install CMakeLists.txt to read cache variables from

    install(FILES "${OpenVINO_BINARY_DIR}/CMakeCache.txt"
            DESTINATION ${DEV_PACKAGE_CMAKE_DIR}
            COMPONENT ${DEVELOPER_PACKAGE_COMPONENT})

    # export all sets of developer package targets

    foreach(component IN LISTS openvino_export_components)
        if(component MATCHES ".*_legacy$")
            continue()
        endif()

        # filter out targets which are installed by OpenVINOConfig.cmake static build case
        set(install_targets)
        foreach(target IN LISTS ${component})
            # if(NOT target IN_LIST openvino_installed_targets)
                message("!! Adding ${target}")
                list(APPEND install_targets ${target})
            # endif()
        endforeach()

        # install all developer targets with prefix and use them during extra modules build
        set(export_set ov_${component}_dev_targets)

        install(TARGETS ${install_targets} EXPORT OpenVINODeveloperTargets
                RUNTIME DESTINATION ${DEV_PACKAGE_ROOT_DIR}/bin COMPONENT ${DEVELOPER_PACKAGE_COMPONENT}
                ARCHIVE DESTINATION ${DEV_PACKAGE_ROOT_DIR}/lib COMPONENT ${DEVELOPER_PACKAGE_COMPONENT}
                LIBRARY DESTINATION ${DEV_PACKAGE_ROOT_DIR}/lib COMPONENT ${DEVELOPER_PACKAGE_COMPONENT})
    endforeach()

    install(EXPORT OpenVINODeveloperTargets
            FILE OpenVINODeveloperTargets.cmake
            NAMESPACE openvino::
            DESTINATION ${DEV_PACKAGE_ROOT_DIR}/cmake
            COMPONENT ${DEVELOPER_PACKAGE_COMPONENT})

    # message(FATAL_ERROR "${install_targets}")

    # TODO: OpenCV

    # TODO: gflags for samples

endfunction()

#
# Add extra modules
#

function(_ov_register_extra_modules)
    set(InferenceEngineDeveloperPackage_DIR "${CMAKE_CURRENT_BINARY_DIR}/build-modules")
    set(OpenVINODeveloperPackage_DIR "${CMAKE_BINARY_DIR}/build-modules")
    set(OpenVINO_DIR "${CMAKE_BINARY_DIR}")

    function(generate_fake_dev_package NS)
        if(NS STREQUAL "openvino")
            set(devconfig_file "${OpenVINODeveloperPackage_DIR}/OpenVINODeveloperPackageConfig.cmake")
        else()
            set(devconfig_file "${InferenceEngineDeveloperPackage_DIR}/InferenceEngineDeveloperPackageConfig.cmake")
        endif()
        file(REMOVE "${devconfig_file}")

        file(WRITE "${devconfig_file}" "\# !! AUTOGENERATED: DON'T EDIT !!\n\n")

        foreach(targets_list IN LISTS ${openvino_export_components})
            foreach(target IN LISTS targets_list)
                file(APPEND "${devconfig_file}" "if(NOT TARGET ${NS}::${target})
    add_library(${NS}::${target} ALIAS ${target})
endif()\n")
            endforeach()
        endforeach()
    endfunction()

    generate_fake_dev_package("openvino")
    generate_fake_dev_package("IE")

    # detect where OPENVINO_EXTRA_MODULES contains folders with CMakeLists.txt
    # other folders are supposed to have sub-folders with CMakeLists.txt
    foreach(module_path IN LISTS OPENVINO_EXTRA_MODULES IE_EXTRA_MODULES)
        get_filename_component(module_path "${module_path}" ABSOLUTE)
        if(EXISTS "${module_path}/CMakeLists.txt")
            list(APPEND extra_modules "${module_path}")
        elseif(module_path)
            file(GLOB extra_modules ${extra_modules} "${module_path}/*")
        endif()
    endforeach()

    # add template plugin
    if(ENABLE_TEMPLATE)
        list(APPEND extra_modules "${OpenVINO_SOURCE_DIR}/src/plugins/template")
    endif()
    list(APPEND extra_modules "${OpenVINO_SOURCE_DIR}/src/core/template_extension")

    # add extra flags for compilation of extra modules:
    # since not all extra modules use OpenVINODeveloperPackage, we have to add these function calls here
    ov_dev_package_no_errors()
    ov_deprecated_no_errors()

    # add each extra module
    foreach(module_path IN LISTS extra_modules)
        if(module_path)
            get_filename_component(module_name "${module_path}" NAME)
            set(build_module ON)
            if(NOT EXISTS "${module_path}/CMakeLists.txt") # if module is built not using cmake
                set(build_module OFF)
            endif()
            if(NOT DEFINED BUILD_${module_name})
                set(BUILD_${module_name} ${build_module} CACHE BOOL "Build ${module_name} extra module" FORCE)
            endif()
            if(BUILD_${module_name})
                message(STATUS "Register ${module_name} to be built in build-modules/${module_name}")
                add_subdirectory("${module_path}" "build-modules/${module_name}")
            endif()
        endif()
    endforeach()
endfunction()

#
# Extra modules support
#

# this OpenVINODeveloperPackageConfig.cmake is not used during extra modules build
# since it's generated after modules are configured
ie_generate_dev_package_config()
ov_generate_dev_package_config()

# extra modules must be registered after inference_engine library
# and all other OpenVINO Core libraries are creared
# because '_ov_register_extra_modules' creates fake InferenceEngineDeveloperPackageConfig.cmake
# with all imported developer targets
_ov_register_extra_modules()

# we need to generate final ov_plugins.hpp with all the information about plugins
ov_generate_plugins_hpp()
# we need to generate final ov_frontends.hpp with all the information about frontends
ov_generate_frontends_hpp()
