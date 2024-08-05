# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(ov_generate_dev_package_config)
    # dummy check that OpenCV is here
    find_package(OpenCV QUIET)
    if(OpenCV_VERSION VERSION_LESS 3.0)
        set(OpenCV_FOUND OFF)
    endif()

    # create a helper target to build all developer package targets
    add_custom_target(ov_dev_targets DEPENDS ${_OPENVINO_DEVELOPER_PACKAGE_TARGETS})

    # filter out targets which are installed by OpenVINOConfig.cmake static build case
    if(openvino_installed_targets)
        list(REMOVE_ITEM _OPENVINO_DEVELOPER_PACKAGE_TARGETS ${openvino_installed_targets})
    endif()
    # export all developer targets with prefix and use them during extra modules build
    export(TARGETS ${_OPENVINO_DEVELOPER_PACKAGE_TARGETS} NAMESPACE openvino::
           APPEND FILE "${CMAKE_BINARY_DIR}/openvino_developer_package_targets.cmake")

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
    set(DEVELOPER_PACKAGE_EXPORT_SET OpenVINODeveloperTargets)

    # create and install main developer package config files
    configure_package_config_file("${OpenVINO_SOURCE_DIR}/cmake/templates/OpenVINODeveloperPackageConfigRelocatable.cmake.in"
                                  "${OpenVINO_BINARY_DIR}/share/OpenVINODeveloperPackageConfig.cmake"
                                  INSTALL_DESTINATION ${DEV_PACKAGE_CMAKE_DIR}
                                  NO_CHECK_REQUIRED_COMPONENTS_MACRO)

    configure_file("${OpenVINO_SOURCE_DIR}/cmake/templates/OpenVINOConfig-version.cmake.in"
                   "${OpenVINO_BINARY_DIR}/share/OpenVINODeveloperPackageConfig-version.cmake" 
                   @ONLY)

    install(FILES "${OpenVINO_BINARY_DIR}/share/OpenVINODeveloperPackageConfig.cmake"
                  "${OpenVINO_BINARY_DIR}/share/OpenVINODeveloperPackageConfig-version.cmake"
            DESTINATION ${DEV_PACKAGE_CMAKE_DIR}
            COMPONENT ${DEVELOPER_PACKAGE_COMPONENT}
            EXCLUDE_FROM_ALL)

    # Install whole 'cmake/developer_package' folder
    install(DIRECTORY "${OpenVINODeveloperScripts_DIR}/"
            DESTINATION "${DEV_PACKAGE_CMAKE_DIR}"
            COMPONENT ${DEVELOPER_PACKAGE_COMPONENT}
            EXCLUDE_FROM_ALL)

    # Install CMakeLists.txt to read cache variables from
    install(FILES "${OpenVINO_BINARY_DIR}/CMakeCache.txt"
            DESTINATION ${DEV_PACKAGE_CMAKE_DIR}
            COMPONENT ${DEVELOPER_PACKAGE_COMPONENT}
            EXCLUDE_FROM_ALL)

    # install developer package targets
    install(TARGETS ${_OPENVINO_DEVELOPER_PACKAGE_TARGETS} EXPORT ${DEVELOPER_PACKAGE_EXPORT_SET}
            RUNTIME DESTINATION ${DEV_PACKAGE_ROOT_DIR}/bin COMPONENT ${DEVELOPER_PACKAGE_COMPONENT} EXCLUDE_FROM_ALL
            ARCHIVE DESTINATION ${DEV_PACKAGE_ROOT_DIR}/lib COMPONENT ${DEVELOPER_PACKAGE_COMPONENT} EXCLUDE_FROM_ALL
            LIBRARY DESTINATION ${DEV_PACKAGE_ROOT_DIR}/lib COMPONENT ${DEVELOPER_PACKAGE_COMPONENT} EXCLUDE_FROM_ALL)

    install(EXPORT ${DEVELOPER_PACKAGE_EXPORT_SET}
            FILE OpenVINODeveloperPackageTargets.cmake
            NAMESPACE openvino::
            DESTINATION ${DEV_PACKAGE_ROOT_DIR}/cmake
            COMPONENT ${DEVELOPER_PACKAGE_COMPONENT}
            EXCLUDE_FROM_ALL)

    # Note: that OpenCV and gflags are explicitly not installed to simplify relocatable
    # OpenVINO Developer package maintainance. OpenVINO_SOURCE_DIR is also unvailable, because
    # relocatable developer package can be used on a different machine where OpenVINO repo is not available
endfunction()

#
# Add extra modules
#

function(_ov_register_extra_modules)
    set(OpenVINODeveloperPackage_DIR "${CMAKE_BINARY_DIR}/build-modules")
    set(OpenVINO_DIR "${CMAKE_BINARY_DIR}")

    function(_ov_generate_fake_developer_package NS)
        set(devconfig_file "${OpenVINODeveloperPackage_DIR}/OpenVINODeveloperPackageConfig.cmake")

        file(REMOVE "${devconfig_file}")
        file(WRITE "${devconfig_file}" "\# !! AUTOGENERATED: DON'T EDIT !!\n\n")

        foreach(exported_target IN LISTS _OPENVINO_DEVELOPER_PACKAGE_TARGETS)
            string(REPLACE "openvino_" "" exported_target_clean_name "${exported_target}")

            file(APPEND "${devconfig_file}" "if(NOT TARGET openvino::${exported_target_clean_name})
    add_library(${NS}::${exported_target_clean_name} ALIAS ${exported_target})
endif()\n")
        endforeach()

        configure_file("${OpenVINO_SOURCE_DIR}/cmake/templates/OpenVINOConfig-version.cmake.in"
                       "${OpenVINODeveloperPackage_DIR}/OpenVINODeveloperPackageConfig-version.cmake" 
                       @ONLY)
    endfunction()

    _ov_generate_fake_developer_package("openvino")

    # detect where OPENVINO_EXTRA_MODULES contains folders with CMakeLists.txt
    # other folders are supposed to have sub-folders with CMakeLists.txt
    foreach(module_path IN LISTS OPENVINO_EXTRA_MODULES)
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
ov_generate_dev_package_config()

# extra modules must be registered after openvino_runtime library
# and all other OpenVINO Core libraries are creared
# because '_ov_register_extra_modules' creates fake OpenVINODeveloperPackageConfig.cmake
# with all imported developer targets
_ov_register_extra_modules()

# we need to generate final ov_plugins.hpp with all the information about plugins
ov_generate_plugins_hpp()
# we need to generate final ov_frontends.hpp with all the information about frontends
ov_generate_frontends_hpp()
