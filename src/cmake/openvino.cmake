# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino)

#
# Add openvino library
#

add_library(${TARGET_NAME}
    $<TARGET_OBJECTS:ngraph_obj>
    $<TARGET_OBJECTS:frontend_common_obj>
    $<TARGET_OBJECTS:inference_engine_obj>
    $<TARGET_OBJECTS:inference_engine_transformations_obj>
    $<TARGET_OBJECTS:inference_engine_lp_transformations_obj>)

add_library(openvino::runtime ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME runtime)

ie_add_vs_version_file(NAME ${TARGET_NAME} FILEDESCRIPTION "OpenVINO runtime library")
ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/core/include>
    $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/frontends/common/include>
    $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/inference/include>
    $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/inference/include/ie>)

target_link_libraries(${TARGET_NAME} PRIVATE ngraph_reference
                                             ngraph_builders
                                             ov_shape_inference
                                             openvino::pugixml
                                             ${CMAKE_DL_LIBS}
                                             Threads::Threads)

if (TBBBIND_2_5_FOUND)
    target_link_libraries(${TARGET_NAME} PRIVATE ${TBBBIND_2_5_IMPORTED_TARGETS})
endif()

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

if(WIN32)
    set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})
endif()

set_ie_threading_interface_for(${TARGET_NAME})
ie_mark_target_as_cc(${TARGET_NAME})

# LTO
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

ie_register_plugins(MAIN_TARGET ${TARGET_NAME})

# Export for build tree

export(TARGETS ${TARGET_NAME} NAMESPACE openvino::
       APPEND FILE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")

install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
        RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${OV_CPACK_COMP_CORE}
        ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${OV_CPACK_COMP_CORE}
        LIBRARY DESTINATION ${OV_CPACK_LIBRARYDIR} COMPONENT ${OV_CPACK_COMP_CORE}
        NAMELINK_COMPONENT ${OV_CPACK_COMP_CORE_DEV}
        INCLUDES DESTINATION ${OV_CPACK_INCLUDEDIR}
                             ${OV_CPACK_INCLUDEDIR}/ie)

# OpenVINO runtime library dev

#
# Add openvin::dev target
#

add_library(${TARGET_NAME}_dev INTERFACE)
add_library(openvino::runtime::dev ALIAS ${TARGET_NAME}_dev)

target_include_directories(${TARGET_NAME}_dev INTERFACE
    $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/common/transformations/include>
    $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/core/dev_api>
    $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/inference/dev_api>
    $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/common/low_precision_transformations/include>
    $<TARGET_PROPERTY:openvino_gapi_preproc,INTERFACE_INCLUDE_DIRECTORIES>)

target_compile_definitions(${TARGET_NAME}_dev INTERFACE
    $<TARGET_PROPERTY:openvino_gapi_preproc,INTERFACE_COMPILE_DEFINITIONS>)

target_link_libraries(${TARGET_NAME}_dev INTERFACE ${TARGET_NAME} openvino::itt openvino::util)

set_ie_threading_interface_for(${TARGET_NAME}_dev)
set_target_properties(${TARGET_NAME}_dev PROPERTIES EXPORT_NAME runtime::dev)

openvino_developer_export_targets(COMPONENT core TARGETS openvino::runtime::dev)

# Install static libraries for case BUILD_SHARED_LIBS=OFF
ov_install_static_lib(${TARGET_NAME}_dev ${OV_CPACK_COMP_CORE})

#
# Install OpenVINO runtime
#

list(APPEND PATH_VARS "IE_INCLUDE_DIR")

ov_add_library_version(${TARGET_NAME})

if(ENABLE_INTEL_GNA)
    list(APPEND PATH_VARS "GNA_PATH")
endif()

ie_cpack_add_component(${OV_CPACK_COMP_CORE}
                       HIDDEN
                       DEPENDS ${core_components})
ie_cpack_add_component(${OV_CPACK_COMP_CORE_DEV}
                       HIDDEN
                       DEPENDS ${OV_CPACK_COMP_CORE} ${core_dev_components})

if(BUILD_SHARED_LIBS)
    install(FILES $<TARGET_FILE_DIR:${TARGET_NAME}>/plugins.xml
            DESTINATION ${OV_CPACK_PLUGINSDIR}
            COMPONENT ${OV_CPACK_COMP_CORE})

    if(ENABLE_TESTS)
        # for InferenceEngineUnitTest
        install(FILES $<TARGET_FILE_DIR:${TARGET_NAME}>/plugins.xml
                DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
    endif()
endif()

# Install cmake scripts

install(EXPORT OpenVINOTargets
        FILE OpenVINOTargets.cmake
        NAMESPACE openvino::
        DESTINATION ${OV_CPACK_OPENVINO_CMAKEDIR}
        COMPONENT ${OV_CPACK_COMP_CORE_DEV})

set(PUBLIC_HEADERS_DIR "${OpenVINO_SOURCE_DIR}/src/inference/include")
set(IE_INCLUDE_DIR "${PUBLIC_HEADERS_DIR}/ie")

configure_package_config_file("${OpenVINO_SOURCE_DIR}/cmake/templates/InferenceEngineConfig.cmake.in"
                              "${CMAKE_BINARY_DIR}/InferenceEngineConfig.cmake"
                               INSTALL_DESTINATION "${CMAKE_INSTALL_PREFIX}"
                               PATH_VARS ${PATH_VARS})

configure_package_config_file("${OpenVINO_SOURCE_DIR}/cmake/templates/OpenVINOConfig.cmake.in"
                              "${CMAKE_BINARY_DIR}/OpenVINOConfig.cmake"
                              INSTALL_DESTINATION "${CMAKE_INSTALL_PREFIX}"
                              PATH_VARS ${PATH_VARS})

set(IE_INCLUDE_DIR "${OV_CPACK_INCLUDEDIR}/ie")
set(IE_TBB_DIR "${IE_TBB_DIR_INSTALL}")
set(IE_TBBBIND_DIR "${IE_TBBBIND_DIR_INSTALL}")
set(GNA_PATH "${OV_CPACK_RUNTIMEDIR}")
if(WIN32)
    set(GNA_PATH "${OV_CPACK_LIBRARYDIR}/../Release")
endif()

configure_package_config_file("${OpenVINO_SOURCE_DIR}/cmake/templates/InferenceEngineConfig.cmake.in"
                              "${CMAKE_BINARY_DIR}/share/InferenceEngineConfig.cmake"
                              INSTALL_DESTINATION ${OV_CPACK_IE_CMAKEDIR}
                              PATH_VARS ${PATH_VARS})

configure_package_config_file("${OpenVINO_SOURCE_DIR}/cmake/templates/OpenVINOConfig.cmake.in"
                              "${CMAKE_BINARY_DIR}/share/OpenVINOConfig.cmake"
                              INSTALL_DESTINATION ${OV_CPACK_OPENVINO_CMAKEDIR}
                              PATH_VARS ${PATH_VARS})

configure_file("${OpenVINO_SOURCE_DIR}/cmake/templates/InferenceEngineConfig-version.cmake.in"
               "${CMAKE_BINARY_DIR}/InferenceEngineConfig-version.cmake" @ONLY)
configure_file("${OpenVINO_SOURCE_DIR}/cmake/templates/OpenVINOConfig-version.cmake.in"
               "${CMAKE_BINARY_DIR}/OpenVINOConfig-version.cmake" @ONLY)

install(FILES "${CMAKE_BINARY_DIR}/share/InferenceEngineConfig.cmake"
              "${CMAKE_BINARY_DIR}/InferenceEngineConfig-version.cmake"
        DESTINATION ${OV_CPACK_IE_CMAKEDIR}
        COMPONENT ${OV_CPACK_COMP_CORE_DEV})

install(FILES "${CMAKE_BINARY_DIR}/share/OpenVINOConfig.cmake"
              "${CMAKE_BINARY_DIR}/OpenVINOConfig-version.cmake"
        DESTINATION ${OV_CPACK_OPENVINO_CMAKEDIR}
        COMPONENT ${OV_CPACK_COMP_CORE_DEV})

# Generate and install openvino.pc pkg-config file

if(ENABLE_PKGCONFIG_GEN)
    # fill in PKGCONFIG_OpenVINO_FRONTENDS
    get_target_property(PKGCONFIG_OpenVINO_FRONTENDS_LIST ov_frontends MANUALLY_ADDED_DEPENDENCIES)
    if(ENABLE_OV_IR_FRONTEND)
        list(REMOVE_ITEM PKGCONFIG_OpenVINO_FRONTENDS_LIST openvino_ir_frontend)
    endif()
    if(ENABLE_OV_TF_FRONTEND)
        list(REMOVE_ITEM PKGCONFIG_OpenVINO_FRONTENDS_LIST openvino_tensorflow_frontend)
    endif()

    foreach(frontend IN LISTS PKGCONFIG_OpenVINO_FRONTENDS_LIST)
        if(PKGCONFIG_OpenVINO_FRONTENDS)
            set(PKGCONFIG_OpenVINO_FRONTENDS "${PKGCONFIG_OpenVINO_FRONTENDS} -l${frontend}")
        else()
            set(PKGCONFIG_OpenVINO_FRONTENDS "-l${frontend}")
        endif()
    endforeach()

    # fill in PKGCONFIG_OpenVINO_REQUIRES_PRIVATE and PKGCONFIG_OpenVINO_PRIVATE_DEPS

    if(ENABLE_SYSTEM_TBB)
        set(PKGCONFIG_OpenVINO_REQUIRES_PRIVATE "tbb")
    elseif(TBB_FOUND)
        if(NOT pkg_config_tbb_lib_dir)
            message(FATAL_ERROR "Internal error: variable 'pkg_config_tbb_lib_dir' is not defined")
        endif()

        set(PKGCONFIG_OpenVINO_PRIVATE_DEPS "-L\${prefix}/${pkg_config_tbb_lib_dir} -ltbb")
    endif()

    if(ENABLE_SYSTEM_PUGIXML)
        pkg_check_modules(PKGCONFIG_pugixml QUIET
                          NO_CMAKE_PATH
                          NO_CMAKE_ENVIRONMENT_PATH
                          pugixml)
        set(pugixml_dep "pugixml = ${PKGCONFIG_pugixml_VERSION}")

        if(pugixml_buggy_pkgconfig)
            if(PKGCONFIG_OpenVINO_PRIVATE_DEPS)
                set(PKGCONFIG_OpenVINO_PRIVATE_DEPS "${PKGCONFIG_OpenVINO_PRIVATE_DEPS} -lpugixml")
            else()
                set(PKGCONFIG_OpenVINO_PRIVATE_DEPS "-lpugixml")
            endif()
        else()
            if(PKGCONFIG_OpenVINO_REQUIRES_PRIVATE)
                set(PKGCONFIG_OpenVINO_REQUIRES_PRIVATE "${PKGCONFIG_OpenVINO_REQUIRES_PRIVATE}, ${pugixml_dep}")
            else()
                set(PKGCONFIG_OpenVINO_REQUIRES_PRIVATE "${pugixml_dep}")
            endif()
        endif()
    endif()

    # define relative paths
    file(RELATIVE_PATH PKGCONFIG_OpenVINO_PREFIX "/${OV_CPACK_RUNTIMEDIR}/pkgconfig" "/")

    set(pkgconfig_in "${OpenVINO_SOURCE_DIR}/cmake/templates/openvino.pc.in")
    set(pkgconfig_out "${OpenVINO_BINARY_DIR}/share/openvino.pc")
    configure_file("${pkgconfig_in}" "${pkgconfig_out}" @ONLY)

    install(FILES "${pkgconfig_out}"
            DESTINATION "${OV_CPACK_RUNTIMEDIR}/pkgconfig"
            COMPONENT ${OV_CPACK_COMP_CORE_DEV})

    if (PKG_CONFIG_VERSION_STRING VERSION_LESS 0.29)
        set(pkgconfig_option "--exists")
    else()
        set(pkgconfig_option "--validate")
    endif()
    add_custom_command(TARGET openvino PRE_BUILD
        COMMAND "${PKG_CONFIG_EXECUTABLE}" "${pkgconfig_option}" "${pkgconfig_out}"
        COMMENT "[pkg-config] validating openvino.pc"
        VERBATIM)
endif()
