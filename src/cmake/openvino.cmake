# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino)

#
# Add openvino library
#

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /ignore:4098")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /ignore:4098")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ignore:4098")

    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /ignore:4286")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /ignore:4286")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ignore:4286")
endif()

add_library(${TARGET_NAME}
    $<TARGET_OBJECTS:openvino_core_obj>
    $<TARGET_OBJECTS:openvino_core_obj_version>
    $<TARGET_OBJECTS:openvino_frontend_common_obj>
    $<TARGET_OBJECTS:openvino_runtime_obj>
    $<TARGET_OBJECTS:openvino_transformations_obj>
    $<TARGET_OBJECTS:openvino_lp_transformations_obj>
    $<$<TARGET_EXISTS:openvino_proxy_plugin_obj>:$<TARGET_OBJECTS:openvino_proxy_plugin_obj>>)

add_library(openvino::runtime ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME runtime)

target_compile_features(${TARGET_NAME} PUBLIC cxx_std_11)

ov_add_vs_version_file(NAME ${TARGET_NAME} FILEDESCRIPTION "OpenVINO runtime library")

target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/core/include>
    $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/frontends/common/include>
    $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/inference/include>)

target_link_libraries(${TARGET_NAME} PRIVATE openvino::reference
                                             openvino::shape_inference
                                             openvino::pugixml
                                             ${CMAKE_DL_LIBS}
                                             Threads::Threads)

if (TBBBIND_2_5_FOUND)
    target_link_libraries(${TARGET_NAME} PRIVATE ${TBBBIND_2_5_IMPORTED_TARGETS})
endif()

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

if(DEFINED OV_GLIBCXX_USE_CXX11_ABI)
    target_compile_definitions(${TARGET_NAME} PUBLIC _GLIBCXX_USE_CXX11_ABI=${OV_GLIBCXX_USE_CXX11_ABI})
endif()

if(WIN32)
    set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})
endif()

if(RISCV64)
    # for std::atomic_bool
    target_link_libraries(${TARGET_NAME} PRIVATE atomic)
endif()

ov_set_threading_interface_for(${TARGET_NAME})
ov_mark_target_as_cc(${TARGET_NAME})

if(TBB_FOUND)
    if(NOT TBB_LIB_INSTALL_DIR)
        message(FATAL_ERROR "Internal error: variable 'TBB_LIB_INSTALL_DIR' is not defined")
    endif()
    # set RPATH / LC_RPATH to TBB library directory
    ov_set_install_rpath(${TARGET_NAME} ${OV_CPACK_RUNTIMEDIR} ${TBB_LIB_INSTALL_DIR})
endif()

# must be called after all target_link_libraries
ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME} EXTRA ${TBB_IMPORTED_TARGETS})

# LTO
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

ov_register_plugins(MAIN_TARGET ${TARGET_NAME})

# Export for build tree

export(TARGETS ${TARGET_NAME} NAMESPACE openvino::
       APPEND FILE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")

if(BUILD_SHARED_LIBS)
    set(archive_comp CORE_DEV)
else()
    set(archive_comp CORE)
endif()

install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
        RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${OV_CPACK_COMP_CORE} ${OV_CPACK_COMP_CORE_EXCLUDE_ALL}
        ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${OV_CPACK_COMP_${archive_comp}} ${OV_CPACK_COMP_${archive_comp}_EXCLUDE_ALL}
        LIBRARY DESTINATION ${OV_CPACK_LIBRARYDIR} COMPONENT ${OV_CPACK_COMP_CORE} ${OV_CPACK_COMP_CORE_EXCLUDE_ALL}
        NAMELINK_COMPONENT ${OV_CPACK_COMP_LINKS} ${OV_CPACK_COMP_LINKS_EXCLUDE_ALL}
        INCLUDES DESTINATION ${OV_CPACK_INCLUDEDIR})

# OpenVINO runtime library dev

#
# Add openvino::runtime::dev target
#

add_library(openvino_runtime_dev INTERFACE)
add_library(openvino::runtime::dev ALIAS openvino_runtime_dev)

target_include_directories(openvino_runtime_dev INTERFACE
    $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/inference/dev_api>)

target_link_libraries(openvino_runtime_dev INTERFACE ${TARGET_NAME} openvino::core::dev)

ov_set_threading_interface_for(openvino_runtime_dev)
set_target_properties(openvino_runtime_dev PROPERTIES EXPORT_NAME runtime::dev)

ov_developer_package_export_targets(TARGET openvino_runtime_dev
                                    INSTALL_INCLUDE_DIRECTORIES "${OpenVINO_SOURCE_DIR}/src/inference/dev_api/")

file(GLOB_RECURSE dev_api_src "${CMAKE_CURRENT_SOURCE_DIR}/OpenVINO_SOURCE_DIR}/src/inference/dev_api/openvino/*.hpp")
ov_add_clang_format_target(openvino_runtime_dev_clang FOR_SOURCES ${plugin_api_src})

ov_ncc_naming_style(FOR_TARGET openvino_runtime_dev
                    SOURCE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/src/inference/dev_api/openvino"
                    ADDITIONAL_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:openvino::runtime,INTERFACE_INCLUDE_DIRECTORIES>)

# Install static libraries for case BUILD_SHARED_LIBS=OFF
ov_install_static_lib(openvino_runtime_dev ${OV_CPACK_COMP_CORE})

#
# Install OpenVINO runtime
#

ov_add_library_version(${TARGET_NAME})

ov_cpack_add_component(${OV_CPACK_COMP_CORE}
                       HIDDEN
                       DEPENDS ${core_components})
ov_cpack_add_component(${OV_CPACK_COMP_CORE_DEV}
                       HIDDEN
                       DEPENDS ${OV_CPACK_COMP_CORE} ${core_dev_components})

ov_cpack_add_component(${OV_CPACK_COMP_LINKS}
                       HIDDEN
                       DEPENDS ${OV_CPACK_COMP_CORE_DEV})

if(ENABLE_PLUGINS_XML)
    install(FILES $<TARGET_FILE_DIR:${TARGET_NAME}>/plugins.xml
            DESTINATION ${OV_CPACK_PLUGINSDIR}
            COMPONENT ${OV_CPACK_COMP_CORE}
            ${OV_CPACK_COMP_CORE_EXCLUDE_ALL})

    if(ENABLE_TESTS)
        # for ov_inference_unit_tests
        install(FILES $<TARGET_FILE_DIR:${TARGET_NAME}>/plugins.xml
                DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
    endif()
endif()

#
# Install cmake scripts
#

install(EXPORT OpenVINOTargets
        FILE OpenVINOTargets.cmake
        NAMESPACE openvino::
        DESTINATION ${OV_CPACK_OPENVINO_CMAKEDIR}
        COMPONENT ${OV_CPACK_COMP_CORE_DEV}
        ${OV_CPACK_COMP_CORE_DEV_EXCLUDE_ALL})

# build tree

if(DNNL_USE_ACL)
    list(APPEND BUILD_PATH_VARS "FIND_ACL_PATH;CMAKE_ARCHIVE_OUTPUT_DIRECTORY")
    set(FIND_ACL_PATH "${intel_cpu_thirdparty_SOURCE_DIR}")
endif()
if(ENABLE_ONEDNN_FOR_GPU)
    list(APPEND BUILD_PATH_VARS "ONEDNN_GPU_LIB_PATH")
endif()

set(OV_TBB_DIR "${TBB_DIR}")

configure_package_config_file("${OpenVINO_SOURCE_DIR}/cmake/templates/OpenVINOConfig.cmake.in"
                              "${CMAKE_BINARY_DIR}/OpenVINOConfig.cmake"
                              INSTALL_DESTINATION "${CMAKE_INSTALL_PREFIX}"
                              PATH_VARS ${PATH_VARS} ${BUILD_PATH_VARS})

# install tree

list(APPEND INSTALL_PATH_VARS "OPENVINO_LIB_DIR")
# remove generator expression at the end, because searching in Release / Debug
# will be done by inside OpenVINOConfig.cmak / ACLConfig.cmake
string(REPLACE "$<CONFIG>" "" OPENVINO_LIB_DIR "${OV_CPACK_LIBRARYDIR}")

set(OV_TBB_DIR "${OV_TBB_DIR_INSTALL}")
set(OV_TBBBIND_DIR "${OV_TBBBIND_DIR_INSTALL}")

configure_package_config_file("${OpenVINO_SOURCE_DIR}/cmake/templates/OpenVINOConfig.cmake.in"
                              "${CMAKE_BINARY_DIR}/share/OpenVINOConfig.cmake"
                              INSTALL_DESTINATION ${OV_CPACK_OPENVINO_CMAKEDIR}
                              PATH_VARS ${PATH_VARS} ${INSTALL_PATH_VARS})

configure_file("${OpenVINO_SOURCE_DIR}/cmake/templates/OpenVINOConfig-version.cmake.in"
               "${CMAKE_BINARY_DIR}/OpenVINOConfig-version.cmake" @ONLY)

install(FILES "${CMAKE_BINARY_DIR}/share/OpenVINOConfig.cmake"
              "${CMAKE_BINARY_DIR}/OpenVINOConfig-version.cmake"
        DESTINATION ${OV_CPACK_OPENVINO_CMAKEDIR}
        COMPONENT ${OV_CPACK_COMP_CORE_DEV}
        ${OV_CPACK_COMP_CORE_DEV_EXCLUDE_ALL})

#
# Generate and install openvino.pc pkg-config file
#

if(ENABLE_PKGCONFIG_GEN)

    ov_cpack_add_component(${OV_CPACK_COMP_PKG_CONFIG}
                        HIDDEN
                        DEPENDS ${OV_CPACK_COMP_CORE_DEV})

    # fill in PKGCONFIG_OpenVINO_DEFINITIONS
    get_target_property(openvino_defs openvino INTERFACE_COMPILE_DEFINITIONS)
    foreach(openvino_def IN LISTS openvino_defs)
        set(PKGCONFIG_OpenVINO_DEFINITIONS "${PKGCONFIG_OpenVINO_DEFINITIONS} -D${openvino_def}")
    endforeach()

    # fill in PKGCONFIG_OpenVINO_FRONTENDS
    get_target_property(PKGCONFIG_OpenVINO_FRONTENDS_LIST ov_frontends MANUALLY_ADDED_DEPENDENCIES)
    if(ENABLE_OV_IR_FRONTEND)
        list(REMOVE_ITEM PKGCONFIG_OpenVINO_FRONTENDS_LIST openvino_ir_frontend)
    endif()

    foreach(frontend IN LISTS PKGCONFIG_OpenVINO_FRONTENDS_LIST)
        if(PKGCONFIG_OpenVINO_FRONTENDS)
            set(PKGCONFIG_OpenVINO_FRONTENDS "${PKGCONFIG_OpenVINO_FRONTENDS} -l${frontend}")
        else()
            set(PKGCONFIG_OpenVINO_FRONTENDS "-l${frontend}")
        endif()
    endforeach()

    # fill in PKGCONFIG_OpenVINO_PRIVATE_DEPS

    if(ENABLE_SYSTEM_TBB)
        set(PKGCONFIG_OpenVINO_PRIVATE_DEPS "-ltbb")
    elseif(TBB_FOUND)
        if(NOT TBB_LIB_INSTALL_DIR)
            message(FATAL_ERROR "Internal error: variable 'TBB_LIB_INSTALL_DIR' is not defined")
        endif()

        set(PKGCONFIG_OpenVINO_PRIVATE_DEPS "-L\${prefix}/${TBB_LIB_INSTALL_DIR} -ltbb")
    endif()

    if(ENABLE_SYSTEM_PUGIXML)
        if(PKGCONFIG_OpenVINO_PRIVATE_DEPS)
            set(PKGCONFIG_OpenVINO_PRIVATE_DEPS "${PKGCONFIG_OpenVINO_PRIVATE_DEPS} -lpugixml")
        else()
            set(PKGCONFIG_OpenVINO_PRIVATE_DEPS "-lpugixml")
        endif()
    endif()

    # define relative paths
    file(RELATIVE_PATH PKGCONFIG_OpenVINO_PREFIX "/${OV_CPACK_RUNTIMEDIR}/pkgconfig" "/")

    set(pkgconfig_in "${OpenVINO_SOURCE_DIR}/cmake/templates/openvino.pc.in")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20 AND OV_GENERATOR_MULTI_CONFIG)
        set(pkgconfig_out "${OpenVINO_BINARY_DIR}/share/$<CONFIG>/openvino.pc")
    else()
        set(pkgconfig_out "${OpenVINO_BINARY_DIR}/share/openvino.pc")
    endif()

    if(PKG_CONFIG_VERSION_STRING VERSION_LESS 0.29)
        set(pkgconfig_option "--exists")
    else()
        set(pkgconfig_option "--validate")
    endif()

    add_custom_command(TARGET openvino POST_BUILD
        COMMAND "${CMAKE_COMMAND}" --config $<CONFIG>
                -D PKG_CONFIG_IN_FILE=${pkgconfig_in}
                -D PKG_CONFIG_OUT_FILE=${pkgconfig_out}
                -D PKGCONFIG_OpenVINO_PREFIX=${PKGCONFIG_OpenVINO_PREFIX}
                -D OV_CPACK_RUNTIMEDIR=${OV_CPACK_RUNTIMEDIR}
                -D OV_CPACK_INCLUDEDIR=${OV_CPACK_INCLUDEDIR}
                -D OpenVINO_VERSION=${OpenVINO_VERSION}
                -D PKGCONFIG_OpenVINO_DEFINITIONS=${PKGCONFIG_OpenVINO_DEFINITIONS}
                -D PKGCONFIG_OpenVINO_FRONTENDS=${PKGCONFIG_OpenVINO_FRONTENDS}
                -D PKGCONFIG_OpenVINO_PRIVATE_DEPS=${PKGCONFIG_OpenVINO_PRIVATE_DEPS}
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/pkg_config_gen.cmake"
        COMMAND "${PKG_CONFIG_EXECUTABLE}" "${pkgconfig_option}" "${pkgconfig_out}"
        COMMENT "[pkg-config] creation and validation of openvino.pc"
        VERBATIM)

    install(FILES "${pkgconfig_out}"
            DESTINATION "${OV_CPACK_RUNTIMEDIR}/pkgconfig"
            COMPONENT ${OV_CPACK_COMP_PKG_CONFIG}
            ${OV_CPACK_COMP_PKG_CONFIG_EXCLUDE_ALL})
endif()
