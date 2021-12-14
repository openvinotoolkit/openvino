# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_runtime)

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()
add_definitions(-DPROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_target(ov_runtime_libraries)

include(cmake/install_tbb.cmake)

add_library(${TARGET_NAME} $<TARGET_OBJECTS:ngraph_obj>
                           $<TARGET_OBJECTS:frontend_common_obj>
                           $<TARGET_OBJECTS:inference_engine_transformations_obj>
                           $<TARGET_OBJECTS:inference_engine_lp_transformations_obj>
                           $<TARGET_OBJECTS:inference_engine_obj>
    )
add_library(openvino::runtime ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME runtime)
ie_add_vs_version_file(NAME ${TARGET_NAME} FILEDESCRIPTION "OpenVINO runtime library")
ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:$<TARGET_PROPERTY:ngraph,INTERFACE_INCLUDE_DIRECTORIES>>
                                                 $<BUILD_INTERFACE:$<TARGET_PROPERTY:frontend_common,INTERFACE_INCLUDE_DIRECTORIES>>
                                                 $<BUILD_INTERFACE:$<TARGET_PROPERTY:inference_engine,INTERFACE_INCLUDE_DIRECTORIES>>
                                                 # WA: frontends have incorrect install paths
                                                 $<INSTALL_INTERFACE:${FRONTEND_INSTALL_INCLUDE}>
    )

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

if(WIN32)
    set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})
endif()

target_link_libraries(${TARGET_NAME} PRIVATE pugixml::static
                                             openvino::itt
                                             openvino::util
                                             ngraph_reference
                                             ngraph_builders
                                             ov_shape_inference
                                             ${CMAKE_DL_LIBS}
                                             Threads::Threads)
if (TBBBIND_2_5_FOUND)
    target_link_libraries(${TARGET_NAME} PRIVATE ${TBBBIND_2_5_IMPORTED_TARGETS})
endif()

# LTO
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

ie_mark_target_as_cc(${TARGET_NAME})
set_ie_threading_interface_for(ov_runtime)

# TODO: Fix Wall for core, frontends, tests and  c binding
add_subdirectory(core)
add_subdirectory(frontends)
# TODO: remove ngraph/ngraph.hpp usage
add_subdirectory(core/tests)
add_subdirectory(bindings)

if(CMAKE_COMPILER_IS_GNUCXX)
    ie_add_compiler_flags(-Wall)
    ie_add_compiler_flags(-Wmissing-declarations)
endif()

add_subdirectory(plugins)
add_subdirectory(inference)
add_subdirectory(common)


# if(ENABLE_OV_IR_FRONTEND)
#     if(BUILD_SHARED_LIBS)
#         add_dependencies(ov_runtime_libraries ir_ov_frontend)
#     endif()
#     # use this one once CVS-69781 is fixed
#     # add_dependencies(inference_engine ir_ov_frontend)
# endif()
# if(ENABLE_OV_ONNX_FRONTEND)
#     add_dependencies(inference_engine onnx_ov_frontend)
# endif()
# 
# if(ENABLE_OV_PDPD_FRONTEND)
#     add_dependencies(inference_engine paddlepaddle_ov_frontend)
# endif()
# 
# if(ENABLE_OV_TF_FRONTEND)
#     add_dependencies(inference_engine tensorflow_ov_frontend)
# endif()

ie_register_plugins(MAIN_TARGET ${TARGET_NAME}
                    POSSIBLE_PLUGINS ov_auto_plugin ov_hetero_plugin ov_intel_gpu_plugin ov_intel_gna_plugin MKLDNNPlugin myriadPlugin)

# Export for build tree

export(TARGETS ${TARGET_NAME} NAMESPACE openvino::
       APPEND FILE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")

install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT core
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT core
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT core
        INCLUDES DESTINATION runtime/include
                             runtime/include/ie)

if(BUILD_SHARED_LIBS)
    install(FILES $<TARGET_FILE_DIR:${TARGET_NAME}>/plugins.xml
            DESTINATION ${IE_CPACK_RUNTIME_PATH}
            COMPONENT core)

    install(FILES $<TARGET_FILE_DIR:${TARGET_NAME}>/plugins.xml DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
endif()

list(APPEND PATH_VARS "IE_INCLUDE_DIR" "IE_NGRAPH_DIR"
                      "IE_PARALLEL_CMAKE")

# Install Inference Engine

ie_cpack_add_component(core REQUIRED DEPENDS ${core_components})
ie_cpack_add_component(core_dev REQUIRED core ${core_dev_components})

# Install cmake scripts

install(EXPORT OpenVINOTargets
        FILE OpenVINOTargets.cmake
        NAMESPACE openvino::
        DESTINATION runtime/cmake
        COMPONENT core_dev)

set(IE_PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/inference/include")
set(IE_NGRAPH_DIR "${CMAKE_BINARY_DIR}/src/core")
set(IE_INCLUDE_DIR "${IE_PUBLIC_HEADERS_DIR}/ie")
set(IE_PARALLEL_CMAKE "${OpenVINO_SOURCE_DIR}/src/cmake/ie_parallel.cmake")

configure_package_config_file("${OpenVINO_SOURCE_DIR}/cmake/templates/InferenceEngineConfig.cmake.in"
                              "${CMAKE_BINARY_DIR}/InferenceEngineConfig.cmake"
                               INSTALL_DESTINATION "${CMAKE_INSTALL_PREFIX}"
                               PATH_VARS ${PATH_VARS})

configure_package_config_file("${OpenVINO_SOURCE_DIR}/cmake/templates/OpenVINOConfig.cmake.in"
                              "${CMAKE_BINARY_DIR}/OpenVINOConfig.cmake"
                              INSTALL_DESTINATION "${CMAKE_INSTALL_PREFIX}"
                              PATH_VARS ${PATH_VARS})

set(IE_INCLUDE_DIR "include/ie")
set(IE_NGRAPH_DIR ".")
set(IE_PARALLEL_CMAKE "cmake/ie_parallel.cmake")

configure_package_config_file("${OpenVINO_SOURCE_DIR}/cmake/templates/InferenceEngineConfig.cmake.in"
                              "${CMAKE_BINARY_DIR}/share/InferenceEngineConfig.cmake"
                              INSTALL_DESTINATION cmake
                              PATH_VARS ${PATH_VARS})

configure_package_config_file("${OpenVINO_SOURCE_DIR}/cmake/templates/OpenVINOConfig.cmake.in"
                              "${CMAKE_BINARY_DIR}/share/OpenVINOConfig.cmake"
                              INSTALL_DESTINATION share
                              PATH_VARS ${PATH_VARS})

configure_file("${OpenVINO_SOURCE_DIR}/cmake/templates/InferenceEngineConfig-version.cmake.in"
               "${CMAKE_BINARY_DIR}/InferenceEngineConfig-version.cmake" @ONLY)
configure_file("${OpenVINO_SOURCE_DIR}/cmake/templates/OpenVINOConfig-version.cmake.in"
               "${CMAKE_BINARY_DIR}/OpenVINOConfig-version.cmake" @ONLY)

install(FILES "${CMAKE_BINARY_DIR}/share/InferenceEngineConfig.cmake"
              "${CMAKE_BINARY_DIR}/InferenceEngineConfig-version.cmake"
              "${OpenVINO_SOURCE_DIR}/src/cmake/ie_parallel.cmake"
        DESTINATION runtime/cmake
        COMPONENT core_dev)

install(FILES "${CMAKE_BINARY_DIR}/share/OpenVINOConfig.cmake"
              "${CMAKE_BINARY_DIR}/OpenVINOConfig-version.cmake"
        DESTINATION runtime/cmake
        COMPONENT core_dev)
