# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "inference_engine")

if(THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO")
    find_package(TBBBIND_2_4 QUIET)
    if (TBBBIND_2_4_FOUND)
        message(STATUS "Static tbbbind_2_4 package was found")
    endif()
endif()

file (GLOB LIBRARY_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/threading/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp_interfaces/interface/*.cpp
      )

# TODO: WA for OneHot pass usage in reshape
set(LEGACY_SRC_ROOT "${IE_MAIN_SOURCE_DIR}/src/legacy_api/src/")
set(LEGACY_LIBRARY_SHARED_SRCS
    "${LEGACY_SRC_ROOT}/transformations/convert_opset1_to_legacy/convert_one_hot_to_one_hot_ie.cpp"
    "${LEGACY_SRC_ROOT}/transformations/convert_opset1_to_legacy/convert_nms_5_to_legacy.cpp"
    "${LEGACY_SRC_ROOT}/ngraph_ops/nms_ie.cpp"
    "${LEGACY_SRC_ROOT}/ngraph_ops/onehot_ie.cpp")

set_source_files_properties(${LEGACY_LIBRARY_SHARED_SRCS} PROPERTIES
    COMPILE_DEFINITIONS "USE_STATIC_IE")

set(IE_STATIC_DEPENDENT_FILES ${CMAKE_CURRENT_SOURCE_DIR}/file_utils.cpp)
list(REMOVE_ITEM LIBRARY_SRC ${IE_STATIC_DEPENDENT_FILES})

file (GLOB LIBRARY_HEADERS
       ${CMAKE_CURRENT_SOURCE_DIR}/*.h
       ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
      )

if (LINUX)
    file (GLOB LIBRARY_SRC
          ${LIBRARY_SRC}
          ${CMAKE_CURRENT_SOURCE_DIR}/os/lin/*.cpp)
    file (GLOB LIBRARY_HEADERS
         ${LIBRARY_HEADERS}
         ${CMAKE_CURRENT_SOURCE_DIR}/os/lin/*.hpp)
elseif (UNIX)
    list (APPEND LIBRARY_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/os/lin/lin_shared_object_loader.cpp)
endif()

if (WIN32)
    file (GLOB LIBRARY_SRC
         ${LIBRARY_SRC}
         ${CMAKE_CURRENT_SOURCE_DIR}/os/win/*.cpp)
    file (GLOB LIBRARY_HEADERS
         ${LIBRARY_HEADERS}
         ${CMAKE_CURRENT_SOURCE_DIR}/os/win/*.hpp)
endif()

if(ENABLE_SSE42)
    file(GLOB SSE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/*.cpp)
    file(GLOB SSE_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/*.hpp)

    list(APPEND LIBRARY_HEADERS ${SSE_HEADERS})
    list(APPEND LIBRARY_SRC ${SSE_SRC})

    ie_sse42_optimization_flags(sse4_2_flags)
    set_source_files_properties(${SSE_SRC} PROPERTIES COMPILE_OPTIONS "${sse4_2_flags}")
    add_definitions(-DHAVE_SSE=1)

    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.16")
        set_source_files_properties(${SSE_SRC} PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
    endif()
endif()

addVersionDefines(ie_version.cpp CI_BUILD_NUMBER)

set (PUBLIC_HEADERS_DIR "${IE_MAIN_SOURCE_DIR}/include")

file (GLOB_RECURSE PUBLIC_HEADERS
       ${PUBLIC_HEADERS_DIR}/*.hpp
       ${PUBLIC_HEADERS_DIR}/*.h
      )

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS} ${PUBLIC_HEADERS})

# Plugin API library

add_library(${TARGET_NAME}_plugin_api INTERFACE)

target_include_directories(${TARGET_NAME}_plugin_api INTERFACE
    "${IE_MAIN_SOURCE_DIR}/src/plugin_api"
    $<TARGET_PROPERTY:${TARGET_NAME}_preproc,INTERFACE_INCLUDE_DIRECTORIES>
    ${PUBLIC_HEADERS_DIR})

target_link_libraries(${TARGET_NAME}_plugin_api INTERFACE pugixml::static openvino::itt)

set_ie_threading_interface_for(${TARGET_NAME}_plugin_api)

file(GLOB_RECURSE plugin_api_src "${IE_MAIN_SOURCE_DIR}/src/plugin_api/*.hpp"
                                 "${IE_MAIN_SOURCE_DIR}/src/plugin_api/*.h")

add_cpplint_target(${TARGET_NAME}_plugin_api_cpplint FOR_SOURCES ${plugin_api_src})

# Create object library

add_library(${TARGET_NAME}_obj OBJECT
            ${LIBRARY_SRC}
            ${LIBRARY_HEADERS}
            ${PUBLIC_HEADERS})

ie_faster_build(${TARGET_NAME}_obj
    UNITY PCH PRIVATE "precomp.hpp"
)

target_compile_definitions(${TARGET_NAME}_obj PRIVATE IMPLEMENT_INFERENCE_ENGINE_API
                                                      $<TARGET_PROPERTY:ngraph::ngraph,INTERFACE_COMPILE_DEFINITIONS>)

target_include_directories(${TARGET_NAME}_obj SYSTEM PRIVATE $<TARGET_PROPERTY:ngraph::ngraph,INTERFACE_INCLUDE_DIRECTORIES>
                                                             $<TARGET_PROPERTY:pugixml::static,INTERFACE_INCLUDE_DIRECTORIES>
                                                             $<TARGET_PROPERTY:ngraph::frontend_manager,INTERFACE_INCLUDE_DIRECTORIES>
                                                             $<TARGET_PROPERTY:xbyak,INTERFACE_INCLUDE_DIRECTORIES>)

target_include_directories(${TARGET_NAME}_obj PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
                                                      "${IE_MAIN_SOURCE_DIR}/src/readers/ir_reader" # for ie_ir_version.hpp
                                                      $<TARGET_PROPERTY:${TARGET_NAME}_legacy,INTERFACE_INCLUDE_DIRECTORIES>
                                                      $<TARGET_PROPERTY:${TARGET_NAME}_transformations,INTERFACE_INCLUDE_DIRECTORIES>
                                                      $<TARGET_PROPERTY:${TARGET_NAME}_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(${TARGET_NAME}_obj PRIVATE ${TARGET_NAME}_reader_api openvino::itt)

set_ie_threading_interface_for(${TARGET_NAME}_obj)
if (TBBBIND_2_4_FOUND)
    target_compile_definitions(${TARGET_NAME}_obj PRIVATE -DTBBBIND_2_4_AVAILABLE)
    target_link_libraries(${TARGET_NAME}_obj PRIVATE ${TBBBIND_2_4_IMPORTED_TARGETS})
endif()

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME}_obj)

# Create shared library file from object library

add_library(${TARGET_NAME} SHARED
            ${IE_STATIC_DEPENDENT_FILES}
            ${LEGACY_LIBRARY_SHARED_SRCS}
            ${vs_version_file}
            $<TARGET_OBJECTS:${TARGET_NAME}_obj>)

ie_add_vs_version_file(NAME ${TARGET_NAME}
                       FILEDESCRIPTION "Inference Engine Core Runtime library")

set_ie_threading_interface_for(${TARGET_NAME})
if (TBBBIND_2_4_FOUND)
    target_compile_definitions(${TARGET_NAME} PRIVATE -DTBBBIND_2_4_AVAILABLE)
    target_link_libraries(${TARGET_NAME} PRIVATE ${TBBBIND_2_4_IMPORTED_TARGETS})
endif()

target_link_libraries(${TARGET_NAME} PRIVATE pugixml::static openvino::itt ${CMAKE_DL_LIBS} Threads::Threads
                                             ngraph ngraph::frontend_manager inference_engine_transformations)

target_include_directories(${TARGET_NAME} INTERFACE
            $<BUILD_INTERFACE:${PUBLIC_HEADERS_DIR}>
            $<INSTALL_INTERFACE:${IE_CPACK_IE_DIR}/include>
    PRIVATE $<TARGET_PROPERTY:${TARGET_NAME}_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:${TARGET_NAME}_legacy,INTERFACE_INCLUDE_DIRECTORIES>)

if(WIN32)
    set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})
endif()

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_ENGINE_API)

ie_register_plugins(MAIN_TARGET ${TARGET_NAME}
                    POSSIBLE_PLUGINS MultiDevicePlugin HeteroPlugin clDNNPlugin GNAPlugin MKLDNNPlugin myriadPlugin)

ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

# Static library used for unit tests which are always built

add_library(${TARGET_NAME}_s STATIC EXCLUDE_FROM_ALL
            $<TARGET_OBJECTS:${TARGET_NAME}_legacy_obj>
            $<TARGET_OBJECTS:${TARGET_NAME}_obj>
            ${IE_STATIC_DEPENDENT_FILES})

set_ie_threading_interface_for(${TARGET_NAME}_s)
if (TBBBIND_2_4_FOUND)
    target_compile_definitions(${TARGET_NAME}_s PRIVATE -DTBBBIND_2_4_AVAILABLE)
    target_link_libraries(${TARGET_NAME}_s PRIVATE ${TBBBIND_2_4_IMPORTED_TARGETS})
endif()

target_include_directories(${TARGET_NAME}_s PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
    $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
    "${IE_MAIN_SOURCE_DIR}/src/legacy_api/src")

if(WIN32)
    set_target_properties(${TARGET_NAME}_s PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME}_s)
endif()

target_link_libraries(${TARGET_NAME}_s PRIVATE openvino::itt ${CMAKE_DL_LIBS} ngraph ngraph::frontend_manager
                                               inference_engine_transformations pugixml::static)

target_compile_definitions(${TARGET_NAME}_s PUBLIC USE_STATIC_IE)

set_target_properties(${TARGET_NAME}_s PROPERTIES EXCLUDE_FROM_ALL ON)

# LTO

set_target_properties(${TARGET_NAME} ${TARGET_NAME}_obj ${TARGET_NAME}_s
                      PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# Export for build tree

export(TARGETS ${TARGET_NAME} NAMESPACE IE::
        APPEND FILE "${CMAKE_BINARY_DIR}/InferenceEngineTargets.cmake")

# Export for developer package

openvino_developer_export_targets(COMPONENT inference_engine TARGETS ${TARGET_NAME}_plugin_api)

# install TBB

list(APPEND core_components ngraph)
list(APPEND PATH_VARS "IE_INCLUDE_DIR" "IE_NGRAPH_DIR"
                      "IE_PARALLEL_CMAKE")

if((THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO") AND TBBROOT MATCHES ${TEMP})
    ie_cpack_add_component(tbb REQUIRED)
    list(APPEND core_components tbb)

    install(DIRECTORY "${TBB}/include"
            DESTINATION ${IE_CPACK_IE_DIR}/external/tbb
            COMPONENT tbb)
    install(DIRECTORY "${TBB}/lib"
            DESTINATION ${IE_CPACK_IE_DIR}/external/tbb
            COMPONENT tbb)
    if(EXISTS "${TBB}/bin")
        install(DIRECTORY "${TBB}/bin"
                DESTINATION ${IE_CPACK_IE_DIR}/external/tbb
                COMPONENT tbb)
    endif()
    install(FILES "${TBB}/LICENSE"
            DESTINATION ${IE_CPACK_IE_DIR}/external/tbb
            COMPONENT tbb)

    set(IE_TBB_DIR_INSTALL "external/tbb/cmake")
    set(IE_TBB_DIR "${TBB_DIR}")
    list(APPEND PATH_VARS "IE_TBB_DIR")

    install(FILES "${TBB}/cmake/TBBConfig.cmake"
                  "${TBB}/cmake/TBBConfigVersion.cmake"
            DESTINATION ${IE_CPACK_IE_DIR}/${IE_TBB_DIR_INSTALL}
            COMPONENT tbb)
endif()

# Install Inference Engine

ie_cpack_add_component(core REQUIRED DEPENDS ${core_components})
ie_cpack_add_component(core_dev REQUIRED core ngraph_dev)

install(DIRECTORY "${PUBLIC_HEADERS_DIR}" DESTINATION ${IE_CPACK_IE_DIR}
        COMPONENT core_dev)

install(TARGETS ${TARGET_NAME} EXPORT InferenceEngineTargets
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT core
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT core
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT core)

install(FILES $<TARGET_FILE_DIR:${TARGET_NAME}>/plugins.xml
        DESTINATION ${IE_CPACK_RUNTIME_PATH}
        COMPONENT core)

# for InferenceEngineUnitTest
if(WIN32)
    install(FILES $<TARGET_FILE_DIR:${TARGET_NAME}>/plugins.xml
            DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
else()
    install(FILES $<TARGET_FILE_DIR:${TARGET_NAME}>/plugins.xml
            DESTINATION tests/lib COMPONENT tests EXCLUDE_FROM_ALL)
endif()

# Install cmake scripts

install(EXPORT InferenceEngineTargets
        FILE InferenceEngineTargets.cmake
        NAMESPACE IE::
        DESTINATION ${IE_CPACK_IE_DIR}/share
        COMPONENT core_dev)

set(IE_NGRAPH_DIR "${CMAKE_BINARY_DIR}/ngraph")
set(IE_INCLUDE_DIR "${PUBLIC_HEADERS_DIR}")
set(IE_PARALLEL_CMAKE "${InferenceEngine_SOURCE_DIR}/cmake/ie_parallel.cmake")

configure_package_config_file("${OpenVINO_SOURCE_DIR}/cmake/templates/InferenceEngineConfig.cmake.in"
                              "${CMAKE_BINARY_DIR}/InferenceEngineConfig.cmake"
                               INSTALL_DESTINATION "${CMAKE_INSTALL_PREFIX}"
                               PATH_VARS ${PATH_VARS})

set(IE_INCLUDE_DIR "include")
set(IE_NGRAPH_DIR "../ngraph/cmake")
set(IE_TBB_DIR "${IE_TBB_DIR_INSTALL}")
set(IE_PARALLEL_CMAKE "share/ie_parallel.cmake")

configure_package_config_file("${OpenVINO_SOURCE_DIR}/cmake/templates/InferenceEngineConfig.cmake.in"
                              "${CMAKE_BINARY_DIR}/share/InferenceEngineConfig.cmake"
                              INSTALL_DESTINATION share
                              PATH_VARS ${PATH_VARS})

configure_file("${OpenVINO_SOURCE_DIR}/cmake/templates/InferenceEngineConfig-version.cmake.in"
               "${CMAKE_BINARY_DIR}/InferenceEngineConfig-version.cmake"
               @ONLY)

install(FILES "${CMAKE_BINARY_DIR}/share/InferenceEngineConfig.cmake"
              "${CMAKE_BINARY_DIR}/InferenceEngineConfig-version.cmake"
              "${InferenceEngine_SOURCE_DIR}/cmake/ie_parallel.cmake"
        DESTINATION ${IE_CPACK_IE_DIR}/share
        COMPONENT core_dev)
