# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME inference_engine_c_api)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB HEADERS ${InferenceEngine_C_API_SOURCE_DIR}/include/*.h)

# create library

add_library(${TARGET_NAME} SHARED ${HEADERS} ${SOURCES})

target_link_libraries(${TARGET_NAME} PRIVATE inference_engine)

target_include_directories(${TARGET_NAME} PUBLIC
    $<INSTALL_INTERFACE:${IE_CPACK_IE_DIR}/include>
    $<BUILD_INTERFACE:${InferenceEngine_C_API_SOURCE_DIR}/include>)

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

ie_add_vs_version_file(NAME ${TARGET_NAME}
                       FILEDESCRIPTION "Inference Engine C API Core Runtime library")

# export

export(TARGETS ${TARGET_NAME} NAMESPACE IE::
       APPEND FILE "${CMAKE_BINARY_DIR}/InferenceEngineTargets.cmake")

# install

ie_cpack_add_component(core_c DEPENDS core)
ie_cpack_add_component(core_c_dev DEPENDS core_c)

install(TARGETS ${TARGET_NAME} EXPORT InferenceEngineTargets
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT core_c
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT core_c
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT core_c)

install(DIRECTORY ${InferenceEngine_C_API_SOURCE_DIR}/include/
        DESTINATION ${IE_CPACK_IE_DIR}/include
        COMPONENT core_c_dev)
