# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_c2)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB HEADERS ${InferenceEngine_C_API_SOURCE_DIR}/include/*.h)

# create library

add_library(${TARGET_NAME} ${HEADERS} ${SOURCES})
add_library(openvino::runtime::c ALIAS ${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PRIVATE openvino)

target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${InferenceEngine_C_API_SOURCE_DIR}/include>)

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

ie_add_vs_version_file(NAME ${TARGET_NAME}
                       FILEDESCRIPTION "Inference Engine C API Core Runtime library")

# export

set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME runtime::c)
export(TARGETS ${TARGET_NAME} NAMESPACE openvino::
       APPEND FILE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")

# install

ie_cpack_add_component(core_c DEPENDS core)
ie_cpack_add_component(core_c_dev DEPENDS core_c)

install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT core_c
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT core_c
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT core_c
        INCLUDES DESTINATION runtime/include/ie)

install(DIRECTORY ${InferenceEngine_C_API_SOURCE_DIR}/include/
        DESTINATION runtime/include/ie
        COMPONENT core_c_dev)
