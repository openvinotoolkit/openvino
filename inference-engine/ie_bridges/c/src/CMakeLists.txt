# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME inference_engine_c_api)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB HEADERS ${InferenceEngine_C_API_SOURCE_DIR}/include/*.h)

# create library

add_library(${TARGET_NAME} SHARED ${HEADERS} ${SOURCES})

target_link_libraries(${TARGET_NAME} PRIVATE inference_engine)

target_include_directories(${TARGET_NAME} PUBLIC "${InferenceEngine_C_API_SOURCE_DIR}/include")

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})

# export

export(TARGETS ${TARGET_NAME} NAMESPACE IE:: APPEND FILE "${CMAKE_BINARY_DIR}/targets.cmake")

# install

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION ${IE_CPACK_LIBRARY_PATH}
        ARCHIVE DESTINATION ${IE_CPACK_LIBRARY_PATH}
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH}
        COMPONENT core)

install(DIRECTORY ${InferenceEngine_C_API_SOURCE_DIR}/include/
        DESTINATION ${IE_CPACK_IE_DIR}/include/
        COMPONENT core)
