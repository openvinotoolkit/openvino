# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "mock_engine")

file (GLOB LIBRARY_SRC
       ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
      )

file (GLOB LIBRARY_HEADERS
       ${CMAKE_CURRENT_SOURCE_DIR}/*.h
       ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
      )

if(UNIX)
    list(REMOVE_ITEM LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/dllmain.cpp)
endif()

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})

# Create module library file from sources.
add_library(${TARGET_NAME} MODULE
            ${LIBRARY_SRC}
            ${LIBRARY_HEADERS})

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(TARGET IE::inference_engine_plugin_api)
    target_link_libraries(${TARGET_NAME} PRIVATE IE::inference_engine_plugin_api)
else()
    target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime::dev)
endif()

target_link_libraries(${TARGET_NAME} PRIVATE inference_engine)

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_ENGINE_PLUGIN)

install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
