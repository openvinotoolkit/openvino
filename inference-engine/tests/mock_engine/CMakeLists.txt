# Copyright (C) 2018 Intel Corporation
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
add_definitions(-DIMPLEMENT_INFERENCE_ENGINE_API)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})

# Properties->C/C++->General->Additional Include Directories
include_directories (${IE_MAIN_SOURCE_DIR}/include
        ${IE_MAIN_SOURCE_DIR}/src/inference_engine
        ${IE_MAIN_SOURCE_DIR}/include
        ${IE_MAIN_SOURCE_DIR}/thirdparty/pugixml/src)

# Create library file from sources.
add_library(${TARGET_NAME} SHARED
            ${LIBRARY_SRC}
            ${LIBRARY_HEADERS})


set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ${TARGET_NAME} PROPERTY COMPILE_PDB_NAME ${TARGET_NAME})
