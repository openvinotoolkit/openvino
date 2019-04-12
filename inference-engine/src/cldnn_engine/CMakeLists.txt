# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "clDNNPlugin")

file (GLOB MAIN_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

file (GLOB LIBRARY_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
        )

addVersionDefines(cldnn_engine.cpp CI_BUILD_NUMBER CLDNN_VERSION)

add_definitions(-DIMPLEMENT_INFERENCE_ENGINE_PLUGIN)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})


#TODO: clDNN for non windows...
if (APPLE)
elseif  (UNIX)
    set(CLDNN_BUILD_PLATFORM Centos7/)
    set(CLDNN_PLATFORM Linux64/)
    set(CLDNN_LIB_FOLDER bin)
    set(CLDNN_LIB_NAME clDNN64)
else ()
    #32 bits platform
    if (NOT "${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
        set(CLDNN_PLATFORM Windows32/)
        set(CLDNN_LIB_NAME clDNN32)
    else()
        set(CLDNN_PLATFORM Windows64/)
        set(CLDNN_LIB_NAME clDNN64)
    endif()
    set(CLDNN_LIB_FOLDER lib)
endif ()

set (CLDNN_TOP_FOLDER ${IE_MAIN_SOURCE_DIR}/thirdparty/clDNN)

# Properties->C/C++->General->Additional Include Directories
include_directories (
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${IE_MAIN_SOURCE_DIR}/include
        ${CLDNN_TOP_FOLDER}/api
        ${CLDNN_TOP_FOLDER}/include
        #${OCL_DIST}/include
        ${IE_MAIN_SOURCE_DIR}/src/inference_engine
        ${IE_MAIN_SOURCE_DIR}/thirdparty/pugixml/src)

# clDnn build configuration, change that if you'd like to debug
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug" )
    set (Configuration ${CMAKE_BUILD_TYPE})
else()
    set (Configuration Release)
endif()

set(CLDNN_LIBRARY clDNN_shlib)

# Create library file from sources.
add_library(${TARGET_NAME} SHARED
    ${MAIN_SRC}
    ${LIBRARY_HEADERS})
target_link_libraries(${TARGET_NAME} ${INTEL_ITT_LIBS} inference_engine ${CLDNN_LIBRARY})

set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})

#copy default global xml file describing the custom kernels and the *.cl files
add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/cldnn_global_custom_kernels $<TARGET_FILE_DIR:${TARGET_NAME}>/cldnn_global_custom_kernels)

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})
