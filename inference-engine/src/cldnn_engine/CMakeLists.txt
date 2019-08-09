# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "clDNNPlugin")

file(GLOB MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

addVersionDefines(cldnn_engine.cpp CI_BUILD_NUMBER CLDNN_VERSION)

ie_add_plugin(NAME ${TARGET_NAME}
	          DEVICE_NAME "GPU"
              SOURCES ${MAIN_SRC} ${LIBRARY_HEADERS}
              VERSION_DEFINES_FOR cldnn_engine.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE ${INTEL_ITT_LIBS} inference_engine clDNN_shlib pugixml)

set (CLDNN_TOP_FOLDER ${IE_MAIN_SOURCE_DIR}/thirdparty/clDNN)
target_include_directories(${TARGET_NAME} PRIVATE
        ${CLDNN_TOP_FOLDER}/api
        ${CLDNN_TOP_FOLDER}/include
        ${IE_MAIN_SOURCE_DIR}/src/inference_engine
        ${IE_MAIN_SOURCE_DIR}/thirdparty/pugixml/src)

# copy default global xml file describing the custom kernels and the *.cl files

add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/cldnn_global_custom_kernels
                                                 $<TARGET_FILE_DIR:${TARGET_NAME}>/cldnn_global_custom_kernels)
