# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "clDNNPlugin")

if(CMAKE_COMPILER_IS_GNUCC)
    ie_add_compiler_flags(-Wno-all)
    ie_add_compiler_flags(-msse4.1 -msse4.2)
    if (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
        ie_add_compiler_flags(-Wno-error=terminate)
    endif()
endif()

if(GPU_DEBUG_CONFIG)
  add_definitions(-DGPU_DEBUG_CONFIG=1)
endif()

file(GLOB_RECURSE MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

addVersionDefines(cldnn_engine.cpp CI_BUILD_NUMBER CLDNN_VERSION)

ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "GPU"
              SOURCES ${MAIN_SRC} ${LIBRARY_HEADERS}
              VERSION_DEFINES_FOR cldnn_engine.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE clDNN_lib pugixml
                                             inference_engine
                                             inference_engine_transformations
                                             inference_engine_lp_transformations
                                             ${NGRAPH_LIBRARIES})

set(CLDNN_TOP_FOLDER "${IE_MAIN_SOURCE_DIR}/thirdparty/clDNN")
target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        $<TARGET_PROPERTY:inference_engine_transformations,INTERFACE_INCLUDE_DIRECTORIES>
        ${CLDNN_TOP_FOLDER}/api)

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# Failed because of OpenCL
# ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

# install

install(FILES "${clDNN_SOURCE_DIR}/kernel_selector/core/cache/cache.json"
        DESTINATION ${IE_CPACK_RUNTIME_PATH}
        COMPONENT gpu)
