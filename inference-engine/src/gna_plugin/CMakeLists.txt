# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(TARGET_NAME "GNAPlugin")

file(GLOB_RECURSE SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

file(GLOB_RECURSE HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

addVersionDefines(gna_plugin_entry_points.cpp CI_BUILD_NUMBER)

find_package(libGNA REQUIRED
             PATHS "${IE_MAIN_SOURCE_DIR}/cmake"
             NO_DEFAULT_PATH)

if(GNA_LIBRARY_VERSION STREQUAL "GNA2")
    set(GNA_LIBRARY_VERSION_NUMBER 2)
else()
    set(GNA_LIBRARY_VERSION_NUMBER 1)
endif()

#
# Shared plugin library
# 

ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "GNA"
              SOURCES ${SOURCES} ${HEADERS})

# saving rpath to GNA shared library be used by CI
log_rpath_from_dir(GNA ${libGNA_LIBRARIES_BASE_PATH})

target_link_libraries(${TARGET_NAME} PRIVATE inference_engine inference_engine_legacy inference_engine_transformations
        Threads::Threads libGNA)
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(${TARGET_NAME}
    PRIVATE
        _NO_MKL_
    PUBLIC
        GNA_LIB_VER=${GNA_LIBRARY_VERSION_NUMBER})

ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

#
# Static version for tests
#

add_library(${TARGET_NAME}_test_static STATIC EXCLUDE_FROM_ALL ${SOURCES} ${HEADERS})

target_compile_definitions(${TARGET_NAME}_test_static
        PRIVATE
            _NO_MKL_
            IMPLEMENT_INFERENCE_ENGINE_PLUGIN
        PUBLIC
            GNA_LIB_VER=${GNA_LIBRARY_VERSION_NUMBER}
            INTEGER_LOW_P
            USE_STATIC_IE)

target_link_libraries(${TARGET_NAME}_test_static PUBLIC inference_engine_preproc_s inference_engine_transformations libGNA::API)
target_include_directories(${TARGET_NAME}_test_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:inference_engine_legacy,INTERFACE_INCLUDE_DIRECTORIES>)
set_target_properties(${TARGET_NAME}_test_static PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME}_test_static)

set_target_properties(${TARGET_NAME} ${TARGET_NAME}_test_static
                      PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# install

file(GLOB_RECURSE source_list "${libGNA_LIBRARIES_BASE_PATH}/*${CMAKE_SHARED_LIBRARY_SUFFIX}*")
install(FILES ${source_list} 
        DESTINATION ${IE_CPACK_IE_DIR}/external/gna/lib
        COMPONENT gna)
