# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT ENABLE_INTEL_GNA)
    return()
endif()

add_subdirectory(legacy)

set(TARGET_NAME "openvino_intel_gna_plugin")

if (ENABLE_INTEL_GNA_DEBUG)
    add_compile_definitions(GNA_DEBUG)
endif()

if(CMAKE_COMPILER_IS_GNUCC)
    ie_add_compiler_flags(-Wno-sign-compare)
    ie_add_compiler_flags(-Wno-parentheses)
    ie_add_compiler_flags(-Wno-comment)
    ie_add_compiler_flags(-Wno-int-in-bool-context)
elseif(OV_COMPILER_IS_CLANG)
    ie_add_compiler_flags(-Wno-int-in-bool-context)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # C4267, 4244 issues from mkl-dnn headers conversion from 'XXX' to 'YYY', possible loss of data
    ie_add_compiler_flags(/wd4267)
    ie_add_compiler_flags(/wd4244)
    # '<': signed/unsigned mismatch
    ie_add_compiler_flags(/wd4018)
    ie_add_compiler_flags(/wd4099)
    ie_add_compiler_flags(/wd4200)
    # C4661: no suitable definition provided for explicit template instantiation request
    ie_add_compiler_flags(/wd4661)
    # C4297: function assumed not to throw an exception but does
    ie_add_compiler_flags(/wd4297)
endif()


file(GLOB_RECURSE SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

file(GLOB_RECURSE HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)

addVersionDefines(src/gna_plugin_entry_points.cpp CI_BUILD_NUMBER)

find_package(libGNA REQUIRED
             PATHS "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
             NO_DEFAULT_PATH)

#
# Shared plugin library
#

ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "GNA"
              SOURCES ${SOURCES} ${HEADERS}
              ADD_CLANG_FORMAT)

# Enable support of CC for the plugin
ie_mark_target_as_cc(${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PRIVATE inference_engine_legacy
        Threads::Threads libGNA)
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_definitions(${TARGET_NAME}
    PRIVATE
        _NO_MKL_
    )

# must be called after all target_link_libraries
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
            INTEGER_LOW_P
            USE_STATIC_IE)

target_link_libraries(${TARGET_NAME}_test_static PUBLIC inference_engine_s inference_engine_transformations libGNA::API)
target_include_directories(${TARGET_NAME}_test_static
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        $<TARGET_PROPERTY:inference_engine_legacy,INTERFACE_INCLUDE_DIRECTORIES>
    PRIVATE
        $<TARGET_PROPERTY:openvino::conditional_compilation,INTERFACE_INCLUDE_DIRECTORIES>)

set_target_properties(${TARGET_NAME} ${TARGET_NAME}_test_static
                      PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# install

file(GLOB_RECURSE gna_libraries "${libGNA_LIBRARIES_BASE_PATH}/*${CMAKE_SHARED_LIBRARY_SUFFIX}*")
set(gna_library_installed OFF)

foreach(gna_lib IN LISTS gna_libraries)
    # install only versioned files like libgna.so.N or libgna.so.N.M
    # and don't install global symlink libgna.so. Exceptions:
    # 1. On Windows we don't have libraries versioning and need to install all .dlls
    # 2. For case of static libraries we also need to install .so since it's required for linkage with final application
    if(gna_lib MATCHES "^.*\.${CMAKE_SHARED_LIBRARY_SUFFIX}(\.[0-9]+)+$" OR
       WIN32 OR NOT BUILD_SHARED_LIBS)
        install(FILES ${gna_lib}
                DESTINATION ${OV_CPACK_RUNTIMEDIR}
                COMPONENT gna)
        set(gna_library_installed ON)
    endif()
endforeach()

if(NOT gna_library_installed)
    message(FATAL_ERROR "Failed to install GNA library (available libraries: ${gna_libraries})")
endif()

# additional install rules for case of static libraries

if(NOT BUILD_SHARED_LIBS)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/libGNAConfig.cmake ${CMAKE_BINARY_DIR} COPYONLY)
    install(FILES "${CMAKE_BINARY_DIR}/libGNAConfig.cmake"
            DESTINATION ${OV_CPACK_IE_CMAKEDIR}
            COMPONENT gna)

    # install .lib file on Windows
    if(WIN32)
        file(GLOB_RECURSE gna_libraries  "${libGNA_LIBRARIES_BASE_PATH}/*${CMAKE_STATIC_LIBRARY_SUFFIX}")
        install(FILES ${gna_libraries}
                DESTINATION ${OV_CPACK_LIBRARYDIR}
                COMPONENT gna)
    endif()
endif()

add_subdirectory(tests)
