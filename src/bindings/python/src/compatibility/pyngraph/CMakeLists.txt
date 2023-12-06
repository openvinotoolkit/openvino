# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 3.13)

project (pyngraph)

if(NOT DEFINED OpenVINO_SOURCE_DIR)
    find_package(OpenVINO REQUIRED)
    find_package(OpenVINODeveloperPackage QUIET
                 PATHS "${InferenceEngineDeveloperPackage_DIR}")
endif()

# Python3_VERSION_MAJOR and Python3_VERSION_MINOR are defined in FindPython3
set(pyversion python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR})

if(OV_GENERATOR_MULTI_CONFIG)
    set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/python/)
else()
    set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python/)
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY})
set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY})
set(CMAKE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY})

# compile options

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # disable warning: This operator was deprecated and will be removed with v0 operation.
    add_compile_options(/wd4996)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wno-deprecated-register -Wno-range-loop-analysis)
elseif(OV_COMPILER_IS_APPLECLANG)
    add_link_options(-stdlib=libc++)
    add_compile_options(-Wno-unused-value -Wno-range-loop-analysis)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # WA for GCC 7.5 "PYBIND11_NOINLINE inline" warning
    add_compile_options(-Wno-error=attributes)
endif()

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # for proper fix need to update pybind to version which does not use PyEval_InitThreads()
    add_compile_options(-Wno-deprecated-declarations -Wno-undef)
endif()

# create target

file(GLOB_RECURSE SOURCES *.cpp)

pybind11_add_module(_${PROJECT_NAME} MODULE NO_EXTRAS ${SOURCES})

target_include_directories(_${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../")

target_link_libraries(_${PROJECT_NAME} PRIVATE openvino::runtime openvino::core::dev)

set_target_properties(_${PROJECT_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# perform copy
add_custom_command(TARGET _${PROJECT_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../ngraph ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/ngraph
)

ov_set_apple_rpath(_${PROJECT_NAME} ${OV_CPACK_PYTHONDIR} ${OV_CPACK_RUNTIMEDIR})

# Install

ov_python_minimal_api(_${PROJECT_NAME})
ov_add_clang_format_target(_${PROJECT_NAME}_clang FOR_TARGETS _${PROJECT_NAME})

ov_cpack_add_component(${OV_CPACK_COMP_PYTHON_OPENVINO}_${pyversion} HIDDEN)

install(TARGETS _${PROJECT_NAME}
        DESTINATION ${OV_CPACK_PYTHONDIR}
        COMPONENT ${OV_CPACK_COMP_PYTHON_OPENVINO}_${pyversion}
        ${OV_CPACK_COMP_PYTHON_OPENVINO_EXCLUDE_ALL})

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../ngraph
        DESTINATION ${OV_CPACK_PYTHONDIR}
        COMPONENT ${OV_CPACK_COMP_PYTHON_OPENVINO}_${pyversion}
        ${OV_CPACK_COMP_PYTHON_OPENVINO_EXCLUDE_ALL}
        USE_SOURCE_PERMISSIONS)

install(DIRECTORY ${OpenVINOPython_SOURCE_DIR}/tests_compatibility
        DESTINATION tests/${PROJECT_NAME}
        COMPONENT tests
        EXCLUDE_FROM_ALL)
