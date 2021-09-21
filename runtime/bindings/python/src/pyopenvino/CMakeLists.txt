# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

project (pyopenvino)

if(DEFINED IE_MAIN_SOURCE_DIR)
    set(InferenceEngine_LIBRARIES inference_engine)
    set(NGRAPH_LIBRARIES ngraph)
    set(FRONTEND_MANAGER_LIBRARIES frontend_manager)
else()
    find_package(InferenceEngineDeveloperPackage REQUIRED)
    find_package(ngraph QUIET)
    find_package(frontend_manager QUIET)
    set(InferenceEngine_LIBRARIES IE::inference_engine)
    set(NGRAPH_LIBRARIES ngraph)
    set(FRONTEND_MANAGER_LIBRARIES frontend_manager)
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY})
set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY})
set(CMAKE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY})

# create target

find_package(Python COMPONENTS Interpreter Development REQUIRED)

if(PYTHON_FOUND)
    set(PYTHON_VERSION python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
    message("Python version={${Python_VERSION}}")
else()
    message(FATAL_ERROR "Python was not found!")
endif()

# compile options

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # disable warning: This operator was deprecated and will be removed with v0 operation.
    add_compile_options(/wd4996)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wno-deprecated-register -Wno-range-loop-analysis)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
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

file(GLOB_RECURSE SOURCES *.cpp)

pybind11_add_module(${PROJECT_NAME} MODULE ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(${PROJECT_NAME} PRIVATE ${InferenceEngine_LIBRARIES} ${NGRAPH_LIBRARIES} ${FRONTEND_MANAGER_LIBRARIES})

# perform copy
if(OpenVINO_SOURCE_DIR)
    add_custom_command(TARGET ${PROJECT_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../openvino ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${PROJECT_NAME}.*.so ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
            )
endif()

if(OpenVINO_SOURCE_DIR OR InferenceEngineDeveloperPackage_FOUND)
    if(COMMAND ie_python_minimal_api)
        ie_python_minimal_api(${PROJECT_NAME})
    endif()

    add_clang_format_target(${PROJECT_NAME}_clang FOR_TARGETS ${PROJECT_NAME})

    ie_cpack_add_component(pyopenvino_${PYTHON_VERSION})

    install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../openvino
            DESTINATION python/${PYTHON_VERSION}
            COMPONENT pyopenvino_${PYTHON_VERSION}
            USE_SOURCE_PERMISSIONS)

    install(TARGETS ${PROJECT_NAME}
            DESTINATION python/${PYTHON_VERSION}/openvino
            COMPONENT pyopenvino_${PYTHON_VERSION})

    ie_cpack(pyopenvino_${PYTHON_VERSION})
endif()
