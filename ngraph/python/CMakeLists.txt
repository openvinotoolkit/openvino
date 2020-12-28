# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

cmake_minimum_required (VERSION 3.13)

project (pyngraph)

if(NOT DEFINED OpenVINO_MAIN_SOURCE_DIR)
    find_package(InferenceEngineDeveloperPackage)
    find_package(ngraph REQUIRED)    
endif()

if(ngraph_FOUND)
    message("ngraph version = {${ngraph_VERSION}}")
endif()

include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY "https://github.com/pybind/pybind11.git"
    GIT_TAG        "v2.5.0"
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)

if(PYTHON_FOUND)
    set(PYTHON_VERSION python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
    message("Python version={${Python_VERSION}}")
else()
    message(FATAL_ERROR "Python was not found!")
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    add_compile_options(-fPIC)
endif()

if (APPLE)
    add_link_options(-stdlib=libc++)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    #disable warning: This operator was deprecated and will be removed with v0 operation.
    add_compile_options(/wd4996) 
endif()

file(GLOB_RECURSE SOURCES src/pyngraph/*.cpp)
pybind11_add_module(_${PROJECT_NAME} MODULE ${SOURCES})
target_include_directories(_${PROJECT_NAME} PRIVATE src)
target_link_libraries(_${PROJECT_NAME} PRIVATE ngraph::ngraph ngraph::onnx_importer)

if(OpenVINO_MAIN_SOURCE_DIR OR InferenceEngineDeveloperPackage_FOUND)
    ie_cpack_add_component(pyngraph_${PYTHON_VERSION})

    install(TARGETS _${PROJECT_NAME} 
            DESTINATION python/${PYTHON_VERSION}
            COMPONENT pyngraph_${PYTHON_VERSION})

    install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/src/ngraph
            DESTINATION python/${PYTHON_VERSION}
            COMPONENT pyngraph_${PYTHON_VERSION}
            USE_SOURCE_PERMISSIONS)

    ie_cpack(pyngraph_${PYTHON_VERSION})
endif()
