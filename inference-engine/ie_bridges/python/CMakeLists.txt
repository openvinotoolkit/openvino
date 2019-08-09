# Defines the CMake commands/policies
cmake_minimum_required (VERSION 3.3)

# Set the project name
project (ie_python_api)
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)

string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} ARCH)
if(ARCH STREQUAL "x86_64" OR ARCH STREQUAL "amd64") # Windows detects Intel's 64-bit CPU as AMD64
    set(ARCH intel64)
elseif(ARCH STREQUAL "i386")
    set(ARCH ia32)
endif()

# in case of independent python api build (out of Inference Engine root Cmake)
if (NOT(IE_MAIN_SOURCE_DIR))
    if("${CMAKE_BUILD_TYPE}" STREQUAL "")
        message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
        set(CMAKE_BUILD_TYPE "Release")
    endif()
    message(STATUS "BUILD_CONFIGURATION: ${CMAKE_BUILD_TYPE}")

    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin/${ARCH})
    if(NOT(WIN32))
        set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE})
    endif()
else()
    if (UNIX OR APPLE)
        # cython generated files requires public visibility. Force visibility required.
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fvisibility=default")
    endif()
endif()

include (UseCython)

if (PYTHONINTERP_FOUND)
    set (PYTHON_VERSION python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
else()
    message(FATAL_ERROR "Python Interpretator was not found!")
endif()

if(WIN32)
    set (PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/python_api/${PYTHON_VERSION}/openvino)
else()
    set (PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python_api/${PYTHON_VERSION}/openvino)
endif()

find_package (InferenceEngine REQUIRED)

set (PYTHON_BRIDGE_SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory (src/openvino/inference_engine)
add_subdirectory (src/openvino/tools/statistics_collector)
