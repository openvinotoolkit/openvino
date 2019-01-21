# Defines the CMake commands/policies
cmake_minimum_required (VERSION 3.3)

# Set the project name
project (ie_python_api)
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7l")
    set (ARCH armv7l)
elseif ("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
    set (ARCH intel64)
else()
    set (ARCH ia32)
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
add_subdirectory (src/openvino/inference_engine/dnn_builder)