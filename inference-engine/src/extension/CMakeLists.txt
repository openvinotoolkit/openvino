# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "ie_cpu_extension")

if (NOT(IE_MAIN_SOURCE_DIR))
    list (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
    include(feature_defs)

    # to use C++11 if samples are built outside of IE repo
    set (CMAKE_CXX_STANDARD 11)
    set (CMAKE_CXX_STANDARD_REQUIRED ON)
    if (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
        set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
    endif()
endif()

# treating warnings as errors
if (WIN32)
    if (TREAT_WARNING_AS_ERROR)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX")
    endif ()
    if (${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251 /wd4275 /wd4267") #disable some warnings
    endif()
else()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif()

if (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

file(GLOB_RECURSE SRC *.cpp)
file(GLOB_RECURSE HDR *.hpp)

add_definitions(-DIMPLEMENT_INFERENCE_ENGINE_API)

include_directories (PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${InferenceEngine_INCLUDE_DIRS}
)

add_library(${TARGET_NAME} SHARED ${SRC} ${HDR})
set_ie_threading_interface_for(${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME "cpu_extension")

target_link_libraries(${TARGET_NAME} PRIVATE ${InferenceEngine_LIBRARIES})

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})

set_target_cpu_flags(${TARGET_NAME})

if (IE_MAIN_SOURCE_DIR)
    export(TARGETS ${TARGET_NAME} NAMESPACE IE:: APPEND FILE "${CMAKE_BINARY_DIR}/targets.cmake")

    # developer package
    export(TARGETS ${TARGET_NAME} NAMESPACE IE:: APPEND FILE "${CMAKE_BINARY_DIR}/targets_developer.cmake")
endif()

