# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0054 NEW)

set(TARGET_NAME InferenceEngineUnitTests)

#rpath enabled for unit tests only
SET (CMAKE_SKIP_RPATH FALSE)

if (UNIX AND NOT APPLE)
    set(ARCH_SPECIFIC_FOLDER_TBB /gcc4.4)
    set(ARCH_SPECIFIC_FOLDER intel64_lin)
    if ("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
        set(ARCH_SPECIFIC_FOLDER intel64_lin)
    else ("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
        set(ARCH_SPECIFIC_FOLDER intel32_lin)
    endif ("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
else ()
    set(ARCH_SPECIFIC_FOLDER_TBB /vc_mt)
    if ("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
        set(ARCH_SPECIFIC_FOLDER intel64_win)
    else ("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
        set(ARCH_SPECIFIC_FOLDER ia32_win)
    endif ("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
endif ()

file(GLOB
        TEST_SRC
        graph_tools/*.cpp
        inference_engine_tests/*.cpp
        inference_engine_tests/cpp_interfaces/*.cpp
        mem_solver/*.cpp
        cnn_network/*.cpp
        shape_infer/*.cpp
        shape_infer/built-in/*.cpp
        topology_verification_tests/*.cpp
        stress_tests/*.cpp
        )

if (ENABLE_MKL_DNN)
    if (THREADING STREQUAL "OMP")
        find_package(OpenMP)
        if (OPENMP_FOUND)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        endif ()
    endif ()

    file(GLOB
            MKLDNN_TESTS
            engines/mkldnn/*.cpp
            engines/mkldnn/graph/layers/extensions/*.cpp
            engines/mkldnn/graph/layers/internal/*.cpp
            engines/mkldnn/graph/structure/*.cpp
            engines/mkldnn/graph/*.cpp)
    file(GLOB
            MKLDNN_TESTS_INCLUDE engines/mkldnn/graph/*.hpp)

    if (USE_BOOST_RE)
        debug_message(STATUS "Adding boost dependency")
        if (VERBOSE_BUILD)
            set(Boost_DEBUG on)
        endif ()
        find_package(Boost REQUIRED COMPONENTS regex)
        link_directories(${Boost_LIBRARY_DIRS})
        include_directories(${Boost_INCLUDE_DIRS})
    endif ()

    include_directories(
            ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/include
            engines/mkldnn/graph)

    source_group("mkldnn" FILES ${MKLDNN_TESTS} ${MKLDNN_TESTS_INCLUDE})
endif ()

file(GLOB
        TEST_INCLUDE
        ${IE_MAIN_SOURCE_DIR}/tests/helpers/*.hpp
        shape_infer/*.hpp)

source_group("src" FILES ${TEST_SRC})
source_group("include" FILES ${TEST_INCLUDE})

include_directories(
        ${IE_MAIN_SOURCE_DIR}/include
        ${IE_MAIN_SOURCE_DIR}/src/inference_engine
        ${IE_MAIN_SOURCE_DIR}/src/mkldnn_plugin
        ${IE_MAIN_SOURCE_DIR}/src/extension
        ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/gflags/include
        mocks)
add_executable(${TARGET_NAME} ${TEST_SRC} ${TEST_INCLUDE} ${MKLDNN_TESTS} ${MKLDNN_TESTS_INCLUDE} ${DLAI_TESTS})
set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE"
COMPILE_PDB_NAME ${TARGET_NAME})

## Mock macros doesn't use "override" specificator
target_compile_options(${TARGET_NAME} PRIVATE $<$<CXX_COMPILER_ID:Clang>: -Wno-inconsistent-missing-override >)

if (FALSE)
    add_custom_command(
            TARGET ${TARGET_NAME}
            POST_BUILD COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/data/*.xml ${LIBRARY_OUTPUT_DIRECTORY}
            POST_BUILD COMMAND cp -R ${IE_MAIN_SOURCE_DIR}/temp/models ${LIBRARY_OUTPUT_DIRECTORY}/models
    )
endif ()

if (MSVC)
    set(PUGI pugixml_mt)
else ()
    set(PUGI pugixml)
endif ()

add_definitions(-DMODELS_PATH="${MODELS_PATH}" -DDATA_PATH="${IE_MAIN_SOURCE_DIR}/tests/data")

target_compile_definitions(${TARGET_NAME} PUBLIC -DUSE_STATIC_IE)

target_link_libraries(${TARGET_NAME}
        gtest
        gmock
        gtest_main
        inference_engine_s
        ie_cpu_extension
        helpers
        ${PUGI}
        ${LIB_DL}
        ${MKLDNN_STATIC_ENGINE}
        ${INTEL_ITT_LIBS}
        ${Boost_REGEX_LIBRARY}
        ${TBB_LIBRARY}
        ${TBBMALLOC_LIBRARY})

if (ENABLE_MKL_DNN)
    target_link_libraries(${TARGET_NAME}
            test_MKLDNNPlugin
            mkldnn)
endif ()

add_test(NAME ${TARGET_NAME}
        COMMAND ${TARGET_NAME})

add_dependencies(${TARGET_NAME} mock_engine)
