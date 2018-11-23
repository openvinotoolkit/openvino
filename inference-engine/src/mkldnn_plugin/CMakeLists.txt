# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
set(TARGET_NAME "MKLDNNPlugin")

if (UNIX AND NOT APPLE)
    set(OS_SPECIFIC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn/os/lin/*.cpp)
endif()

file(GLOB SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/nodes/*.cpp
        ${OS_SPECIFIC_SRC}
)

file(GLOB HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/nodes/*.h
)

addVersionDefines(mkldnn_plugin.cpp CI_BUILD_NUMBER MKL_VERSION)

if(WIN32)
    add_definitions(-DIMPLEMENT_INFERENCE_ENGINE_PLUGIN)
endif()

include_directories(
        ${IE_MAIN_SOURCE_DIR}/include
        ${IE_MAIN_SOURCE_DIR}/src/inference_engine
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn
        ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/src/common
        ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/include
)

add_library(${TARGET_NAME} SHARED ${SOURCES} ${HEADERS})

if (THREADING STREQUAL "TBB")
    target_compile_definitions(${TARGET_NAME} PUBLIC -DIE_THREAD=IE_THREAD_TBB -DMKLDNN_THR=MKLDNN_THR_TBB)
    target_include_directories(${TARGET_NAME} PUBLIC ${TBB_INCLUDE_DIRS})
    target_link_libraries(${TARGET_NAME} debug ${TBB_LIBRARIES_RELEASE} optimized ${TBB_LIBRARIES_RELEASE})
elseif (THREADING STREQUAL "OMP")
    target_compile_definitions(${TARGET_NAME} PUBLIC -DIE_THREAD=IE_THREAD_OMP -DMKLDNN_THR=MKLDNN_THR_OMP)
    enable_omp()
    if(ENABLE_INTEL_OMP)
        target_link_libraries(${TARGET_NAME} ${intel_omp_lib})
    endif()
else()
    target_compile_definitions(${TARGET_NAME} PUBLIC -DIE_THREAD=IE_THREAD_SEQ -DMKLDNN_THR=MKLDNN_THR_SEQ)
endif()

target_link_libraries(${TARGET_NAME} inference_engine ${INTEL_ITT_LIBS} mkldnn)
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})

add_library(test_${TARGET_NAME} STATIC ${SOURCES} ${HEADERS})

if (THREADING STREQUAL "TBB")
    target_compile_definitions(test_${TARGET_NAME} PUBLIC -DIE_THREAD=IE_THREAD_TBB -DMKLDNN_THR=MKLDNN_THR_TBB)
    target_include_directories(test_${TARGET_NAME} PUBLIC ${TBB_INCLUDE_DIRS})
    target_link_libraries(test_${TARGET_NAME} debug ${TBB_LIBRARIES_RELEASE} optimized ${TBB_LIBRARIES_RELEASE})
elseif (THREADING STREQUAL "OMP")
    target_compile_definitions(test_${TARGET_NAME} PUBLIC -DIE_THREAD=IE_THREAD_OMP -DMKLDNN_THR=MKLDNN_THR_OMP)
    if(ENABLE_INTEL_OMP)
        target_link_libraries(test_${TARGET_NAME} ${intel_omp_lib})
    endif()
else()
    target_compile_definitions(test_${TARGET_NAME} PUBLIC -DIE_THREAD=IE_THREAD_SEQ -DMKLDNN_THR=MKLDNN_THR_SEQ)
endif()

target_link_libraries(test_${TARGET_NAME} inference_engine_s mkldnn)
set_target_properties(test_${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME test_${TARGET_NAME})
