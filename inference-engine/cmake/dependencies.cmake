# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)

#we have number of dependencies stored on ftp
include(dependency_solver)

set_temp_directory(TEMP "${IE_MAIN_SOURCE_DIR}")

include(ExternalProject)

if (ENABLE_SAME_BRANCH_FOR_MODELS)
    branchName(MODELS_BRANCH)
else()
    set(MODELS_BRANCH "master")
endif()

include(linux_name)
if(COMMAND get_linux_name)
    get_linux_name(LINUX_OS_NAME)
endif()

if (ENABLE_MYRIAD)
    RESOLVE_DEPENDENCY(VPU_FIRMWARE_MA2450
            ARCHIVE_UNIFIED firmware_ma2450_676.zip
            TARGET_PATH "${TEMP}/vpu/firmware/ma2450"
            ENVIRONMENT "VPU_FIRMWARE_MA2450"
            FOLDER)
    debug_message(STATUS "ma2450=" ${VPU_FIRMWARE_MA2450})
endif ()

if (ENABLE_MYRIAD)
    RESOLVE_DEPENDENCY(VPU_FIRMWARE_MA2X8X
            ARCHIVE_UNIFIED firmware_ma2x8x_mdk_R8_9.zip
            TARGET_PATH "${TEMP}/vpu/firmware/ma2x8x"
            ENVIRONMENT "VPU_FIRMWARE_MA2X8X"
            FOLDER)
    debug_message(STATUS "ma2x8x=" ${VPU_FIRMWARE_MA2X8X})
endif ()

## enable cblas_gemm from OpenBLAS package
if (GEMM STREQUAL "OPENBLAS")
if(NOT BLAS_LIBRARIES OR NOT BLAS_INCLUDE_DIRS)
    find_package(BLAS REQUIRED)
    if(BLAS_FOUND)
        find_path(BLAS_INCLUDE_DIRS cblas.h)
    else()
        message(ERROR "OpenBLAS not found: install OpenBLAS or set -DBLAS_INCLUDE_DIRS=<path to dir with cblas.h> and -DBLAS_LIBRARIES=<path to libopenblas.so or openblas.lib>")
    endif()
endif()
debug_message(STATUS "openblas=" ${BLAS_LIBRARIES})
endif ()

#MKL-ml package
if (GEMM STREQUAL "MKL")
if(NOT MKLROOT)
    message(FATAL_ERROR "MKLROOT not found: install MKL and set -DMKLROOT=<path_to_MKL>")
endif()
set(MKL ${MKLROOT})
debug_message(STATUS "mkl_ml=" ${MKLROOT})
endif ()

## Intel OMP package
if (THREADING STREQUAL "OMP")
if (WIN32)
    RESOLVE_DEPENDENCY(OMP
            ARCHIVE_WIN "iomp.zip"
            TARGET_PATH "${TEMP}/omp"
            ENVIRONMENT "OMP"
            VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
elseif(LINUX)
    set(OMP "-fopenmp")
else(APPLE)
    RESOLVE_DEPENDENCY(OMP
            ARCHIVE_MAC "iomp_20190130_mac.tgz"
            TARGET_PATH "${TEMP}/omp"
            ENVIRONMENT "OMP"
            VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
endif()
log_rpath_from_dir(OMP "${OMP}/lib")
debug_message(STATUS "intel_omp=" ${OMP})
endif ()

## TBB package
if (THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO")
if (WIN32)
    #TODO: add target_path to be platform specific as well, to avoid following if
    RESOLVE_DEPENDENCY(TBB
            ARCHIVE_WIN "tbb2019_20181010_win.zip" #TODO: windows zip archive created incorrectly using old name for folder
            TARGET_PATH "${TEMP}/tbb"
            ENVIRONMENT "TBBROOT"
            VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
elseif(LINUX)
    RESOLVE_DEPENDENCY(TBB
            ARCHIVE_LIN "tbb2019_20181010_lin.tgz"
            TARGET_PATH "${TEMP}/tbb"
            ENVIRONMENT "TBBROOT")
else(APPLE)
    RESOLVE_DEPENDENCY(TBB
            ARCHIVE_MAC "tbb2019_20190414_mac.tgz"
            TARGET_PATH "${TEMP}/tbb"
            ENVIRONMENT "TBBROOT"
            VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
endif()
log_rpath_from_dir(TBB "${TBB}/lib")
debug_message(STATUS "tbb=" ${TBB})
endif ()

if (ENABLE_OPENCV)
  set(OPENCV_VERSION "4.1.1")
  set(OPENCV_BUILD "595")
  set(OPENCV_SUFFIX "")
if (WIN32)
    RESOLVE_DEPENDENCY(OPENCV
            ARCHIVE_WIN "opencv_${OPENCV_VERSION}-${OPENCV_BUILD}.zip"
            TARGET_PATH "${TEMP}/opencv_${OPENCV_VERSION}"
            ENVIRONMENT "OpenCV_DIR"
            VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")
    log_rpath_from_dir(OPENCV "\\opencv_${OPENCV_VERSION}\\bin")
    set( ENV{OpenCV_DIR} ${OPENCV}/cmake )
elseif(APPLE)
    RESOLVE_DEPENDENCY(OPENCV
            ARCHIVE_MAC "opencv_${OPENCV_VERSION}-${OPENCV_BUILD}_osx.tar.xz"
            TARGET_PATH "${TEMP}/opencv_${OPENCV_VERSION}_osx"
            ENVIRONMENT "OpenCV_DIR"
            VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")
    log_rpath_from_dir(OPENCV "opencv_${OPENCV_VERSION}_osx/lib")
    set( ENV{OpenCV_DIR} ${OPENCV}/cmake )
elseif(LINUX)
    if (${LINUX_OS_NAME} STREQUAL "Ubuntu 16.04")
        set(OPENCV_SUFFIX "ubuntu16")
    elseif (${LINUX_OS_NAME} STREQUAL "Ubuntu 18.04")
        set(OPENCV_SUFFIX "ubuntu18")
    elseif (${LINUX_OS_NAME} STREQUAL "CentOS 7")
        set(OPENCV_SUFFIX "centos7")
    elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "armv7l" AND
            (${LINUX_OS_NAME} STREQUAL "Debian 9" OR
             ${LINUX_OS_NAME} STREQUAL "Raspbian 9" OR
             ${LINUX_OS_NAME} STREQUAL "Debian 10" OR
             ${LINUX_OS_NAME} STREQUAL "Raspbian 10"))
        set(OPENCV_SUFFIX "debian9arm")
    endif()
endif()

if (OPENCV_SUFFIX)
    RESOLVE_DEPENDENCY(OPENCV
            ARCHIVE_LIN "opencv_${OPENCV_VERSION}-${OPENCV_BUILD}_${OPENCV_SUFFIX}.tar.xz"
            TARGET_PATH "${TEMP}/opencv_${OPENCV_VERSION}_${OPENCV_SUFFIX}"
            ENVIRONMENT "OpenCV_DIR"
            VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")
    log_rpath_from_dir(OPENCV "opencv_${OPENCV_VERSION}_${OPENCV_SUFFIX}/lib")
    set( ENV{OpenCV_DIR} ${OPENCV}/cmake )
endif()

debug_message(STATUS "opencv=" ${OPENCV})
set(OpenCV_DIR "${OPENCV}" CACHE PATH "Path to OpenCV in temp directory")
endif()


include(ie_parallel)

if (ENABLE_GNA)
    RESOLVE_DEPENDENCY(GNA
            ARCHIVE_UNIFIED "gna_20181120.zip"
            TARGET_PATH "${TEMP}/gna")
endif()

configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/share/InferenceEngineConfig.cmake.in"
        "${CMAKE_BINARY_DIR}/share/InferenceEngineConfig.cmake"
        @ONLY)

configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/share/InferenceEngineConfig-version.cmake.in"
        "${CMAKE_BINARY_DIR}/share/InferenceEngineConfig-version.cmake"
        COPYONLY)

configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/ie_parallel.cmake"
        "${CMAKE_BINARY_DIR}/share/ie_parallel.cmake"
        COPYONLY)
