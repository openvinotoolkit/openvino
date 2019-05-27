# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)

#features trigger supported by build system
include(check_features)
include(debug)

#we have number of dependencies stored on ftp
include(dependency_solver)

#prepare temporary folder
if (DEFINED ENV{${DL_SDK_TEMP}} AND NOT $ENV{${DL_SDK_TEMP}} STREQUAL "")
    if (WIN32)
        string(REPLACE "\\" "\\\\" TEMP $ENV{${DL_SDK_TEMP}})
    else(WIN32)
        set(TEMP $ENV{${DL_SDK_TEMP}})
    endif(WIN32)

    if (ENABLE_ALTERNATIVE_TEMP)
        set(ALTERNATIVE_PATH ${IE_MAIN_SOURCE_DIR}/temp)
    endif()
else ()
    message(STATUS "DL_SDK_TEMP envionment not set")
    set(TEMP ${IE_MAIN_SOURCE_DIR}/temp)
endif ()


include(ExternalProject)

if (ENABLE_SAME_BRANCH_FOR_MODELS)
    branchName(MODELS_BRANCH)
else()
    set(MODELS_BRANCH "master")
endif()

if (ENABLE_MYRIAD)
    RESOLVE_DEPENDENCY(VPU_FIRMWARE_MA2450
            ARCHIVE_UNIFIED firmware_ma2450_491.zip
            TARGET_PATH "${TEMP}/vpu/firmware/ma2450"
            ENVIRONMENT "VPU_FIRMWARE_MA2450"
            FOLDER)
    debug_message(STATUS "ma2450=" ${VPU_FIRMWARE_MA2450})
endif ()

if (ENABLE_MYRIAD)
    RESOLVE_DEPENDENCY(VPU_FIRMWARE_MA2480
            ARCHIVE_UNIFIED firmware_ma2480_mdk_R7_9.zip
            TARGET_PATH "${TEMP}/vpu/firmware/ma2480"
            ENVIRONMENT "VPU_FIRMWARE_MA2480"
            FOLDER)
    debug_message(STATUS "ma2480=" ${VPU_FIRMWARE_MA2480})
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
    RESOLVE_DEPENDENCY(OMP
            ARCHIVE_LIN "iomp.tgz"
            TARGET_PATH "${TEMP}/omp"
            ENVIRONMENT "OMP"
            VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
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
if (THREADING STREQUAL "TBB")
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
if (WIN32)
    RESOLVE_DEPENDENCY(OPENCV
            ARCHIVE_WIN "opencv_4.1.0-0437.zip"
            TARGET_PATH "${TEMP}/opencv_4.1.0"
            ENVIRONMENT "OpenCV_DIR"
            VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")
    log_rpath_from_dir(OPENCV "\\opencv_4.1.0\\bin")
    set( ENV{OpenCV_DIR} ${OPENCV}/cmake )
elseif(APPLE)
    RESOLVE_DEPENDENCY(OPENCV
            ARCHIVE_MAC "opencv_4.1.0-0437_osx.tar.xz"
            TARGET_PATH "${TEMP}/opencv_4.1.0_osx"
            ENVIRONMENT "OpenCV_DIR"
            VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")
    log_rpath_from_dir(OPENCV "opencv_4.1.0_osx/lib")
    set( ENV{OpenCV_DIR} ${OPENCV}/cmake )
elseif(LINUX)
if (${LINUX_OS_NAME} STREQUAL "Ubuntu 16.04")
    RESOLVE_DEPENDENCY(OPENCV
            ARCHIVE_LIN "opencv_4.1.0-0437_ubuntu16.tar.xz"
            TARGET_PATH "${TEMP}/opencv_4.1.0_ubuntu16"
            ENVIRONMENT "OpenCV_DIR"
            VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")
    log_rpath_from_dir(OPENCV "opencv_4.1.0_ubuntu16/lib")
elseif (${LINUX_OS_NAME} STREQUAL "Ubuntu 18.04")
    RESOLVE_DEPENDENCY(OPENCV
            ARCHIVE_LIN "opencv_4.1.0-0437_ubuntu18.tar.xz"
            TARGET_PATH "${TEMP}/opencv_4.1.0_ubuntu18"
            ENVIRONMENT "OpenCV_DIR"
            VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")
    log_rpath_from_dir(OPENCV "opencv_4.1.0_ubuntu18/lib")
elseif (${LINUX_OS_NAME} STREQUAL "CentOS 7")
    RESOLVE_DEPENDENCY(OPENCV
            ARCHIVE_LIN "opencv_4.1.0-0437_centos7.tar.xz"
            TARGET_PATH "${TEMP}/opencv_4.1.0_centos"
            ENVIRONMENT "OpenCV_DIR"
            VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")
    log_rpath_from_dir(OPENCV "opencv_4.1.0_centos/lib")
elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "armv7l" AND
        (${LINUX_OS_NAME} STREQUAL "Debian 9" OR
         ${LINUX_OS_NAME} STREQUAL "Raspbian 9"))
    RESOLVE_DEPENDENCY(OPENCV
            ARCHIVE_LIN "opencv_4.1.0-0437_debian9arm.tar.xz"
            TARGET_PATH "${TEMP}/opencv_4.1.0_debian9arm"
            ENVIRONMENT "OpenCV_DIR"
            VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")
    log_rpath_from_dir(OPENCV "opencv_4.1.0_debian9arm/lib")
endif()
    set( ENV{OpenCV_DIR} ${OPENCV}/cmake )
endif()
debug_message(STATUS "opencv=" ${OPENCV})
endif()


include(ie_parallel)

if (ENABLE_GNA)
    RESOLVE_DEPENDENCY(GNA
            ARCHIVE_UNIFIED "gna_20181120.zip"
            TARGET_PATH "${TEMP}/gna")
endif()

configure_file(
        "${CMAKE_SOURCE_DIR}/cmake/share/InferenceEngineConfig.cmake.in"
        "${CMAKE_BINARY_DIR}/share/InferenceEngineConfig.cmake"
        @ONLY)

configure_file(
        "${CMAKE_SOURCE_DIR}/cmake/share/InferenceEngineConfig-version.cmake.in"
        "${CMAKE_BINARY_DIR}/share/InferenceEngineConfig-version.cmake"
        COPYONLY)

configure_file(
        "${CMAKE_SOURCE_DIR}/cmake/ie_parallel.cmake"
        "${CMAKE_BINARY_DIR}/share/ie_parallel.cmake"
        COPYONLY)
