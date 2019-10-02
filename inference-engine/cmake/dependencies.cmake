# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)

#we have number of dependencies stored on ftp
include(dependency_solver)

set_temp_directory(TEMP "${IE_MAIN_SOURCE_DIR}")
if (CMAKE_CROSSCOMPILING)
    set(CMAKE_STAGING_PREFIX "${TEMP}")
endif()

include(ExternalProject)

include(linux_name)
if(COMMAND get_linux_name)
    get_linux_name(LINUX_OS_NAME)
endif()

if (ENABLE_MYRIAD)
    include(vpu_dependencies)
endif()

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
                ARCHIVE_MAC "tbb2019_20190414_v1_mac.tgz"
                TARGET_PATH "${TEMP}/tbb"
                ENVIRONMENT "TBBROOT"
                VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
    endif()
    log_rpath_from_dir(TBB "${TBB}/lib")
    debug_message(STATUS "tbb=" ${TBB})
endif ()

if (ENABLE_OPENCV)
    set(OPENCV_VERSION "4.1.2")
    set(OPENCV_BUILD "624")
    set(OPENCV_SUFFIX "")
    if (WIN32)
        RESOLVE_DEPENDENCY(OPENCV
                ARCHIVE_WIN "opencv_${OPENCV_VERSION}-${OPENCV_BUILD}.zip"
                TARGET_PATH "${TEMP}/opencv_${OPENCV_VERSION}"
                ENVIRONMENT "OpenCV_DIR"
                VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")
        log_rpath_from_dir(OPENCV "\\opencv_${OPENCV_VERSION}\\bin")
    elseif(APPLE)
        RESOLVE_DEPENDENCY(OPENCV
                ARCHIVE_MAC "opencv_${OPENCV_VERSION}-${OPENCV_BUILD}_osx.tar.xz"
                TARGET_PATH "${TEMP}/opencv_${OPENCV_VERSION}_osx"
                ENVIRONMENT "OpenCV_DIR"
                VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")
        log_rpath_from_dir(OPENCV "opencv_${OPENCV_VERSION}_osx/lib")
    elseif(LINUX)
        if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "armv7l")
            set(OPENCV_SUFFIX "debian9arm")
        elseif (${LINUX_OS_NAME} STREQUAL "Ubuntu 16.04")
            set(OPENCV_SUFFIX "ubuntu16")
        elseif (${LINUX_OS_NAME} STREQUAL "Ubuntu 18.04")
            set(OPENCV_SUFFIX "ubuntu18")
        elseif (${LINUX_OS_NAME} STREQUAL "CentOS 7")
            set(OPENCV_SUFFIX "centos7")
        endif()
    endif()

    if (OPENCV_SUFFIX)
        RESOLVE_DEPENDENCY(OPENCV
                ARCHIVE_LIN "opencv_${OPENCV_VERSION}-${OPENCV_BUILD}_${OPENCV_SUFFIX}.tar.xz"
                TARGET_PATH "${TEMP}/opencv_${OPENCV_VERSION}_${OPENCV_SUFFIX}"
                ENVIRONMENT "OpenCV_DIR"
                VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")
        log_rpath_from_dir(OPENCV "opencv_${OPENCV_VERSION}_${OPENCV_SUFFIX}/lib")
    endif()
    debug_message(STATUS "opencv=" ${OPENCV})
    # OpenCV_DIR should point to cmake folder within the specified OpenCV binary package.
    # It's required to successsfully find OpenCV libs using find_package(OpenCV ...) command.
    # So, the cached OpenCV_DIR variable should be update if custom value wasn't previously set here.
    if (NOT DEFINED ENV{OpenCV_DIR})
        set(OpenCV_DIR "${OPENCV}/cmake" CACHE PATH "Path to OpenCV in temp directory")
    endif()
endif()

include(ie_parallel)

if (ENABLE_GNA)
    if (GNA_LIBRARY_VERSION STREQUAL "GNA1")
        RESOLVE_DEPENDENCY(GNA
                ARCHIVE_UNIFIED "gna_20181120.zip"
                TARGET_PATH "${TEMP}/gna")
    elseif(GNA_LIBRARY_VERSION STREQUAL "GNA1_1401")
        set(GNA_VERSION "01.00.00.1401")
        RESOLVE_DEPENDENCY(GNA
                ARCHIVE_UNIFIED "GNA_${GNA_VERSION}.zip"
                TARGET_PATH "${TEMP}/gna_${GNA_VERSION}"
                VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+.[0-9]+).*")
    endif()
    debug_message(STATUS "gna=" ${GNA})
endif()

if (ENABLE_ROCKHOPER)
    set(rh_decoder_version "Rockhopper_1.0.0.682")

    set(INCLUDE_RH_DECODER "include(\"\$\{IE_ROOT_DIR\}/share/ie_rh_decoder.cmake\")")

    RESOLVE_DEPENDENCY(RH_Decoder
            ARCHIVE_UNIFIED "${rh_decoder_version}.zip"
            TARGET_PATH "${TEMP}/${rh_decoder_version}"
            VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+.[0-9]+).*")

    configure_file(
            "${IE_MAIN_SOURCE_DIR}/cmake/InitRHDecoder.cmake.in"
            "${CMAKE_BINARY_DIR}/share/ie_rh_decoder.cmake"
            @ONLY)

    list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR}/share)
    # for inference engine in tree build - lets include this finder
    include(ie_rh_decoder)
endif()

configure_file(
        "${IE_MAIN_SOURCE_DIR}/cmake/share/InferenceEngineConfig.cmake.in"
        "${CMAKE_BINARY_DIR}/share/InferenceEngineConfig.cmake"
        @ONLY)

configure_file(
        "${IE_MAIN_SOURCE_DIR}/cmake/share/InferenceEngineConfig-version.cmake.in"
        "${CMAKE_BINARY_DIR}/share/InferenceEngineConfig-version.cmake"
        COPYONLY)

configure_file(
        "${IE_MAIN_SOURCE_DIR}/cmake/ie_parallel.cmake"
        "${CMAKE_BINARY_DIR}/share/ie_parallel.cmake"
        COPYONLY)
