# Copyright (C) 2018 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 2.8)

#features trigger supported by build system
include(check_features)
include(debug)

#we have number of dependencies stored on ftp
include(dependency_solver)

#prepare temporary folder
if (DEFINED ENV{${DL_SDK_TEMP}})
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

set(MODELS_PATH "${TEMP}/models")
debug_message(STATUS "MODELS_PATH=" ${MODELS_PATH})

#clDNN
if (ENABLE_CLDNN AND NOT ENABLE_CLDNN_BUILD)
if(NOT IE_SUBMODULE_IN_CLDNN)
    RESOLVE_DEPENDENCY(CLDNN
            ARCHIVE_UNIFIED "cldnn-main-03988.zip"
            TARGET_PATH "${TEMP}/clDNN"
            ENVIRONMENT "CLDNN"
            VERSION_REGEX ".*_(([a-z]+-)?[a-z]+-[0-9]+)---.*"
            FOLDER) #new cldnn package dont have toplevel cldnn folder
    debug_message(STATUS "clDNN=" ${CLDNN})
endif ()
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
if (GEMM STREQUAL "MKL" OR ENABLE_INTEL_OMP)
if (WIN32)
    RESOLVE_DEPENDENCY(MKL
            ARCHIVE_WIN "mkltiny_win_20180512.zip"
            TARGET_PATH "${TEMP}/mkltiny_win_20180512"
            ENVIRONMENT "MKLROOT"
            VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
elseif(LINUX)
    RESOLVE_DEPENDENCY(MKL
            ARCHIVE_LIN "mkltiny_lnx_20180511.tgz"
            TARGET_PATH "${TEMP}/mkltiny_lnx_20180511"
            ENVIRONMENT "MKLROOT"
            VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
endif()
debug_message(STATUS "mkl_ml=" ${MKL})
endif ()

if (ENABLE_OPENCV)
if (WIN32)
    RESOLVE_DEPENDENCY(OPENCV
            ARCHIVE_WIN "opencv_3.4.3.zip"
            TARGET_PATH "${TEMP}/opencv"
            ENVIRONMENT "OpenCV_DIR"
            VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")
    log_rpath_from_dir(OPENCV "\\opencv\\x64\\vc14\\bin")
    set( ENV{OpenCV_DIR} ${OPENCV} )
elseif(LINUX)
if (${LINUX_OS_NAME} STREQUAL "Ubuntu 16.04")
    RESOLVE_DEPENDENCY(OPENCV
            ARCHIVE_LIN "opencv_3.4.3_ubuntu16.tar.bz2"
            TARGET_PATH "${TEMP}/opencv_ubuntu16"
            ENVIRONMENT "OpenCV_DIR"
            VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")
    log_rpath_from_dir(OPENCV "opencv_ubuntu16/lib")
elseif (${LINUX_OS_NAME} STREQUAL "CentOS 7")
    RESOLVE_DEPENDENCY(OPENCV
            ARCHIVE_LIN "opencv_3.4.3_centos7.tar.bz2"
            TARGET_PATH "${TEMP}/opencv_centos7"
            ENVIRONMENT "OpenCV_DIR"
            VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")
    log_rpath_from_dir(OPENCV "opencv_centos7/lib")
endif()
    set( ENV{OpenCV_DIR} ${OPENCV}/share )
endif()
debug_message(STATUS "opencv=" ${OPENCV})
endif()

include(omp)
