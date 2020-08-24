# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)

include(models)

#we have number of dependencies stored on ftp
include(dependency_solver)

if (CMAKE_CROSSCOMPILING)
    set(CMAKE_STAGING_PREFIX "${TEMP}")
endif()

include(ExternalProject)

if (ENABLE_SAME_BRANCH_FOR_MODELS)
    branchName(MODELS_BRANCH)
else()
    set(MODELS_BRANCH "master")
endif()


if (ENABLE_DATA)
    add_models_repo(${ENABLE_DATA} "data:https://github.com/openvinotoolkit/testdata.git")
    set(MODELS_PATH "${TEMP}/models/src/data")
    set(DATA_PATH "${MODELS_PATH}")
endif()

message(STATUS "MODELS_PATH=" ${MODELS_PATH})

fetch_models_and_validation_set()

include(linux_name)
if(COMMAND get_linux_name)
    get_linux_name(LINUX_OS_NAME)
endif()

include(CMakeParseArguments)

if (ENABLE_MYRIAD)
    include(vpu_dependencies)
endif()

## enable cblas_gemm from OpenBLAS package
if (ENABLE_MKL_DNN AND GEMM STREQUAL "OPENBLAS")
    if(AARCH64)
        if(DEFINED ENV{THIRDPARTY_SERVER_PATH})
            set(IE_PATH_TO_DEPS "$ENV{THIRDPARTY_SERVER_PATH}")
        elseif(DEFINED THIRDPARTY_SERVER_PATH)
            set(IE_PATH_TO_DEPS "${THIRDPARTY_SERVER_PATH}")
        else()
            message(WARNING "OpenBLAS is not found!")
        endif()

        if(DEFINED IE_PATH_TO_DEPS)
            reset_deps_cache(OpenBLAS_DIR)

            RESOLVE_DEPENDENCY(OpenBLAS
                    ARCHIVE_LIN "keembay/openblas_0.3.7_yocto_kmb.tar.xz"
                    TARGET_PATH "${TEMP}/openblas_0.3.7_yocto_kmb"
                    ENVIRONMENT "OpenBLAS_DIR")

            update_deps_cache(OpenBLAS_DIR "${OpenBLAS}/lib/cmake/openblas" "Path to OpenBLAS package folder")

            find_package(OpenBLAS QUIET)

            if(OpenBLAS_FOUND)
                set(BLAS_FOUND TRUE)
                set(BLAS_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIRS})
                set(BLAS_LIBRARIES ${OpenBLAS_LIBRARIES})
            endif()

            unset(IE_PATH_TO_DEPS)
        endif()
    endif()

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

## MKL-ML package
if (GEMM STREQUAL "MKL")
    if(NOT MKLROOT)
        message(FATAL_ERROR "MKLROOT not found: install MKL and set -DMKLROOT=<path_to_MKL>")
    endif()
    set(MKL ${MKLROOT})
    debug_message(STATUS "mkl_ml=" ${MKLROOT})
endif ()

## Intel OMP package
if (THREADING STREQUAL "OMP")
    reset_deps_cache(OMP)
    if (WIN32 AND X86_64)
        RESOLVE_DEPENDENCY(OMP
                ARCHIVE_WIN "iomp.zip"
                TARGET_PATH "${TEMP}/omp"
                ENVIRONMENT "OMP"
                VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
    elseif(LINUX AND X86_64)
        RESOLVE_DEPENDENCY(OMP
                ARCHIVE_LIN "iomp.tgz"
                TARGET_PATH "${TEMP}/omp"
                ENVIRONMENT "OMP"
                VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
    elseif(APPLE AND X86_64)
        RESOLVE_DEPENDENCY(OMP
                ARCHIVE_MAC "iomp_20190130_mac.tgz"
                TARGET_PATH "${TEMP}/omp"
                ENVIRONMENT "OMP"
                VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*")
    else()
        message(FATAL_ERROR "Intel OMP is not available on current platform")
    endif()
    update_deps_cache(OMP "${OMP}" "Path to OMP root folder")
    log_rpath_from_dir(OMP "${OMP}/lib")
    debug_message(STATUS "intel_omp=" ${OMP})
endif ()

## TBB package
if (THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO")
    reset_deps_cache(TBBROOT)

    if(NOT DEFINED TBB_DIR AND NOT DEFINED ENV{TBB_DIR})
        if (WIN32 AND X86_64)
            #TODO: add target_path to be platform specific as well, to avoid following if
            RESOLVE_DEPENDENCY(TBB
                    ARCHIVE_WIN "tbb2020_20200415_win.zip"
                    TARGET_PATH "${TEMP}/tbb"
                    ENVIRONMENT "TBBROOT")
        elseif(ANDROID)  # Should be before LINUX due LINUX is detected as well
            RESOLVE_DEPENDENCY(TBB
                    ARCHIVE_ANDROID "tbb2020_20200404_android.tgz"
                    TARGET_PATH "${TEMP}/tbb"
                    ENVIRONMENT "TBBROOT")
        elseif(LINUX AND X86_64)
            RESOLVE_DEPENDENCY(TBB
                    ARCHIVE_LIN "tbb2020_20200415_lin_strip.tgz"
                    TARGET_PATH "${TEMP}/tbb")
        elseif(LINUX AND AARCH64)
            RESOLVE_DEPENDENCY(TBB
                    ARCHIVE_LIN "keembay/tbb2020_38404_kmb.tgz"
                    TARGET_PATH "${TEMP}/tbb_yocto"
                    ENVIRONMENT "TBBROOT")
        elseif(APPLE AND X86_64)
            RESOLVE_DEPENDENCY(TBB
                    ARCHIVE_MAC "tbb2020_20200404_mac.tgz"
                    TARGET_PATH "${TEMP}/tbb"
                    ENVIRONMENT "TBBROOT")
        else()
            message(FATAL_ERROR "TBB is not available on current platform")
        endif()
    else()
        if(DEFINED TBB_DIR)
            get_filename_component(TBB ${TBB_DIR} DIRECTORY)
        else()
            get_filename_component(TBB $ENV{TBB_DIR} DIRECTORY)
        endif()
    endif()

    update_deps_cache(TBBROOT "${TBB}" "Path to TBB root folder")

    if (WIN32)
        log_rpath_from_dir(TBB "${TBB}/bin")
    else ()
        log_rpath_from_dir(TBB "${TBB}/lib")
    endif ()
    debug_message(STATUS "tbb=" ${TBB})
endif ()

if (ENABLE_OPENCV)
    reset_deps_cache(OpenCV_DIR)

    set(OPENCV_VERSION "4.3.0")
    set(OPENCV_BUILD "060")
    set(OPENCV_BUILD_YOCTO "073")

    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        if(DEFINED ENV{THIRDPARTY_SERVER_PATH})
            set(IE_PATH_TO_DEPS "$ENV{THIRDPARTY_SERVER_PATH}")
        elseif(DEFINED THIRDPARTY_SERVER_PATH)
            set(IE_PATH_TO_DEPS "${THIRDPARTY_SERVER_PATH}")
        else()
            message(WARNING "OpenCV is not found!")
        endif()

        if(DEFINED IE_PATH_TO_DEPS)
            set(OPENCV_SUFFIX "yocto_kmb")
            set(OPENCV_BUILD "${OPENCV_BUILD_YOCTO}")

            RESOLVE_DEPENDENCY(OPENCV
                    ARCHIVE_LIN "opencv/opencv_${OPENCV_VERSION}-${OPENCV_BUILD}_${OPENCV_SUFFIX}.txz"
                    TARGET_PATH "${TEMP}/opencv_${OPENCV_VERSION}_${OPENCV_SUFFIX}/opencv"
                    ENVIRONMENT "OpenCV_DIR"
                    VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")

            unset(IE_PATH_TO_DEPS)
        endif()
    else()
        if (WIN32 AND X86_64)
            RESOLVE_DEPENDENCY(OPENCV
                    ARCHIVE_WIN "opencv/opencv_${OPENCV_VERSION}-${OPENCV_BUILD}.txz"
                    TARGET_PATH "${TEMP}/opencv_${OPENCV_VERSION}/opencv"
                    ENVIRONMENT "OpenCV_DIR"
                    VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")
        elseif(APPLE AND X86_64)
            RESOLVE_DEPENDENCY(OPENCV
                    ARCHIVE_MAC "opencv/opencv_${OPENCV_VERSION}-${OPENCV_BUILD}_osx.txz"
                    TARGET_PATH "${TEMP}/opencv_${OPENCV_VERSION}_osx/opencv"
                    ENVIRONMENT "OpenCV_DIR"
                    VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")
        elseif(LINUX)
            if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
                set(OPENCV_SUFFIX "yocto_kmb")
                set(OPENCV_BUILD "${OPENCV_BUILD_YOCTO}")
            elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "armv7l")
                set(OPENCV_SUFFIX "debian9arm")
            elseif (${LINUX_OS_NAME} STREQUAL "CentOS 7" OR CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.9")
                set(OPENCV_SUFFIX "centos7")
            elseif (${LINUX_OS_NAME} STREQUAL "Ubuntu 16.04")
                set(OPENCV_SUFFIX "ubuntu16")
            elseif (${LINUX_OS_NAME} STREQUAL "Ubuntu 18.04")
                set(OPENCV_SUFFIX "ubuntu18")
            elseif (${LINUX_OS_NAME} STREQUAL "Ubuntu 20.04")
                set(OPENCV_SUFFIX "ubuntu20")
            else()
                message(FATAL_ERROR "OpenCV is not available on current platform")
            endif()
            RESOLVE_DEPENDENCY(OPENCV
                    ARCHIVE_LIN "opencv/opencv_${OPENCV_VERSION}-${OPENCV_BUILD}_${OPENCV_SUFFIX}.txz"
                    TARGET_PATH "${TEMP}/opencv_${OPENCV_VERSION}_${OPENCV_SUFFIX}/opencv"
                    ENVIRONMENT "OpenCV_DIR"
                    VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*")
        endif()

    endif()

    if(ANDROID)
        set(ocv_cmake_path "${OPENCV}/sdk/native/jni/")
    else()
        set(ocv_cmake_path "${OPENCV}/cmake")
    endif()

    update_deps_cache(OpenCV_DIR "${ocv_cmake_path}" "Path to OpenCV package folder")

    if(WIN32)
        log_rpath_from_dir(OPENCV "${OpenCV_DIR}/../bin")
    elseif(ANDROID)
        log_rpath_from_dir(OPENCV "${OpenCV_DIR}/../../../lib")
    else()
        log_rpath_from_dir(OPENCV "${OpenCV_DIR}/../lib")
    endif()
    debug_message(STATUS "opencv=" ${OPENCV})
endif()

include(ie_parallel)

if (ENABLE_GNA)
    reset_deps_cache(
            GNA
            GNA_PLATFORM_DIR
            GNA_KERNEL_LIB_NAME
            GNA_LIBS_LIST
            GNA_LIB_DIR
            libGNA_INCLUDE_DIRS
            libGNA_LIBRARIES_BASE_PATH)
    if (GNA_LIBRARY_VERSION STREQUAL "GNA1")
        RESOLVE_DEPENDENCY(GNA
                ARCHIVE_UNIFIED "GNA/gna_20181120.zip"
                TARGET_PATH "${TEMP}/gna")
    else()
        if(GNA_LIBRARY_VERSION STREQUAL "GNA1_1401")
            set(GNA_VERSION "01.00.00.1401")
        endif()
        if(GNA_LIBRARY_VERSION STREQUAL "GNA2")
            set(GNA_VERSION "02.00.00.1047.1")
        endif()
        RESOLVE_DEPENDENCY(GNA
                ARCHIVE_UNIFIED "GNA/GNA_${GNA_VERSION}.zip"
                TARGET_PATH "${TEMP}/gna_${GNA_VERSION}"
                VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+.[0-9]+).*")
    endif()
    update_deps_cache(GNA "${GNA}" "Path to GNA root folder")
    debug_message(STATUS "gna=" ${GNA})
endif()

if (ENABLE_SPEECH_DEMO)
    reset_deps_cache(SPEECH_LIBS_AND_DEMOS)
    if(DEFINED ENV{THIRDPARTY_SERVER_PATH})
        set(IE_PATH_TO_DEPS "$ENV{THIRDPARTY_SERVER_PATH}")
    elseif(DEFINED THIRDPARTY_SERVER_PATH)
        set(IE_PATH_TO_DEPS "${THIRDPARTY_SERVER_PATH}")
    else()
        message(WARNING "Unable to locate Speech Demo")
    endif()
    if(DEFINED IE_PATH_TO_DEPS)
        if (WIN32 AND X86_64)
            RESOLVE_DEPENDENCY(SPEECH_LIBS_AND_DEMOS
                    ARCHIVE_WIN "speech_demo_1.0.0.746_windows.zip"
                    VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+.[0-9]+).*"
                    TARGET_PATH "${TEMP}/speech_demo_1.0.0.746")
            debug_message(STATUS "speech_libs_and_demos=" ${SPEECH_LIBS_AND_DEMOS})
        elseif (LINUX AND X86_64)
            if (${LINUX_OS_NAME} STREQUAL "CentOS 7" OR CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.9")
                RESOLVE_DEPENDENCY(SPEECH_LIBS_AND_DEMOS
                    ARCHIVE_LIN "speech_demo_1.0.0.746_centos.tgz"
                    VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+.[0-9]+).*"
                    TARGET_PATH "${TEMP}/speech_demo_1.0.0.746")
                debug_message(STATUS "speech_libs_and_demos=" ${SPEECH_LIBS_AND_DEMOS})
            else()
                RESOLVE_DEPENDENCY(SPEECH_LIBS_AND_DEMOS
                    ARCHIVE_LIN "speech_demo_1.0.0.746_linux.tgz"
                    VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+.[0-9]+).*"
                    TARGET_PATH "${TEMP}/speech_demo_1.0.0.746")
                debug_message(STATUS "speech_libs_and_demos=" ${SPEECH_LIBS_AND_DEMOS})
            endif()
        else()
            message(FATAL_ERROR "Speech Demo is not available on current platform")
        endif()
        unset(IE_PATH_TO_DEPS)
    endif()
    update_deps_cache(SPEECH_LIBS_AND_DEMOS "${SPEECH_LIBS_AND_DEMOS}" "Path to SPEECH_LIBS_AND_DEMOS root folder")
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
