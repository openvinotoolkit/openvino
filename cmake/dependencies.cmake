# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)

ov_set_temp_directory(TEMP "${CMAKE_SOURCE_DIR}")

## Intel OMP package
if(THREADING STREQUAL "OMP")
    # check whether the compiler supports OpenMP at all
    find_package(OpenMP)

    if(NOT OpenMP_CXX_FOUND)
        message(WARNING "Compiler does not support OpenMP standard. Falling back to SEQ threading")
        set(THREADING "SEQ")
        set(ENABLE_INTEL_OPENMP OFF)
    endif()

    if(ENABLE_INTEL_OPENMP)
        reset_deps_cache(OMP)
        if(WIN32 AND X86_64)
            RESOLVE_DEPENDENCY(INTEL_OMP
                    ARCHIVE_WIN "iomp.zip"
                    TARGET_PATH "${TEMP}/omp"
                    ENVIRONMENT "INTEL_OMP"
                    VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*"
                    SHA256 "62c68646747fb10f19b53217cb04a1e10ff93606f992e6b35eb8c31187c68fbf"
                    USE_NEW_LOCATION TRUE)
        elseif(LINUX AND X86_64 AND OPENVINO_GNU_LIBC)
            RESOLVE_DEPENDENCY(INTEL_OMP
                    ARCHIVE_LIN "iomp.tgz"
                    TARGET_PATH "${TEMP}/omp"
                    ENVIRONMENT "INTEL_OMP"
                    VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*"
                    SHA256 "7832b16d82513ee880d97c27c7626f9525ebd678decf6a8fe6c38550f73227d9"
                    USE_NEW_LOCATION TRUE)
        elseif(APPLE AND X86_64)
            RESOLVE_DEPENDENCY(INTEL_OMP
                    ARCHIVE_MAC "iomp_20190130_mac.tgz"
                    TARGET_PATH "${TEMP}/omp"
                    ENVIRONMENT "INTEL_OMP"
                    VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*"
                    SHA256 "591ea4a7e08bbe0062648916f42bded71d24c27f00af30a8f31a29b5878ea0cc"
                    USE_NEW_LOCATION TRUE)
        endif()

        if(INTEL_OMP)
            update_deps_cache(INTEL_OMP "${INTEL_OMP}" "Path to OMP root folder")
            debug_message(STATUS "intel_omp=" ${INTEL_OMP})

            ov_cpack_add_component(omp HIDDEN)
            file(GLOB_RECURSE source_list "${INTEL_OMP}/*${CMAKE_SHARED_LIBRARY_SUFFIX}*")
            install(FILES ${source_list}
                    DESTINATION ${OV_CPACK_RUNTIMEDIR}
                    COMPONENT intel_omp
                    ${OV_CPACK_COMP_OPENMP_EXCLUDE_ALL})
        endif()
    endif()
endif()

## TBB package
unset(_ov_download_tbb_done CACHE)

#
# The function downloads prebuilt TBB package
# NOTE: the function should be used if system TBB is not found
# or ENABLE_SYSTEM_TBB is OFF
#
function(ov_download_tbb)
    if(_ov_download_tbb_done OR NOT THREADING MATCHES "^(TBB|TBB_AUTO)$")
        return()
    endif()
    set(_ov_download_tbb_done ON CACHE INTERNAL "Whether prebuilt TBB is already downloaded")

    reset_deps_cache(TBBROOT TBB_DIR)

    if(DEFINED ENV{THIRDPARTY_SERVER_PATH})
        set(IE_PATH_TO_DEPS "$ENV{THIRDPARTY_SERVER_PATH}")
    elseif(DEFINED THIRDPARTY_SERVER_PATH)
        set(IE_PATH_TO_DEPS "${THIRDPARTY_SERVER_PATH}")
    endif()

    if(NOT DEFINED ENV{TBBROOT} AND (DEFINED ENV{TBB_DIR} OR DEFINED TBB_DIR))
        if(DEFINED ENV{TBB_DIR})
            set(TBB_DIR "$ENV{TBB_DIR}")
        endif()
        set(TEMP_ROOT "${TBB_DIR}")
        while(NOT EXISTS "${TEMP_ROOT}/include")
            get_filename_component(TEMP_ROOT_PARENT ${TEMP_ROOT} PATH)
            if(TEMP_ROOT_PARENT STREQUAL TEMP_ROOT)
                # to prevent recursion
                message(FATAL_ERROR "${TBB_DIR} does not contain 'include' folder. Please, unset TBB_DIR")
            endif()
            set(TEMP_ROOT "${TEMP_ROOT_PARENT}")
        endwhile()
        set(TBBROOT ${TEMP_ROOT})
    endif()

    if(WIN32 AND X86_64)
        # TODO: add target_path to be platform specific as well, to avoid following if
        # build oneTBB 2021.2.1 with Visual Studio 2019 (MSVC 14.21)
        RESOLVE_DEPENDENCY(TBB
                ARCHIVE_WIN "oneapi-tbb-2021.2.5-win-pdb.zip"
                TARGET_PATH "${TEMP}/tbb"
                ENVIRONMENT "TBBROOT"
                SHA256 "8acb2bd35769aac20597bc0ce1f7608e993b48dee18ee1cc144a05406b6b2613"
                USE_NEW_LOCATION TRUE)
    elseif(ANDROID AND X86_64)
        RESOLVE_DEPENDENCY(TBB
                ARCHIVE_ANDROID "tbb2020_20200404_android.tgz"
                TARGET_PATH "${TEMP}/tbb"
                ENVIRONMENT "TBBROOT"
                SHA256 "f42d084224cc2d643314bd483ad180b081774608844000f132859fca3e9bf0ce"
                USE_NEW_LOCATION TRUE)
    elseif(LINUX AND X86_64 AND OPENVINO_GNU_LIBC AND OV_LIBC_VERSION VERSION_GREATER_EQUAL 2.17)
        # build oneTBB 2021.2.1 with gcc 4.8 (glibc 2.17)
        RESOLVE_DEPENDENCY(TBB
                ARCHIVE_LIN "oneapi-tbb-2021.13.0-lin-release.tgz"
                TARGET_PATH "${TEMP}/tbb"
                ENVIRONMENT "TBBROOT"
                SHA256 "fd2e889323cd5458be750eefdc026ce6791c723fae60b146c2511a5caeaf01c5"
                USE_NEW_LOCATION TRUE)
    elseif(YOCTO_AARCH64)
        RESOLVE_DEPENDENCY(TBB
                ARCHIVE_LIN "keembay/tbb2020_38404_kmb_lic.tgz"
                TARGET_PATH "${TEMP}/tbb_yocto"
                ENVIRONMENT "TBBROOT"
                SHA256 "321261ff2eda6d4568a473cb883262bce77a93dac599f7bd65d2918bdee4d75b"
                USE_NEW_LOCATION TRUE)
    elseif(APPLE AND X86_64)
        # build oneTBB 2021.2.1 with OS version 11.4
        RESOLVE_DEPENDENCY(TBB
                ARCHIVE_MAC "oneapi-tbb-2021.13.0-mac-canary.tgz"
                TARGET_PATH "${TEMP}/tbb"
                ENVIRONMENT "TBBROOT"
                SHA256 "f26a8ae579c4e843781b139c6b74325ae48b58cb2a7a31a0982acda5343f0dd8"
                USE_NEW_LOCATION TRUE)
    elseif(WIN32 AND AARCH64)
        # build oneTBB 2021.2.1 with Visual Studio 2022 (MSVC 14.35)
        RESOLVE_DEPENDENCY(TBB
                ARCHIVE_WIN "oneapi-tbb-2021.2.5-win-arm64-trim.zip"
                TARGET_PATH "${TEMP}/tbb"
                ENVIRONMENT "TBBROOT"
                SHA256 "c26b7593e1808c2dd15a768cd4ea1ee14aa0aa2dacb210b86e326ab7960d2473"
                USE_NEW_LOCATION TRUE)
    elseif(LINUX AND AARCH64 AND OPENVINO_GNU_LIBC AND OV_LIBC_VERSION VERSION_GREATER_EQUAL 2.17)
        # build oneTBB with glibc 2.17
        RESOLVE_DEPENDENCY(TBB
                ARCHIVE_LIN "oneapi-tbb-2021.13.0-lin-arm64-release.tgz"
                TARGET_PATH "${TEMP}/tbb"
                ENVIRONMENT "TBBROOT"
                SHA256 "6e1106735714600474440c134df25b40a225d40b44c2102d7ff23e0482834faa"
                USE_NEW_LOCATION TRUE)
    elseif(APPLE AND AARCH64)
        # build oneTBB with export MACOSX_DEPLOYMENT_TARGET=11.0
        RESOLVE_DEPENDENCY(TBB
                ARCHIVE_MAC "oneapi-tbb-2021.13.0-mac-arm64-canary.tgz"
                TARGET_PATH "${TEMP}/tbb"
                ENVIRONMENT "TBBROOT"
                SHA256 "fb4be1dd03044a97475c45a0cf4576e502b4b64048e98e019520b0720fc255aa"
                USE_NEW_LOCATION TRUE)
    else()
        message(WARNING "Prebuilt TBB is not available on current platform")
    endif()

    update_deps_cache(TBBROOT "${TBB}" "Path to TBB root folder")
    if(EXISTS "${TBBROOT}/lib/cmake/TBB/TBBConfig.cmake")
        # oneTBB case
        update_deps_cache(TBB_DIR "${TBBROOT}/lib/cmake/TBB" "Path to TBB cmake folder")
    elseif(EXISTS "${TBBROOT}/lib/cmake/tbb/TBBConfig.cmake")
        # oneTBB release package version less than 2021.6.0
        # or TBB from oneAPI package
        update_deps_cache(TBB_DIR "${TBBROOT}/lib/cmake/tbb" "Path to TBB cmake folder")
    elseif(EXISTS "${TBBROOT}/lib64/cmake/TBB/TBBConfig.cmake")
        # 64-bits oneTBB case
        update_deps_cache(TBB_DIR "${TBBROOT}/lib64/cmake/TBB" "Path to TBB cmake folder")
    elseif(EXISTS "${TBBROOT}/cmake/TBBConfig.cmake")
        # custom downloaded or user provided TBB
        update_deps_cache(TBB_DIR "${TBBROOT}/cmake" "Path to TBB cmake folder")
    else()
        message(WARNING "Failed to find TBBConfig.cmake in ${TBBROOT} tree. Custom TBBConfig.cmake will be used")
    endif()

    debug_message(STATUS "tbb=" ${TBB})
    debug_message(STATUS "tbb_dir=" ${TBB_DIR})
    debug_message(STATUS "tbbroot=" ${TBBROOT})
endfunction()

## TBBBind_2_5 package
unset(_ov_download_tbbbind_2_5_done CACHE)

#
# The function downloads static prebuilt TBBBind_2_5 package
# NOTE: the function should be called only we have TBB with version less 2021
#
function(ov_download_tbbbind_2_5)
    if(_ov_download_tbbbind_2_5_done OR NOT ENABLE_TBBBIND_2_5)
        return()
    endif()
    set(_ov_download_tbbbind_2_5_done ON CACHE INTERNAL "Whether prebuilt TBBBind_2_5 is already downloaded")

    reset_deps_cache(TBBBIND_2_5_ROOT TBBBIND_2_5_DIR)

    if(DEFINED ENV{THIRDPARTY_SERVER_PATH})
        set(IE_PATH_TO_DEPS "$ENV{THIRDPARTY_SERVER_PATH}")
    elseif(DEFINED THIRDPARTY_SERVER_PATH)
        set(IE_PATH_TO_DEPS "${THIRDPARTY_SERVER_PATH}")
    endif()

    if(WIN32 AND X86_64)
        RESOLVE_DEPENDENCY(TBBBIND_2_5
                ARCHIVE_WIN "tbbbind_2_5_static_win_v2.zip"
                TARGET_PATH "${TEMP}/tbbbind_2_5"
                ENVIRONMENT "TBBBIND_2_5_ROOT"
                SHA256 "49ae93b13a13953842ff9ae8d01681b269b5b0bc205daf18619ea9a828c44bee"
                USE_NEW_LOCATION TRUE)
    elseif(LINUX AND X86_64 AND OPENVINO_GNU_LIBC AND OV_LIBC_VERSION VERSION_GREATER_EQUAL 2.17)
        RESOLVE_DEPENDENCY(TBBBIND_2_5
                ARCHIVE_LIN "tbbbind_2_5_static_lin_v4.tgz"
                TARGET_PATH "${TEMP}/tbbbind_2_5"
                ENVIRONMENT "TBBBIND_2_5_ROOT"
                SHA256 "4ebf30246530795f066fb9616e6707c6b17be7a65d29d3518b578a769dd54eea"
                USE_NEW_LOCATION TRUE)
    else()
        # TMP: for Apple Silicon TBB does not provide TBBBind
        if(NOT (APPLE AND AARCH64))
            message(WARNING "prebuilt TBBBIND_2_5 is not available.
Build oneTBB from sources and set TBBROOT environment var before OpenVINO cmake configure")
        endif()
        return()
    endif()

    update_deps_cache(TBBBIND_2_5_ROOT "${TBBBIND_2_5}" "Path to TBBBIND_2_5 root folder")
    update_deps_cache(TBBBIND_2_5_DIR "${TBBBIND_2_5}/cmake" "Path to TBBBIND_2_5 cmake folder")
endfunction()
