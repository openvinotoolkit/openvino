# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)

# TBB library is stored on ftp
include(dependency_solver)

# "MKL-DNN library based on OMP or TBB or Sequential implementation: TBB|OMP|SEQ"
if(ARM)
    set(THREADING_DEFAULT "SEQ")
else()
    set(THREADING_DEFAULT "TBB")
endif()

set(THREADING "${THREADING_DEFAULT}" CACHE STRING "Threading")

set_property(CACHE THREADING PROPERTY STRINGS "TBB" "TBB_AUTO" "OMP" "SEQ")

list (APPEND IE_OPTIONS THREADING)

if (NOT THREADING STREQUAL "TBB" AND
    NOT THREADING STREQUAL "TBB_AUTO" AND
    NOT THREADING STREQUAL "OMP" AND
    NOT THREADING STREQUAL "SEQ")
    message(FATAL_ERROR "THREADING should be set to TBB, TBB_AUTO, OMP or SEQ. Default option is ${THREADING_DEFAULT}")
endif()

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

if (THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO")
    find_package(TBB COMPONENTS tbb tbbmalloc)
    if (TBB_FOUND)
        if (${TBB_VERSION} VERSION_LESS 2020)
            ext_message(WARNING "TBB version is less than OpenVINO recommends to use.\
                                 Some TBB related features like NUMA-aware tbb::task_arena\
                                 execution will be disabled.")
        endif()
    else ()
        ext_message(WARNING "TBB was not found by the configured TBB_DIR/TBBROOT path. \
                             SEQ method will be used.")
    endif ()
endif()

function(set_ie_threading_interface_for TARGET_NAME)
    get_target_property(target_type ${TARGET_NAME} TYPE)
    if(target_type STREQUAL "INTERFACE_LIBRARY")
        set(LINK_TYPE "INTERFACE")
    else()
        set(LINK_TYPE "PUBLIC")
    endif()

    function(ov_target_link_libraries TARGET_NAME LINK_TYPE)
        if(CMAKE_VERSION VERSION_LESS "3.12.0")
            if(NOT target_type STREQUAL "OBJECT_LIBRARY")
                target_link_libraries(${TARGET_NAME} ${LINK_TYPE} ${ARGN})
            else()
                # Object library may not link to anything.
                # To add interface include directories and compile options explicitly.
                foreach(ITEM IN LISTS ARGN)
                    if(TARGET ${ITEM})
                        get_target_property(compile_options ${ITEM} INTERFACE_COMPILE_OPTIONS)
                        if (compile_options)
                            target_compile_options(${TARGET_NAME} ${LINK_TYPE} ${compile_options})
                        endif()
                        get_target_property(compile_definitions ${ITEM} INTERFACE_COMPILE_DEFINITIONS)
                        if (compile_options)
                            target_compile_definitions(${TARGET_NAME} ${LINK_TYPE} ${compile_definitions})
                        endif()
                    endif()
                endforeach()
            endif()
        else()
            target_link_libraries(${TARGET_NAME} ${LINK_TYPE} ${ARGN})
        endif()

        # include directories as SYSTEM
        foreach(library IN LISTS ARGN)
            if(TARGET ${library})
                get_target_property(include_directories ${library} INTERFACE_INCLUDE_DIRECTORIES)
                if(include_directories)
                    target_include_directories(${TARGET_NAME} SYSTEM BEFORE ${LINK_TYPE} ${include_directories})
                endif()
            endif()
        endforeach()
    endfunction()

    set(IE_THREAD_DEFINE "IE_THREAD_SEQ")

    if (THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO")
        if (TBB_FOUND)
            set(IE_THREAD_DEFINE "IE_THREAD_TBB")
            ov_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} ${TBB_IMPORTED_TARGETS})
        else ()
            ext_message(WARNING "TBB was not found by the configured TBB_DIR path. \
                                 SEQ method will be used for ${TARGET_NAME}")
        endif ()
    elseif (THREADING STREQUAL "OMP")
        if (WIN32)
            set(omp_lib_name libiomp5md)
        else ()
            set(omp_lib_name iomp5)
        endif ()

        if (NOT IE_MAIN_SOURCE_DIR)
            if (WIN32)
                set(lib_rel_path ${IE_LIB_REL_DIR})
                set(lib_dbg_path ${IE_LIB_DBG_DIR})
            else ()
                set(lib_rel_path ${IE_EXTERNAL_DIR}/omp/lib)
                set(lib_dbg_path ${lib_rel_path})
            endif ()
        else ()
            set(lib_rel_path ${OMP}/lib)
            set(lib_dbg_path ${lib_rel_path})
        endif ()

        if (NOT OMP_LIBRARIES_RELEASE)
            find_library(OMP_LIBRARIES_RELEASE ${omp_lib_name} ${lib_rel_path} NO_DEFAULT_PATH)
            ext_message(STATUS "OMP Release lib: ${OMP_LIBRARIES_RELEASE}")
            if (NOT LINUX)
                find_library(OMP_LIBRARIES_DEBUG ${omp_lib_name} ${lib_dbg_path} NO_DEFAULT_PATH)
                if (OMP_LIBRARIES_DEBUG)
                    ext_message(STATUS "OMP Debug lib: ${OMP_LIBRARIES_DEBUG}")
                else ()
                    ext_message(WARNING "OMP Debug binaries are missed.")
                endif ()
            endif ()
        endif ()

        if (NOT OMP_LIBRARIES_RELEASE)
            ext_message(WARNING "Intel OpenMP not found. Intel OpenMP support will be disabled. ${IE_THREAD_DEFINE} is defined")
        else ()
            set(IE_THREAD_DEFINE "IE_THREAD_OMP")

            if (WIN32)
                target_compile_options(${TARGET_NAME} ${LINK_TYPE} ${OpenMP_CXX_FLAGS} /openmp)
                target_compile_options(${TARGET_NAME} ${LINK_TYPE} ${OpenMP_CXX_FLAGS} /Qopenmp)
                ov_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} "-nodefaultlib:vcomp")
            else()
                target_compile_options(${TARGET_NAME} ${LINK_TYPE} ${OpenMP_CXX_FLAGS} -fopenmp)
            endif ()

            # Debug binaries are optional.
            if (OMP_LIBRARIES_DEBUG AND NOT LINUX)
                if (WIN32)
                    ov_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} "$<$<CONFIG:DEBUG>:${OMP_LIBRARIES_DEBUG}>;$<$<NOT:$<CONFIG:DEBUG>>:${OMP_LIBRARIES_RELEASE}>")
                else()
                    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
                        ov_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} ${OMP_LIBRARIES_DEBUG})
                    else()
                        ov_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} ${OMP_LIBRARIES_RELEASE})
                    endif ()
                endif ()
            else ()
                # Link Release library to all configurations.
                ov_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} ${OMP_LIBRARIES_RELEASE})
            endif ()
        endif ()

    endif ()

    target_compile_definitions(${TARGET_NAME} ${LINK_TYPE} -DIE_THREAD=${IE_THREAD_DEFINE})

    if (NOT THREADING STREQUAL "SEQ")
        find_package(Threads REQUIRED)
        ov_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} ${CMAKE_THREAD_LIBS_INIT})
    endif()
endfunction(set_ie_threading_interface_for)
