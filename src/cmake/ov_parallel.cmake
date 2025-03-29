# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_package(PkgConfig QUIET)

function(_ov_get_tbb_location tbb_target _tbb_lib_location_var)
    if(NOT TBB_FOUND)
        return()
    endif()

    function(_get_target_location target lib_location_var)
        if(NOT TARGET ${target})
            message(FATAL_ERROR "Internal error: ${target} does not represent a target")
        endif()

        # check that target is imported
        get_target_property(is_imported ${target} IMPORTED)
        if(NOT is_imported)
            return()
        endif()

        get_target_property(_imported_configs ${target} IMPORTED_CONFIGURATIONS)
        if(NOT _imported_configs)
            # if IMPORTED_CONFIGURATIONS property is not set, then set a common list
            set(_imported_configs RELEASE DEBUG NONE)
            if(NOT OV_GENERATOR_MULTI_CONFIG)
                string(TOUPPER ${CMAKE_BUILD_TYPE} _build_type)
                list(APPEND _imported_configs ${_build_type})
            endif()
        endif()

        # generate a list of locations
        foreach(_imported_config IN LISTS _imported_configs)
            list(APPEND _location_properties IMPORTED_LOCATION_${_imported_config})
        endforeach()
        # add some more locations which are used by package managers
        list(APPEND _location_properties IMPORTED_LOCATION)

        foreach(_location_property IN LISTS _location_properties)
            get_target_property(_lib_location ${target} ${_location_property})
            if(_lib_location)
                set(${lib_location_var} "${_lib_location}" PARENT_SCOPE)
                break()
            endif()
        endforeach()
    endfunction()

    macro(_get_target_location_and_return _tbb_target)
        _get_target_location(${_tbb_target} "_tbb_lib_location")
        if(_tbb_lib_location)
            set(${_tbb_lib_location_var} "${_tbb_lib_location}" PARENT_SCOPE)
            return()
        endif()
    endmacro()

    # handle INTERFACE_LINK_LIBRARIES
    get_target_property(_tbb_interface_link_libraries ${tbb_target} INTERFACE_LINK_LIBRARIES)
    # pkg-config can set multiple libraries as interface, need to filter out
    foreach(tbb_lib IN LISTS _tbb_interface_link_libraries)
        # handle cases like in conan: $<$<CONFIG:Release>:CONAN_LIB::onetbb_TBB_tbb_tbb_RELEASE>
        if(${tbb_lib} MATCHES "CONAN_LIB::([A-Za-z0-9_]*)")
            set(tbb_lib_parsed "CONAN_LIB::${CMAKE_MATCH_1}")
            _get_target_location_and_return(${tbb_lib_parsed})
        elseif(tbb_lib MATCHES "${CMAKE_SHARED_LIBRARY_PREFIX}tbb${CMAKE_SHARED_LIBRARY_SUFFIX}")
            # tbb_lib just a full path to a library itself
            set(${_tbb_lib_location_var} "${tbb_lib}" PARENT_SCOPE)
            return()
        endif()
    endforeach()

    # handle case of usual target
    _get_target_location_and_return(${tbb_target})

   message(FATAL_ERROR "Failed to detect TBB library location")
endfunction()

macro(ov_find_package_tbb)
    if(THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO" AND NOT TBB_FOUND)
        # conan generates TBBConfig.cmake files, which follows cmake's
        # SameMajorVersion scheme, while TBB itself follows AnyNewerVersion one
        # see https://cmake.org/cmake/help/latest/module/CMakePackageConfigHelpers.html#generating-a-package-version-file
        set(PKG_CONFIG_SEARCH ON)
        if(CMAKE_TOOLCHAIN_FILE MATCHES "conan_toolchain.cmake" OR CONAN_EXPORTED)
            set(_ov_minimal_tbb_version 2021.0)
        elseif(LINUX AND AARCH64)
            # CVS-126984: system TBB is not very stable on Linux ARM64
            set(_ov_minimal_tbb_version 2021.0)
            # on Ubuntu22.04, tbb2020 can be installed by "apt install libtbb2-dev",
            # after installation, TBB_VERSION is missed in tbb.pc,
            # so here skip pkg_search_module for tbb to avoid using TBB 2020
            # that does not meet the minimun version number requirements.
            set(PKG_CONFIG_SEARCH OFF)
        else()
            set(_ov_minimal_tbb_version 2017.0)
        endif()

        if(NOT ENABLE_SYSTEM_TBB)
            if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
                set(_no_cmake_install_prefix NO_CMAKE_INSTALL_PREFIX)
            endif()

            # Note, we explicitly:
            # don't set NO_CMAKE_PATH to allow -DTBB_DIR=XXX
            # don't set NO_CMAKE_ENVIRONMENT_PATH to allow env TBB_DIR=XXX
            set(_find_package_no_args NO_PACKAGE_ROOT_PATH
                                      NO_SYSTEM_ENVIRONMENT_PATH
                                      NO_CMAKE_PACKAGE_REGISTRY
                                      NO_CMAKE_SYSTEM_PATH
                                      ${_no_cmake_install_prefix}
                                      NO_CMAKE_SYSTEM_PACKAGE_REGISTRY)

            unset(_no_cmake_install_prefix)
        endif()

        find_package(TBB ${_ov_minimal_tbb_version} QUIET COMPONENTS tbb tbbmalloc
                     ${_find_package_no_args})

        if(NOT TBB_FOUND)
            # remove invalid TBB_DIR=TBB_DIR-NOTFOUND from cache
            unset(TBB_DIR CACHE)
            unset(TBB_DIR)

            # try tbb.pc from system
            if(ENABLE_SYSTEM_TBB AND PkgConfig_FOUND AND PKG_CONFIG_SEARCH)
                macro(_ov_pkg_config_tbb_unset)
                    # unset since it affects OpenVINOConfig.cmake.in
                    unset(tbb_FOUND)
                    unset(tbb_FOUND CACHE)
                endmacro()
                pkg_search_module(tbb QUIET
                                  IMPORTED_TARGET
                                  # we need to set GLOBAL in order to create ALIAS later
                                  # ALIAS creation for non-GLOBAL targets is available since cmake 3.18
                                  ${OV_PkgConfig_VISIBILITY}
                                  tbb)
                if(tbb_FOUND)
                    # parse version
                    string(REGEX REPLACE "~.*" "" tbb_VERSION_PATCHED "${tbb_VERSION}")
                    if(tbb_VERSION_PATCHED AND tbb_VERSION_PATCHED VERSION_LESS _ov_minimal_tbb_version)
                        _ov_pkg_config_tbb_unset()
                        message(WARNING "Found TBB ${tbb_VERSION} via ${PKG_CONFIG_EXECUTABLE} while OpenVINO requies ${_ov_minimal_tbb_version} at least")
                    elseif(TARGET PkgConfig::tbb)
                        add_library(TBB::tbb ALIAS PkgConfig::tbb)
                        set(TBB_VERSION ${tbb_VERSION})
                        set(TBB_FOUND ${tbb_FOUND})

                        # note: for python wheels we need to find and install tbbmalloc as well
                        _ov_get_tbb_location(PkgConfig::tbb tbb_loc)
                        string(REPLACE "tbb" "tbbmalloc" tbbmalloc_loc "${tbb_loc}")
                        if(EXISTS "${tbbmalloc_loc}")
                            add_library(TBB::tbbmalloc SHARED IMPORTED)
                            set_target_properties(TBB::tbbmalloc PROPERTIES IMPORTED_LOCATION ${tbbmalloc_loc})
                        endif()

                        message(STATUS "${PKG_CONFIG_EXECUTABLE}: tbb (${tbb_VERSION}) is found at ${tbb_PREFIX}")
                    else()
                        _ov_pkg_config_tbb_unset()

                        if(CPACK_GENERATOR STREQUAL "^(DEB|RPM|CONDA-FORGE|BREW|CONAN|VCPKG)$")
                            # package managers require system TBB
                            set(message_type FATAL_ERROR)
                        else()
                            set(message_type WARNING)
                        endif()
                        message(${message_type} "cmake v${CMAKE_VERSION} contains bug in function 'pkg_search_module', need to update to at least v3.16.0 version")
                    endif()
                endif()
            endif()

            if(NOT TBB_FOUND)
                # system TBB failed to be found
                set(ENABLE_SYSTEM_TBB OFF CACHE BOOL "" FORCE)

                # TBB on system is not found, download prebuilt one
                # if TBBROOT env variable is not defined
                ov_download_tbb()

                # fallback variant for TBB 2018 and older where TBB have not had cmake interface
                if(DEFINED TBBROOT OR DEFINED ENV{TBBROOT})
                    # note: if TBB older than 2017.0 is passed, cmake will skip it and THREADING=SEQ will be used
                    set(_tbb_paths PATHS "${OpenVINODeveloperScripts_DIR}/tbb")
                endif()

                # try to find one more time
                find_package(TBB ${_ov_minimal_tbb_version} QUIET COMPONENTS tbb tbbmalloc
                             # TBB_DIR can be provided by ov_download_tbb
                             HINTS ${TBB_DIR}
                             ${_tbb_paths}
                             ${_find_package_no_args}
                             NO_CMAKE_PATH
                             NO_CMAKE_ENVIRONMENT_PATH)
            endif()
        endif()

        # WA for oneTBB: it does not define TBB_IMPORTED_TARGETS
        if(TBB_FOUND AND NOT TBB_IMPORTED_TARGETS)
            foreach(target TBB::tbb TBB::tbbmalloc TBB::tbbbind_2_5)
                if(TARGET ${target})
                    list(APPEND TBB_IMPORTED_TARGETS ${target})
                endif()
            endforeach()

            if(WIN32 AND TARGET TBB::tbbbind_2_5)
                # some package managers provide hwloc.pc file within installation package
                # let's try it first
                if(PkgConfig_FOUND)
                    pkg_search_module(HWLOC QUIET
                                      IMPORTED_TARGET
                                      hwloc)
                endif()

                if(TARGET PkgConfig::HWLOC)
                    # dependency is satisfied
                    add_library(HWLOC::hwloc_2_5 ALIAS PkgConfig::HWLOC)
                else()
                    # Add HWLOC::hwloc_2_5 target to check via ApiValidator
                    get_target_property(imported_configs TBB::tbbbind_2_5 IMPORTED_CONFIGURATIONS)
                    foreach(imported_config RELEASE RELWITHDEBINFO DEBUG)
                        if(imported_config IN_LIST imported_configs)
                            get_target_property(TBBbind_location TBB::tbbbind_2_5 IMPORTED_LOCATION_${imported_config})
                            get_filename_component(TBB_dir "${TBBbind_location}" DIRECTORY)
                            break()
                        endif()
                    endforeach()

                    set(hwloc_dll_name "${CMAKE_SHARED_LIBRARY_PREFIX}hwloc${CMAKE_SHARED_LIBRARY_SUFFIX}")
                    find_file(HWLOC_DLL NAMES ${hwloc_dll_name} PATHS "${TBB_dir}" DOC "Path to hwloc.dll")

                    if(HWLOC_DLL)
                        add_library(HWLOC::hwloc_2_5 SHARED IMPORTED)
                        set_property(TARGET HWLOC::hwloc_2_5 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
                        set_target_properties(HWLOC::hwloc_2_5 PROPERTIES IMPORTED_LOCATION_RELEASE "${HWLOC_DLL}")
                    endif()
                endif()
            endif()
        endif()

        if(NOT TBB_FOUND)
            set(THREADING "SEQ")
            set(ENABLE_TBBBIND_2_5 OFF)
            message(WARNING "TBB was not found by the configured TBB_DIR / TBBROOT path.\
                             SEQ method will be used.")
        else()
            message(STATUS "TBB (${TBB_VERSION}) is found at ${TBB_DIR}")
        endif()

        unset(_find_package_no_args)
    endif()
endmacro()

macro(ov_find_package_openmp)
    # check whether the compiler supports OpenMP at all
    find_package(OpenMP)

    # check if Intel OpenMP is downloaded and override system library
    if(THREADING STREQUAL "OMP")
        if(INTEL_OMP)
            if(WIN32)
                set(iomp_lib_name libiomp5md)
            else()
                set(iomp_lib_name iomp5)
            endif()

            set(lib_rel_path ${INTEL_OMP}/lib)
            set(lib_dbg_path ${lib_rel_path})

            find_library(INTEL_OMP_LIBRARIES_RELEASE NAMES ${iomp_lib_name} PATHS ${lib_rel_path} REQUIRED NO_DEFAULT_PATH)
            list(APPEND iomp_imported_configurations RELEASE)

            # try to find debug libraries as well
            if(WIN32)
                set(iomp_link_flags
                    # avoid linking default OpenMP
                    # https://learn.microsoft.com/en-us/cpp/parallel/openmp/reference/openmp-library-reference?view=msvc-170
                    INTERFACE_LINK_OPTIONS -nodefaultlib:vcomp)

                # location of .lib file
                string(REPLACE ".dll" ".lib" INTEL_OMP_IMPLIB_RELEASE "${INTEL_OMP_LIBRARIES_RELEASE}")
                set(iomp_implib_location_release
                    IMPORTED_IMPLIB_RELEASE "${INTEL_OMP_IMPLIB_RELEASE}")

                find_library(INTEL_OMP_LIBRARIES_DEBUG NAMES ${iomp_lib_name} PATHS ${lib_dbg_path} NO_DEFAULT_PATH)
                if(INTEL_OMP_LIBRARIES_DEBUG)
                    string(REPLACE ".dll" ".lib" INTEL_OMP_IMPLIB_DEBUG "${INTEL_OMP_LIBRARIES_DEBUG}")

                    list(APPEND iomp_imported_configurations DEBUG)
                    set(iomp_imported_locations_debug
                        IMPORTED_LOCATION_DEBUG "${INTEL_OMP_LIBRARIES_DEBUG}"
                        IMPORTED_IMPLIB_DEBUG "${INTEL_OMP_IMPLIB_DEBUG}")
                else()
                    set(iomp_map_imported_debug_configuration MAP_IMPORTED_CONFIG_DEBUG Release)
                    message(WARNING "OMP Debug binaries are missed.")
                endif()
            endif()

            # create imported target
            if(NOT TARGET IntelOpenMP::OpenMP_CXX)
                add_library(IntelOpenMP::OpenMP_CXX SHARED IMPORTED)
                set_property(TARGET IntelOpenMP::OpenMP_CXX APPEND PROPERTY
                    IMPORTED_CONFIGURATIONS ${iomp_imported_configurations})
                set_target_properties(IntelOpenMP::OpenMP_CXX PROPERTIES
                    # reuse standard OpenMP compiler flags
                    INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS}
                    # location of release library (both .dll and lib.)
                    IMPORTED_LOCATION_RELEASE "${INTEL_OMP_LIBRARIES_RELEASE}"
                    ${iomp_implib_location_release}
                    # the same for debug libs
                    ${iomp_imported_locations_debug}
                    # linker flags to override system OpenMP
                    ${iomp_link_flags}
                    # map imported configurations if required
                    ${iomp_map_imported_debug_configuration}
                    MAP_IMPORTED_CONFIG_MINSIZEREL Release
                    MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
            endif()
        endif()

        # falling back to system OpenMP then
        if(NOT TARGET IntelOpenMP::OpenMP_CXX)
            ov_target_link_libraries_as_system(${TARGET_NAME} ${LINK_TYPE} OpenMP::OpenMP_CXX)
        endif()
    endif()
endmacro()

function(ov_set_threading_interface_for TARGET_NAME)
    if(THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO" AND NOT TBB_FOUND)
        # find TBB
        ov_find_package_tbb()

        # set variables to parent scope to prevent multiple invocations of find_package(TBB)
        # at the same CMakeLists.txt; invocations in different directories are allowed
        set(THREADING ${THREADING} PARENT_SCOPE)
        set(TBB_FOUND ${TBB_FOUND} PARENT_SCOPE)
        set(TBB_IMPORTED_TARGETS ${TBB_IMPORTED_TARGETS} PARENT_SCOPE)
        set(TBB_VERSION ${TBB_VERSION} PARENT_SCOPE)
        set(TBB_DIR ${TBB_DIR} PARENT_SCOPE)
        set(ENABLE_SYSTEM_TBB ${ENABLE_SYSTEM_TBB} PARENT_SCOPE)
        set(ENABLE_TBBBIND_2_5 ${ENABLE_TBBBIND_2_5} PARENT_SCOPE)
    endif()

    get_target_property(target_type ${TARGET_NAME} TYPE)

    if(target_type STREQUAL "INTERFACE_LIBRARY")
        set(LINK_TYPE "INTERFACE")
        set(COMPILE_DEF_TYPE "INTERFACE")
    elseif(target_type MATCHES "^(EXECUTABLE|OBJECT_LIBRARY|MODULE_LIBRARY)$")
        set(LINK_TYPE "PRIVATE")
        set(COMPILE_DEF_TYPE "PRIVATE")
    elseif(target_type STREQUAL "STATIC_LIBRARY")
        # Affected libraries: openvino_runtime_s
        # they don't have TBB in public headers => PRIVATE
        set(LINK_TYPE "PRIVATE")
        set(COMPILE_DEF_TYPE "PUBLIC")
    elseif(target_type STREQUAL "SHARED_LIBRARY")
        # Affected libraries: 'openvino' only
        set(LINK_TYPE "PRIVATE")
        set(COMPILE_DEF_TYPE "PUBLIC")
    else()
        message(WARNING "Unknown target type")
    endif()

    set(_ov_thread_define "OV_THREAD_SEQ")

    if(NOT TARGET openvino::threading)
        add_library(openvino_threading INTERFACE)
        add_library(openvino::threading ALIAS openvino_threading)
    endif()

    if(THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO")
        if(TBB_FOUND)
            set(_ov_thread_define "OV_THREAD_TBB")
            set(_ov_threading_lib TBB::tbb)
        else()
            set(THREADING "SEQ" PARENT_SCOPE)
            message(WARNING "TBB was not found by the configured TBB_DIR path.\
                             SEQ method will be used for ${TARGET_NAME}")
        endif()
    elseif(THREADING STREQUAL "OMP")
        ov_find_package_openmp()

        if(TARGET IntelOpenMP::OpenMP_CXX)
            set(_ov_thread_define "OV_THREAD_OMP")
            set(_ov_threading_lib IntelOpenMP::OpenMP_CXX)
        elseif(TARGET OpenMP::OpenMP_CXX)
            set(_ov_thread_define "OV_THREAD_OMP")
            set(_ov_threading_lib OpenMP::OpenMP_CXX)
        else()
            message(FATAL_ERROR "Internal error: OpenMP is not supported by compiler. Switch to SEQ should be performed before")
        endif()
    endif()

    if(_ov_threading_lib)
        # populate properties of openvino::threading
        set_target_properties(openvino_threading PROPERTIES INTERFACE_LINK_LIBRARIES ${_ov_threading_lib})

        # perform linkage with target
        ov_target_link_libraries_as_system(${TARGET_NAME} ${LINK_TYPE} ${_ov_threading_lib})
    endif()

    target_compile_definitions(${TARGET_NAME} ${COMPILE_DEF_TYPE} OV_THREAD=${_ov_thread_define})

    if(NOT THREADING STREQUAL "SEQ")
        find_package(Threads REQUIRED)
        ov_target_link_libraries_as_system(${TARGET_NAME} ${LINK_TYPE} Threads::Threads)
    endif()
endfunction(ov_set_threading_interface_for)
