# Copyright (C) 2018-2023 Intel Corporation
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
            set(_imported_configs RELEASE NONE)
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
        if(CMAKE_TOOLCHAIN_FILE MATCHES "conan_toolchain.cmake" OR CONAN_EXPORTED)
            set(_ov_minimal_tbb_version 2021.0)
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
            if(ENABLE_SYSTEM_TBB AND PkgConfig_FOUND)
                macro(_ov_pkg_config_tbb_unset)
                    # unset since it affects OpenVINOConfig.cmake.in
                    unset(tbb_FOUND)
                    unset(tbb_FOUND CACHE)
                endmacro()
                pkg_search_module(tbb QUIET
                                  IMPORTED_TARGET
                                  # we need to set GLOBAL in order to create ALIAS later
                                  # ALIAS creation for non-GLOBAL targets is available since cmake 3.18
                                  ${OV_PkgConfig_VISILITY}
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
    elseif(target_type STREQUAL "EXECUTABLE" OR target_type STREQUAL "OBJECT_LIBRARY" OR
           target_type STREQUAL "MODULE_LIBRARY")
        set(LINK_TYPE "PRIVATE")
        set(COMPILE_DEF_TYPE "PUBLIC")
    elseif(target_type STREQUAL "STATIC_LIBRARY")
        # Affected libraries: inference_engine_s, openvino_gapi_preproc_s
        # they don't have TBB in public headers => PRIVATE
        set(LINK_TYPE "PRIVATE")
        set(COMPILE_DEF_TYPE "PUBLIC")
    elseif(target_type STREQUAL "SHARED_LIBRARY")
        # Affected libraries: inference_engine only
        # TODO: why TBB propogates its headers to inference_engine?
        set(LINK_TYPE "PRIVATE")
        set(COMPILE_DEF_TYPE "PUBLIC")
    else()
        message(WARNING "Unknown target type")
    endif()

    function(_ov_target_link_libraries TARGET_NAME LINK_TYPE)
        target_link_libraries(${TARGET_NAME} ${LINK_TYPE} ${ARGN})

        # include directories as SYSTEM
        foreach(library IN LISTS ARGN)
            if(TARGET ${library})
                get_target_property(include_directories ${library} INTERFACE_INCLUDE_DIRECTORIES)
                if(include_directories)
                    foreach(include_directory IN LISTS include_directories)
                        # cannot include /usr/include headers as SYSTEM
                        if(NOT "${include_directory}" MATCHES "^/usr.*$")
                            target_include_directories(${TARGET_NAME} SYSTEM
                                ${LINK_TYPE} $<BUILD_INTERFACE:${include_directory}>)
                        else()
                            set(_system_library ON)
                        endif()
                    endforeach()
                endif()
            endif()
        endforeach()

        if(_system_library)
            # if we deal with system library (e.i. having /usr/include as header paths)
            # we cannot use SYSTEM key word for such library
            set_target_properties(${TARGET_NAME} PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)
        endif()
    endfunction()

    set(IE_THREAD_DEFINE "IE_THREAD_SEQ")
    set(OV_THREAD_DEFINE "OV_THREAD_SEQ")

    if (THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO")
        if (TBB_FOUND)
            set(IE_THREAD_DEFINE "IE_THREAD_TBB")
            set(OV_THREAD_DEFINE "OV_THREAD_TBB")
            _ov_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} TBB::tbb)
            target_compile_definitions(${TARGET_NAME} ${COMPILE_DEF_TYPE} TBB_PREVIEW_WAITING_FOR_WORKERS=1)
        else ()
            set(THREADING "SEQ" PARENT_SCOPE)
            message(WARNING "TBB was not found by the configured TBB_DIR path.\
                             SEQ method will be used for ${TARGET_NAME}")
        endif ()
    elseif (THREADING STREQUAL "OMP")
        if (WIN32)
            set(omp_lib_name libiomp5md)
        else ()
            set(omp_lib_name iomp5)
        endif ()

        if (NOT OpenVINO_SOURCE_DIR)
            # TODO: dead code since ov_parallel.cmake is not used outside of OpenVINO build
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
            message(STATUS "OMP Release lib: ${OMP_LIBRARIES_RELEASE}")
            if (NOT LINUX)
                find_library(OMP_LIBRARIES_DEBUG ${omp_lib_name} ${lib_dbg_path} NO_DEFAULT_PATH)
                if (OMP_LIBRARIES_DEBUG)
                    message(STATUS "OMP Debug lib: ${OMP_LIBRARIES_DEBUG}")
                else ()
                    message(WARNING "OMP Debug binaries are missed.")
                endif ()
            endif ()
        endif ()

        if (NOT OMP_LIBRARIES_RELEASE)
            message(WARNING "Intel OpenMP not found. Intel OpenMP support will be disabled. ${IE_THREAD_DEFINE} is defined")
            set(THREADING "SEQ" PARENT_SCOPE)
        else ()
            set(IE_THREAD_DEFINE "IE_THREAD_OMP")
            set(OV_THREAD_DEFINE "OV_THREAD_OMP")

            if (WIN32)
                target_compile_options(${TARGET_NAME} ${LINK_TYPE} ${OpenMP_CXX_FLAGS} /openmp)
                target_compile_options(${TARGET_NAME} ${LINK_TYPE} ${OpenMP_CXX_FLAGS} /Qopenmp)
                _ov_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} "-nodefaultlib:vcomp")
            else()
                target_compile_options(${TARGET_NAME} ${LINK_TYPE} ${OpenMP_CXX_FLAGS} -fopenmp)
            endif ()

            # Debug binaries are optional.
            if (OMP_LIBRARIES_DEBUG AND NOT LINUX)
                if (WIN32)
                    _ov_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} "$<$<CONFIG:DEBUG>:${OMP_LIBRARIES_DEBUG}>;$<$<NOT:$<CONFIG:DEBUG>>:${OMP_LIBRARIES_RELEASE}>")
                else()
                    # TODO: handle multi-config generators case
                    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
                        _ov_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} ${OMP_LIBRARIES_DEBUG})
                    else()
                        _ov_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} ${OMP_LIBRARIES_RELEASE})
                    endif ()
                endif ()
            else ()
                # Link Release library to all configurations.
                _ov_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} ${OMP_LIBRARIES_RELEASE})
            endif ()
        endif ()
    endif ()

    target_compile_definitions(${TARGET_NAME} ${COMPILE_DEF_TYPE} -DIE_THREAD=${IE_THREAD_DEFINE})
    target_compile_definitions(${TARGET_NAME} ${COMPILE_DEF_TYPE} -DOV_THREAD=${OV_THREAD_DEFINE})

    if (NOT THREADING STREQUAL "SEQ")
        find_package(Threads REQUIRED)
        _ov_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} Threads::Threads)
    endif()
endfunction(ov_set_threading_interface_for)

# deprecated

function(set_ie_threading_interface_for TARGET_NAME)
    message(WARNING "'set_ie_threading_interface_for' is deprecated. Please use 'ov_set_threading_interface_for' instead.")
    ov_set_threading_interface_for(${TARGET_NAME})
endfunction(set_ie_threading_interface_for)
