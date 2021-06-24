# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(set_ie_threading_interface_for TARGET_NAME)
    macro(ext_message TRACE_LEVEL)
         if (TRACE_LEVEL STREQUAL FATAL_ERROR)
             if(InferenceEngine_FIND_REQUIRED)
                 message(FATAL_ERROR "${ARGN}")
             elseif(NOT InferenceEngine_FIND_QUIETLY)
                 message(WARNING "${ARGN}")
             endif()
             return()
         elseif(NOT InferenceEngine_FIND_QUIETLY)
             message(${TRACE_LEVEL} "${ARGN}")
         endif ()
    endmacro()

    if (THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO" AND NOT TBB_FOUND)
        if(IEDevScripts_DIR)
            find_package(TBB COMPONENTS tbb tbbmalloc
                         PATHS IEDevScripts_DIR
                         NO_CMAKE_FIND_ROOT_PATH
                         NO_DEFAULT_PATH)
        else()
            find_dependency(TBB COMPONENTS tbb tbbmalloc)
        endif()
        set(TBB_FOUND ${TBB_FOUND} PARENT_SCOPE)
        set(TBB_IMPORTED_TARGETS ${TBB_IMPORTED_TARGETS} PARENT_SCOPE)
        set(TBB_VERSION ${TBB_VERSION} PARENT_SCOPE)
        if (NOT TBB_FOUND)
            ext_message(WARNING "TBB was not found by the configured TBB_DIR/TBBROOT path.\
                                SEQ method will be used.")
        endif ()
    endif()

    get_target_property(target_type ${TARGET_NAME} TYPE)

    if(target_type STREQUAL "INTERFACE_LIBRARY")
        set(LINK_TYPE "INTERFACE")
    elseif(target_type STREQUAL "EXECUTABLE" OR target_type STREQUAL "OBJECT_LIBRARY" OR
           target_type STREQUAL "MODULE_LIBRARY")
        set(LINK_TYPE "PRIVATE")
    elseif(target_type STREQUAL "STATIC_LIBRARY")
        # Affected libraries: inference_engine_s, inference_engine_preproc_s
        # they don't have TBB in public headers => PRIVATE
        set(LINK_TYPE "PRIVATE")
    elseif(target_type STREQUAL "SHARED_LIBRARY")
        # Affected libraries: inference_engine only
        # TODO: why TBB propogates its headers to inference_engine?
        set(LINK_TYPE "PRIVATE")
    else()
        ext_message(WARNING "Unknown target type")
    endif()

    function(ie_target_link_libraries TARGET_NAME LINK_TYPE)
        if(CMAKE_VERSION VERSION_LESS "3.12.0")
            if(NOT target_type STREQUAL "OBJECT_LIBRARY")
                target_link_libraries(${TARGET_NAME} ${LINK_TYPE} ${ARGN})
            else()
                # Object library may not link to anything.
                # To add interface include definitions and compile options explicitly.
                foreach(ITEM IN LISTS ARGN)
                    if(TARGET ${ITEM})
                        get_target_property(compile_options ${ITEM} INTERFACE_COMPILE_OPTIONS)
                        if (compile_options)
                            target_compile_options(${TARGET_NAME} ${LINK_TYPE} ${compile_options})
                        endif()
                        get_target_property(compile_definitions ${ITEM} INTERFACE_COMPILE_DEFINITIONS)
                        if (compile_definitions)
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
            ie_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} ${TBB_IMPORTED_TARGETS})
        else ()
            ext_message(WARNING "TBB was not found by the configured TBB_DIR path.\
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
                ie_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} "-nodefaultlib:vcomp")
            else()
                target_compile_options(${TARGET_NAME} ${LINK_TYPE} ${OpenMP_CXX_FLAGS} -fopenmp)
            endif ()

            # Debug binaries are optional.
            if (OMP_LIBRARIES_DEBUG AND NOT LINUX)
                if (WIN32)
                    ie_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} "$<$<CONFIG:DEBUG>:${OMP_LIBRARIES_DEBUG}>;$<$<NOT:$<CONFIG:DEBUG>>:${OMP_LIBRARIES_RELEASE}>")
                else()
                    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
                        ie_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} ${OMP_LIBRARIES_DEBUG})
                    else()
                        ie_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} ${OMP_LIBRARIES_RELEASE})
                    endif ()
                endif ()
            else ()
                # Link Release library to all configurations.
                ie_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} ${OMP_LIBRARIES_RELEASE})
            endif ()
        endif ()

    endif ()

    target_compile_definitions(${TARGET_NAME} ${LINK_TYPE} -DIE_THREAD=${IE_THREAD_DEFINE})

    if (NOT THREADING STREQUAL "SEQ")
        find_package(Threads REQUIRED)
        ie_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} ${CMAKE_THREAD_LIBS_INIT})
    endif()
endfunction(set_ie_threading_interface_for)
