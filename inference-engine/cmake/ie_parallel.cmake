# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(set_ie_threading_interface_for TARGET_NAME)
    if (THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO" AND NOT TBB_FOUND)
        find_package(TBB COMPONENTS tbb tbbmalloc)
        set("TBB_FOUND" ${TBB_FOUND} PARENT_SCOPE)
        set("TBB_IMPORTED_TARGETS" ${TBB_IMPORTED_TARGETS} PARENT_SCOPE)
        set("TBB_VERSION" ${TBB_VERSION} PARENT_SCOPE)
        if (NOT TBB_FOUND)
            ext_message(WARNING "TBB was not found by the configured TBB_DIR/TBBROOT path.\
                                SEQ method will be used.")
        endif ()
    endif()

    get_target_property(target_type ${TARGET_NAME} TYPE)
    if(target_type STREQUAL "INTERFACE_LIBRARY")
        set(LINK_TYPE "INTERFACE")
    elseif(target_type STREQUAL "EXECUTABLE" OR target_type STREQUAL "OBJECT_LIBRARY")
        set(LINK_TYPE "PRIVATE")
    else()
        set(LINK_TYPE "PUBLIC")
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
    endif ()

    target_compile_definitions(${TARGET_NAME} ${LINK_TYPE} -DIE_THREAD=${IE_THREAD_DEFINE})

    if (NOT THREADING STREQUAL "SEQ")
        find_package(Threads REQUIRED)
        ie_target_link_libraries(${TARGET_NAME} ${LINK_TYPE} ${CMAKE_THREAD_LIBS_INIT})
    endif()
endfunction(set_ie_threading_interface_for)
