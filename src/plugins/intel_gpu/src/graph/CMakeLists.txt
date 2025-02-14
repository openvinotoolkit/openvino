# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_intel_gpu_graph")

file(GLOB SUBDIRS "${CMAKE_CURRENT_SOURCE_DIR}/impls/*")

foreach(SUBDIR IN LISTS SUBDIRS)
    if(IS_DIRECTORY ${SUBDIR})
        get_filename_component(SUBDIR_NAME ${SUBDIR} NAME)
        if (NOT SUBDIR_NAME STREQUAL "registry")
            list(APPEND AVAILABLE_IMPL_TYPES ${SUBDIR_NAME})
            list(APPEND EXCLUDE_PATHS ${SUBDIR})
        endif()
    endif()
endforeach()

set(COMMON_COMPILE_OPTIONS $<$<CONFIG:Release>:$<IF:$<CXX_COMPILER_ID:MSVC>,/Os,-Os>>)
set(COMMON_INCLUDE_DIRS $<BUILD_INTERFACE:${MAIN_DIR}/src>
                        $<BUILD_INTERFACE:${INCLUDE_DIR}>
                        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

set(COMMON_LINK_LIBRARIES openvino::shape_inference # for tensor accessor
                          openvino_intel_gpu_kernels # for cl_kernel_data_serializer
                          openvino_intel_gpu_runtime
                          openvino::itt
                          openvino::runtime::dev
                          openvino::runtime
)

function(ov_gpu_add_backend_target)
    set(options
    )
    set(oneValueRequiredArgs
        NAME
    )
    set(multiValueArgs
        INCLUDES                      # Extra include directories
        LINK_LIBRARIES                # Link libraries (in form of target name or file name)
    )
    cmake_parse_arguments(ARG "${options}" "${oneValueRequiredArgs}" "${multiValueArgs}" ${ARGN} )

    set(TARGET_INCLUDES ${COMMON_INCLUDE_DIRS} ${ARG_INCLUDES})
    set(TARGET_LINK_LIBRARIES ${COMMON_LINK_LIBRARIES} ${ARG_LINK_LIBRARIES})

    foreach(lib IN LISTS TARGET_LINK_LIBRARIES)
        list(APPEND TARGET_INCLUDES $<TARGET_PROPERTY:${lib},INTERFACE_INCLUDE_DIRECTORIES>)
    endforeach()

    ov_add_target(
        NAME ${ARG_NAME}
        TYPE OBJECT
        ADD_CPPLINT
        INCLUDES
            PRIVATE
                ${TARGET_INCLUDES}
        ${ARG_UNPARSED_ARGUMENTS}
    )

    target_compile_options(${TARGET_NAME} PRIVATE ${COMMON_COMPILE_OPTIONS})
    ov_set_threading_interface_for(${TARGET_NAME})
endfunction()

foreach(IMPL_TYPE IN LISTS AVAILABLE_IMPL_TYPES)
    add_subdirectory(impls/${IMPL_TYPE})
    if (TARGET openvino_intel_gpu_${IMPL_TYPE}_obj)
        list(APPEND OBJ_FILES $<TARGET_OBJECTS:openvino_intel_gpu_${IMPL_TYPE}_obj>)
    endif()
endforeach()

ov_add_target(
    NAME ${TARGET_NAME}
    TYPE STATIC
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    EXCLUDED_SOURCE_PATHS
        ${EXCLUDE_PATHS}
    OBJECT_FILES
        ${OBJ_FILES}
    INCLUDES
        PUBLIC
            ${COMMON_INCLUDE_DIRS}
    LINK_LIBRARIES
        PUBLIC
            ${COMMON_LINK_LIBRARIES}
            openvino::reference # for loop primitive subroutines
    ADD_CPPLINT
)

target_compile_options(${TARGET_NAME} PRIVATE ${COMMON_COMPILE_OPTIONS})

ov_set_threading_interface_for(${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})
