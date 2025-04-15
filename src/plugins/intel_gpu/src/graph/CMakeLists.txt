# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_intel_gpu_graph")

file(GLOB SUBDIRS "${CMAKE_CURRENT_SOURCE_DIR}/impls/*")

find_host_package(Python3 REQUIRED COMPONENTS Interpreter)

foreach(SUBDIR IN LISTS SUBDIRS)
    if(IS_DIRECTORY ${SUBDIR})
        get_filename_component(SUBDIR_NAME ${SUBDIR} NAME)
        list(APPEND AVAILABLE_IMPL_TYPES ${SUBDIR_NAME})
        list(APPEND EXCLUDE_PATHS ${SUBDIR})
    endif()
endforeach()

set(COMMON_COMPILE_OPTIONS $<$<CONFIG:Release>:$<IF:$<CXX_COMPILER_ID:MSVC>,/Os,-Os -Wno-suggest-override>>)
set(COMMON_INCLUDE_DIRS $<BUILD_INTERFACE:${MAIN_DIR}/src>
                        $<BUILD_INTERFACE:${INCLUDE_DIR}>
                        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/impls>
)

set(COMMON_LINK_LIBRARIES openvino::shape_inference # for tensor accessor
                          openvino_intel_gpu_kernels # for cl_kernel_data_serializer
                          openvino_intel_gpu_runtime
                          openvino::itt
                          openvino::runtime::dev
                          openvino::runtime
)

define_property(GLOBAL PROPERTY EXTRA_LINK_LIBRARIES_GLOBAL BRIEF_DOCS "All link libs" FULL_DOCS "Link libraries collection from all backends")
set_property(GLOBAL PROPERTY EXTRA_LINK_LIBRARIES_GLOBAL "")

macro(ov_gpu_add_backend_target)
    set(options
    )
    set(oneValueRequiredArgs
        NAME
    )
    set(multiValueArgs
        INCLUDES                      # Extra include directories
        LINK_LIBRARIES                # Link libraries (in form of target name or file name)
        BYPASS                        # All other args that must be passed as is to ov_add_target call
    )
    cmake_parse_arguments(ARG "${options}" "${oneValueRequiredArgs}" "${multiValueArgs}" ${ARGN} )

    set(TARGET_INCLUDES ${COMMON_INCLUDE_DIRS} ${ARG_INCLUDES})
    set(TARGET_LINK_LIBRARIES ${COMMON_LINK_LIBRARIES} ${ARG_LINK_LIBRARIES})
    set(TARGET_DEFINITIONS "")

    foreach(lib IN LISTS TARGET_LINK_LIBRARIES)
        list(APPEND TARGET_INCLUDES $<TARGET_PROPERTY:${lib},INTERFACE_INCLUDE_DIRECTORIES>)
        list(APPEND TARGET_DEFINITIONS $<TARGET_PROPERTY:${lib},INTERFACE_COMPILE_DEFINITIONS>)
    endforeach()

    ov_add_target(
        NAME ${ARG_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        TYPE OBJECT
        ADD_CPPLINT
        INCLUDES
            PRIVATE
                ${TARGET_INCLUDES}
        ${ARG_BYPASS}
    )

    get_property(CURRENT_LIBS GLOBAL PROPERTY EXTRA_LINK_LIBRARIES_GLOBAL)
    list(APPEND CURRENT_LIBS ${ARG_LINK_LIBRARIES})
    set_property(GLOBAL PROPERTY EXTRA_LINK_LIBRARIES_GLOBAL ${CURRENT_LIBS})

    target_compile_options(${TARGET_NAME} PRIVATE ${COMMON_COMPILE_OPTIONS})
    target_compile_definitions(${TARGET_NAME} PRIVATE ${TARGET_DEFINITIONS})
    ov_set_threading_interface_for(${TARGET_NAME})

    # We use onednn headers all over the graph module, so we have to append includes to all backends and add a dependency between targets
    if (ENABLE_ONEDNN_FOR_GPU)
        target_include_directories(${ARG_NAME} SYSTEM PRIVATE $<TARGET_PROPERTY:onednn_gpu_tgt,INTERFACE_INCLUDE_DIRECTORIES>)
        add_dependencies(openvino_intel_gpu_${IMPL_TYPE}_obj onednn_gpu_tgt)
    endif()
endmacro()

set(CODEGEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/common_utils/kernels_db_gen.py")
set(CODEGEN_TEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/common_utils/test_kernels_db_gen.py")

enable_testing()

add_custom_target(run_codegen_test
    COMMAND ${Python3_EXECUTABLE} -B ${CODEGEN_TEST_SCRIPT}
    DEPENDS "${CODEGEN_SCRIPT}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(NAME TestOCLCodePreprocessing
         COMMAND ${Python3_EXECUTABLE} -B ${CODEGEN_TEST_SCRIPT}
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

foreach(IMPL_TYPE IN LISTS AVAILABLE_IMPL_TYPES)
    add_subdirectory(impls/${IMPL_TYPE})
    if (TARGET openvino_intel_gpu_${IMPL_TYPE}_obj)
        list(APPEND OBJ_FILES $<TARGET_OBJECTS:openvino_intel_gpu_${IMPL_TYPE}_obj>)
    endif()
endforeach()

get_property(EXTRA_LINK_LIBRARIES GLOBAL PROPERTY EXTRA_LINK_LIBRARIES_GLOBAL)

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
            ${EXTRA_LINK_LIBRARIES}
            openvino::reference # for loop primitive subroutines
    DEPENDENCIES
        run_codegen_test
    ADD_CPPLINT
)

target_compile_options(${TARGET_NAME} PRIVATE ${COMMON_COMPILE_OPTIONS})

ov_set_threading_interface_for(${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})
