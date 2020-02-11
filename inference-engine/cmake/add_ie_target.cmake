# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#[[
function to create CMake target and setup its options in a declarative style.
Example:
addIeTarget(
   NAME core_lib
   TYPE shared
   ROOT ${CMAKE_CURRENT_SOURCE_DIR}
   INCLUDES
        ${SDL_INCLUDES}
        /some/specific/path
   LINK_LIBRARIES
        ie::important_plugin
)
#]]
function(addIeTarget)
    set(options
        )
    set(oneValueRequiredArgs
        TYPE # type of target, shared|static|executable. shared and static correspond to add_library, executable to add_executable.
        NAME # name of target
        ROOT # directory will used for source files globbing root.
        )
    set(oneValueOptionalArgs
        )
    set(multiValueArgs
        INCLUDES                   # Extra include directories.
        LINK_LIBRARIES             # Link libraries (in form of target name or file name)
        DEPENDENCIES               # compile order dependencies (no link implied)
        DEFINES                    # extra preprocessor definitions
        ADDITIONAL_SOURCE_DIRS     # list of directories, which will be used to search for source files in addition to ROOT.
        )
    cmake_parse_arguments(ARG "${options}" "${oneValueRequiredArgs};${oneValueOptionalArgs}" "${multiValueArgs}" ${ARGN} )

    # sanity checks
    foreach(argName ${oneValueRequiredArgs})
        if (NOT ARG_${argName})
            message(SEND_ERROR "Argument '${argName}' is required.")
        endif()
    endforeach()
    if (ARG_UNPARSED_ARGUMENTS)
        message(SEND_ERROR "Unexpected parameters have passed to function: ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    # adding files to target
    set(includeSearch)
    set(sourceSearch)
    foreach(directory ${ARG_ROOT} ${ARG_ADDITIONAL_SOURCE_DIRS})
        list(APPEND includeSearch ${directory}/*.h ${directory}/*.hpp)
        list(APPEND sourceSearch  ${directory}/*.cpp)
    endforeach()

    file(GLOB_RECURSE includes ${includeSearch})
    file(GLOB_RECURSE sources  ${sourceSearch})

    source_group("include" FILES ${includes})
    source_group("src"     FILES ${sources})

    # defining a target
    if (ARG_TYPE STREQUAL executable)
        add_executable(${ARG_NAME} ${sources} ${includes})
    elseif(ARG_TYPE STREQUAL static OR ARG_TYPE STREQUAL shared)
        string(TOUPPER ${ARG_TYPE} type)
        add_library(${ARG_NAME} ${type} ${sources} ${includes})
    else()
        message(SEND_ERROR "Invalid target type: ${ARG_TYPE}")
    endif()

    # filling target properties
    set_property(TARGET ${ARG_NAME} PROPERTY CXX_STANDARD 11)
    set_property(TARGET ${ARG_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
    if (ARG_DEFINES)
        target_compile_definitions(${ARG_NAME} PRIVATE ${ARG_DEFINES})
    endif()
    if (ARG_INCLUDES)
        target_include_directories(${ARG_NAME} PRIVATE ${ARG_INCLUDES})
    endif()
    if (ARG_LINK_LIBRARIES)
        target_link_libraries(${ARG_NAME} PRIVATE ${ARG_LINK_LIBRARIES})
    endif()
    if (ARG_DEPENDENCIES)
        add_dependencies(${ARG_NAME} ${ARG_DEPENDENCIES})
    endif()
endfunction()

#[[
Wrapper function over addIeTarget, that also adds a test with the same name.
You could use
addIeTargetTest( ... LABELS labelOne labelTwo )
also to provide labels for that test.
Important: you MUST pass LABELS as last argument, otherwise it will consume any parameters that come after.
#]]
function(addIeTargetTest)
    set(options
        )
    set(oneValueRequiredArgs
        NAME
        )
    set(oneValueOptionalArgs
        )
    set(multiValueArgs
        LABELS
        )
    cmake_parse_arguments(ARG "${options}" "${oneValueRequiredArgs};${oneValueOptionalArgs}" "${multiValueArgs}" ${ARGN} )

    addIeTarget(TYPE executable NAME ${ARG_NAME} ${ARG_UNPARSED_ARGUMENTS})

    add_test(NAME ${ARG_NAME} COMMAND ${ARG_NAME})
    set_property(TEST ${ARG_NAME} PROPERTY LABELS ${ARG_LABELS})
endfunction()
