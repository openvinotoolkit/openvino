# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#[[
function to create CMake target and setup its options in a declarative style.
Example:
addIeTarget(
   NAME core_lib
   ADD_CPPLINT
   DEVELOPER_PACKAGE
   TYPE SHARED
   ROOT ${CMAKE_CURRENT_SOURCE_DIR}
   ADDITIONAL_SOURCE_DIRS
        /some/additional/sources
   EXCLUDED_SOURCE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/unnecessary_sources/
   INCLUDES
        ${SDL_INCLUDES}
        /some/specific/path
   LINK_LIBRARIES
        ie::important_plugin
   EXPORT_DEPENDENCIES
        dependency_lib_to_export
   DEPENDENCIES
        dependencies
   OBJECT_FILES
        object libraries
)
#]]
function(addIeTarget)
    set(options
        ADD_CPPLINT                   # Enables code style checks for the target
        DEVELOPER_PACKAGE             # Enables exporting of the target through the developer package
        )
    set(oneValueRequiredArgs
        TYPE # type of target, SHARED|STATIC|EXECUTABLE. SHARED and STATIC correspond to add_library, EXECUTABLE to add_executable
        NAME # name of target
        ROOT # root directory to be used for recursive search of source files
        )
    set(oneValueOptionalArgs
        )
    set(multiValueArgs
        INCLUDES                      # Extra include directories
        LINK_LIBRARIES                # Link libraries (in form of target name or file name)
        DEPENDENCIES                  # compile order dependencies (no link implied)
        DEFINES                       # extra preprocessor definitions
        ADDITIONAL_SOURCE_DIRS        # list of directories which will be used to recursive search of source files in addition to ROOT
        OBJECT_FILES                  # list of object files to be additionally built into the target
        EXCLUDED_SOURCE_DIRS          # list of directories excluded from the global recursive search of source files
        LINK_LIBRARIES_WHOLE_ARCHIVE  # list of static libraries to link, each object file should be used and not discarded
        LINK_FLAGS                    # list of extra commands to linker
        EXPORT_DEPENDENCIES           # list of the dependencies to be exported with the target through the developer package
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

    # remove unnecessary directories
    if (ARG_EXCLUDED_SOURCE_DIRS)
        list(FILTER includes EXCLUDE REGEX "${ARG_EXCLUDED_SOURCE_DIRS}/*")
        list(FILTER sources EXCLUDE REGEX "${ARG_EXCLUDED_SOURCE_DIRS}/*")
    endif()

    source_group("include" FILES ${includes})
    source_group("src"     FILES ${sources})

    set(all_sources)
    list(APPEND all_sources ${sources} ${includes} ${ARG_OBJECT_FILES})

    # defining a target
    if (ARG_TYPE STREQUAL EXECUTABLE)
        add_executable(${ARG_NAME} ${all_sources})
    elseif(ARG_TYPE STREQUAL STATIC OR ARG_TYPE STREQUAL SHARED)
        add_library(${ARG_NAME} ${type} ${all_sources})
    else()
        message(SEND_ERROR "Invalid target type ${ARG_TYPE} specified for target name ${ARG_NAME}")
    endif()

    ieTargetLinkWholeArchive(${ARG_NAME} ${ARG_LINK_LIBRARIES_WHOLE_ARCHIVE})

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
    if (ARG_LINK_FLAGS)
        get_target_property(oldLinkFlags ${ARG_NAME} LINK_FLAGS)
        string(REPLACE ";" " " ARG_LINK_FLAGS "${ARG_LINK_FLAGS}")
        set_target_properties(${ARG_NAME} PROPERTIES LINK_FLAGS "${oldLinkFlags} ${ARG_LINK_FLAGS}")
    endif()
    if (ARG_ADD_CPPLINT)
        # code style
        add_cpplint_target(${ARG_NAME}_cpplint FOR_TARGETS ${ARG_NAME})
    endif()
    if (ARG_DEVELOPER_PACKAGE)
        # developer package
        ie_developer_export_targets(${ARG_NAME})
        if (ARG_EXPORT_DEPENDENCIES)
            ie_developer_export_targets(${ARG_NAME} ${ARG_EXPORT_DEPENDENCIES})
        endif()
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

    addIeTarget(TYPE EXECUTABLE NAME ${ARG_NAME} ${ARG_UNPARSED_ARGUMENTS})

    add_test(NAME ${ARG_NAME} COMMAND ${ARG_NAME})
    set_property(TEST ${ARG_NAME} PROPERTY LABELS ${ARG_LABELS})
endfunction()
