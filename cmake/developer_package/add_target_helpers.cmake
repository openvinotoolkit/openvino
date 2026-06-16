# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#[[
function to create CMake target and setup its options in a declarative style.
Example:
ov_add_target(
   NAME core_lib
   ADD_CLANG_FORMAT
   TYPE <SHARED / STATIC / EXECUTABLE>
   ROOT ${CMAKE_CURRENT_SOURCE_DIR}
   ADDITIONAL_SOURCE_DIRS
        /some/additional/sources
   EXCLUDED_SOURCE_PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/unnecessary_sources/
   INCLUDES
        ${SDL_INCLUDES}
        /some/specific/path
   LINK_LIBRARIES
        link_dependencies
   DEPENDENCIES
        dependencies
        openvino::important_plugin
   OBJECT_FILES
        object libraries
   DEFINES
        DEF1 DEF2
   LINK_LIBRARIES_WHOLE_ARCHIVE
        lib1 lib2
   LINK_FLAGS
        flag1 flag2
)
#]]
function(ov_add_target)
    set(options
        ADD_CLANG_FORMAT              # Enables code style checks for the target
        )
    set(oneValueRequiredArgs
        TYPE # type of target, SHARED|STATIC|EXECUTABLE. SHARED and STATIC correspond to add_library, EXECUTABLE to add_executable
        NAME # name of target
        )
    set(oneValueOptionalArgs
        ROOT # root directory to be used for recursive search of source files (optional when SOURCES is provided)
        )
    set(multiValueArgs
        SOURCES                       # Explicit source file list; when provided ROOT glob is skipped
        INCLUDES                      # Extra include directories
        LINK_LIBRARIES                # Link libraries (in form of target name or file name)
        DEPENDENCIES                  # compile order dependencies (no link implied)
        DEFINES                       # extra preprocessor definitions
        ADDITIONAL_SOURCE_DIRS        # list of directories which will be used to recursive search of source files in addition to ROOT
        OBJECT_FILES                  # list of object files to be additionally built into the target
        EXCLUDED_SOURCE_PATHS         # list of paths excluded from the global recursive search of source files
        LINK_LIBRARIES_WHOLE_ARCHIVE  # list of static libraries to link, each object file should be used and not discarded
        LINK_FLAGS                    # list of extra commands to linker
        )
    cmake_parse_arguments(ARG "${options}" "${oneValueRequiredArgs};${oneValueOptionalArgs}" "${multiValueArgs}" ${ARGN})

    # sanity checks
    foreach(argName IN LISTS oneValueRequiredArgs)
        if (NOT ARG_${argName})
            message(SEND_ERROR "Argument '${argName}' is required.")
        endif()
    endforeach()
    if(ARG_UNPARSED_ARGUMENTS)
        message(SEND_ERROR "Unexpected parameters have passed to function: ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    # adding files to target
    if(ARG_SOURCES)
        # Explicit list provided — skip glob entirely
        set(includes)
        set(sources ${ARG_SOURCES})
    elseif(ARG_ROOT)
        set(includeSearch)
        set(sourceSearch)

        foreach(directory ${ARG_ROOT} ${ARG_ADDITIONAL_SOURCE_DIRS})
            list(APPEND includeSearch ${directory}/*.h ${directory}/*.hpp)
            list(APPEND sourceSearch ${directory}/*.cpp)
        endforeach()

        file(GLOB_RECURSE includes ${includeSearch})
        file(GLOB_RECURSE sources ${sourceSearch})

        # remove unnecessary directories
        foreach(excludedDir IN LISTS ARG_EXCLUDED_SOURCE_PATHS)
            list(FILTER includes EXCLUDE REGEX "${excludedDir}.*")
            list(FILTER sources EXCLUDE REGEX "${excludedDir}.*")
        endforeach()

        source_group("include" FILES ${includes})
        source_group("src"     FILES ${sources})
    else()
        message(SEND_ERROR "Either 'ROOT' or 'SOURCES' argument is required.")
    endif()

    set(all_sources ${sources} ${includes} ${ARG_OBJECT_FILES})

    # defining a target
    if(ARG_TYPE STREQUAL EXECUTABLE)
        add_executable(${ARG_NAME} ${all_sources})
    elseif(ARG_TYPE STREQUAL STATIC OR ARG_TYPE STREQUAL SHARED OR ARG_TYPE STREQUAL OBJECT)
        add_library(${ARG_NAME} ${ARG_TYPE} ${all_sources})
    else()
        message(SEND_ERROR "Invalid target type ${ARG_TYPE} specified for target name ${ARG_NAME}")
    endif()

    ov_target_link_whole_archive(${ARG_NAME} ${ARG_LINK_LIBRARIES_WHOLE_ARCHIVE})

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
    if (ARG_ADD_CLANG_FORMAT)
        # code style
        ov_add_clang_format_target(${ARG_NAME}_clang FOR_TARGETS ${ARG_NAME})
    endif()
endfunction()

#[[
Creates one small test executable per source file, each registered as a CTest test.
Intended for local iterative development — build and run a single file without
recompiling the whole suite. Not intended for CI use.

Usage:
  ov_add_test_target_per_source(
    NAME           <base-name>        # prefix for generated target names: <NAME>_<stem>
    SOURCES        file1.cpp ...      # source files to split into individual targets
    LINK_LIBRARIES <lib1> <lib2>      # same libs as the combined target
    LABELS         <label1> <label2>
    GTEST_DISCOVER                    # register per-TEST() CTest entries for each target
  )

Example targets produced for NAME=ov_util_tests:
  ov_util_tests_file_util_test
  ov_util_tests_graph_comparator_tests
  ...

Build / run one file:
  cmake --build <dir> --target ov_util_tests_file_util_test
  ctest --test-dir <dir> -R "ov_util_tests_file_util_test"
#]]
function(ov_add_test_target_per_source)
    if(NOT ENABLE_TESTS_PER_SOURCE)
        return()
    endif()

    set(options
        GTEST_DISCOVER
    )
    set(oneValueRequiredArgs
        NAME
    )
    set(multiValueArgs
        SOURCES
        LINK_LIBRARIES
        LABELS
    )
    cmake_parse_arguments(ARG "${options}" "${oneValueRequiredArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_NAME)
        message(SEND_ERROR "ov_add_test_target_per_source: NAME is required.")
        return()
    endif()

    if(NOT ARG_SOURCES)
        message(SEND_ERROR "ov_add_test_target_per_source: SOURCES is required.")
        return()
    endif()

    set(_has_gtest FALSE)

    if(ARG_GTEST_DISCOVER AND(NOT CMAKE_CROSSCOMPILING OR CMAKE_CROSSCOMPILING_EMULATOR))
        include(GoogleTest)

        if(COMMAND gtest_discover_tests AND(TARGET GTest::gtest OR TARGET GTest::GTest))
            set(_has_gtest TRUE)
        endif()
    endif()

    foreach(_src IN LISTS ARG_SOURCES)
        get_filename_component(_stem "${_src}" NAME_WE)
        set(_target "${ARG_NAME}_${_stem}")

        add_executable(${_target} ${_src})

        if(ARG_LINK_LIBRARIES)
            target_link_libraries(${_target} PRIVATE ${ARG_LINK_LIBRARIES})
        endif()

        add_test(NAME ${_target} COMMAND ${_target})

        if(ARG_LABELS)
            set_property(TEST ${_target} PROPERTY LABELS ${ARG_LABELS})
        endif()

        if(_has_gtest)
            gtest_discover_tests(${_target}
                DISCOVERY_MODE POST_BUILD
                PROPERTIES LABELS "${ARG_LABELS}"
            )
        endif()
    endforeach()
endfunction()

#[[
Wrapper function over ov_add_target that also registers the executable as a CTest test.
All ov_add_target parameters (ROOT, SOURCES, LINK_LIBRARIES, INCLUDES, DEFINES,
DEPENDENCIES, EXCLUDED_SOURCE_PATHS, etc.) are forwarded transparently.

Usage:
  ov_add_test_target(
    NAME              <target-name>
    LABELS            <label1> <label2>
    GTEST_DISCOVER                          # register per-TEST() CTest entries (requires GTest)
    TESTS_PER_SOURCE                        # delegate to ov_add_test_target_per_source (local dev only)
    COMPONENT         <cpack-component>     # default: tests
  )

Note: LABELS must be the last keyword argument when ROOT or other ov_add_target
  pass-through args are used, as it consumes all following tokens.
#]]
function(ov_add_test_target)
    set(options
        GTEST_DISCOVER # Register per-TEST() CTest entries via gtest_discover_tests (requires GTest)
        TESTS_PER_SOURCE # Create one executable per source file for local dev (not for CI)
    )
    set(oneValueRequiredArgs
        NAME
        )
    set(oneValueOptionalArgs
        COMPONENT
        )
    set(multiValueArgs
        LABELS
        )
    cmake_parse_arguments(ARG "${options}" "${oneValueRequiredArgs};${oneValueOptionalArgs}" "${multiValueArgs}" ${ARGN} )
    if (NOT DEFINED ARG_COMPONENT)
        set(ARG_COMPONENT tests)
    endif()

    ov_add_target(TYPE EXECUTABLE NAME ${ARG_NAME} ${ARG_UNPARSED_ARGUMENTS})

    if(EMSCRIPTEN)
        set(JS_BIN_NAME "${ARG_NAME}.js")
        set(JS_APP_NAME "${ARG_NAME}_js.js")
        set(JS_TEST_APP "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${JS_APP_NAME}")
        file(WRITE   ${JS_TEST_APP} "// Copyright (C) 2018-2026 Intel Corporation\n")
        file(APPEND  ${JS_TEST_APP} "// SPDX-License-Identifier: Apache-2.0\n")
        file(APPEND  ${JS_TEST_APP} "//\n")
        file(APPEND  ${JS_TEST_APP} "// JS test app\n")
        file(APPEND  ${JS_TEST_APP} "const createModule = require(\"./${JS_BIN_NAME}\");\n")
        file(APPEND  ${JS_TEST_APP} "createModule().then(function(Module) {});\n")
        file(APPEND  ${JS_TEST_APP} " ")
        # node version>= 16.8.0, else need add "--experimental-wasm-threads --experimental-wasm-bulk-memory" option
        add_test(NAME ${ARG_NAME} COMMAND node ${JS_TEST_APP})
    else()
        add_test(NAME ${ARG_NAME} COMMAND ${ARG_NAME})
    endif()
    if(ARG_LABELS)
        set_property(TEST ${ARG_NAME} PROPERTY LABELS ${ARG_LABELS})
    endif()

    if(ARG_GTEST_DISCOVER AND(NOT CMAKE_CROSSCOMPILING OR CMAKE_CROSSCOMPILING_EMULATOR))
        include(GoogleTest OPTIONAL RESULT_VARIABLE has_gtest)

        if(has_gtest)
            gtest_discover_tests(${ARG_NAME}
                DISCOVERY_MODE POST_BUILD
                PROPERTIES LABELS "${ARG_LABELS}"
            )
        endif()
    endif()

    if(ARG_TESTS_PER_SOURCE)
        set(_gtest_discover)

        if(ARG_GTEST_DISCOVER)
            set(_gtest_discover GTEST_DISCOVER)
        endif()

        ov_add_test_target_per_source(
            NAME ${ARG_NAME}
            LABELS ${ARG_LABELS}
            ${_gtest_discover}
            ${ARG_UNPARSED_ARGUMENTS})
    endif()

    install(TARGETS ${ARG_NAME}
            RUNTIME DESTINATION tests
            COMPONENT ${ARG_COMPONENT}
            EXCLUDE_FROM_ALL)
endfunction()
