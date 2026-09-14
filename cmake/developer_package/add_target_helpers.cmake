# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#[[
function to create CMake target and setup its options in a declarative style.

SOURCE LISTING — preferred vs. legacy
--------------------------------------
SOURCES (preferred)
  Pass an explicit list of source and header files.  CMake sees the exact file set at configure time, incremental
  builds are reliable, and there are no hidden glob-related staleness issues.
  See src/common/util/CMakeLists.txt for the canonical example.

ROOT (legacy / deprecated)
  Pass a directory root; the function will GLOB_RECURSE for *.cpp / *.h / *.hpp under that directory (and any
  ADDITIONAL_SOURCE_DIRS).  New targets should NOT use ROOT.  Existing targets should be migrated to explicit
  SOURCES lists when the opportunity arises.

Migration pattern — converting ROOT to SOURCES:
  1. Replace ROOT ${CMAKE_CURRENT_SOURCE_DIR} with an explicit SOURCES list.
  2. Remove ADDITIONAL_SOURCE_DIRS and EXCLUDED_SOURCE_PATHS (handle in SOURCES).
  3. Use target_sources() for platform-specific files (WIN32/UNIX conditionals).
  See src/common/util/CMakeLists.txt for a worked example.

Preferred example (SOURCES):
ov_add_target(
   NAME core_lib
   ADD_CLANG_FORMAT
   TYPE <SHARED / STATIC / EXECUTABLE>
   SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/foo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/foo.hpp
   INCLUDES
        ${SDL_INCLUDES}
        /some/specific/path
   LINK_LIBRARIES
        link_dependencies
   DEPENDENCIES
        dependencies
        openvino::important_plugin
   DEFINES
        DEF1 DEF2
)

Legacy example (ROOT — avoid for new targets):
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
        ROOT # [LEGACY] root directory for GLOB_RECURSE source discovery; prefer SOURCES for new targets
    )
    set(multiValueArgs
        SOURCES                       # [PREFERRED] Explicit source file list; when provided ROOT glob is skipped
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
        ADD_CLANG_FORMAT
    )
    set(oneValueRequiredArgs
        NAME
    )
    set(multiValueArgs
        SOURCES
        LINK_LIBRARIES
        LABELS
        INCLUDES
        # accept but ignore
        DEPENDENCIES
        DEFINES
        LINK_LIBRARIES_WHOLE_ARCHIVE
        LINK_FLAGS
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
        # Use the subdirectory-qualified stem to avoid collisions when two source files have the same basename
        # but live in different subdirectories.
        file(RELATIVE_PATH _rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_src}")
        get_filename_component(_stem_dir "${_rel}" DIRECTORY)
        get_filename_component(_stem_base "${_rel}" NAME_WE)

        if(_stem_dir)
            string(REPLACE "/" "_" _stem_dir "${_stem_dir}")
            set(_stem "${_stem_dir}_${_stem_base}")
        else()
            set(_stem "${_stem_base}")
        endif()
        set(_target "${ARG_NAME}_${_stem}")

        add_executable(${_target} EXCLUDE_FROM_ALL ${_src})

        if(ARG_LINK_LIBRARIES)
            target_link_libraries(${_target} PRIVATE ${ARG_LINK_LIBRARIES})
        endif()

        if (ARG_INCLUDES)
            target_include_directories(${_target} PRIVATE ${ARG_INCLUDES})
        endif()

        add_test(NAME ${_target} COMMAND ${_target})

        if(ARG_LABELS)
            set_property(TEST ${_target} PROPERTY LABELS ${ARG_LABELS})
        endif()

        if(_has_gtest)
            # PRE_TEST keeps enumeration out of the build: it runs under ctest
            gtest_discover_tests(${_target}
                DISCOVERY_MODE PRE_TEST
                DISCOVERY_TIMEOUT 300
                PROPERTIES LABELS "${ARG_LABELS}"
            )
        endif()
    endforeach()
endfunction()

#[[
Wrapper over ov_add_target that builds a test executable and registers it as a CTest test.
All ov_add_target parameters (SOURCES, ROOT, LINK_LIBRARIES, INCLUDES, DEFINES, DEPENDENCIES,
EXCLUDED_SOURCE_PATHS, etc.) are forwarded transparently.

SOURCE LISTING — preferred vs. legacy
--------------------------------------
SOURCES (preferred)
  Pass an explicit list of source files.  CMake sees the exact file set at configure time, incremental builds are
  reliable, and there are no hidden glob-related staleness issues.
  When SOURCES is used, pass CHECK_SOURCES_LISTED (see below) to guard against accidentally unlisted files.

ROOT (legacy / deprecated)
  Pass a directory root; the function will GLOB_RECURSE for *.cpp under that directory.
  New test targets should NOT use ROOT. The automatic completeness check is skipped in this mode, since GLOB
  already guarantees every file on disk is picked up.

Migration pattern — converting ROOT to SOURCES:
  1. Replace ROOT ${CMAKE_CURRENT_SOURCE_DIR} with an explicit SOURCES list (or include() a sources.cmake).
  2. Remove ADDITIONAL_SOURCE_DIRS and EXCLUDED_SOURCE_PATHS (handle in the SOURCES list).
  See src/core/tests/CMakeLists.txt for a worked example.

Automatic completeness check (opt-in)
---------------------------------------
The completeness check itself (every *.cpp under a scanned directory must be part of the target's SOURCES) is
implemented as an internal helper, private to ov_add_test_target - there is no standalone function to call from a
CMakeLists.txt. It is enabled via:

  CHECK_SOURCES_LISTED             Opt-in: run the check (requires SOURCES, not ROOT). The check is registered via
                                    cmake_language(DEFER CALL ...) to run at the end of the current directory's
                                    CMakeLists.txt processing, so EXCLUDE_TARGETS may safely name targets that do
                                    not exist yet at the point ov_add_test_target() is called (e.g. targets created
                                    by an add_subdirectory() that happens later in the same file) - see
                                    src/core/tests/CMakeLists.txt for such a case.

Extra parameters tune the check (meaningful only together with CHECK_SOURCES_LISTED; passing any of them
without it has no effect and triggers a warning):

  CHECK_SOURCES_DIRECTORY          <dir>               Directory to scan (default: CMAKE_CURRENT_SOURCE_DIR)
  CHECK_SOURCES_EXTENSIONS         <ext1> [<ext2>]     Extensions to scan (default: cpp)
  CHECK_SOURCES_EXCLUDE_FILES      <file1> [<file2>]   Files never matching the target's raw SOURCES property
                                                        (e.g. listed behind a generator expression). Must be
                                                        plain, unconditional paths (see example below)
                                                        instead of wrapping the path in a genex.
  CHECK_SOURCES_EXCLUDE_DIRECTORIES <dir1> [<dir2>]    Directories skipped entirely by the scan
  CHECK_SOURCES_EXCLUDE_TARGETS    <tgt1> [<tgt2>]     Other targets whose SOURCES should count as "listed" too
                                                        (e.g. a mock/benchmark target built from files that live
                                                        next to the main target's sources)

The recommended pattern is to keep the exclusion list next to the source list it applies to, as a dedicated
variable in the target's sources.cmake, e.g.:

  # sources.cmake
  set(MY_TESTS_SRCS
      ${CMAKE_CURRENT_LIST_DIR}/foo_test.cpp
      $<$<BOOL:${ENABLE_DEBUG_CAPS}>:${CMAKE_CURRENT_LIST_DIR}/debug_only_test.cpp>
  )
  # Files above are listed behind a generator expression - excluded from the completeness check when the
  # condition is false. Note: the exclude list itself uses a plain if(), not a genex - CHECK_SOURCES_EXCLUDE_FILES
  # is compared as a raw string and would never match a genex-wrapped entry.
  set(MY_TESTS_CHECK_SOURCES_EXCLUDE_FILES)

  if(NOT ENABLE_DEBUG_CAPS)
      list(APPEND MY_TESTS_CHECK_SOURCES_EXCLUDE_FILES
          ${CMAKE_CURRENT_LIST_DIR}/debug_only_test.cpp
      )
  endif()

  ov_check_all_sources_listed(TARGET my_unit_tests
      DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

Legacy example (ROOT — avoid for new targets):
  ov_add_test_target(
    NAME              my_unit_tests
    ROOT              ${CMAKE_CURRENT_SOURCE_DIR}
    EXCLUDED_SOURCE_PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/skip_this/
    INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}
    LINK_LIBRARIES
        common_test_utils
        openvino::runtime
    DEPENDENCIES
        openvino_template_extension
    GTEST_DISCOVER
    LABELS
        OV UNIT
  )

Test-specific parameters (in addition to all ov_add_target parameters):
  NAME              <target-name>          (required) — also becomes the CTest test name
  LABELS            <label1> <label2>      CTest LABELS property; must come after all pass-through args
  GTEST_DISCOVER                           Register per-TEST() CTest entries via gtest_discover_tests
  TESTS_PER_SOURCE                         Create one executable per source file (local dev only, not CI)
  COMPONENT         <cpack-component>      CPack install component (default: tests)

  # CMakeLists.txt
  include(sources.cmake)
  ov_add_test_target(
    NAME              my_unit_tests
    SOURCES
        ${MY_TESTS_SRCS}
    INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}
    LINK_LIBRARIES
        common_test_utils
        openvino::runtime
    DEPENDENCIES
        openvino_template_extension
    GTEST_DISCOVER
    CHECK_SOURCES_LISTED
    CHECK_SOURCES_EXCLUDE_FILES
        ${MY_TESTS_CHECK_SOURCES_EXCLUDE_FILES}
    LABELS
        OV UNIT
  )

Legacy example (ROOT — avoid for new targets):
  ov_add_test_target(
    NAME              my_unit_tests
    ROOT              ${CMAKE_CURRENT_SOURCE_DIR}
    EXCLUDED_SOURCE_PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/skip_this/
    INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}
    LINK_LIBRARIES
        common_test_utils
        openvino::runtime
    DEPENDENCIES
        openvino_template_extension
    GTEST_DISCOVER
    LABELS
        OV UNIT
  )

Test-specific parameters (in addition to all ov_add_target parameters):
  NAME              <target-name>          (required) — also becomes the CTest test name
  LABELS            <label1> <label2>      CTest LABELS property; must come after all pass-through args
  GTEST_DISCOVER                           Register per-TEST() CTest entries via gtest_discover_tests
  TESTS_PER_SOURCE                         Create one executable per source file (local dev only, not CI)
  COMPONENT         <cpack-component>      CPack install component (default: tests)
  See "Automatic completeness check (opt-in)" above for CHECK_SOURCES_LISTED, CHECK_SOURCES_DIRECTORY,
  CHECK_SOURCES_EXTENSIONS, CHECK_SOURCES_EXCLUDE_FILES, CHECK_SOURCES_EXCLUDE_DIRECTORIES and
  CHECK_SOURCES_EXCLUDE_TARGETS.

Note: LABELS, CHECK_SOURCES_LISTED, CHECK_SOURCES_DIRECTORY, CHECK_SOURCES_EXTENSIONS,
  CHECK_SOURCES_EXCLUDE_FILES, CHECK_SOURCES_EXCLUDE_DIRECTORIES and CHECK_SOURCES_EXCLUDE_TARGETS must all come
  after every ov_add_target pass-through argument (SOURCES/ROOT, LINK_LIBRARIES, DEPENDENCIES, INCLUDES, etc.).
  Each of these keywords greedily consumes every token that follows until the next keyword *recognized by
  ov_add_test_target itself* - a pass-through argument like LINK_LIBRARIES placed after one of them would
  silently be swallowed and never reach ov_add_target.
#]]
function(ov_add_test_target)
    set(options
        GTEST_DISCOVER # Register per-TEST() CTest entries via gtest_discover_tests (requires GTest)
        TESTS_PER_SOURCE # Create one executable per source file for local dev (not for CI)
        CHECK_SOURCES_LISTED # Opt-in: run the completeness check, deferred to the end of the directory scope (requires SOURCES)
    )
    set(oneValueRequiredArgs
        NAME
        )
    set(oneValueOptionalArgs
        COMPONENT
        CHECK_SOURCES_DIRECTORY # Directory scanned by the automatic completeness check (default: CMAKE_CURRENT_SOURCE_DIR)
    )
    set(multiValueArgs
        LABELS
        CHECK_SOURCES_EXTENSIONS # Extensions scanned by the automatic completeness check (default: cpp)
        CHECK_SOURCES_EXCLUDE_FILES # Files never matching the target's raw SOURCES property
        CHECK_SOURCES_EXCLUDE_DIRECTORIES # Directories skipped entirely by the completeness check
        CHECK_SOURCES_EXCLUDE_TARGETS # Other targets whose SOURCES should count as "listed" too
    )
    cmake_parse_arguments(ARG "${options}" "${oneValueRequiredArgs};${oneValueOptionalArgs}" "${multiValueArgs}" ${ARGN} )
    if (NOT DEFINED ARG_COMPONENT)
        set(ARG_COMPONENT tests)
    endif()

    ov_add_target(TYPE EXECUTABLE NAME ${ARG_NAME} ${ARG_UNPARSED_ARGUMENTS})

    if(NOT ARG_CHECK_SOURCES_LISTED)
        if(DEFINED ARG_CHECK_SOURCES_DIRECTORY OR ARG_CHECK_SOURCES_EXTENSIONS OR ARG_CHECK_SOURCES_EXCLUDE_FILES
            OR ARG_CHECK_SOURCES_EXCLUDE_DIRECTORIES OR ARG_CHECK_SOURCES_EXCLUDE_TARGETS)
            message(WARNING "ov_add_test_target(${ARG_NAME}): CHECK_SOURCES_* arguments were passed without "
                "CHECK_SOURCES_LISTED - they have no effect.")
        endif()
    else()
        # Peek at SOURCES/ROOT without consuming them, so they are still forwarded to ov_add_target above as-is.
        cmake_parse_arguments(_ov_test_check "" "ROOT" "SOURCES" ${ARG_UNPARSED_ARGUMENTS})

        if(_ov_test_check_ROOT)
            message(SEND_ERROR "ov_add_test_target(${ARG_NAME}): CHECK_SOURCES_LISTED "
                "is not available for ROOT-based (legacy) targets - ROOT-based targets already glob every file "
                "on disk. Migrate to SOURCES to use the completeness check.")
        elseif(NOT _ov_test_check_SOURCES)
            message(SEND_ERROR "ov_add_test_target(${ARG_NAME}): CHECK_SOURCES_LISTED "
                "requires SOURCES (ROOT-based targets already glob every file on disk).")
        else()
            if(NOT ARG_CHECK_SOURCES_DIRECTORY)
                set(ARG_CHECK_SOURCES_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
            endif()

            # cmake_language(DEFER CALL ...) only evaluates variable references in its <args> at the time the
            # deferred call actually runs (end of the current directory's CMakeLists.txt) - by then this
            # function's local ARG_* variables no longer exist. Wrapping the registration in
            # cmake_language(EVAL CODE ...) bakes the current values in now as literal bracket-quoted arguments,
            # instead of having them re-evaluated (as empty) later.
            cmake_language(EVAL CODE "
                cmake_language(DEFER DIRECTORY [[${CMAKE_CURRENT_SOURCE_DIR}]] CALL _ov_check_all_sources_listed
                    TARGET [[${ARG_NAME}]]
                    DIRECTORY [[${ARG_CHECK_SOURCES_DIRECTORY}]]
                    EXTENSIONS [[${ARG_CHECK_SOURCES_EXTENSIONS}]]
                    EXCLUDE_FILES [[${ARG_CHECK_SOURCES_EXCLUDE_FILES}]]
                    EXCLUDE_DIRECTORIES [[${ARG_CHECK_SOURCES_EXCLUDE_DIRECTORIES}]]
                    EXCLUDE_TARGETS [[${ARG_CHECK_SOURCES_EXCLUDE_TARGETS}]])
            ")
        endif()
    endif()

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
            # PRE_TEST keeps enumeration out of the build: it runs under ctest
            gtest_discover_tests(${ARG_NAME}
                DISCOVERY_MODE PRE_TEST
                DISCOVERY_TIMEOUT 300
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

#[=[
Validates that every source file matching the given extensions that exists under DIRECTORY is also listed in the SOURCES
property of TARGET.  Emits SEND_ERROR for each unlisted file so the developer sees all problems before configure stops.

This is an internal helper, private to ov_add_test_target - it is not meant to be called directly from a
CMakeLists.txt.

Usage:
_ov_check_all_sources_listed(
    TARGET      <target-name>
    DIRECTORY   <root-directory-to-scan>
    [EXTENSIONS <ext1> [<ext2> ...]]   # defaults to "cpp"
)
]=]
function(_ov_check_all_sources_listed)
    cmake_parse_arguments(_ov_casl "" "TARGET;DIRECTORY" "EXTENSIONS;EXCLUDE_DIRECTORIES;EXCLUDE_TARGETS;EXCLUDE_FILES" ${ARGN})

    if(NOT _ov_casl_TARGET)
        message(FATAL_ERROR "_ov_check_all_sources_listed: TARGET is required")
    endif()

    if(NOT _ov_casl_DIRECTORY)
        message(FATAL_ERROR "_ov_check_all_sources_listed: DIRECTORY is required")
    endif()

    if(NOT _ov_casl_EXTENSIONS)
        set(_ov_casl_EXTENSIONS cpp)
    endif()

    get_target_property(_ov_casl_registered_sources ${_ov_casl_TARGET} SOURCES)

    # Collect sources from all excluded targets
    set(_ov_casl_excluded_sources)

    foreach(_ov_casl_excl_target IN LISTS _ov_casl_EXCLUDE_TARGETS)
        if(TARGET ${_ov_casl_excl_target})
            get_target_property(_ov_casl_excl_target_sources ${_ov_casl_excl_target} SOURCES)

            if(_ov_casl_excl_target_sources)
                list(APPEND _ov_casl_excluded_sources ${_ov_casl_excl_target_sources})
            endif()
        endif()
    endforeach()

    set(_ov_casl_glob_patterns)

    foreach(_ov_casl_ext IN LISTS _ov_casl_EXTENSIONS)
        list(APPEND _ov_casl_glob_patterns "${_ov_casl_DIRECTORY}/*.${_ov_casl_ext}")
    endforeach()

    file(GLOB_RECURSE _ov_casl_disk_files CONFIGURE_DEPENDS ${_ov_casl_glob_patterns})

    foreach(_ov_casl_file IN LISTS _ov_casl_disk_files)
        # Skip individually excluded files
        if(_ov_casl_file IN_LIST _ov_casl_EXCLUDE_FILES)
            continue()
        endif()

        # Skip files that belong to an excluded target
        if(_ov_casl_file IN_LIST _ov_casl_excluded_sources)
            continue()
        endif()

        # Skip files under any excluded directory
        set(_ov_casl_excluded FALSE)

        foreach(_ov_casl_excl_dir IN LISTS _ov_casl_EXCLUDE_DIRECTORIES)
            cmake_path(IS_PREFIX _ov_casl_excl_dir "${_ov_casl_file}" NORMALIZE _ov_casl_is_prefix)

            if(_ov_casl_is_prefix)
                set(_ov_casl_excluded TRUE)
                break()
            endif()
        endforeach()

        if(_ov_casl_excluded)
            continue()
        endif()

        if(NOT _ov_casl_file IN_LIST _ov_casl_registered_sources)
            message(SEND_ERROR
                "[${_ov_casl_TARGET}] '${_ov_casl_file}' exists on disk but is not listed in target sources and "
                "will not be compiled. Add it to the appropriate CMakeLists.txt or sources.cmake.")
        endif()
    endforeach()
endfunction()
