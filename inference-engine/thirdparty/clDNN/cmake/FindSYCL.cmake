# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindSYCL
-------

SYCL Library to verify SYCL compatability of CMAKE_CXX_COMPILER
and passes relevant compiler flags.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``SYCL_FOUND``
  True if the system has the SYCL library.
``SYCL_LANGUAGE_VERSION``
  The SYCL language spec version by Compiler.
``SYCL_INCLUDE_DIR``
  Include directories needed to use SYCL.
``SYCL_IMPLEMENTATION_ID``
  The SYCL compiler variant.
``SYCL_FLAGS``
  SYCL specific flags for the compiler.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``SYCL_INCLUDE_DIR``
  The directory containing ``sycl.hpp``.
``SYCL_LIBRARY_DIR``
  The path to the SYCL library.
``SYCL_FLAGS``
  SYCL specific flags for the compiler.
``SYCL_LANGUAGE_VERSION``
  The SYCL language spec version by Compiler.


.. note::

  For now, user needs to set -DCMAKE_CXX_COMPILER or environment of
  CXX pointing to SYCL compatible compiler  ( eg: clang++, icpx)

  Note: do not set to DPCPP compiler. If set to a Compiler family
  that supports dpcpp ( eg: IntelLLVM) both DPCPP and SYCL
  features are enabeld.

  And add this package to user's Cmake config file.

  .. code-block:: cmake

    find_package(SYCL REQUIRED)

#]=======================================================================]

include(FindPackageHandleStandardArgs)

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  # TODO add dependency package module checks, if any
endif()

find_program(SYCL_COMPILER clang++)
if (NOT SYCL_COMPILER)
    message(FATAL_ERROR "DPC++ compiler was not found")
endif()

message("DPCPP=${SYCL_COMPILER}")

# Function to write a test case to verify SYCL features.

function(SYCL_FEATURE_TEST_WRITE src)

  set(pp_if "#if")
  set(pp_endif "#endif")

  set(SYCL_TEST_CONTENT "")
  string(APPEND SYCL_TEST_CONTENT "#include <iostream>\nusing namespace std;\n")
  string(APPEND SYCL_TEST_CONTENT "int main(){\n")

  # Feature tests goes here

  string(APPEND SYCL_TEST_CONTENT "${pp_if} defined(SYCL_LANGUAGE_VERSION)\n")
  string(APPEND SYCL_TEST_CONTENT "cout << \"SYCL_LANGUAGE_VERSION=\"<<SYCL_LANGUAGE_VERSION<<endl;\n")
  string(APPEND SYCL_TEST_CONTENT "${pp_endif}\n")

  string(APPEND SYCL_TEST_CONTENT "return 0;}\n")

  file(WRITE ${src} "${SYCL_TEST_CONTENT}")

endfunction()

# Function to Build the feature check test case.

function(SYCL_FEATURE_TEST_BUILD TEST_SRC_FILE TEST_EXE)

  # Convert CXX Flag string to list
  set(SYCL_CXX_FLAGS_LIST "${SYCL_FLAGS}")
  separate_arguments(SYCL_CXX_FLAGS_LIST)

  # Spawn a process to build the test case.
  execute_process(
    COMMAND "${SYCL_COMPILER}"
    ${SYCL_CXX_FLAGS_LIST}
    ${TEST_SRC_FILE}
    "-o"
    ${TEST_EXE}
    WORKING_DIRECTORY ${SYCL_TEST_DIR}
    OUTPUT_VARIABLE output ERROR_VARIABLE output
    COMMAND_ECHO STDOUT
    OUTPUT_FILE ${SYCL_TEST_DIR}/Compile.log
    RESULT_VARIABLE result
    TIMEOUT 20
    )

  # Verify if test case build properly.
  if(result)
    message("SYCL feature test compile failed!")
    message("compile output is: ${output}")
  endif()

  # TODO: what to do if it doesn't build

endfunction()

# Function to run the test case to generate feature info.

function(SYCL_FEATURE_TEST_RUN TEST_EXE)

  # Spawn a process to run the test case.

  execute_process(
    COMMAND ${TEST_EXE}
    WORKING_DIRECTORY ${SYCL_TEST_DIR}
    OUTPUT_VARIABLE output ERROR_VARIABLE output
    RESULT_VARIABLE result
    TIMEOUT 20
    )

  # Verify the test execution output.
  if(test_result)
    set(SYCL_FOUND False)
    message(FATAL_ERROR "SYCL: feature test execution failed!!")
  endif()
  # TODO: what iff the result is false.. error or ignore?

  set( test_result "${result}" PARENT_SCOPE)
  set( test_output "${output}" PARENT_SCOPE)

endfunction()


# Function to extract the information from test execution.
function(SYCL_FEATURE_TEST_EXTRACT test_output)

  string(REGEX REPLACE "\n" ";" test_output_list "${test_output}")

  set(SYCL_LANGUAGE_VERSION "")
  foreach(strl ${test_output_list})
     if(${strl} MATCHES "^SYCL_LANGUAGE_VERSION=([A-Za-z0-9_]+)$")
       string(REGEX REPLACE "^SYCL_LANGUAGE_VERSION=" "" extracted_sycl_lang "${strl}")
       set(SYCL_LANGUAGE_VERSION ${extracted_sycl_lang})
     endif()
  endforeach()

  set(SYCL_LANGUAGE_VERSION "${SYCL_LANGUAGE_VERSION}" PARENT_SCOPE)
endfunction()

if(SYCL_COMPILER)
  # TODO ensure CMAKE_LINKER and CMAKE_CXX_COMPILER are same/supports SYCL.
  # set(CMAKE_LINKER ${SYCL_COMPILER})

  # use REALPATH to resolve symlinks
  get_filename_component(_REALPATH_SYCL_COMPILER "${SYCL_COMPILER}" REALPATH)
  get_filename_component(SYCL_BIN_DIR "${_REALPATH_SYCL_COMPILER}" DIRECTORY)
  get_filename_component(SYCL_PACKAGE_DIR "${SYCL_BIN_DIR}" DIRECTORY CACHE)

  # Find Include path from binary
  find_file(SYCL_INCLUDE_DIR
    NAMES
      include
    HINTS
      ${SYCL_PACKAGE_DIR}
    NO_DEFAULT_PATH
      )

  # Find Library directory
  find_file(SYCL_LIBRARY_DIR
    NAMES
      lib lib64
    HINTS
      ${SYCL_PACKAGE_DIR}
    NO_DEFAULT_PATH
      )

  find_library(SYCL_LIBRARY
    NAMES sycl
    PATHS
      ${SYCL_LIBRARY_DIR}
  )
endif()

set(SYCL_FLAGS "-fsycl")

# And now test the assumptions.

# Create a clean working directory.
set(SYCL_TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/TESTSYCL")
file(REMOVE_RECURSE ${SYCL_TEST_DIR})
file(MAKE_DIRECTORY ${SYCL_TEST_DIR})

# Create the test source file
set(TEST_SRC_FILE "${SYCL_TEST_DIR}/sycl_features.cpp")
set(TEST_EXE "${TEST_SRC_FILE}.exe")
SYCL_FEATURE_TEST_WRITE(${TEST_SRC_FILE})

# Build the test and create test executable
SYCL_FEATURE_TEST_BUILD(${TEST_SRC_FILE} ${TEST_EXE})

# Execute the test to extract information
SYCL_FEATURE_TEST_RUN(${TEST_EXE})

# Extract test output for information
SYCL_FEATURE_TEST_EXTRACT(${test_output})

# As per specification, all the SYCL compatible compilers should
# define macro  SYCL_LANGUAGE_VERSION
string(COMPARE EQUAL "${SYCL_LANGUAGE_VERSION}" "" nosycllang)
if(nosycllang)
  set(SYCL_FOUND False)
  message(FATAL_ERROR "SYCL: It appears that the ${SYCL_COMPILER} does not support SYCL")
endif()

message(STATUS "The SYCL compiler is ${SYCL_COMPILER}")
message(STATUS "The SYCL Flags are ${SYCL_FLAGS}")
message(STATUS "The SYCL Language Version is ${SYCL_LANGUAGE_VERSION}")

find_package_handle_standard_args(
  SYCL
  FOUND_VAR SYCL_FOUND
  REQUIRED_VARS SYCL_INCLUDE_DIR SYCL_LIBRARY_DIR SYCL_FLAGS
  VERSION_VAR SYCL_LANGUAGE_VERSION
  REASON_FAILURE_MESSAGE "${SYCL_REASON_FAILURE}")

# Include in Cache
set(SYCL_LANGUAGE_VERSION "${SYCL_LANGUAGE_VERSION}" CACHE STRING "SYCL Language version")
set(SYCL_INCLUDE_DIR "${SYCL_INCLUDE_DIR}" CACHE FILEPATH "SYCL Include directory")
set(SYCL_LIBRARY_DIR "${SYCL_LIBRARY_DIR}" CACHE FILEPATH "SYCL Library Directory")
set(SYCL_FLAGS "${SYCL_FLAGS}" CACHE STRING "SYCL flags for the compiler")
set(SYCL_COMPILER "${SYCL_COMPILER}" CACHE STRING "SYCL compiler path")
set(SYCL_LIBRARY "${SYCL_LIBRARY}" CACHE STRING "SYCL library")
