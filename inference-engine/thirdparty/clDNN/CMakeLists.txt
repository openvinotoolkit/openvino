# Copyright (c) 2016 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required (VERSION 3.1 FATAL_ERROR)
include(CheckCXXCompilerFlag)

# Register Intel helper modules.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/utils/build/cmake/modules")
include(IntelHelpers)


# ======================================================================================================
# ==================================== BUILD CONFIGURATIONS (part 1) ===================================
# ======================================================================================================
# The section must be before project definition which is a point for configurations generation
# for multi-configuration generators.

# Available configuration types:
# Two standard configurations.
set(CMAKE_CONFIGURATION_TYPES
    "Debug"
    "Release"
  )
set(CMAKE_CONFIGURATION_TYPES ${CMAKE_CONFIGURATION_TYPES} CACHE INTERNAL "Available build configurations.")

# ======================================================================================================
# ======================================================================================================
# ======================================================================================================

# Name of project (helper constant variable).
set(CLDNN__PROJ_NAME "clDNN")

# Default languages: C, C++.
project("${CLDNN__PROJ_NAME}")


# ======================================================================================================
# ==================================== BUILD CONFIGURATIONS (part 2) ===================================
# ======================================================================================================
# The section must be after project definition and its supported languages because otherwise
# used variables are not available yet.

#intel_custom_build_add("<custom config>" "Release") # No custom configuration at the momeent.
# Populating global property with list of debug configurations.
set_property(GLOBAL PROPERTY DEBUG_CONFIGURATIONS "Debug")
# Use solution folders.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# ======================================================================================================
# ======================================================================================================
# ======================================================================================================


# ======================================================================================================
# ====================================== HELPER CONSTANT VARIABLES =====================================
# ======================================================================================================

# Path which points to main directory of project.
set(CLDNN__MAIN_DIR      "${CMAKE_CURRENT_SOURCE_DIR}")

# Path which points to directory with common dependencies (internal and 3-rd party).
set(CLDNN__COMMON_DIR    "${CMAKE_CURRENT_SOURCE_DIR}/common")

# Path which points to directory with interface for framework.
set(CLDNN__API_DIR       "${CMAKE_CURRENT_SOURCE_DIR}/api")

# Path which points to directory with interface extension for framework.
set(CLDNN__API_EXTENSION_DIR       "${CMAKE_CURRENT_SOURCE_DIR}/api_extension")

# Path which points to directory with interface for framework.
set(CLDNN__KERNEL_SELECTOR_DIR       "${CMAKE_CURRENT_SOURCE_DIR}/kernel_selector")

# Path which points to directory with binaries for Intel OpenCL SDK ICD (Installable Client Driver).
set(CLDNN__IOCL_ICD_DIR  "${CLDNN__COMMON_DIR}/intel_ocl_icd")

# Path which points to directory with C++ bindings for OpenCL (header files + wrapper that disables specific warnings).
set(CLDNN__KHR_CLHPP_DIR "${CLDNN__COMMON_DIR}/khronos_ocl_clhpp")

# Path which points to directory with fused version of googletest framework (with fused googlemock as well).
set(CLDNN__GTEST_DIR     "${CLDNN__COMMON_DIR}/googletest-fused")

# Build targets settings.

# Path which points to default root directory for compilation output.
set(CLDNN_BUILD__DEFAULT_OUT_ROOT "${CMAKE_BINARY_DIR}")
set(CLDNN__OUTPUT_DIR, "${CMAKE_BINARY_DIR}")

# Prefix for all targets in internal pass.
set(CLDNN_BUILD__PROJ_NAME_PREFIX "")

# Single/multi-configuration generator helpers.
if(CMAKE_CFG_INTDIR STREQUAL ".")
  set(CLDNN__TARGET_CFG_VAR "${CMAKE_BUILD_TYPE}")
  set(CLDNN__MULTI_CFG_GEN  NO)
else()
  set(CLDNN__TARGET_CFG_VAR "${CMAKE_CFG_INTDIR}")
  set(CLDNN__MULTI_CFG_GEN  YES)
endif()

# Code generation settings.

# Path which points to root directory where code generated elements are created.
set(CLDNN__CODEGEN_BASEDIR "${CMAKE_CURRENT_BINARY_DIR}/codegen")
# Path which points to root directory where code generated elements are created
# (specific to build configuration).
set(CLDNN__CODEGEN_DIR     "${CLDNN__CODEGEN_BASEDIR}/${CLDNN__TARGET_CFG_VAR}")

# Path which points to automatically included directory with code generated elements
# (to support "copy-if-different" optimization).
set(CLDNN__CODEGEN_INCDIR  "${CLDNN__CODEGEN_DIR}/include")

# Version file.
set(CLDNN__VERSION_FILE_NAME "${CLDNN__MAIN_DIR}/version.json")

# ======================================================================================================
# ======================================================================================================
# ======================================================================================================


# ======================================================================================================
# =========================================== HELPER FUNCTIONS =========================================
# ======================================================================================================

# Writes debug message in project (it will be presented only if debug is enabled).
#
# @param text Text to present.
function(cldnn_dmsg text)
  if(CLDNN__CMAKE_DEBUG)
    message(STATUS "[clDNN][D] ${text}")
  endif()
endfunction()

# ======================================================================================================

# Locates and detects all versions of specific component.
#
# The highest version of component is also returned as default version to use.
#
# @param retVarNameVersions       Name of variable placeholder with all versions of specified component
#                                 which are available in the build.
# @param retVarNamePaths          Name of variable placeholder with all paths to versions of specified
#                                 component which are available in build. The paths map to corresponding
#                                 versions from retVarNameVersions (1:1).
# @param retVarNameDefaultVersion Default version to use. It is highest specified version, or 0 if no
#                                 version of component was found.
# @param componentName            Descriptive name of a component. It is presentated in warning and
#                                 debug messages.
# @param componentBaseDir         Base directory for component (directory containing version
#                                 subdirectories).
# @param [componentReqFile [...]] Optional list of component required files to check for existence.
#                                 The componentReqFile can contain file(GLOB) wildcards. If multiple
#                                 are specified, checked version is valid when all files are existing
#                                 (or GLOB return at least one file for each entry).
function(cldnn_locate_component_versions retVarNameVersions retVarNamePaths retVarNameDefaultVersion
                                         componentName componentBaseDir)
  if(NOT EXISTS "${componentBaseDir}")
    cldnn_dmsg("Base directory for ${componentName} does not exist at ${componentBaseDir}.")
    set("${retVarNameVersions}"       ""  PARENT_SCOPE)
    set("${retVarNamePaths}"          ""  PARENT_SCOPE)
    set("${retVarNameDefaultVersion}" "0" PARENT_SCOPE)
    return()
  endif()

  # Detecting all versions of added component in the project.
  file(GLOB __CLDNN_F_CompVersionPaths RELATIVE "${componentBaseDir}" "${componentBaseDir}/[0-9rRvV]*")
  set(__CLDNN_F_CompVersions "")
  set(__CLDNN_F_CompPaths    "")
  foreach(__CLDNN_F_CompVersionPath ${__CLDNN_F_CompVersionPaths})
    string(REPLACE ";" "\;" __CLDNN_F_CompVersionPath "${__CLDNN_F_CompVersionPath}") # [WA#1] Must escape ; again if occurred in item.
    if(__CLDNN_F_CompVersionPath MATCHES "^[rRvV]?([0-9]+([\\._][0-9]+([\\._][0-9]+([\\._][0-9]+)?)?)?)")
      string(REPLACE "_" "." __CLDNN_F_CompVersion "${CMAKE_MATCH_1}")
      set(__CLDNN_F_CompPath "${componentBaseDir}/${__CLDNN_F_CompVersionPath}")

      set(__CLDNN_F_CompPathValid TRUE)
      foreach(__CLDNN_F_CompReqFile ${ARGN})
        string(REPLACE ";" "\;" __CLDNN_F_CompReqFile "${__CLDNN_F_CompReqFile}") # [WA#1] Must escape ; again if occurred in item.
        set(__CLDNN_F_CompReqFileGlob "${__CLDNN_F_CompPath}/${__CLDNN_F_CompReqFile}")

        file(GLOB __CLDNN_F_CompReqTargets ${__CLDNN_F_CompReqFileGlob})
        list(LENGTH __CLDNN_F_CompReqTargets __CLDNN_F_CompReqTargetsCount)
        if(__CLDNN_F_CompReqTargetsCount LESS 1)
            message(WARNING "[clDNN] ${componentName} at \"${__CLDNN_F_CompPath}\" does not contain required files. It will be omitted.")
            set(__CLDNN_F_CompPathValid FALSE)
            break()
        endif()
      endforeach()
      unset(__CLDNN_F_CompReqFile)
      unset(__CLDNN_F_CompReqFileGlob)
      unset(__CLDNN_F_CompReqTargets)
      unset(__CLDNN_F_CompReqTargetsCount)

      if(__CLDNN_F_CompPathValid)
        cldnn_dmsg("Found internal ${componentName} (version: ${__CLDNN_F_CompVersion}) at path: ${__CLDNN_F_CompPath}")
        list(APPEND __CLDNN_F_CompVersions "${__CLDNN_F_CompVersion}")
        list(APPEND __CLDNN_F_CompPaths    "${__CLDNN_F_CompPath}")
      endif()
    endif()
  endforeach()
  unset(__CLDNN_F_CompVersionPaths)
  unset(__CLDNN_F_CompVersionPath)
  unset(__CLDNN_F_CompVersion)
  unset(__CLDNN_F_CompPath)
  unset(__CLDNN_F_CompPathValid)

  # Selecting highest version of component as a default version.
  set(__CLDNN_F_CompDefaultVersion "0")
  foreach(__CLDNN_F_CompVersion ${__CLDNN_F_CompVersions})
    if(__CLDNN_F_CompVersion VERSION_GREATER __CLDNN_F_CompDefaultVersion)
      set(__CLDNN_F_CompDefaultVersion "${__CLDNN_F_CompVersion}")
    endif()
  endforeach()
  unset(__CLDNN_F_CompVersion)
  cldnn_dmsg("Selected default version of internal ${componentName}: ${__CLDNN_F_CompDefaultVersion}")

  set("${retVarNameVersions}"       ${__CLDNN_F_CompVersions}         PARENT_SCOPE)
  set("${retVarNamePaths}"          ${__CLDNN_F_CompPaths}            PARENT_SCOPE)
  set("${retVarNameDefaultVersion}" "${__CLDNN_F_CompDefaultVersion}" PARENT_SCOPE)
endfunction()


# ======================================================================================================
# ======================================================================================================
# ======================================================================================================


# Expressing CMake setting for current build configuration as option and providing a way to correct it (treat as case-insensitive).
if(DEFINED CMAKE_BUILD_TYPE)
  string(TOLOWER "${CMAKE_BUILD_TYPE}" __CLDNN_BuildType)
  string(STRIP "${__CLDNN_BuildType}" __CLDNN_BuildType)

  if(__CLDNN_BuildType MATCHES "^release$")
    set(__CLDNN_BuildType "Release")
  elseif(__CLDNN_BuildType MATCHES "^debug$")
    set(__CLDNN_BuildType "Debug")
  else()
    set(__CLDNN_BuildType "Debug")
    if(NOT CLDNN__MULTI_CFG_GEN)
      message(WARNING "[clDNN] CMAKE_BUILD_TYPE: Unknown build configuration. The following configurations are available: ${CMAKE_CONFIGURATION_TYPES}.
          The \"${__CLDNN_BuildType}\" configuration will be used.

          This value has meaning only for single-configuration generators (like Make). It will be ignored for MSVC/XCode."
        )
    endif()
  endif()
else()
  set(__CLDNN_BuildType "Debug")
  if(NOT CLDNN__MULTI_CFG_GEN)
    message(WARNING "[clDNN] CMAKE_BUILD_TYPE: No build configuration specified. The following configurations are available: ${CMAKE_CONFIGURATION_TYPES}.
        The \"${__CLDNN_BuildType}\" configuration will be used.

        This value has meaning only for single-configuration generators (like Make). It will be ignored for MSVC/XCode."
      )
  endif()
endif()
set(CMAKE_BUILD_TYPE "${__CLDNN_BuildType}")
unset(__CLDNN_BuildType)

# ======================================================================================================

# Detecting, setting and validating target architecture for compilation.

set(__CLDNN_RequestedArch "")

intel_arch_detect(__CLDNN_DetectedArch_Target __CLDNN_DetectedArch_Host __CLDNN_RequestedArch)
intel_arch_validate(__CLDNN_DetectedArchValid_Target "${__CLDNN_DetectedArch_Target}")
unset(__CLDNN_DetectedArch_Host)
unset(__CLDNN_RequestedArch)

if(DEFINED CLDNN__ARCHITECTURE_TARGET)
  intel_arch_normalize(__CLDNN_Arch_Target "${CLDNN__ARCHITECTURE_TARGET}")
elseif(__CLDNN_DetectedArchValid_Target)
  set(__CLDNN_Arch_Target "${__CLDNN_DetectedArch_Target}")
  message("[clDNN] CLDNN__ARCHITECTURE_TARGET: Target architecture is not specified. Trying to deduce it from context.")
else()
  message(FATAL_ERROR "[clDNN] CLDNN__ARCHITECTURE_TARGET: Target architecture is not specified and cannot be deduced from context.
      Please specify one, e.g. Windows32, Linux64, Android32, Darwin32, ..."
    )
endif()
set(CLDNN__ARCHITECTURE_TARGET "${__CLDNN_Arch_Target}")
unset(__CLDNN_Arch_Target)
unset(__CLDNN_DetectedArchValid_Target)

intel_arch_validate(__CLDNN_ArchValid_Target "${CLDNN__ARCHITECTURE_TARGET}")
if(NOT __CLDNN_ArchValid_Target)
  message(FATAL_ERROR "[clDNN] CLDNN__ARCHITECTURE_TARGET: Target architecture \"${CLDNN__ARCHITECTURE_TARGET}\" is invalid.
      Please specify correct one, e.g. Windows32, Linux64, Android32, Darwin32, ..."
    )
endif()
unset(__CLDNN_ArchValid_Target)

# ======================================================================================================

# Detecting all versions of added Intel OpenCL SDK ICDs in the project.
if(NOT EXISTS "${CLDNN__IOCL_ICD_DIR}")
  if(NOT CLDNN__IOCL_ICD_USE_EXTERNAL)
    message(FATAL_ERROR "[clDNN] Root for ICDs for Intel OpenCL SDK is invalid or does not exist.
      Please make sure that ICDs are added into \"${CLDNN__IOCL_ICD_DIR}\".")
  endif()
endif()

cldnn_locate_component_versions(__CLDNN_IOclIcdVersions __CLDNN_IOclIcdPaths __CLDNN_IOclIcdDefaultVersion
    "ICD for Intel OpenCL SDK" "${CLDNN__IOCL_ICD_DIR}"
    "*/include/CL/opencl.h"
  )

if(__CLDNN_IOclIcdDefaultVersion VERSION_EQUAL "0")
  if(NOT CLDNN__IOCL_ICD_USE_EXTERNAL)
    message(FATAL_ERROR "[clDNN] Intel OpenCL SDK ICD required for build cannot be located.")
  endif()
endif()

# ======================================================================================================
# ============================================ CMAKE OPTIONS ===========================================
# ======================================================================================================

# Current build configuration (only for single-configuration generators).
set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "Current build configuration (only for single-configuration generators)." FORCE)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})

# ======================================================================================================

# Target architecture for compilation.
set(CLDNN__ARCHITECTURE_TARGET "${CLDNN__ARCHITECTURE_TARGET}" CACHE STRING "Target architecture for compilation." FORCE)
if(MSVC)
  set_property(CACHE CLDNN__ARCHITECTURE_TARGET PROPERTY STRINGS "Windows32" "Windows64")
elseif(DEFINED XCODE_VERSION)
  set_property(CACHE CLDNN__ARCHITECTURE_TARGET PROPERTY STRINGS "Darwin32" "Darwin64")
endif()

# ======================================================================================================
if(DEFINED CLDNN__OUTPUT_DIR)
  set(CLDNN__OUTPUT_BIN_DIR "${CLDNN__OUTPUT_DIR}" CACHE PATH "Output directory path where the final exetuables, examples and tests will be stored.")
  set(CLDNN__OUTPUT_LIB_DIR "${CLDNN__OUTPUT_DIR}" CACHE PATH "Output directory path where the final libraries will be stored.")
elseif(NOT DEFINED CLDNN__OUTPUT_BIN_DIR AND NOT DEFINED CLDNN__OUTPUT_LIB_DIR)
  # Output directory path where the final libraries, examples and tests will be stored.
  if(CLDNN__MULTI_CFG_GEN)
    # Multi-configuration generators automatically append build type subdirectory.
    set(__CLDNN_OutDir "${CLDNN_BUILD__DEFAULT_OUT_ROOT}/${CLDNN__ARCHITECTURE_TARGET}")
  else()
    set(__CLDNN_OutDir "${CLDNN_BUILD__DEFAULT_OUT_ROOT}/${CLDNN__ARCHITECTURE_TARGET}/${CMAKE_BUILD_TYPE}")
  endif()
  set(CLDNN__OUTPUT_BIN_DIR "${__CLDNN_OutDir}" CACHE PATH "Output directory path where the final exetuables, examples and tests will be stored.")
  set(CLDNN__OUTPUT_LIB_DIR "${CLDNN__OUTPUT_BIN_DIR}" CACHE PATH "Output directory path where the final libraries will be stored.")
  unset(__CLDNN_OutDir)
endif()
# ======================================================================================================

# Intel OpenCL SDK ICD: Version of ICD used to build clDNN.
set(CLDNN__IOCL_ICD_VERSION "${__CLDNN_IOclIcdDefaultVersion}" CACHE STRING "Intel OpenCL SDK ICD: Version of ICD used to build clDNN framework.")
mark_as_advanced(CLDNN__IOCL_ICD_VERSION)
set_property(CACHE CLDNN__IOCL_ICD_VERSION PROPERTY STRINGS ${__CLDNN_IOclIcdVersions})

# ======================================================================================================

set(CLDNN__IOCL_ICD_USE_EXTERNAL OFF CACHE BOOL "Intel OpenCL SDK ICD: Try to use externally-intalled Intel OpenCL SDK ICD.")
mark_as_advanced(CLDNN__IOCL_ICD_USE_EXTERNAL)

# ======================================================================================================

# Include and build: Core of clDNN framework.
set(CLDNN__INCLUDE_CORE ON CACHE BOOL "Include and build: clDNN core.")
mark_as_advanced(CLDNN__INCLUDE_CORE)

# ======================================================================================================

# Include and build: Kernel selector for clDNN framework.
set(CLDNN__INCLUDE_KERNEL_SELECTOR ON CACHE BOOL "Include and build: clDNN kernel selector.")
mark_as_advanced(CLDNN__INCLUDE_KERNEL_SELECTOR)

# ======================================================================================================

# Include and build: Tests (unit tests and small acceptance tests) for clDNN framework.
set(CLDNN__INCLUDE_TESTS ON CACHE BOOL "Include and build: clDNN framework's tests.")
mark_as_advanced(CLDNN__INCLUDE_TESTS)

# ======================================================================================================

# Include and build: Core Internal Tests (unit tests and small acceptance tests) for core internal clDNN framework mechanisms.
set(CLDNN__INCLUDE_CORE_INTERNAL_TESTS ON CACHE BOOL "Include and build: clDNN framework's core internal tests.")
mark_as_advanced(CLDNN__INCLUDE_CORE_INTERNAL_TESTS)

# ======================================================================================================


# Include and build: clDNN tutorial.
set(CLDNN__INCLUDE_TUTORIAL ON CACHE BOOL "Include and build: clDNN Tutorial.")
mark_as_advanced(CLDNN__INCLUDE_TUTORIAL)

# ======================================================================================================

# Run (requires CLDNN__INCLUDE_TESTS to be true): Tests (unit tests and small acceptance tests) for clDNN framework.
set(CLDNN__RUN_TESTS OFF CACHE BOOL "Run: clDNN framework's tests.")
mark_as_advanced(CLDNN__RUN_TESTS)

# ======================================================================================================

# Run (requires CLDNN__INCLUDE_CORE_INTERNAL_TESTS to be true): Tests (unit tests and small acceptance core internal tests) for clDNN framework.
set(CLDNN__RUN_CORE_INTERNAL_TESTS OFF CACHE BOOL "Run: clDNN framework's core internal tests.")
mark_as_advanced(CLDNN__RUN_CORE_INTERNAL_TESTS)

# ======================================================================================================

# Compile / Link: Use static C++ Runtime library.
set(CLDNN__COMPILE_LINK_USE_STATIC_RUNTIME OFF CACHE BOOL "Compile / Link: Use static version of C++ Runtime library instead of shared one.")
mark_as_advanced(CLDNN__COMPILE_LINK_USE_STATIC_RUNTIME)

# ======================================================================================================

# Compile / Link: Allow unsafe binary size optimizations.
set(CLDNN__COMPILE_LINK_ALLOW_UNSAFE_SIZE_OPT ON CACHE BOOL "Compile / Link: Allow unsafe binary size optimizations.")
mark_as_advanced(CLDNN__COMPILE_LINK_ALLOW_UNSAFE_SIZE_OPT)

# ======================================================================================================

# CMake: Enables debug trace messages in adapter project.
set(CLDNN__CMAKE_DEBUG OFF CACHE BOOL "CMake: Enables debug trace messages in clDNN project.")
mark_as_advanced(CLDNN__CMAKE_DEBUG)

# ======================================================================================================
# ======================================================================================================

# Minimum versions of compilers.
set(CLDNN__MIN_COMPILER_VERSION__MSVC  "19.0")
set(CLDNN__MIN_COMPILER_VERSION__CLANG "3.5")
set(CLDNN__MIN_COMPILER_VERSION__ICC   "17.0")
set(CLDNN__MIN_COMPILER_VERSION__GCC   "4.8")

# ======================================================================================================

# Checking whether tests can be run.
if((NOT CLDNN__INCLUDE_TESTS) AND CLDNN__RUN_TESTS)
  message(WARNING "[clDNN] CLDNN__RUN_TESTS: Selected running of tests, but test are not built. Option will be disabled.")
  set(CLDNN__RUN_TESTS OFF)
endif()

# ======================================================================================================

# Checking whether tests can be run.
if((NOT CLDNN__INCLUDE_CORE_INTERNAL_TESTS) AND CLDNN__RUN_CORE_INTERNAL_TESTS)
  message(WARNING "[clDNN] CLDNN__INCLUDE_CORE_INTERNAL_TESTS: Selected running of core internal tests, but test are not built. Option will be disabled.")
  set(CLDNN__RUN_CORE_INTERNAL_TESTS OFF)
endif()

# ======================================================================================================

# Check for python 2.7 interpreter (required tool).
find_package(PythonInterp 2.7)
if(NOT PYTHONINTERP_FOUND)
  message(WARNING "[clDNN] Project requires Python 2.7 interpreter to build (with python loader). CMake could not detect it correctly.
      If you have installed this interpreter, please disregard this warning or specify PYTHON_EXECUTABLE in CMake command-line."
    )
endif()

# ======================================================================================================

# Setting helper variables for component paths.
intel_arch_get_os(__CLDNN_TargetOs "${CLDNN__ARCHITECTURE_TARGET}")
string(TOLOWER "${__CLDNN_TargetOs}" __CLDNN_TargetOs)

intel_arch_get_cpu(__CLDNN_TargetCpu "${CLDNN__ARCHITECTURE_TARGET}")
string(TOLOWER "${__CLDNN_TargetCpu}" __CLDNN_TargetCpu)

if(__CLDNN_TargetCpu STREQUAL "32")
  set(__CLDNN_TargetCpuDir "x86")
elseif(__CLDNN_TargetCpu STREQUAL "64")
  set(__CLDNN_TargetCpuDir "x64")
else()
  set(__CLDNN_TargetCpuDir "${__CLDNN_TargetCpu}")
endif()

if((CMAKE_C_COMPILER_ID MATCHES "^Clang$") OR (CMAKE_CXX_COMPILER_ID MATCHES "^Clang$"))
  set(__CLDNN_TargetCompilerDir "clang")
else()
  set(__CLDNN_TargetCompilerDir ".")
endif()

# ======================================================================================================

# Selecting Intel OpenCL SDK version and path.
if (CLDNN__IOCL_ICD_USE_EXTERNAL)
  if(NOT DEFINED ENV{INTELOCLSDKROOT})
    message(FATAL_ERROR "[clDNN] CLDNN__IOCL_ICD_USE_EXTERNAL: Usage of external Intel OpenCL SDK ICD was selected, but the INTELOCLSDKROOT environment variable cannot be found.
      Make sure that SDK is installed on the host.")
  endif()
  if(NOT EXISTS "$ENV{INTELOCLSDKROOT}/include/CL/opencl.h")
    message(FATAL_ERROR "[clDNN] CLDNN__IOCL_ICD_USE_EXTERNAL: Usage of external Intel OpenCL SDK ICD was selected, but the INTELOCLSDKROOT environment variable points to invalid directory.
      Make sure that SDK is installed on the host.")
  endif()
  set(CLDNN__IOCL_ICD_ROOT "$ENV{INTELOCLSDKROOT}")
else()
  list(FIND __CLDNN_IOclIcdVersions "${CLDNN__IOCL_ICD_VERSION}" __CLDNN_IOclIcdVersionIdx)
  if(__CLDNN_IOclIcdVersionIdx LESS 0)
    message(FATAL_ERROR "[clDNN] CLDNN__IOCL_ICD_VERSION: Selected version of Intel OpenCL SDK ICD cannot be found.
      Please use one of following values: ${__CLDNN_IOclIcdVersions}.")
  endif()
  list(GET __CLDNN_IOclIcdPaths ${__CLDNN_IOclIcdVersionIdx} CLDNN__IOCL_ICD_ROOT)
endif()

set(CLDNN__IOCL_ICD_ROOT "${CLDNN__IOCL_ICD_ROOT}" CACHE INTERNAL "Path to Intel OpenCL SDK ICD used to build clDNN framework.")

# Select SDK subdirectories with headers, binaries and libraries (based on architecture, cpu, generator and SDK type).
if(CLDNN__IOCL_ICD_USE_EXTERNAL)
  set(CLDNN__IOCL_ICD_INCDIRS "${CLDNN__IOCL_ICD_ROOT}/include" CACHE INTERNAL "Paths to interface headers for Intel OpenCL SDK ICD.")
  set(CLDNN__IOCL_ICD_STLDIRS "${CLDNN__IOCL_ICD_ROOT}/lib/${__CLDNN_TargetCpuDir}" CACHE INTERNAL "Paths to static libraries for Intel OpenCL SDK ICD.")

  set(__CLDNN_FindLibSuffixes ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_SHARED_LIBRARY_SUFFIX}")
  find_library(CLDNN_FIND__IOCL_ICD_SHLPATH
    NAMES OpenCL
    PATHS "${CLDNN__IOCL_ICD_ROOT}"
          "${CLDNN__IOCL_ICD_ROOT}/bin"
          "${CLDNN__IOCL_ICD_ROOT}/bin/${__CLDNN_TargetCpuDir}"
          "${CLDNN__IOCL_ICD_ROOT}/lib"
          "${CLDNN__IOCL_ICD_ROOT}/lib/${__CLDNN_TargetCpuDir}"
    NO_CMAKE_FIND_ROOT_PATH
  )
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${__CLDNN_FindLibSuffixes})

  if(CLDNN_FIND__IOCL_ICD_SHLPATH)
    get_filename_component(__CLDNN_IOclIcdSharedLibDir "${CLDNN_FIND__IOCL_ICD_SHLPATH}" DIRECTORY)
    set(CLDNN__IOCL_ICD_SHLDIRS "${__CLDNN_IOclIcdSharedLibDir}" CACHE INTERNAL "Paths to shared libraries for Intel OpenCL SDK ICD.")
  else()
    message(FATAL_ERROR "[clDNN] CLDNN__IOCL_ICD_USE_EXTERNAL: Usage of external Intel OpenCL SDK ICD was selected, but the ICD shared library cannot be located.
      Make sure that SDK is installed on the host.")
  endif()

  unset(CLDNN_FIND__IOCL_ICD_SHLPATH CACHE)
  unset(__CLDNN_FindLibSuffixes)
  unset(__CLDNN_IOclIcdSharedLibDir)
else()
  set(CLDNN__IOCL_ICD_INCDIRS "${CLDNN__IOCL_ICD_ROOT}/${__CLDNN_TargetOs}/include" CACHE INTERNAL "Paths to interface headers for Intel OpenCL SDK ICD.")
  set(CLDNN__IOCL_ICD_STLDIRS "${CLDNN__IOCL_ICD_ROOT}/${__CLDNN_TargetOs}/${CLDNN__TARGET_CFG_VAR}/lib/${__CLDNN_TargetCpuDir}" CACHE INTERNAL "Paths to static libraries for Intel OpenCL SDK ICD.")
  set(CLDNN__IOCL_ICD_SHLDIRS "${CLDNN__IOCL_ICD_ROOT}/${__CLDNN_TargetOs}/${CLDNN__TARGET_CFG_VAR}/bin/${__CLDNN_TargetCpuDir}" CACHE INTERNAL "Paths to shared libraries for Intel OpenCL SDK ICD.")
endif()

# Select link directory based on targeted OS.
# - on Windows: static libraries directory.
# - on others:  shared libraries directory.
if(__CLDNN_TargetOs MATCHES "^windows$")
  set(CLDNN__IOCL_ICD_LIBDIRS ${CLDNN__IOCL_ICD_STLDIRS} CACHE INTERNAL "Paths to libraries to link for Intel OpenCL SDK ICD.")
else()
  set(CLDNN__IOCL_ICD_LIBDIRS ${CLDNN__IOCL_ICD_SHLDIRS} CACHE INTERNAL "Paths to libraries to link for Intel OpenCL SDK ICD.")
endif()

unset(__CLDNN_IOclIcdVersions)
unset(__CLDNN_IOclIcdPaths)
unset(__CLDNN_IOclIcdDefaultVersion)
unset(__CLDNN_IOclIcdVersionIdx)


# ======================================================================================================
set(CLDNN_UTILS__RAPIDJSON_INCDIRS "utils/rapidjson" CACHE INTERNAL "Paths to interface headers for rapidjson.")
# ====================================== Version Calculation ===========================================

if(EXISTS "${CLDNN__VERSION_FILE_NAME}")
  file(READ "${CLDNN__VERSION_FILE_NAME}" __CLDNN_VersionFileContent)

  # Extracting JSON object with version.
  set(__CLDNN_VersionJsonObject "")
  if("${__CLDNN_VersionFileContent}" MATCHES "\\\"version\\\"[ \t\n\r]*:[ \t\n\r]*\\{[ \t\n\r]*(.*)[ \t\n\r]*\\}")
    set(__CLDNN_VersionJsonObject "${CMAKE_MATCH_1}")
  endif()

  # Extracting field values.
  set(__CLDNN_VersionJsonMajor   "0")
  set(__CLDNN_VersionJsonMinor   "-1")
  set(__CLDNN_VersionJsonBuild   "-1")
  set(__CLDNN_VersionJsonRevBase "0")
  set(__CLDNN_VersionJsonRevMin  "-1")
  if("${__CLDNN_VersionJsonObject}" MATCHES "\\\"major\\\"[ \t\n\r]*:[ \t\n\r]*([-+]?[0-9]+)[ \t\n\r#,}]")
    set(__CLDNN_VersionJsonMajor "${CMAKE_MATCH_1}")
  endif()
  if("${__CLDNN_VersionJsonObject}" MATCHES "\\\"minor\\\"[ \t\n\r]*:[ \t\n\r]*([-+]?[0-9]+)[ \t\n\r#,}]")
    set(__CLDNN_VersionJsonMinor "${CMAKE_MATCH_1}")
  endif()
  if("${__CLDNN_VersionJsonObject}" MATCHES "\\\"build\\\"[ \t\n\r]*:[ \t\n\r]*([-+]?[0-9]+)[ \t\n\r#,}]")
    set(__CLDNN_VersionJsonBuild "${CMAKE_MATCH_1}")
  endif()
  if("${__CLDNN_VersionJsonObject}" MATCHES "\\\"revision_base\\\"[ \t\n\r]*:[ \t\n\r]*([-+]?[0-9]+)[ \t\n\r#,}]")
    set(__CLDNN_VersionJsonRevBase "${CMAKE_MATCH_1}")
  endif()
  if("${__CLDNN_VersionJsonObject}" MATCHES "\\\"revision_min\\\"[ \t\n\r]*:[ \t\n\r]*([-+]?[0-9]+)[ \t\n\r#,}]")
    set(__CLDNN_VersionJsonRevMin "${CMAKE_MATCH_1}")
  endif()

  # Fetching revision identifier (first from env. variable, next from CMake variable).
  if(DEFINED "ENV{CLDNN__REV_ID}")
    set(__CLDNN_VersionRevId "$ENV{CLDNN__REV_ID}")
  elseif(DEFINED CLDNN__REV_ID)
    set(__CLDNN_VersionRevId "${CLDNN__REV_ID}")
  else()
    set(__CLDNN_VersionRevId "0")
  endif()

  # Calculating version components (WA for lack of support for unary operators in CMake math()).
  math(EXPR CLDNN__VERSION_MAJOR    "(0${__CLDNN_VersionJsonMajor})")
  math(EXPR CLDNN__VERSION_MINOR    "(0${__CLDNN_VersionJsonMinor})")
  math(EXPR CLDNN__VERSION_BUILD    "(0${__CLDNN_VersionJsonBuild})")
  math(EXPR CLDNN__VERSION_REVISION "(0${__CLDNN_VersionRevId})-(0${__CLDNN_VersionJsonRevBase})")
  math(EXPR __CLDNN_VersionJsonRevMin "(0${__CLDNN_VersionJsonRevMin})")
  if("${CLDNN__VERSION_REVISION}" LESS "${__CLDNN_VersionJsonRevMin}")
    set(CLDNN__VERSION_REVISION "${__CLDNN_VersionJsonRevMin}")
  endif()

  unset(__CLDNN_VersionFileContent)
  unset(__CLDNN_VersionJsonObject)
  unset(__CLDNN_VersionJsonMajor)
  unset(__CLDNN_VersionJsonMinor)
  unset(__CLDNN_VersionJsonBuild)
  unset(__CLDNN_VersionJsonRevBase)
  unset(__CLDNN_VersionJsonRevMin)
else()
  set(CLDNN__VERSION_MAJOR    "0")
  set(CLDNN__VERSION_MINOR    "-1")
  set(CLDNN__VERSION_BUILD    "-1")
  set(CLDNN__VERSION_REVISION "-1")
endif()

# ============================================= Status =================================================

# Display status.
message(STATUS "[clDNN] ======================== ${CLDNN__PROJ_NAME} Project =======================")
message(STATUS "[clDNN] Version:          ${CLDNN__VERSION_MAJOR}.${CLDNN__VERSION_MINOR}.${CLDNN__VERSION_BUILD}.${CLDNN__VERSION_REVISION}")
message(STATUS "[clDNN]")
message(STATUS "[clDNN] Build type:       ${CMAKE_BUILD_TYPE} (for single-configuration generators)")
message(STATUS "[clDNN] Av. build types:  ${CMAKE_CONFIGURATION_TYPES} (for multi-configuration generators)")
message(STATUS "[clDNN]")
message(STATUS "[clDNN] Output bin directory:")
message(STATUS "[clDNN]  - \"${CLDNN__OUTPUT_BIN_DIR}\"")
message(STATUS "[clDNN] Output lib directory:")
message(STATUS "[clDNN]  - \"${CLDNN__OUTPUT_LIB_DIR}\"")
message(STATUS "[clDNN] Architecture:")
message(STATUS "[clDNN]  - target: ${CLDNN__ARCHITECTURE_TARGET}    (detected: ${__CLDNN_DetectedArch_Target})")
message(STATUS "[clDNN]")
message(STATUS "[clDNN]")
message(STATUS "[clDNN] Advanced:")
if (CLDNN__IOCL_ICD_USE_EXTERNAL)
  message(STATUS "[clDNN]  - ICD version used to build:         N/A (installed externally)")
else()
  message(STATUS "[clDNN]  - ICD version used to build:         ${CLDNN__IOCL_ICD_VERSION}")
endif()
message(STATUS "[clDNN]")
message(STATUS "[clDNN]  - Include/Build cldnn core:          ${CLDNN__INCLUDE_CORE}")
message(STATUS "[clDNN]  - Include/Build kernel selector:     ${CLDNN__INCLUDE_KERNEL_SELECTOR}")
message(STATUS "[clDNN]  - Include/Build tests:               ${CLDNN__INCLUDE_TESTS}")
message(STATUS "[clDNN]  - Include/Build core internal tests: ${CLDNN__INCLUDE_CORE_INTERNAL_TESTS}")
message(STATUS "[clDNN]  - Include/Build tutorial:            ${CLDNN__INCLUDE_TUTORIAL}")
message(STATUS "[clDNN]")
message(STATUS "[clDNN]  - Run tests:                     ${CLDNN__RUN_TESTS}")
message(STATUS "[clDNN]  - Run core internal tests:       ${CLDNN__RUN_CORE_INTERNAL_TESTS}")
message(STATUS "[clDNN]")
message(STATUS "[clDNN]  - Use static C++ Runtime:        ${CLDNN__COMPILE_LINK_USE_STATIC_RUNTIME}")
message(STATUS "[clDNN]  - Allow unsafe size opts:        ${CLDNN__COMPILE_LINK_ALLOW_UNSAFE_SIZE_OPT}")
message(STATUS "[clDNN]  - CMake debug trace:             ${CLDNN__CMAKE_DEBUG}")
message(STATUS "[clDNN]")
message(STATUS "[clDNN]")
message(STATUS "[clDNN] ICD:")
message(STATUS "[clDNN]  - Root:            ${CLDNN__IOCL_ICD_ROOT}")
message(STATUS "[clDNN]     + Headers:      ${CLDNN__IOCL_ICD_INCDIRS}")
message(STATUS "[clDNN]     + Static libs:  ${CLDNN__IOCL_ICD_STLDIRS}")
message(STATUS "[clDNN]     + Shared libs:  ${CLDNN__IOCL_ICD_SHLDIRS}")
message(STATUS "[clDNN]     + Libs to link: ${CLDNN__IOCL_ICD_LIBDIRS}")
message(STATUS "[clDNN] =============================================================================")
unset(__CLDNN_DetectedArch_Target)


# ======================================================================================================
# ==================================== COMMON BUILD CONFIGURATION ======================================
# ======================================================================================================

# =================================== Main targets names and labels ====================================

set(CLDNN_BUILD__PROJ__clDNN       "${CLDNN_BUILD__PROJ_NAME_PREFIX}clDNN_shlib")
set(CLDNN_BUILD__PROJ_LABEL__clDNN "clDNN")

# ================================================ Outputs =============================================

# Old.
set(EXECUTABLE_OUTPUT_PATH "${CLDNN__OUTPUT_BIN_DIR}")
set(LIBRARY_OUTPUT_PATH    "${CLDNN__OUTPUT_LIB_DIR}")
# New.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CLDNN__OUTPUT_LIB_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CLDNN__OUTPUT_LIB_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CLDNN__OUTPUT_BIN_DIR}")


# Main targets' output names.
intel_arch_get_cpu(CLDNN__OUT_CPU_SUFFIX "${CLDNN__ARCHITECTURE_TARGET}")

set(CLDNN_BUILD__PROJ_OUTPUT_NAME__clDNN "clDNN${CLDNN__OUT_CPU_SUFFIX}")


# RPATH for executables (Linux, Android, Mac)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

intel_arch_get_os(__CLDNN_TargetOs "${CLDNN__ARCHITECTURE_TARGET}")
if(__CLDNN_TargetOs MATCHES "^Darwin$")
  set(CMAKE_INSTALL_RPATH "@executable_path")
else()
  set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")
endif()
unset(__CLDNN_TargetOs)

# ======================================================================================================

cldnn_dmsg("${CLDNN__PROJ_NAME} Settings:")
cldnn_dmsg(" - project file name: ${CLDNN_BUILD__PROJ__clDNN}")
cldnn_dmsg(" - project label:     ${CLDNN_BUILD__PROJ_LABEL__clDNN}")
cldnn_dmsg(" - library name:      ${CLDNN_BUILD__PROJ_OUTPUT_NAME__clDNN}")
cldnn_dmsg(" - used generator:    ${CMAKE_GENERATOR}")
cldnn_dmsg("    + platform:       ${CMAKE_GENERATOR_PLATFORM}")
cldnn_dmsg("    + toolset:        ${CMAKE_GENERATOR_TOOLSET}")
cldnn_dmsg(" - crosscompiling:    ${CMAKE_CROSSCOMPILING}")
if(CMAKE_CROSSCOMPILING)
  cldnn_dmsg("    + toolchain file: ${CMAKE_TOOLCHAIN_FILE}")
endif()
cldnn_dmsg(" - compiler (C++):    ${CMAKE_CXX_COMPILER_ID}")
cldnn_dmsg("    + version:        ${CMAKE_CXX_COMPILER_VERSION}")
cldnn_dmsg(" - compiler (C):      ${CMAKE_C_COMPILER_ID}")
cldnn_dmsg("    + version:        ${CMAKE_C_COMPILER_VERSION}")

# ============================== Abstraction of compiler and linker options ============================

include("${CMAKE_CURRENT_SOURCE_DIR}/CMakeCompilerLinkerOpts.txt" NO_POLICY_SCOPE)

# ======================================= Generic compiler options =====================================
# Selecting C++ Runtime.
if(CLDNN__COMPILE_LINK_USE_STATIC_RUNTIME)
  set(__CLDNN_RtType      "RtMultiThreadedStatic")
  set(__CLDNN_RtTypeDebug "RtMultiThreadedStaticDebug")
else()
  set(__CLDNN_RtType      "RtMultiThreadedShared")
  set(__CLDNN_RtTypeDebug "RtMultiThreadedSharedDebug")
endif()

foreach(__CLDNN_CompilerFlagName IN ITEMS "CMAKE_CXX_FLAGS" "CMAKE_C_FLAGS")
  # Change some generic settings of compiler.
  # NOTE: Debug info generation is enabled for all build configuration, because it is separate on Windows
  #       and we will use "strip" command on Linux and Android (to separate it).
  intel_config_flag_apply_settings(
      CompilerOptions
      "${__CLDNN_CompilerFlagName}"
      ALL_PATTERN ""
      SET
        CompileAsDefault
        ExceptionsEnabled
        MultiProcessorCompilation
        DeadCodeEliminate
        ExtensionsEnabled
        TreatWarnAsErrorEnabled
        WarnLevel4
        NoFastMath
        StackProtector
    )
  intel_config_flag_apply_settings(
      CompilerOptions
      "${__CLDNN_CompilerFlagName}"
      ALL_PATTERN_NOINHERIT ""
      SET
        OptimizeSize
        "${__CLDNN_RtType}"
    )
  intel_config_flag_apply_settings(
      CompilerOptions
      "${__CLDNN_CompilerFlagName}"
      PATTERN "^Debug"
      SET
        OptimizeDisabled
        "${__CLDNN_RtTypeDebug}"
    )

  # Adding needed settings specific to MSVC.
  if(MSVC)
    if(CMAKE_COMPILER_IS_INTEL)
      intel_config_flag_apply_settings(
          CompilerOptions
          "${__CLDNN_CompilerFlagName}"
          ALL_PATTERN ""
          SET_RAW
            "/wd177"
            "/wd367"
            "/wd411"
            "/wd2415"
            "/wd3280"
            "/wd3346"
            "/wd11074"
            "/wd11076"
        )
    else()
      if(__CLDNN_TargetCpu STREQUAL "32")
        intel_config_flag_apply_settings(
            CompilerOptions
            "${__CLDNN_CompilerFlagName}"
            ALL_PATTERN ""
            SET_RAW
            "/arch:SSE2"
            "/sdl"
        )
      else()
        intel_config_flag_apply_settings(
                CompilerOptions
                "${__CLDNN_CompilerFlagName}"
                ALL_PATTERN ""
                SET_RAW
                "/sdl"
            )
      endif()
    endif()
  elseif(CMAKE_COMPILER_IS_INTEL)
    message(FATAL_ERROR "TODO Support native ICC")
  # Adding needed settings specific to GCC.
  # NOTE: Following options can be needed in the future (although some not recommended: NR):
  # [NR] -fno-short-enums
  # [NR] -fno-tree-pre
  #      -fno-omit-frame-pointer
  #      -Wa,--noexecstack
  #      -fkeep-inline-functions
  elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    intel_config_flag_apply_settings(
        CompilerOptions
        "${__CLDNN_CompilerFlagName}"
        ALL_PATTERN ""
        SET_RAW
          -pipe
          -fmessage-length=0
          -fno-strict-aliasing
          -W
          -Wno-unknown-pragmas
          -Wwrite-strings
          -Wswitch
          -Wformat
          -Wformat-security
          -Wno-error=missing-field-initializers
          -Wno-error=unused-parameter
          -Wno-error=unused-function
          -march=corei7
          -mstackrealign
          -msse
          -msse2
          -msse3
          -mssse3
          -msse4
          -msse4.1
          -msse4.2
          -fvisibility=hidden
          -finline
          -finline-functions
          -finline-limit=300
          -funswitch-loops
          -fPIE
          -fPIC
          -Wl,--no-undefined
      )
  elseif((CMAKE_C_COMPILER_ID MATCHES "^Clang$") OR (CMAKE_CXX_COMPILER_ID MATCHES "^Clang$"))
    intel_config_flag_apply_settings(
        CompilerOptions
        "${__CLDNN_CompilerFlagName}"
        ALL_PATTERN ""
        SET_RAW
          -pipe
          -fvisibility=hidden
          -fvisibility-inlines-hidden
          -Wall
          -Wno-covered-switch-default
          -Wextra
          -Wno-unused-parameter
          -Wno-gnu
          -pedantic
          -finline
          -msse4.2
          -fPIE
          -fPIC
      )
  endif()
endforeach()

# C++ only options.
intel_config_flag_apply_settings(
    CompilerOptions
    CMAKE_CXX_FLAGS
    ALL_PATTERN ""
    SET
      RttiDisabled
      StandardCxx11
  )

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  intel_config_flag_apply_settings(
      CompilerOptions
      CMAKE_CXX_FLAGS
      ALL_PATTERN ""
      SET_RAW
        -fno-operator-names
        -fpermissive
        -fvisibility-inlines-hidden
    )
endif()

# NOTE: When compiling with Clang, use the flags below for C++ files. These flags cannot be enabled when compiling
#       C files.
if((CMAKE_C_COMPILER_ID MATCHES "^Clang$") OR (CMAKE_CXX_COMPILER_ID MATCHES "^Clang$"))
  intel_config_flag_apply_settings(
      CompilerOptions
      CMAKE_CXX_FLAGS
      ALL_PATTERN ""
      SET_RAW
        -stdlib=libc++
    )
endif()

unset(__CLDNN_RtType)
unset(__CLDNN_RtTypeDebug)
unset(__CLDNN_CompilerFlagName)

# ======================================== Generic linker options ======================================

# Additional libraries that needs to be linked to shared objects/executables.
set(CLDNN__SYSTEM_LINK_LIBRARIES)

foreach(__CLDNN_LinkerFlagName IN ITEMS "CMAKE_EXE_LINKER_FLAGS" "CMAKE_SHARED_LINKER_FLAGS")
  # Change some generic settings of linker.
  # NOTE: Debug info generation is enabled for all build configuration, because it is separate on Windows
  #       and we will use "strip" command on Linux and Android (to separate it).
  intel_config_flag_apply_settings(
      LinkerOptions
      "${__CLDNN_LinkerFlagName}"
      ALL_PATTERN ""
      SET
        DeadCodeEliminate
        IdenticalCodeDataFold
        IncrementalDisabled
    )
endforeach()

# Force static linking of common libraries on Android for shared objects.
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  intel_config_flag_apply_settings(
        LinkerOptions
        CMAKE_EXE_LINKER_FLAGS
        ALL_PATTERN ""
        SET_RAW
          -pie
          -Wl,-z,noexecstack,-z,relro,-z,now
      )
  intel_config_flag_apply_settings(
        LinkerOptions
        CMAKE_SHARED_LINKER_FLAGS
        ALL_PATTERN ""
        SET_RAW
          -Wl,-z,noexecstack,-z,relro,-z,now
      )
  list(APPEND CLDNN__SYSTEM_LINK_LIBRARIES "dl")
endif()

if((CMAKE_C_COMPILER_ID MATCHES "^Clang$") OR (CMAKE_CXX_COMPILER_ID MATCHES "^Clang$"))
  intel_config_flag_apply_settings(
      LinkerOptions
      CMAKE_SHARED_LINKER_FLAGS
      ALL_PATTERN ""
      SET_RAW
        -shared
        -Wl,-undefined,dynamic_lookup
        -Wl,-headerpad_max_install_names
    )

  list(APPEND CLDNN__SYSTEM_LINK_LIBRARIES "c++" "c++abi" "supc++" "dl")
endif()

unset(__CLDNN_LinkerFlagName)

# ======================================================================================================

cldnn_dmsg(" - compile/link flags:")
foreach(__CLDNN_Flag IN ITEMS "CMAKE_CXX_FLAGS" "CMAKE_C_FLAGS" "CMAKE_EXE_LINKER_FLAGS" "CMAKE_SHARED_LINKER_FLAGS")
  cldnn_dmsg("    + ${__CLDNN_Flag}:     ${${__CLDNN_Flag}}")
  foreach(__CLDNN_CfgType ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER "${__CLDNN_Flag}_${__CLDNN_CfgType}" __CLDNN_CfgFlag)
    if(DEFINED "${__CLDNN_CfgFlag}")
      cldnn_dmsg("    + ${__CLDNN_CfgFlag}:     ${${__CLDNN_CfgFlag}}")
    endif()
  endforeach()
  cldnn_dmsg("")
endforeach()
unset(__CLDNN_Flag)
unset(__CLDNN_CfgType)
unset(__CLDNN_CfgFlag)

# ======================================================================================================

# Clean-up of helper variables for component paths.
unset(__CLDNN_TargetOs)
unset(__CLDNN_TargetCpu)
unset(__CLDNN_TargetCpuDir)
unset(__CLDNN_TargetCompilerDir)

# ============================== Generic compiler preprocessor definitions =============================

set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS
    CLDNN_CMAKE
  )

if (MSVC)
#  set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS
#      _SCL_SECURE_NO_WARNINGS
#    )
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS
    _GLIBCXX_USE_CXX11_ABI=0
    $<$<CONFIG:Release>:_FORTIFY_SOURCE=2>
  )
endif()

# ===================================== Include/Link directories =======================================

include_directories(
    ${CLDNN__IOCL_ICD_INCDIRS}
	${CLDNN_UTILS__RAPIDJSON_INCDIRS}
    "${CLDNN__KHR_CLHPP_DIR}"
    "${CLDNN__CODEGEN_INCDIR}"
  )
link_directories(
    ${CLDNN__IOCL_ICD_LIBDIRS}
  )

# =================================== Link targets and dependencies ====================================
if(CLDNN__INCLUDE_CORE)
  add_subdirectory(src)
  add_subdirectory(api_test_builds)
endif()
if(CLDNN__INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
if(CLDNN__INCLUDE_CORE_INTERNAL_TESTS)
  add_subdirectory(tests_core_internal)
endif()
if(CLDNN__INCLUDE_KERNEL_SELECTOR)
  add_subdirectory(kernel_selector)
endif()
if(CLDNN__INCLUDE_TUTORIAL)
  add_subdirectory(tutorial)
endif()

add_subdirectory(docs)


# ======================================================================================================
# ======================================================================================================
# ======================================================================================================
