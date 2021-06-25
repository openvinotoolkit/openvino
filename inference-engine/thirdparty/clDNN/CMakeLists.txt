# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 3.1)
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
if(NOT DEFINED CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_CONFIGURATION_TYPES
        "Debug"
        "Release"
        "RelWithDebInfo"
      )
endif()

# ======================================================================================================
# ======================================================================================================
# ======================================================================================================

# Name of project (helper constant variable).
set(CLDNN__PROJ_NAME "clDNN")

# Default languages: C, C++.
project("${CLDNN__PROJ_NAME}")

if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()

# ======================================================================================================
# ==================================== BUILD CONFIGURATIONS (part 2) ===================================
# ======================================================================================================
# The section must be after project definition and its supported languages because otherwise
# used variables are not available yet.

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
# ======================================================================================================
if(CLDNN_THREADING MATCHES "SEQ")
    add_definitions(-DCLDNN_THREADING=CLDNN_THREADING_SEQ)
elseif(CLDNN_THREADING MATCHES "TBB")
    add_definitions(-DCLDNN_THREADING=CLDNN_THREADING_TBB)
else()
    add_definitions(-DCLDNN_THREADING=CLDNN_THREADING_THREADPOOL)
endif()

if(GPU_DEBUG_CONFIG)
  add_definitions(-DGPU_DEBUG_CONFIG=1)
endif()

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

# Path which points to directory with runtime
set(CLDNN__RUNTIME_DIR       "${CMAKE_CURRENT_SOURCE_DIR}/runtime")

# Path which points to directory with C++ bindings for OpenCL (header files + wrapper that disables specific warnings).
set(CLDNN__KHR_CLHPP_DIR "${CLDNN__COMMON_DIR}/include")

# Build targets settings.

# Path which points to default root directory for compilation output.
set(CLDNN_BUILD__DEFAULT_OUT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/build/out")

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

if (NOT "${CLDNN__TARGET_CFG_VAR}" MATCHES "^(Release|Debug)$")
    message(STATUS "[clDNN] ${CLDNN__TARGET_CFG_VAR} configuration is not supported for prebuilt dependencies. Release will be used")
    set (CLDNN__TARGET_CFG_VAR "Release")
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
  elseif(__CLDNN_BuildType MATCHES "^relwithdebinfo$")
    set(__CLDNN_BuildType "RelWithDebInfo")
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


# Include and build: clDNN tutorial.
set(CLDNN__INCLUDE_TUTORIAL ON CACHE BOOL "Include and build: clDNN Tutorial.")
mark_as_advanced(CLDNN__INCLUDE_TUTORIAL)

# ======================================================================================================

# Run (requires CLDNN__INCLUDE_TESTS to be true): Tests (unit tests and small acceptance tests) for clDNN framework.
set(CLDNN__RUN_TESTS OFF CACHE BOOL "Run: clDNN framework's tests.")
mark_as_advanced(CLDNN__RUN_TESTS)

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

# Check for python 3 interpreter (required tool).
find_package(PythonInterp 3 QUIET)
if(NOT PYTHONINTERP_FOUND)
  message(WARNING "[clDNN] Project requires Python 3.x interpreter to build (with python loader). CMake could not detect it correctly.
      If you have installed this interpreter, please disregard this warning or specify PYTHON_EXECUTABLE in CMake command-line."
    )
endif()

# ======================================================================================================

# Setting helper variables for component paths.
intel_arch_get_os(__CLDNN_TargetOs "${CLDNN__ARCHITECTURE_TARGET}")
string(TOLOWER "${__CLDNN_TargetOs}" __CLDNN_TargetOs)

intel_arch_get_cpu(__CLDNN_TargetCpu "${CLDNN__ARCHITECTURE_TARGET}")
string(TOLOWER "${__CLDNN_TargetCpu}" __CLDNN_TargetCpu)

if((CMAKE_C_COMPILER_ID MATCHES "^Clang$") OR (CMAKE_CXX_COMPILER_ID MATCHES "^Clang$"))
  set(__CLDNN_TargetCompilerDir "clang")
else()
  set(__CLDNN_TargetCompilerDir ".")
endif()

# ======================================================================================================
set(CLDNN_UTILS__RAPIDJSON_INCDIRS "utils/rapidjson" CACHE INTERNAL "Paths to interface headers for rapidjson.")
# ====================================== Version Calculation ===========================================

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
message(STATUS "[clDNN]")
message(STATUS "[clDNN]  - Include/Build cldnn core:          ${CLDNN__INCLUDE_CORE}")
message(STATUS "[clDNN]  - Include/Build kernel selector:     ${CLDNN__INCLUDE_KERNEL_SELECTOR}")
message(STATUS "[clDNN]  - Include/Build tests:               ${CLDNN__INCLUDE_TESTS}")
message(STATUS "[clDNN]  - Include/Build tutorial:            ${CLDNN__INCLUDE_TUTORIAL}")
message(STATUS "[clDNN]")
message(STATUS "[clDNN]  - Run tests:                     ${CLDNN__RUN_TESTS}")
message(STATUS "[clDNN]")
message(STATUS "[clDNN]  - Use static C++ Runtime:        ${CLDNN__COMPILE_LINK_USE_STATIC_RUNTIME}")
message(STATUS "[clDNN]  - Allow unsafe size opts:        ${CLDNN__COMPILE_LINK_ALLOW_UNSAFE_SIZE_OPT}")
message(STATUS "[clDNN]  - CMake debug trace:             ${CLDNN__CMAKE_DEBUG}")
message(STATUS "[clDNN]")
message(STATUS "[clDNN]")
message(STATUS "[clDNN] =============================================================================")
unset(__CLDNN_DetectedArch_Target)


# ======================================================================================================
# ==================================== COMMON BUILD CONFIGURATION ======================================
# ======================================================================================================

# =================================== Main targets names and labels ====================================

set(CLDNN_BUILD__PROJ__clDNN       "${CLDNN_BUILD__PROJ_NAME_PREFIX}clDNN_lib")
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
    if(UNIX)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-warning=68,654,1125")
    endif()
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
          -Wno-error=pedantic
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
      RttiEnabled
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

  list(APPEND CLDNN__SYSTEM_LINK_LIBRARIES "c++" "c++abi" "dl")
  if(NOT ANDROID)
    list(APPEND CLDNN__SYSTEM_LINK_LIBRARIES "supc++")
  endif()
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

# ============================== Generic compiler preprocessor definitions =============================

set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS
    CLDNN_CMAKE
  )

# ===================================== Include/Link directories =======================================

include_directories(
    ${CLDNN_UTILS__RAPIDJSON_INCDIRS}
    "${CLDNN__CODEGEN_INCDIR}"
    "${CLDNN__API_DIR}"
  )

include_directories(SYSTEM
   "${CLDNN__KHR_CLHPP_DIR}"
  )
# =================================== Link targets and dependencies ====================================
if(CLDNN__INCLUDE_CORE)
  add_subdirectory(src)
  add_subdirectory(runtime)
  add_subdirectory(api_test_builds)
endif()
if(CLDNN__INCLUDE_TESTS)
  add_subdirectory(tests)
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
