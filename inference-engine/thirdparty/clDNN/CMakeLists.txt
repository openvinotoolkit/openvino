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

# Path which points to directory with interface for framework.
set(CLDNN__API_DIR       "${CMAKE_CURRENT_SOURCE_DIR}/api")

# Path which points to directory with interface for framework.
set(CLDNN__KERNEL_SELECTOR_DIR       "${CMAKE_CURRENT_SOURCE_DIR}/kernel_selector")

# Path which points to directory with runtime
set(CLDNN__RUNTIME_DIR       "${CMAKE_CURRENT_SOURCE_DIR}/runtime")

# Build targets settings.

# Prefix for all targets in internal pass.
set(CLDNN_BUILD__PROJ_NAME_PREFIX "")

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

# Target architecture for compilation.
set(CLDNN__ARCHITECTURE_TARGET "${CLDNN__ARCHITECTURE_TARGET}" CACHE STRING "Target architecture for compilation." FORCE)
if(MSVC)
  set_property(CACHE CLDNN__ARCHITECTURE_TARGET PROPERTY STRINGS "Windows32" "Windows64")
elseif(DEFINED XCODE_VERSION)
  set_property(CACHE CLDNN__ARCHITECTURE_TARGET PROPERTY STRINGS "Darwin32" "Darwin64")
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

# Compile / Link: Use static C++ Runtime library.
set(CLDNN__COMPILE_LINK_USE_STATIC_RUNTIME OFF CACHE BOOL "Compile / Link: Use static version of C++ Runtime library instead of shared one.")
mark_as_advanced(CLDNN__COMPILE_LINK_USE_STATIC_RUNTIME)

# ======================================================================================================

# Compile / Link: Allow unsafe binary size optimizations.
set(CLDNN__COMPILE_LINK_ALLOW_UNSAFE_SIZE_OPT ON CACHE BOOL "Compile / Link: Allow unsafe binary size optimizations.")
mark_as_advanced(CLDNN__COMPILE_LINK_ALLOW_UNSAFE_SIZE_OPT)

# ======================================================================================================
# ======================================================================================================

# Minimum versions of compilers.
set(CLDNN__MIN_COMPILER_VERSION__MSVC  "19.0")
set(CLDNN__MIN_COMPILER_VERSION__CLANG "3.5")
set(CLDNN__MIN_COMPILER_VERSION__ICC   "17.0")
set(CLDNN__MIN_COMPILER_VERSION__GCC   "4.8")

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
intel_arch_get_cpu(__CLDNN_TargetCpu "${CLDNN__ARCHITECTURE_TARGET}")
string(TOLOWER "${__CLDNN_TargetCpu}" __CLDNN_TargetCpu)

# ======================================================================================================
set(CLDNN_UTILS__RAPIDJSON_INCDIRS "utils/rapidjson" CACHE INTERNAL "Paths to interface headers for rapidjson.")

# ======================================================================================================
# ==================================== COMMON BUILD CONFIGURATION ======================================
# ======================================================================================================

# =================================== Main targets names and labels ====================================

set(CLDNN_BUILD__PROJ__clDNN       "${CLDNN_BUILD__PROJ_NAME_PREFIX}clDNN_lib")
set(CLDNN_BUILD__PROJ_LABEL__clDNN "clDNN")

# ================================================ Outputs =============================================

# Main targets' output names.
intel_arch_get_cpu(CLDNN__OUT_CPU_SUFFIX "${CLDNN__ARCHITECTURE_TARGET}")

set(CLDNN_BUILD__PROJ_OUTPUT_NAME__clDNN "clDNN${CLDNN__OUT_CPU_SUFFIX}")


# RPATH for executables (Linux, Android, Mac)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

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

# Clean-up of helper variables for component paths.
unset(__CLDNN_TargetCpu)

# ===================================== Include/Link directories =======================================

include_directories(
    ${CLDNN_UTILS__RAPIDJSON_INCDIRS}
    "${CLDNN__CODEGEN_INCDIR}"
    "${CLDNN__API_DIR}"
  )

# =================================== Link targets and dependencies ====================================
if(CLDNN__INCLUDE_CORE)
  add_subdirectory(src)
  add_subdirectory(runtime)
endif()
if(CLDNN__INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
if(CLDNN__INCLUDE_KERNEL_SELECTOR)
  add_subdirectory(kernel_selector)
endif()


# ======================================================================================================
# ======================================================================================================
# ======================================================================================================
