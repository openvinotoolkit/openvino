# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was OpenVINODeveloperPackageConfigRelocatable.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

####################################################################################

include(CMakeFindDependencyMacro)

# Variables to export in plugin's projects

set(ov_options "CPACK_GENERATOR;ENABLE_LTO;OS_FOLDER;USE_BUILD_TYPE_SUBFOLDER;CMAKE_COMPILE_WARNING_AS_ERROR;ENABLE_QSPECTRE;ENABLE_INTEGRITYCHECK;ENABLE_SANITIZER;ENABLE_UB_SANITIZER;ENABLE_THREAD_SANITIZER;ENABLE_COVERAGE;ENABLE_SSE42;ENABLE_AVX2;ENABLE_AVX512F;BUILD_SHARED_LIBS;ENABLE_LIBRARY_VERSIONING;ENABLE_FASTER_BUILD;ENABLE_CPPLINT;ENABLE_CPPLINT_REPORT;ENABLE_CLANG_FORMAT;ENABLE_NCC_STYLE;ENABLE_UNSAFE_LOCATIONS;ENABLE_FUZZING;ENABLE_PROXY;ENABLE_INTEL_CPU;ENABLE_ARM_COMPUTE_CMAKE;ENABLE_TESTS;ENABLE_INTEL_GPU;ENABLE_ONEDNN_FOR_GPU;ENABLE_DEBUG_CAPS;ENABLE_GPU_DEBUG_CAPS;ENABLE_CPU_DEBUG_CAPS;ENABLE_SNIPPETS_DEBUG_CAPS;ENABLE_PROFILING_ITT;ENABLE_PROFILING_FILTER;ENABLE_PROFILING_FIRST_INFERENCE;SELECTIVE_BUILD;ENABLE_DOCS;ENABLE_PKGCONFIG_GEN;THREADING;ENABLE_TBBBIND_2_5;ENABLE_TBB_RELEASE_ONLY;ENABLE_MULTI;ENABLE_AUTO;ENABLE_AUTO_BATCH;ENABLE_HETERO;ENABLE_TEMPLATE;ENABLE_PLUGINS_XML;ENABLE_FUNCTIONAL_TESTS;ENABLE_SAMPLES;ENABLE_OV_ONNX_FRONTEND;ENABLE_OV_PADDLE_FRONTEND;ENABLE_OV_IR_FRONTEND;ENABLE_OV_PYTORCH_FRONTEND;ENABLE_OV_IR_FRONTEND;ENABLE_OV_TF_FRONTEND;ENABLE_OV_TF_LITE_FRONTEND;ENABLE_SNAPPY_COMPRESSION;ENABLE_STRICT_DEPENDENCIES;ENABLE_SYSTEM_TBB;ENABLE_SYSTEM_PUGIXML;ENABLE_SYSTEM_FLATBUFFERS;ENABLE_SYSTEM_OPENCL;ENABLE_SYSTEM_PROTOBUF;ENABLE_SYSTEM_SNAPPY;ENABLE_PYTHON_PACKAGING;ENABLE_JS;ENABLE_OPENVINO_DEBUG")
list(APPEND ov_options CPACK_GENERATOR)

if(APPLE)
    list(APPEND ov_options CMAKE_OSX_ARCHITECTURES CMAKE_OSX_DEPLOYMENT_TARGET)
endif()

get_property(_OV_GENERATOR_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(_OV_GENERATOR_MULTI_CONFIG)
    list(APPEND ov_options CMAKE_CONFIGURATION_TYPES)
    if(CMAKE_GENERATOR MATCHES "^Ninja Multi-Config$")
        list(APPEND ov_options CMAKE_DEFAULT_BUILD_TYPE)
    endif()
else()
    list(APPEND ov_options CMAKE_BUILD_TYPE)
endif()
unset(_OV_GENERATOR_MULTI_CONFIG)

file(TO_CMAKE_PATH "${CMAKE_CURRENT_LIST_DIR}" cache_path)

message(STATUS "The following CMake options are exported from OpenVINO Developer package")
message(" ")
foreach(option IN LISTS ov_options)
    if(NOT DEFINED "${option}")
        load_cache("${cache_path}" READ_WITH_PREFIX "" ${option})
    endif()
    message("    ${option}: ${${option}}")
endforeach()
message(" ")

# Restore TBB installation directory (requires for proper LC_RPATH on macOS with SIP)
load_cache("${cache_path}" READ_WITH_PREFIX "" TBB_INSTALL_DIR)

# activate generation of plugins.xml
set(ENABLE_PLUGINS_XML ON)

# Disable warning as error for private components
set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)

#
# Content
#

# OpenVINO_DIR is supposed to be set as an environment variable
find_dependency(OpenVINO)

find_dependency(OpenVINODeveloperScripts
                PATHS "${CMAKE_CURRENT_LIST_DIR}"
                NO_CMAKE_FIND_ROOT_PATH
                NO_DEFAULT_PATH)

_ov_find_tbb()
_ov_find_pugixml()

include("${CMAKE_CURRENT_LIST_DIR}/OpenVINODeveloperPackageTargets.cmake")
#
# Extra Compile Flags
#

# don't fail on strict compilation options in 3rd party modules
ov_dev_package_no_errors()

# Don't threat deprecated API warnings as errors in 3rd party apps
ov_deprecated_no_errors()
