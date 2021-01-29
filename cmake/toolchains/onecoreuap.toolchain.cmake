# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Define CMAKE_SYSTEM_VERSION if not defined
#

if(NOT DEFINED CMAKE_SYSTEM_VERSION)
    # Sometimes CMAKE_HOST_SYSTEM_VERSION has form 10.x.y while we need
    # form 10.x.y.z Adding .0 at the end fixes the issue
    if(CMAKE_HOST_SYSTEM_VERSION MATCHES "^10\.0\.[0-9]+$")
        set(CMAKE_SYSTEM_VERSION "${CMAKE_HOST_SYSTEM_VERSION}.0")
    else()
        set(CMAKE_SYSTEM_VERSION "${CMAKE_HOST_SYSTEM_VERSION}")
    endif()
endif()

if(NOT DEFINED CMAKE_SYSTEM_PROCESSOR)
    set(CMAKE_SYSTEM_PROCESSOR ${CMAKE_HOST_SYSTEM_PROCESSOR})
endif()

message(STATUS "Building for Windows OneCore compliance (using OneCoreUap.lib, ${CMAKE_SYSTEM_VERSION})")

#
# OneCore flags
#

set(_onecoreuap_arch "x64")
if(CMAKE_GENERATOR_PLATFORM)
    set(_onecoreuap_arch ${CMAKE_GENERATOR_PLATFORM})
endif()

if(_onecoreuap_arch STREQUAL "x64")
    # Forcefull make VS search for C++ libreries in these folders prior to other c++ standard libraries localizations.
    add_link_options("/LIBPATH:\"\$\(VC_LibraryPath_VC_x64_OneCore\)\"")

    set(CMAKE_C_STANDARD_LIBRARIES "\$\(UCRTContentRoot\)lib/\$\(TargetUniversalCRTVersion\)/um/\$\(Platform\)/OneCoreUap.lib" CACHE STRING "" FORCE)
    set(CMAKE_CXX_STANDARD_LIBRARIES "\$\(UCRTContentRoot\)lib/\$\(TargetUniversalCRTVersion\)/um/\$\(Platform\)/OneCoreUap.lib" CACHE STRING "" FORCE)
elseif(_onecoreuap_arch STREQUAL "X86")
    add_link_options("/LIBPATH:\"\$\(VCInstallDir\)lib/onecore\"")
    add_link_options("/LIBPATH:\"\$\(VC_LibraryPath_VC_x86_OneCore\)\"")

    set(CMAKE_C_STANDARD_LIBRARIES "\$\(UCRTContentRoot\)lib/\$\(TargetUniversalCRTVersion\)/um/x86/OneCoreUap.lib" CACHE STRING "" FORCE)
    set(CMAKE_CXX_STANDARD_LIBRARIES "\$\(UCRTContentRoot\)lib/\$\(TargetUniversalCRTVersion\)/um/x86/OneCoreUap.lib" CACHE STRING "" FORCE)
else()
    message(FATAL_ERROR "Unsupported architecture ${_onecoreuap_arch}. Only X86 or X86_64 are supported")
endif()

unset(_onecoreuap_arch)

# compile flags

set(includes "/I\"\$\(UniversalCRT_IncludePath\)\"")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${includes}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${includes}")
unset(includes)

# linker flags

foreach(lib kernel32 user32 advapi32 ole32 mscoree combase)
    set(linker_flags "/NODEFAULTLIB:${lib}.lib ${linker_flags}")
endforeach()

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${linker_flags}")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${linker_flags}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${linker_flags}")
unset(linker_flags)

#
# Flags for 3rd party projects
#

set(use_static_runtime ON)

if(use_static_runtime)
    foreach(lang C CXX)
        foreach(build_type "" "_DEBUG" "_MINSIZEREL" "_RELEASE" "_RELWITHDEBINFO")
            set(flag_var "CMAKE_${lang}_FLAGS${build_type}")
            string(REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endforeach()
    endforeach()
endif()

function(onecoreuap_set_runtime var)
    set(${var} ${use_static_runtime} CACHE BOOL "" FORCE)
endfunction()

# ONNX
onecoreuap_set_runtime(ONNX_USE_MSVC_STATIC_RUNTIME)
# pugixml
onecoreuap_set_runtime(STATIC_CRT)
# protobuf
onecoreuap_set_runtime(protobuf_MSVC_STATIC_RUNTIME)
# clDNN
onecoreuap_set_runtime(CLDNN__COMPILE_LINK_USE_STATIC_RUNTIME)

unset(use_static_runtime)
