# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Define CMAKE_SYSTEM_VERSION if not defined
#

if(NOT DEFINED CMAKE_SYSTEM_VERSION)
    # Sometimes CMAKE_HOST_SYSTEM_VERSION has form 10.x.y while we need
    # form 10.x.y.z Adding .0 at the end fixes the issue
    if(CMAKE_HOST_SYSTEM_VERSION MATCHES "^10\\.0\\.[0-9]+$")
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
    # Forcefull make VS search for C++ libraries in these folders prior to other c++ standard libraries localizations.
    add_link_options("/LIBPATH:\"\$\(VC_LibraryPath_VC_x64_OneCore\)\"")

    set(CMAKE_C_STANDARD_LIBRARIES_INIT "\$\(UCRTContentRoot\)lib/\$\(TargetUniversalCRTVersion\)/um/\$\(Platform\)/OneCoreUap.lib" CACHE STRING "" FORCE)
    set(CMAKE_CXX_STANDARD_LIBRARIES_INIT "\$\(UCRTContentRoot\)lib/\$\(TargetUniversalCRTVersion\)/um/\$\(Platform\)/OneCoreUap.lib" CACHE STRING "" FORCE)
elseif(_onecoreuap_arch STREQUAL "X86")
    add_link_options("/LIBPATH:\"\$\(VCInstallDir\)lib/onecore\"")
    add_link_options("/LIBPATH:\"\$\(VC_LibraryPath_VC_x86_OneCore\)\"")

    set(CMAKE_C_STANDARD_LIBRARIES_INIT "\$\(UCRTContentRoot\)lib/\$\(TargetUniversalCRTVersion\)/um/x86/OneCoreUap.lib" CACHE STRING "" FORCE)
    set(CMAKE_CXX_STANDARD_LIBRARIES_INIT "\$\(UCRTContentRoot\)lib/\$\(TargetUniversalCRTVersion\)/um/x86/OneCoreUap.lib" CACHE STRING "" FORCE)
else()
    message(FATAL_ERROR "Unsupported architecture ${_onecoreuap_arch}. Only X86 or X86_64 are supported")
endif()

unset(_onecoreuap_arch)

# compile flags
if(CMAKE_GENERATOR MATCHES "Ninja")
    set(includes "/I\"\$\$\(UniversalCRT_IncludePath\)\"")
else()
    set(includes "/I\"\$\(UniversalCRT_IncludePath\)\"")
endif()

set(CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT} ${includes}")
set(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} ${includes}")
unset(includes)

# linker flags

foreach(lib kernel32 user32 advapi32 ole32 mscoree combase)
    set(linker_flags "/NODEFAULTLIB:${lib}.lib ${linker_flags}")
endforeach()

set(CMAKE_SHARED_LINKER_FLAGS_INIT "${CMAKE_SHARED_LINKER_FLAGS_INIT} ${linker_flags}")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "${CMAKE_MODULE_LINKER_FLAGS_INIT} ${linker_flags}")
set(CMAKE_EXE_LINKER_FLAGS_INIT "${CMAKE_EXE_LINKER_FLAGS_INIT} ${linker_flags}")
unset(linker_flags)

#
# Static runtime to overcome apiValidator tool restrictions
#

include("${CMAKE_CURRENT_LIST_DIR}/mt.runtime.win32.toolchain.cmake")
