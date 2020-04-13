# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Target system specific flags

if(CMAKE_CL_64)
  set(MSVC64 ON)
endif()

if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpmachine
                  OUTPUT_VARIABLE OPENVINO_GCC_TARGET_MACHINE
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(OPENVINO_GCC_TARGET_MACHINE MATCHES "amd64|x86_64|AMD64")
    set(MINGW64 ON)
  endif()
endif()

if(MSVC64 OR MINGW64)
  set(X86_64 ON)
elseif(MINGW OR (MSVC AND NOT CMAKE_CROSSCOMPILING))
  set(X86 ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
  set(X86_64 ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i686.*|i386.*|x86.*|amd64.*|AMD64.*")
  set(X86 ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm.*|ARM.*)")
  set(ARM ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*)")
  set(AARCH64 ON)
endif()

if(UNIX AND NOT APPLE)
    set(LINUX ON)
endif()
