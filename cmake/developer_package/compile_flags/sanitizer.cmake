# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(CheckCXXCompilerFlag)

if (ENABLE_SANITIZER)
    set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -fsanitize=address")
    check_cxx_compiler_flag("-fsanitize-recover=address" SANITIZE_RECOVER_ADDRESS_SUPPORTED)
    if (SANITIZE_RECOVER_ADDRESS_SUPPORTED)
        set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -fsanitize-recover=address")
    endif()

    set(SANITIZER_LINKER_FLAGS "${SANITIZER_LINKER_FLAGS} -fsanitize=address")
endif()

if (ENABLE_UB_SANITIZER)
    # TODO: Remove -fno-sanitize=null as thirdparty/ocl/clhpp_headers UBSAN compatibility resolved:
    # https://github.com/KhronosGroup/OpenCL-CLHPP/issues/17
    set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -fsanitize=undefined -fno-sanitize=null")
    check_cxx_compiler_flag("-fsanitize-recover=undefined" SANITIZE_RECOVER_UNDEFINED_SUPPORTED)
    if (SANITIZE_RECOVER_UNDEFINED_SUPPORTED)
        set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -fsanitize-recover=undefined")
    endif()

    set(SANITIZER_LINKER_FLAGS "${SANITIZER_LINKER_FLAGS} -fsanitize=undefined")
endif()

if (ENABLE_THREAD_SANITIZER)
    set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -fsanitize=thread")
    set(SANITIZER_LINKER_FLAGS "${SANITIZER_LINKER_FLAGS} -fsanitize=thread")
endif()

# common sanitizer options
if (DEFINED SANITIZER_COMPILER_FLAGS)
    # ensure sumbols are present
    set(SANITIZER_COMPILER_FLAGS "-g -fno-omit-frame-pointer")
    # prevent unloading libraries at runtime, so sanitizer can resolve their symbols
    set(SANITIZER_LINKER_FLAGS "${SANITIZER_LINKER_FLAGS} -Wl,-z,nodelete")

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(SANITIZER_LINKER_FLAGS "${SANITIZER_LINKER_FLAGS} -fuse-ld=gold")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$" AND NOT WIN32)
        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 8.0)
            set(SANITIZER_LINKER_FLAGS "${SANITIZER_LINKER_FLAGS} -fuse-ld=lld")
        endif()
    endif()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SANITIZER_COMPILER_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZER_COMPILER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${SANITIZER_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${SANITIZER_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SANITIZER_LINKER_FLAGS}")
endif()