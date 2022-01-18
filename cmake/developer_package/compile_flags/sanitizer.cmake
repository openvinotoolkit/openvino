# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(CheckCXXCompilerFlag)

if (ENABLE_SANITIZER)
    if (WIN32)
        check_cxx_compiler_flag("/fsanitize=address" SANITIZE_ADDRESS_SUPPORTED)
        if (SANITIZE_ADDRESS_SUPPORTED)
            set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} /fsanitize=address")
        else()
            message(FATAL_ERROR "Address sanitizer is not supported by current compiler.\n"
            "Please, check requirements:\n"
            "https://github.com/openvinotoolkit/openvino/wiki/AddressSanitizer-and-LeakSanitizer")
        endif()
    else()
        set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -fsanitize=address")
        check_cxx_compiler_flag("-fsanitize-recover=address" SANITIZE_RECOVER_ADDRESS_SUPPORTED)
        if (SANITIZE_RECOVER_ADDRESS_SUPPORTED)
            set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -fsanitize-recover=address")
        endif()
        set(SANITIZER_LINKER_FLAGS "${SANITIZER_LINKER_FLAGS} -fsanitize=address")
    endif()
endif()

if (ENABLE_UB_SANITIZER)
    if (WIN32)
        message(FATAL_ERROR "UndefinedBehavior sanitizer is not supported in Windows")
    endif()
    # TODO: Remove -fno-sanitize=null as thirdparty/ocl/clhpp_headers UBSAN compatibility resolved:
    # https://github.com/KhronosGroup/OpenCL-CLHPP/issues/17
    set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -fsanitize=undefined -fno-sanitize=null")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # TODO: Remove -Wno-maybe-uninitialized after CVS-61143 fix
        set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -Wno-maybe-uninitialized")
    endif()
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
    # ensure symbols are present
    if (NOT WIN32)
        set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -g -fno-omit-frame-pointer")
        if(NOT OV_COMPILER_IS_CLANG)
            # GPU plugin tests compilation is slow with -fvar-tracking-assignments on GCC.
            # Clang has no var-tracking-assignments.
            set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -fno-var-tracking-assignments")
        endif()
        # prevent unloading libraries at runtime, so sanitizer can resolve their symbols
        set(SANITIZER_LINKER_FLAGS "${SANITIZER_LINKER_FLAGS} -Wl,-z,nodelete")

        if(OV_COMPILER_IS_CLANG AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 8.0)
            set(SANITIZER_LINKER_FLAGS "${SANITIZER_LINKER_FLAGS} -fuse-ld=lld")
        endif()
    else()
        set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} /Oy-")
    endif()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SANITIZER_COMPILER_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZER_COMPILER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${SANITIZER_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${SANITIZER_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SANITIZER_LINKER_FLAGS}")
endif()