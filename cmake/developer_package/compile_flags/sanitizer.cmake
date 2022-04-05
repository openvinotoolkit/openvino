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
    # Mute -fsanitize=function Indirect call of a function through a function pointer of the wrong type.
    #   Sample cases:
    #       call to function GetAPIVersion through pointer to incorrect function type 'void *(*)()'
    # Mute -fsanitize=alignment Use of a misaligned pointer or creation of a misaligned reference. Also sanitizes assume_aligned-like attributes.
    #   Sample cases:
    #       VPU_FixedMaxHeapTest.DefaultConstructor test case load of misaligned address 0x62000000187f for type 'const DataType', which requires 4 byte alignment
    # Mute -fsanitize=bool Load of a bool value which is neither true nor false.
    #   Samples cases:
    #       ie_c_api_version.apiVersion test case load of value 32, which is not a valid value for type 'bool'
    # Mute -fsanitize=enum Load of a value of an enumerated type which is not in the range of representable values for that enumerated type.
    #   Samples cases:
    #       load of value 4294967295, which is not a valid value for type 'const (anonymous namespace)::onnx::Field'
    set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -fsanitize=undefined -fno-sanitize=null -fno-sanitize=alignment -fno-sanitize=bool -fno-sanitize=enum")
    if(OV_COMPILER_IS_CLANG)
        set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -fno-sanitize=function")
    endif()

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