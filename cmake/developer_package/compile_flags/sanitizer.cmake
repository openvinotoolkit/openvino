# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(CheckCXXCompilerFlag)

if (ENABLE_SANITIZER)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # the flag is available since MSVC 2019 16.9
        # see https://learn.microsoft.com/en-us/cpp/build/reference/fsanitize?view=msvc-160
        check_cxx_compiler_flag("/fsanitize=address" SANITIZE_ADDRESS_SUPPORTED)
        if (SANITIZE_ADDRESS_SUPPORTED)
            set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} /fsanitize=address")
        else()
            message(FATAL_ERROR "Address sanitizer is not supported by current compiler.\n"
            "Please, check requirements:\n"
            "https://github.com/openvinotoolkit/openvino/wiki/AddressSanitizer-and-LeakSanitizer")
        endif()
    elseif(OV_COMPILER_IS_CLANG)
        set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -fsanitize=address -fsanitize-blacklist=${OpenVINO_SOURCE_DIR}/tests/sanitizers/asan/ignore.txt")
        if(BUILD_SHARED_LIBS)
            set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -shared-libasan")
        endif()

        check_cxx_compiler_flag("-fsanitize-recover=address" SANITIZE_RECOVER_ADDRESS_SUPPORTED)
        if (SANITIZE_RECOVER_ADDRESS_SUPPORTED)
            set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -fsanitize-recover=address")
        endif()

        set(SANITIZER_LINKER_FLAGS "${SANITIZER_LINKER_FLAGS} -fsanitize=address -fsanitize-blacklist=${OpenVINO_SOURCE_DIR}/tests/sanitizers/asan/ignore.txt")
        if(BUILD_SHARED_LIBS)
            set(SANITIZER_LINKER_FLAGS "${SANITIZER_LINKER_FLAGS} -shared-libasan")
        endif()
    elseif(CMAKE_COMPILER_IS_GNUCXX)
        set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -fsanitize=address")

        check_cxx_compiler_flag("-fsanitize-recover=address" SANITIZE_RECOVER_ADDRESS_SUPPORTED)
        if (SANITIZE_RECOVER_ADDRESS_SUPPORTED)
            set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -fsanitize-recover=address")
        endif()
        set(SANITIZER_LINKER_FLAGS "${SANITIZER_LINKER_FLAGS} -fsanitize=address")
    else()
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
    endif()
endif()

if(ENABLE_UB_SANITIZER)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message(FATAL_ERROR "UndefinedBehavior sanitizer is not supported in Windows with MSVC compiler. Please, use clang-cl or mingw")
    endif()

    # TODO: Remove -fno-sanitize=null as thirdparty/ocl/clhpp_headers UBSAN compatibility resolved:
    # https://github.com/KhronosGroup/OpenCL-CLHPP/issues/17
    # Mute -fsanitize=function Indirect call of a function through a function pointer of the wrong type.
    #   Sample cases:
    #       call to function get_api_version through pointer to incorrect function type 'void *(*)()'
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

    if(CMAKE_COMPILER_IS_GNUCXX)
        # TODO: Remove -Wno-maybe-uninitialized after CVS-61143 is fixed
        set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -Wno-maybe-uninitialized")
    endif()
    check_cxx_compiler_flag("-fsanitize-recover=undefined" SANITIZE_RECOVER_UNDEFINED_SUPPORTED)
    if(SANITIZE_RECOVER_UNDEFINED_SUPPORTED)
        set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -fsanitize-recover=undefined")
    endif()
    
    if(OV_COMPILER_IS_CLANG)
        set(SANITIZER_LINKER_FLAGS "${SANITIZER_LINKER_FLAGS} -lubsan")
    else()
        set(SANITIZER_LINKER_FLAGS "${SANITIZER_LINKER_FLAGS} -fsanitize=undefined")
    endif()
endif()

if(ENABLE_THREAD_SANITIZER)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message(FATAL_ERROR "Thread sanitizer is not supported in Windows with MSVC compiler. Please, use clang-cl or mingw")
    elseif(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
        set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -fsanitize=thread")
        if(OV_COMPILER_IS_CLANG)
            set(SANITIZER_LINKER_FLAGS "${SANITIZER_LINKER_FLAGS} -ltsan")
        else()
            set(SANITIZER_LINKER_FLAGS "${SANITIZER_LINKER_FLAGS} -fsanitize=thread")
        endif()
    else()
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
    endif()
endif()

# common sanitizer options
if(DEFINED SANITIZER_COMPILER_FLAGS)
    # ensure symbols are present
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} /Oy-")
    elseif(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
        set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -g -fno-omit-frame-pointer")
        if(CMAKE_COMPILER_IS_GNUCXX)
            # GPU plugin tests compilation is slow with -fvar-tracking-assignments on GCC.
            # Clang has no var-tracking-assignments.
            set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -fno-var-tracking-assignments")
        endif()
        # prevent unloading libraries at runtime, so sanitizer can resolve their symbols
        if(NOT OV_COMPILER_IS_APPLECLANG)
            set(SANITIZER_LINKER_FLAGS "${SANITIZER_LINKER_FLAGS} -Wl,-z,nodelete")

            # clang does not provide rpath if -shared-libasan is used
            # https://stackoverflow.com/questions/68571138/asan-dynamic-runtime-is-missing-on-ubuntu-18, https://bugs.llvm.org/show_bug.cgi?id=51271
            if(BUILD_SHARED_LIBS AND ENABLE_SANITIZER AND OV_COMPILER_IS_CLANG)
                execute_process(COMMAND ${CMAKE_CXX_COMPILER} --print-file-name libclang_rt.asan-x86_64.so
                    OUTPUT_VARIABLE OV_LIBASAN_FILEPATH)
                get_filename_component(LIBASAN_DIRNAME ${OV_LIBASAN_FILEPATH} PATH)
                set(SANITIZER_LINKER_FLAGS "${SANITIZER_LINKER_FLAGS},-rpath=${LIBASAN_DIRNAME}")
            endif()

            if(OV_COMPILER_IS_CLANG AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 8.0)
                set(SANITIZER_LINKER_FLAGS "${SANITIZER_LINKER_FLAGS} -fuse-ld=lld")
            endif()
        endif()
    else()
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
    endif()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SANITIZER_COMPILER_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZER_COMPILER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${SANITIZER_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${SANITIZER_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SANITIZER_LINKER_FLAGS}")
endif()
