# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

macro(enable_fuzzing)
    # Enable (libFuzzer)[https://llvm.org/docs/LibFuzzer.html] if supported.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # see https://learn.microsoft.com/en-us/cpp/build/reference/fsanitize?view=msvc-160#remarks
        set(FUZZING_COMPILER_FLAGS "/fsanitize=fuzzer")
    elseif(OV_COMPILER_IS_CLANG)
        set(FUZZING_COMPILER_FLAGS "-fsanitize=fuzzer-no-link -fprofile-instr-generate -fcoverage-mapping")
        set(FUZZING_LINKER_FLAGS "-fsanitize-coverage=trace-pc-guard -fprofile-instr-generate")
    endif()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${FUZZING_COMPILER_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FUZZING_COMPILER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${FUZZING_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${FUZZING_LINKER_FLAGS}")

    unset(FUZZING_COMPILER_FLAGS)
    unset(FUZZING_LINKER_FLAGS)
endmacro()

function(add_fuzzer FUZZER_EXE_NAME FUZZER_SOURCES)
    add_executable(${FUZZER_EXE_NAME} ${FUZZER_SOURCES})
    target_link_libraries(${FUZZER_EXE_NAME} PRIVATE fuzz-testhelper)
    if(ENABLE_FUZZING)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            # no extra flags are required
        elseif(OV_COMPILER_IS_CLANG)
            set_target_properties(${FUZZER_EXE_NAME} PROPERTIES LINK_FLAGS "-fsanitize=fuzzer")
        endif()
    endif()
endfunction(add_fuzzer)
