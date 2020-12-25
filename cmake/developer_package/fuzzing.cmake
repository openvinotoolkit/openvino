# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

macro(enable_fuzzing)
    # Enable (libFuzzer)[https://llvm.org/docs/LibFuzzer.html] if supported.
    set(FUZZING_COMPILER_FLAGS "-fsanitize=fuzzer-no-link -fprofile-instr-generate -fcoverage-mapping")
    set(FUZZING_LINKER_FLAGS "-fsanitize-coverage=trace-pc-guard -fprofile-instr-generate")

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
        set_target_properties(${FUZZER_EXE_NAME} PROPERTIES LINK_FLAGS "-fsanitize=fuzzer")
    endif()
endfunction(add_fuzzer)
