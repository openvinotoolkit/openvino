# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(enable_fuzzing)
    # Enable (libFuzzer)[https://llvm.org/docs/LibFuzzer.html] if supported.
    if(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$" AND NOT WIN32)
        # Communicate libfuzzer is enabled
        set(WITH_LIBFUZZER ON PARENT_SCOPE)
        add_compile_definitions(WITH_LIBFUZZER)

        # Enable libfuzzer and code coverage
        set(FUZZING_COMPILER_FLAGS "-fsanitize=fuzzer-no-link -fprofile-instr-generate -fcoverage-mapping")
        set(FUZZING_LINKER_FLAGS "-fsanitize-coverage=trace-pc-guard -fprofile-instr-generate")

        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${FUZZING_COMPILER_FLAGS}" PARENT_SCOPE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FUZZING_COMPILER_FLAGS}" PARENT_SCOPE)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${FUZZING_LINKER_FLAGS}" PARENT_SCOPE)
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${FUZZING_LINKER_FLAGS}" PARENT_SCOPE)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${FUZZING_LINKER_FLAGS}")
    endif()
endfunction(enable_fuzzing)


function(add_fuzzer FUZZER_EXE_NAME FUZZER_SOURCES)
    add_executable(${FUZZER_EXE_NAME} ${FUZZER_SOURCES})
    if(WITH_LIBFUZZER)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=fuzzer" PARENT_SCOPE)
    endif()
    target_link_libraries(${FUZZER_EXE_NAME} PRIVATE fuzz-testhelper)
endfunction(add_fuzzer)
