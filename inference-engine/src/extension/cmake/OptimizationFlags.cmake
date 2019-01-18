#
# service functions:
#   set_target_cpu_flags
#   set_target_vectorizer_report_flags
#   print_target_compiler_options


# set processor speicif compilation options, based either on
# externally defined ENABLE_SSE42/ENABLE_AVX2 options or
# based on detected host processor featured (HAVE_SSE/HAVE_AVX2)
# Note, when ENABLE_AVX2 option is on by any means then ENABLE_SSE42 option
# will be turned on if not set explicitely


function(set_target_cpu_flags TARGET_NAME)
    # if have cpuid info and not requested specific cpu features externally
    # turn on cpu specific compile options based on detected features
    # of host processor
    # if don't have cpuid info or cpu specific features explicitly requested
    # set compile options based on requested features
    if(${HAVE_CPUID_INFO})
        # ENABLE_SSE42, ENABLE_AVX2, ENABLE_AVX512 weren't set explicitly,
        # so derive it from host cpu features
        if( (NOT DEFINED ENABLE_SSE42) AND (NOT DEFINED ENABLE_AVX2) AND (NOT DEFINED ENABLE_AVX512F) )
            set(ENABLE_SSE42   ${HAVE_SSE42})
            set(ENABLE_AVX2    ${HAVE_AVX2})
            set(ENABLE_AVX512F ${HAVE_AVX512F})
        endif()
        # ENABLE_SSE42 was set explicitly, ENABLE_AVX2 and ENABLE_AVX512F were not defined.
        # Consider as request to build for Atom, turn off AVX2 and AVX512
        if( (${ENABLE_SSE42}) AND (NOT DEFINED ENABLE_AVX2) AND (NOT DEFINED ENABLE_AVX512F) )
            set(ENABLE_AVX2    OFF)
            set(ENABLE_AVX512F OFF)
        endif()
        # ENABLE_AVX2 was set explicitly, ENABLE_SSE42 and ENABLE_AVX512F were not defined
        # Consider as request to build for Core, turn on SSE42 as supported feature
        if( (NOT DEFINED ENABLE_SSE42) AND (${ENABLE_AVX2}) AND (NOT DEFINED ENABLE_AVX512F) )
            set(ENABLE_SSE42   ON)
            set(ENABLE_AVX512F OFF)
        endif()
        # ENABLE_AVX512 was set explicitly, ENABLE_SSE42 and ENABLE_AVX2 were not defined
        # Consider as request to build for Xeon (Skylake server and later), turn on SSE42 and AVX2 as supported feature
        if( (NOT DEFINED ENABLE_SSE42) AND (NOT DEFINED ENABLE_AVX2) AND (${ENABLE_AVX512F}) )
            set(ENABLE_SSE42 ON)
            set(ENABLE_AVX2  ON)
        endif()
        # Compiler doesn't support AVX512 instructuion set
        if( (${ENABLE_AVX512F}) AND (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU) AND (NOT (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)) )
            set(ENABLE_AVX512F OFF)
            ext_message(WARNING "Compiler doesn't support AVX512 instructuion set")
        endif()
    endif()

    if(WIN32)
        if(${ENABLE_AVX512F})
            target_compile_definitions(${TARGET_NAME} PUBLIC "-DHAVE_SSE")
            target_compile_definitions(${TARGET_NAME} PUBLIC "-DHAVE_AVX2")
            target_compile_definitions(${TARGET_NAME} PUBLIC "-DHAVE_AVX512F")
            if(${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
                target_compile_options(${TARGET_NAME} PUBLIC "/QxCOMMON-AVX512")
                target_compile_options(${TARGET_NAME} PUBLIC "/Qvc14")
            endif()
            if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
                ext_message(WARNING "MSVC Compiler doesn't support AVX512 instructuion set")
            endif()
        elseif(${ENABLE_AVX2})
            target_compile_definitions(${TARGET_NAME} PUBLIC "-DHAVE_SSE")
            target_compile_definitions(${TARGET_NAME} PUBLIC "-DHAVE_AVX2")
            if(${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
                target_compile_options(${TARGET_NAME} PUBLIC "/QxCORE-AVX2")
                target_compile_options(${TARGET_NAME} PUBLIC "/Qvc14")
            endif()
            if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
                target_compile_options(${TARGET_NAME} PUBLIC "/arch:AVX2")
            endif()
        elseif(${ENABLE_SSE42})
            target_compile_definitions(${TARGET_NAME} PUBLIC "-DHAVE_SSE")
            if(${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
                target_compile_options(${TARGET_NAME} PUBLIC "/arch:SSE4.2")
                target_compile_options(${TARGET_NAME} PUBLIC "/QxSSE4.2")
                target_compile_options(${TARGET_NAME} PUBLIC "/Qvc14")
            endif()
            if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
                target_compile_options(${TARGET_NAME} PUBLIC "/arch:SSE4.2")
            endif()
        endif()
    endif()
    if(UNIX)
        if(${ENABLE_AVX512F})
            target_compile_definitions(${TARGET_NAME} PUBLIC "-DHAVE_SSE")
            target_compile_definitions(${TARGET_NAME} PUBLIC "-DHAVE_AVX2")
            target_compile_definitions(${TARGET_NAME} PUBLIC "-DHAVE_AVX512F")
            if(${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
                target_compile_options(${TARGET_NAME} PUBLIC "-xCOMMON-AVX512")
            endif()
            if(${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
                target_compile_options(${TARGET_NAME} PUBLIC "-mavx512f")
                target_compile_options(${TARGET_NAME} PUBLIC "-mfma")
            endif()
        elseif(${ENABLE_AVX2})
            target_compile_definitions(${TARGET_NAME} PUBLIC "-DHAVE_SSE")
            target_compile_definitions(${TARGET_NAME} PUBLIC "-DHAVE_AVX2")
            if(${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
                target_compile_options(${TARGET_NAME} PUBLIC "-march=core-avx2")
                target_compile_options(${TARGET_NAME} PUBLIC "-xCORE-AVX2")
                target_compile_options(${TARGET_NAME} PUBLIC "-mtune=core-avx2")
            endif()
            if(${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
                target_compile_options(${TARGET_NAME} PUBLIC "-mavx2")
                target_compile_options(${TARGET_NAME} PUBLIC "-mfma")
            endif()
            if(${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
                target_compile_options(${TARGET_NAME} PUBLIC "-mavx2")
                target_compile_options(${TARGET_NAME} PUBLIC "-mfma")
            endif()
        elseif(${ENABLE_SSE42})
            target_compile_definitions(${TARGET_NAME} PUBLIC "-DHAVE_SSE")
            if(${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
                target_compile_options(${TARGET_NAME} PUBLIC "-msse4.2")
                target_compile_options(${TARGET_NAME} PUBLIC "-xSSE4.2")
            endif()
            if(${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
                target_compile_options(${TARGET_NAME} PUBLIC "-msse4.2")
            endif()
        endif()
    endif()
endfunction()


# function set vectorization report flags in case of
# Intel compiler (might be useful for analisys of which loops were not
# vectorized and why)
function(set_target_vectorizer_report_flags TARGET_NAME)
    if(WIN32)
        if(${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
            target_compile_options(${TARGET_NAME} PUBLIC "/Qopt-report=3")
            target_compile_options(${TARGET_NAME} PUBLIC "/Qopt-report-format=vs")
            target_compile_options(${TARGET_NAME} PUBLIC "/Qopt-report-per-object")
        endif()
    endif()
    if(UNIX)
        if(${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
            target_compile_options(${TARGET_NAME} PUBLIC "-qopt-report=3")
            target_compile_options(${TARGET_NAME} PUBLIC "-qopt-report-format=text")
            target_compile_options(${TARGET_NAME} PUBLIC "-qopt-report-per-object")
        endif()
    endif()
endfunction()


# function print target compiler options to console
function(print_target_compiler_options TARGET_NAME)

    if(NOT TARGET ${TARGET_NAME})
        ext_message(WARNING "There is no target named '${TARGET_NAME}'")
        return()
    endif()

    ext_message(STATUS "Target ${TARGET_NAME}")
    ext_message(STATUS "    compiler definitions:")
    get_target_property(TARGET_COMPILE_DEFINITIONS ${TARGET_NAME} COMPILE_DEFINITIONS)
    if(TARGET_COMPILE_DEFINITIONS)
        ext_message(STATUS "        ${TARGET_COMPILE_DEFINITIONS}")
    else()
        ext_message(STATUS "        not set")
    endif()
    ext_message(STATUS "    compiler options:")
    get_target_property(TARGET_COMPILE_OPTIONS ${TARGET_NAME} COMPILE_OPTIONS)
    if(TARGET_COMPILE_OPTIONS)
        ext_message(STATUS "        ${TARGET_COMPILE_OPTIONS}")
    else()
        ext_message(STATUS "        not set")
    endif()
    ext_message(STATUS "    compiler flags:")
    get_target_property(TARGET_COMPILE_FLAGS ${TARGET_NAME} COMPILE_FLAGS)
    if(TARGET_COMPILE_FLAGS)
        ext_message(STATUS "        ${TARGET_COMPILE_FLAGS}")
    else()
        ext_message(STATUS "        not set")
    endif()

    ext_message(STATUS "    CXX_FLAGS:")
    ext_message(STATUS "        ${CMAKE_CXX_FLAGS}")

endfunction()
