# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG OR (UNIX AND OV_COMPILER_IS_INTEL_LLVM))
    set(OV_C_CXX_FLAGS "${OV_C_CXX_FLAGS} -Wformat -Wformat-security")

    if (NOT ENABLE_SANITIZER)
        if(EMSCRIPTEN)
            # emcc does not support fortification, see:
            # https://stackoverflow.com/questions/58854858/undefined-symbol-stack-chk-guard-in-libopenh264-so-when-building-ffmpeg-wit
        else()
            # ASan does not support fortification https://github.com/google/sanitizers/issues/247
            set(OV_C_CXX_FLAGS "${OV_C_CXX_FLAGS} -D_FORTIFY_SOURCE=2")
        endif()
    endif()

    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -pie")

    if(CMAKE_COMPILER_IS_GNUCXX)
        set(OV_C_CXX_FLAGS "${OV_C_CXX_FLAGS} -fno-strict-overflow -fno-delete-null-pointer-checks -fwrapv")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
            set(OV_C_CXX_FLAGS "${OV_C_CXX_FLAGS} -fstack-protector-all")
        else()
            set(OV_C_CXX_FLAGS "${OV_C_CXX_FLAGS} -fstack-protector-strong")
        endif()
        if (NOT ENABLE_SANITIZER)
            # Remove all symbol table and relocation information from the executable
            set(OV_C_CXX_FLAGS "${OV_C_CXX_FLAGS} -s")
        endif()
        if(NOT MINGW AND NOT APPLE)
            set(OV_LINKER_FLAGS "${OV_LINKER_FLAGS} -z noexecstack -z relro -z now")
        endif()
    elseif(OV_COMPILER_IS_CLANG)
        if(EMSCRIPTEN)
            # emcc does not support fortification
            # https://stackoverflow.com/questions/58854858/undefined-symbol-stack-chk-guard-in-libopenh264-so-when-building-ffmpeg-wit
        else()
            set(OV_C_CXX_FLAGS "${OV_C_CXX_FLAGS} -fstack-protector-all")
        endif()
    elseif(OV_COMPILER_IS_INTEL_LLVM)
        set(OV_C_CXX_FLAGS "${OV_C_CXX_FLAGS} -fstack-protector-strong")
        set(OV_LINKER_FLAGS "${OV_LINKER_FLAGS} -z noexecstack -z relro -z now")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR (OV_COMPILER_IS_INTEL_LLVM AND WIN32))
    set(OV_C_CXX_FLAGS "${OV_C_CXX_FLAGS} /sdl /guard:cf")
    set(OV_LINKER_FLAGS "${OV_LINKER_FLAGS} /guard:cf")
endif()

if(ENABLE_QSPECTRE)
    set(OV_C_CXX_FLAGS "${OV_C_CXX_FLAGS} /Qspectre")
endif()

if(ENABLE_INTEGRITYCHECK)
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /INTEGRITYCHECK")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR (OV_COMPILER_IS_INTEL_LLVM AND WIN32))
    # add sdl required flags to both Debug and Release on Windows
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OV_C_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OV_C_CXX_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OV_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${OV_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OV_LINKER_FLAGS}")
else()
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OV_C_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OV_C_CXX_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} ${OV_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "${CMAKE_MODULE_LINKER_FLAGS_RELEASE} ${OV_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${OV_LINKER_FLAGS}")
endif()

unset(OV_C_CXX_FLAGS)
unset(OV_LINKER_FLAGS)
