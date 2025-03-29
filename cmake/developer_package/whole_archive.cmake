# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#[[
function links static library without removing any symbol from it.

ov_target_link_whole_archive(<target name> <lib1> [<lib2> ...])
Example:
ov_target_link_whole_archive("FunctionalTests" "CommonLib" "AnotherLib")

#]]

function(ov_target_link_whole_archive targetName)
    foreach(staticLib IN LISTS ARGN)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            # CMake does not support generator expression in LINK_FLAGS, so we workaround it a little bit:
            # passing same static library as normal link (to get build deps working, and includes too), than using WHOLEARCHIVE option
            # it's important here to not use slash '/' for option !
            if(CMAKE_GENERATOR MATCHES "Visual Studio")
                # MSBuild is unhappy when parsing double quotes in combination with WHOLEARCHIVE flag.
                # remove quotes from path - so build path with spaces not supported, but it's better than nothing.
                list(APPEND libs ${staticLib}
                    "-WHOLEARCHIVE:$<TARGET_FILE:${staticLib}>"
                    )
                if(CMAKE_CURRENT_BINARY_DIR MATCHES " ")
                    message(WARNING "Visual Studio CMake generator may cause problems if your build directory contains spaces. "
                        "Remove spaces from path or select different generator.")
                endif()
            else()
                list(APPEND libs ${staticLib}
                    "-WHOLEARCHIVE:\"$<TARGET_FILE:${staticLib}>\""
                    )
            endif()
        elseif(OV_COMPILER_IS_APPLECLANG)
            list(APPEND libs
                "-Wl,-all_load"
                ${staticLib}
                "-Wl,-noall_load"
                )
        else()
            # non-Apple Clang and GCC / MinGW
            list(APPEND libs
                "-Wl,--whole-archive"
                ${staticLib}
                "-Wl,--no-whole-archive"
                )
        endif()
    endforeach()
    if(libs)
        target_link_libraries(${targetName} PRIVATE ${libs})
    endif()
endfunction()
