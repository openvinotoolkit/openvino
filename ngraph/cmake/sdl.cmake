#===============================================================================
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# Manage secure Development Lifecycle-related compiler flags
#===============================================================================

if(SDL_cmake_included)
    return()
endif()
set(SDL_cmake_included true)

if(UNIX)
    list(APPEND NGRAPH_COMMON_FLAGS -Wformat -Wformat-security)
    list(APPEND NGRAPH_COMMON_FLAGS -O2 -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=2)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
            list(APPEND NGRAPH_COMMON_FLAGS -fstack-protector-all)
        else()
            list(APPEND NGRAPH_COMMON_FLAGS -fstack-protector-strong)
        endif()

        # GCC might be very paranoid for partial structure initialization, e.g.
        #   struct { int a, b; } s = { 0, };
        # However the behavior is triggered by `Wmissing-field-initializers`
        # only. To prevent warnings on users' side who use the library and turn
        # this warning on, let's use it too. Applicable for the library sources
        # and interfaces only (tests currently rely on that fact heavily)
        # set(CMAKE_SRC_CCXX_FLAGS "${CMAKE_SRC_CCXX_FLAGS} -Wmissing-field-initializers")
        # set(CMAKE_EXAMPLE_CCXX_FLAGS "${CMAKE_EXAMPLE_CCXX_FLAGS} -Wmissing-field-initializers")
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        list(APPEND NGRAPH_COMMON_FLAGS -fstack-protector-all)
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        list(APPEND NGRAPH_COMMON_FLAGS -fstack-protector)
    endif()
    if(APPLE)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-bind_at_load")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-bind_at_load")
    else()
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
    endif()
endif()
