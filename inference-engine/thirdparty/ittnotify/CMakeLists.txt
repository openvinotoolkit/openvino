#   Intel® Single Event API
#
#   This file is provided under the BSD 3-Clause license.
#   Copyright (c) 2015, Intel Corporation
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#       Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#       Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#       Neither the name of the Intel Corporation nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
#   IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#********************************************************************************************************************************************************************************************************************************************************************************************

set(ITT_PUBLIC_HDRS
    include/ittnotify.h
    include/jitprofiling.h
    include/libittnotify.h
    include/llvm_jit_event_listener.hpp
)

file(GLOB ITT_SRCS "src/ittnotify/*.c" "src/ittnotify/*.h")

add_library(ittnotify STATIC ${ITT_SRCS} ${ITT_PUBLIC_HDRS})

if(APPLE)
    SET_TARGET_PROPERTIES(ittnotify PROPERTIES OUTPUT_NAME ittnotify)
else()
    if(ARCH_64)
        SET_TARGET_PROPERTIES(ittnotify PROPERTIES OUTPUT_NAME ittnotify64)
    else()
        SET_TARGET_PROPERTIES(ittnotify PROPERTIES OUTPUT_NAME ittnotify32)
    endif()
endif()

target_include_directories(ittnotify PUBLIC ./src/ittnotify)

if (NOT WIN32)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    TARGET_LINK_LIBRARIES(ittnotify dl)
endif()

if (ANDROID)
    target_link_libraries(ittnotify log)
endif()

SET_TARGET_PROPERTIES(ittnotify PROPERTIES LINKER_LANGUAGE C)

target_include_directories(ittnotify
    PUBLIC include src/ittnotify
)
