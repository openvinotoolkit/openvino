#===============================================================================
# Copyright 2018 Intel Corporation
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

# Auxiliary build functions
#===============================================================================

if(utils_cmake_included)
    return()
endif()
set(utils_cmake_included true)

# Register new executable/test
#   name -- name of the executable
#   srcs -- list of source, if many must be enclosed with ""
#   test -- "test" to mark executable as a test, "" otherwise
#   arg4 -- (optional) list of extra library dependencies
function(register_exe name srcs test)
    add_executable(${name} ${srcs})
    target_link_libraries(${name} ${LIB_NAME} ${EXTRA_LIBS} ${ARGV3})
    if("${test}" STREQUAL "test")
        add_test(${name} ${name})
        if(WIN32 OR MINGW)
            set_property(TEST ${name} PROPERTY ENVIRONMENT "PATH=${CTESTCONFIG_PATH};$ENV{PATH}")
            configure_file(${CMAKE_SOURCE_DIR}/config_template.vcxproj.user ${name}.vcxproj.user @ONLY)
        endif()
    endif()
endfunction()

# Append to a variable
#   var = var + value
macro(append var value)
    set(${var} "${${var}} ${value}")
endmacro()

# Set variable depending on condition:
#   var = cond ? val_if_true : val_if_false
macro(set_ternary var condition val_if_true val_if_false)
    if (${condition})
        set(${var} "${val_if_true}")
    else()
        set(${var} "${val_if_false}")
    endif()
endmacro()

# Conditionally set a variable
#   if (cond) var = value
macro(set_if condition var value)
    if (${condition})
        set(${var} "${value}")
    endif()
endmacro()

# Conditionally append
#   if (cond) var = var + value
macro(append_if condition var value)
    if (${condition})
        append(${var} "${value}")
    endif()
endmacro()
