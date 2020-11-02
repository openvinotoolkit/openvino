# ******************************************************************************
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
# ******************************************************************************

# function check existence of cmake and environment variable and read their value
# If the variable is not defined then the supplied default value used
# example:
#    ngraph_var(NGRAPH_VARIABLE_NAME DEFAULT "OFF")
#        checks if variable NGRAPH_VARIABLE_NAME was passed in via the cmake command line
#            if found then passed in value is used
#            else checks for the existence of the environment variable NGRAPH_VARIABLE_NAME
#                if found it's value is used
#        if none of the above then the default value is used
function(NGRAPH_VAR)
    set(options)
    set(oneValueArgs DEFAULT)
    set(multiValueArgs)
    cmake_parse_arguments(NGRAPH_VAR "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if (NOT DEFINED ${NGRAPH_VAR_UNPARSED_ARGUMENTS})
        set(${NGRAPH_VAR_UNPARSED_ARGUMENTS} ${NGRAPH_VAR_DEFAULT} PARENT_SCOPE)
        if(DEFINED ENV{${NGRAPH_VAR_UNPARSED_ARGUMENTS}})
            set(${NGRAPH_VAR_UNPARSED_ARGUMENTS} $ENV{${NGRAPH_VAR_UNPARSED_ARGUMENTS}} PARENT_SCOPE)
        endif()
    endif()
endfunction()
