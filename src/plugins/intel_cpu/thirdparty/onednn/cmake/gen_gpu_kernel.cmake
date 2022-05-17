#===============================================================================
# Copyright 2019-2021 Intel Corporation
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

# Generates cpp file with GPU kernel or header code stored as string
# Parameters:
#   CL_FILE    -- path to the kernel source or header file
#   GEN_FILE   -- path to the generated cpp file
#===============================================================================

file(READ ${CL_FILE} cl_file_lines)

# Remove C++ style comments
string(REGEX REPLACE "//[^\n]*\n" "\n" cl_file_lines "${cl_file_lines}")
# Remove repeated whitespaces
string(REGEX REPLACE " +" " " cl_file_lines "${cl_file_lines}")
# Remove leading whitespaces
string(REGEX REPLACE "\n " "\n" cl_file_lines "${cl_file_lines}")
# Remove empty lines
string(REGEX REPLACE "\n+" "\n" cl_file_lines "${cl_file_lines}")

string(LENGTH "${cl_file_lines}" len)
if(len GREATER 65535)
    message(WARNING "Windows requires string literals to fit in 65535 bytes. Please split ${CL_FILE}.")
endif()

get_filename_component(cl_file_name ${CL_FILE} NAME_WE)
get_filename_component(cl_file_ext ${CL_FILE} EXT)

# Split string into concatenated parts to circumvent the limitation on Windows
string(REGEX REPLACE "\n" " )==\"\"\\\\n\"\nR\"==(" cl_file_lines "${cl_file_lines}")

if(cl_file_ext STREQUAL ".cl")
    set(cl_file_contents  "const char *${cl_file_name}_kernel = R\"==(${cl_file_lines})==\";")
elseif(cl_file_ext STREQUAL ".h")
    set(cl_file_contents  "const char *${cl_file_name}_header = R\"==(${cl_file_lines})==\";")
else()
    message(FATAL_ERROR "Unknown file extensions: ${cl_file_ext}")
endif()

set(cl_file_contents "namespace ocl {\n${cl_file_contents}\n}")
set(cl_file_contents "namespace gpu {\n${cl_file_contents}\n}")
set(cl_file_contents "namespace impl {\n${cl_file_contents}\n}")
set(cl_file_contents "namespace dnnl {\n${cl_file_contents}\n}")

file(WRITE ${GEN_FILE} "${cl_file_contents}")
