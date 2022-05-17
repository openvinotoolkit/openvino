#===============================================================================
# Copyright 2020 Intel Corporation
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

find_library(PI_CUDA_LIBRARIES
    NAMES pi_cuda libpi_cuda.so  PATHS
      PATH_SUFFIXES lib)

find_package_handle_standard_args(PI_CUDA REQUIRED_VARS PI_CUDA_LIBRARIES)

if(TARGET PI_CUDA::PI_CUDA OR NOT PI_CUDA_FOUND)
    return()
endif()

add_library(PI_CUDA::PI_CUDA UNKNOWN IMPORTED)
set_target_properties(PI_CUDA::PI_CUDA PROPERTIES
    IMPORTED_LOCATION ${PI_CUDA_LIBRARIES})

mark_as_advanced(PI_CUDA_LIBRARIES)
