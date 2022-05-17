#===============================================================================
# Copyright 2020 Intel Corporation
# Copyright 2020 Codeplay Software Limited
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

find_package(CUDA 10.0 REQUIRED)
find_package(Threads REQUIRED)

find_path(CUBLAS_INCLUDE_DIR "cublas_v2.h"
          HINTS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
find_library(CUBLAS_LIBRARY cublas)
find_library(CUDA_DRIVER_LIBRARY cuda)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuBLAS
    REQUIRED_VARS
        CUBLAS_INCLUDE_DIR
        CUDA_INCLUDE_DIRS
        CUBLAS_LIBRARY
        CUDA_LIBRARIES
        CUDA_DRIVER_LIBRARY
)

if(NOT TARGET cuBLAS::cuBLAS)
    add_library(cuBLAS::cuBLAS SHARED IMPORTED)
    set_target_properties(cuBLAS::cuBLAS PROPERTIES
        IMPORTED_LOCATION ${CUBLAS_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES
        "${CUBLAS_INCLUDE_DIR};${CUDA_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES
        "Threads::Threads;${CUDA_DRIVER_LIBRARY};${CUDA_LIBRARIES}"
	INTERFACE_COMPILE_DEFINITIONS CUDA_NO_HALF)
endif()
