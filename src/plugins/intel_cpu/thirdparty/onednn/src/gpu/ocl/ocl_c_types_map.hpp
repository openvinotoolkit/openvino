/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_OCL_OCL_C_TYPES_MAP_HPP
#define GPU_OCL_OCL_C_TYPES_MAP_HPP

#include "oneapi/dnnl/dnnl_ocl_types.h"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

using memory_kind_t = dnnl_ocl_interop_memory_kind_t;
namespace memory_kind {
const memory_kind_t usm = dnnl_ocl_interop_usm;
const memory_kind_t buffer = dnnl_ocl_interop_buffer;
} // namespace memory_kind

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
