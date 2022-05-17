/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#include <assert.h>
#include <cinttypes>
#include <stdio.h>

#include "oneapi/dnnl/dnnl_debug.h"
#include "oneapi/dnnl/dnnl_types.h"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#define DPRINT(...) \
    do { \
        int l = snprintf(str + written_len, str_len, __VA_ARGS__); \
        if (l < 0) return l; \
        if ((size_t)l >= str_len) return -1; \
        written_len += l; \
        str_len -= l; \
    } while (0)

const char *dnnl_runtime2str(unsigned runtime) {
    switch (runtime) {
        case DNNL_RUNTIME_NONE: return "none";
        case DNNL_RUNTIME_SEQ: return "sequential";
        case DNNL_RUNTIME_OMP: return "OpenMP";
        case DNNL_RUNTIME_TBB: return "TBB";
        case DNNL_RUNTIME_TBB_AUTO: return "TBB_AUTO";
        case DNNL_RUNTIME_OCL: return "OpenCL";
        case DNNL_RUNTIME_THREADPOOL: return "threadpool";
#ifdef DNNL_WITH_SYCL
        case DNNL_RUNTIME_SYCL: return "DPC++";
#endif
        default: return "unknown";
    }
}
