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

#ifndef GPU_OCL_VERBOSE_HPP
#define GPU_OCL_VERBOSE_HPP

#include <cstdio>

#include "gpu/compute/device_info.hpp"
#include "gpu/ocl/ocl_engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

void print_verbose_header() {
    ocl_engine_factory_t factory(engine_kind::gpu);
    for (size_t i = 0; i < factory.count(); ++i) {
        engine_t *eng_ptr = nullptr;
        factory.engine_create(&eng_ptr, i);
        std::unique_ptr<ocl_gpu_engine_t, engine_deleter_t> eng;
        eng.reset(utils::downcast<ocl_gpu_engine_t *>(eng_ptr));
        auto *dev_info = eng ? eng->device_info() : nullptr;

        auto s_name = dev_info ? dev_info->name() : "unknown";
        auto s_ver = dev_info ? dev_info->runtime_version().str() : "unknown";

        printf("dnnl_verbose,info,gpu,engine,%d,name:%s,driver_version:%s\n",
                (int)i, s_name.c_str(), s_ver.c_str());
    }
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
