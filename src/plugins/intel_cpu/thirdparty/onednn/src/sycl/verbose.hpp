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

#ifndef SYCL_VERBOSE_HPP
#define SYCL_VERBOSE_HPP

#include <cstdio>

#include "gpu/compute/device_info.hpp"
#include "sycl/sycl_engine.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

void print_verbose_header(engine_kind_t kind) {
    sycl_engine_factory_t factory(kind);
    for (size_t i = 0; i < factory.count(); ++i) {
        engine_t *eng_ptr = nullptr;
        factory.engine_create(&eng_ptr, i);
        std::unique_ptr<sycl_engine_base_t, engine_deleter_t> eng;
        eng.reset(utils::downcast<sycl_engine_base_t *>(eng_ptr));
        auto *dev_info = eng ? eng->device_info() : nullptr;

        auto s_engine_kind = (kind == engine_kind::cpu ? "cpu" : "gpu");
        auto s_backend = eng ? to_string(eng->backend()) : "unknown";
        auto s_name = dev_info ? dev_info->name() : "unknown";
        auto s_ver = dev_info ? dev_info->runtime_version().str() : "unknown";

        printf("dnnl_verbose,info,%s,engine,%d,backend:%s,name:%s,driver_"
               "version:%s\n",
                s_engine_kind, (int)i, s_backend.c_str(), s_name.c_str(),
                s_ver.c_str());
    }
}

void print_verbose_header() {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
    print_verbose_header(engine_kind::cpu);
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
    print_verbose_header(engine_kind::gpu);
#endif
}

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif
