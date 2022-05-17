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

#ifndef GPU_OCL_OCL_GPU_ENGINE_ID_HPP
#define GPU_OCL_OCL_GPU_ENGINE_ID_HPP

#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct ocl_gpu_engine_id_impl_t : public engine_id_impl_t {

    ocl_gpu_engine_id_impl_t(cl_device_id device, cl_context context,
            engine_kind_t kind, runtime_kind_t runtime_kind, size_t index)
        : engine_id_impl_t(kind, runtime_kind, index)
        , device_(device, true)
        , context_(context, true) {}

    ~ocl_gpu_engine_id_impl_t() override = default;

private:
    bool compare_resource(const engine_id_impl_t *id_impl) const override {
        const auto *typed_id
                = utils::downcast<const ocl_gpu_engine_id_impl_t *>(id_impl);
        return device_ == typed_id->device_ && context_ == typed_id->context_;
    }

    size_t hash_resource() const override {
        size_t seed = 0;
        seed = hash_combine(seed, device_.get());
        seed = hash_combine(seed, context_.get());
        return seed;
    }

    ocl_wrapper_t<cl_device_id> device_;
    ocl_wrapper_t<cl_context> context_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
