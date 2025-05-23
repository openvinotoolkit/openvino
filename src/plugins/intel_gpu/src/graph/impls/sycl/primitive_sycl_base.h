// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive_inst.h"
#include "intel_gpu/runtime/memory.hpp"
#include "registry/registry.hpp"
#include "runtime/ocl/ocl_event.hpp"

#include <vector>

#include "sycl/sycl.hpp"

namespace cldnn {
namespace sycl {

static std::mutex cacheAccessMutex;

template <class PType>
struct typed_primitive_sycl_impl : public typed_primitive_impl<PType> {
    const engine* _engine;

    typed_primitive_sycl_impl(const engine& engine, const ExecutionConfig& config, std::shared_ptr<WeightsReorderParams> weights_reorder = nullptr)
        : typed_primitive_impl<PType>(weights_reorder, "sycl_kernel"),
        _engine(&engine) { }

    typed_primitive_sycl_impl() : typed_primitive_impl<PType>({}, "undef"), _engine(nullptr) {
    }

    bool is_cpu() const override { return false; }
    bool is_onednn() const override { return false; }

protected:
    void init_kernels(const kernels_cache&, const kernel_impl_params&) override { }

    void set_arguments_impl(typed_primitive_inst<PType>& instance) override { }
    void set_arguments_impl(typed_primitive_inst<PType>& instance, kernel_arguments_data& args) override { }

    static event::ptr to_ocl_event(stream& stream, ::sycl::event e) {
        if (stream.get_queue_type() == QueueTypes::out_of_order) {
            auto native_events = ::sycl::get_native<::sycl::backend::opencl, ::sycl::event>(e);
            std::vector<event::ptr> events;
            for (auto& e : native_events) {
                events.push_back(std::make_shared<ocl::ocl_event>(cl::Event(e, true)));
            }
            return stream.aggregate_events(events);
        } else {
            return nullptr;
        }
    }

    std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params&) const override {
        return {};
    }
};

}  // namespace sycl
}  // namespace cldnn
