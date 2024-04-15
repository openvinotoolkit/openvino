// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive_inst.h"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/file_util.hpp"
#include "to_string_utils.h"
#include "register.hpp"
#include "utils.hpp"
#include "runtime/ocl/ocl_event.hpp"

#include "quantize_inst.h"
#include "reorder_inst.h"

#include <vector>
#include <list>
#include <utility>

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

    typed_primitive_sycl_impl()
        : typed_primitive_impl<PType>({}, "undef"),
          _engine(nullptr){
    }

    bool is_cpu() const override { return false; }
    bool is_onednn() const override { return false; }

    void save(BinaryOutputBuffer& ob) const override {
    }

    void load(BinaryInputBuffer& ib) override {
    }

protected:

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override { }

    void set_arguments_impl(typed_primitive_inst<PType>& instance) override {
        if (instance.can_be_optimized())
            return;
    }

    void update_dispatch_data(const kernel_impl_params& impl_params) override {}

    void set_arguments_impl(typed_primitive_inst<PType>& instance, kernel_arguments_data& args) override {
        if (instance.can_be_optimized()) {
            return;
        }
    }

    event::ptr execute_impl(const std::vector<event::ptr>& /* events */,
                            typed_primitive_inst<PType>& instance) override {
        auto& network = instance.get_network();
        auto& stream = network.get_stream();
        auto net_id = network.get_id();
        event::ptr event;


        return event;
    }

    static event::ptr to_ocl_event(stream& stream, ::sycl::event e) {
        if (stream.get_queue_type() == QueueTypes::out_of_order) {
            auto native_events = get_native<::sycl::backend::opencl, ::sycl::event>(e);
            std::vector<event::ptr> events;
            for (auto& e : native_events) {
                events.push_back(std::make_shared<ocl::ocl_event>(cl::Event(e, true)));
            }
            return events.empty() ? stream.create_user_event(true) : stream.group_events(events);
        } else {
            return stream.create_user_event(true);
        }
    }

    std::vector<layout> get_internal_buffer_layouts_impl() const override {
        return {};
    }
};

}  // namespace sycl
}  // namespace cldnn
