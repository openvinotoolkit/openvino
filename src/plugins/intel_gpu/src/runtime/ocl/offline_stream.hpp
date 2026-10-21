// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "openvino/core/except.hpp"

#include <memory>
#include <vector>

namespace cldnn {
namespace ocl {

// No-op stream for the offline engine: sync/wait are harmless no-ops, every execution method throws
// because nothing is ever enqueued during a compile-only build.
class offline_stream : public stream {
public:
    offline_stream() : stream(QueueTypes::in_order, SyncMethods::none) {}

    void flush() const override {}
    void finish() const override {}
    void wait() override {}

    void set_arguments(kernel&, const kernel_arguments_desc&, const kernel_arguments_data&) override {
        OPENVINO_THROW("[GPU offline] stream::set_arguments is not available on the compile-only engine");
    }
    event::ptr enqueue_kernel(kernel&,
                              const kernel_arguments_desc&,
                              const kernel_arguments_data&,
                              std::vector<event::ptr> const&,
                              bool /*is_output_event*/ = false) override {
        OPENVINO_THROW("[GPU offline] stream::enqueue_kernel is not available on the compile-only engine");
    }
    event::ptr enqueue_marker(std::vector<event::ptr> const&, bool /*is_output_event*/ = false) override {
        OPENVINO_THROW("[GPU offline] stream::enqueue_marker is not available on the compile-only engine");
    }
    void enqueue_barrier(const std::vector<event::ptr>& /*deps*/ = {}) override {
        OPENVINO_THROW("[GPU offline] stream::enqueue_barrier is not available on the compile-only engine");
    }
    event::ptr group_events(std::vector<event::ptr> const&) override {
        OPENVINO_THROW("[GPU offline] stream::group_events is not available on the compile-only engine");
    }
    void wait_for_events(const std::vector<event::ptr>&) override {
        OPENVINO_THROW("[GPU offline] stream::wait_for_events is not available on the compile-only engine");
    }
    event::ptr create_user_event(bool /*set*/) override {
        OPENVINO_THROW("[GPU offline] stream::create_user_event is not available on the compile-only engine");
    }
    event::ptr create_base_event() override {
        OPENVINO_THROW("[GPU offline] stream::create_base_event is not available on the compile-only engine");
    }
    std::unique_ptr<surfaces_lock> create_surfaces_lock(const std::vector<memory::ptr>&) const override {
        OPENVINO_THROW("[GPU offline] stream::create_surfaces_lock is not available on the compile-only engine");
    }

#ifdef ENABLE_ONEDNN_FOR_GPU
    dnnl::stream& get_onednn_stream() override {
        OPENVINO_THROW("[GPU offline] stream::get_onednn_stream is not available on the compile-only engine");
    }
#endif
};

}  // namespace ocl
}  // namespace cldnn
