// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_stream.hpp"

#include <memory>
#include <mutex>

#include "intel_gpu/runtime/memory.hpp"
#include "openvino/core/except.hpp"
#include "vulkan_engine.hpp"
#include "vulkan_event.hpp"

namespace cldnn {
namespace vulkan {
namespace {

class empty_surfaces_lock final : public surfaces_lock {};

}  // namespace

vulkan_stream::vulkan_stream(const vulkan_engine& engine) : stream(QueueTypes::in_order, SyncMethods::none), _engine(engine) {}

vulkan_stream::vulkan_stream(const vulkan_engine& engine, const ExecutionConfig& config)
    : stream(config.get_queue_type(), stream::get_expected_sync_method(config)),
      _engine(engine) {}

void vulkan_stream::flush() const {}

void vulkan_stream::finish() const {
    std::lock_guard<std::mutex> lock(_engine.get_queue_mutex());
    const auto result = vkQueueWaitIdle(_engine.get_compute_queue());
    OPENVINO_ASSERT(result == VK_SUCCESS, "[GPU][Vulkan] vkQueueWaitIdle failed with VkResult ", static_cast<int>(result));
}

void vulkan_stream::wait() {
    finish();
}

void vulkan_stream::set_arguments(kernel&, const kernel_arguments_desc&, const kernel_arguments_data&) {
    OPENVINO_THROW("[GPU][Vulkan] Kernel argument binding is not implemented yet");
}

event::ptr vulkan_stream::enqueue_kernel(kernel&, const kernel_arguments_desc&, const kernel_arguments_data&, const std::vector<event::ptr>&, bool) {
    OPENVINO_THROW("[GPU][Vulkan] Kernel dispatch is not implemented yet");
}

event::ptr vulkan_stream::enqueue_marker(const std::vector<event::ptr>& dependencies, bool) {
    wait_for_events(dependencies);
    return create_user_event(true);
}

void vulkan_stream::enqueue_barrier() {
    finish();
}

event::ptr vulkan_stream::group_events(const std::vector<event::ptr>& dependencies) {
    wait_for_events(dependencies);
    return create_user_event(true);
}

void vulkan_stream::wait_for_events(const std::vector<event::ptr>& events) {
    for (const auto& event : events) {
        if (event != nullptr) {
            event->wait();
        }
    }
}

event::ptr vulkan_stream::create_user_event(bool set) {
    return std::make_shared<vulkan_event>(set);
}

event::ptr vulkan_stream::create_base_event() {
    return std::make_shared<vulkan_event>();
}

std::unique_ptr<surfaces_lock> vulkan_stream::create_surfaces_lock(const std::vector<memory::ptr>&) const {
    return std::make_unique<empty_surfaces_lock>();
}

#ifdef ENABLE_ONEDNN_FOR_GPU
dnnl::stream& vulkan_stream::get_onednn_stream() {
    OPENVINO_THROW("[GPU][Vulkan] oneDNN stream interop is not supported");
}
#endif

}  // namespace vulkan
}  // namespace cldnn
