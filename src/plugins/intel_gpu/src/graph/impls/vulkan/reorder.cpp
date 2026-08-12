// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder.hpp"

#include <memory>

#include "intel_gpu/runtime/stream.hpp"
#include "registry/implementation_map.hpp"

namespace cldnn {
namespace vulkan {
namespace {

bool is_copy_compatible_format(format fmt) {
    return fmt == format::any || format::is_default_format(fmt);
}

struct reorder_impl : typed_primitive_impl<reorder> {
    using parent = typed_primitive_impl<reorder>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::vulkan::reorder_impl)

    reorder_impl() : parent("vulkan_reorder") {}

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<reorder_impl>(*this);
    }

    bool is_cpu() const override {
        return false;
    }

    bool requires_lockable_input() const override {
        return false;
    }

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}

    event::ptr execute_impl(const std::vector<event::ptr>& events, reorder_inst& instance) override {
        auto& stream = instance.get_network().get_stream();
        stream.wait_for_events(events);
        return instance.output_memory_ptr()->copy_from(stream, *instance.input_memory_ptr(), true);
    }
};

}  // namespace

bool ReorderImplementationManager::validate_impl(const program_node& node) const {
    OPENVINO_ASSERT(node.is_type<reorder>(), "[GPU][Vulkan] Invalid node type passed to Reorder manager");
    if (node.get_program().get_engine().runtime_type() != runtime_types::vulkan) {
        return false;
    }

    const auto& input_layout = node.get_input_layout(0);
    const auto& output_layout = node.get_output_layout(0);
    return input_layout.data_type == data_types::f32 && output_layout.data_type == data_types::f32 && is_copy_compatible_format(input_layout.format) &&
           is_copy_compatible_format(output_layout.format) && !input_layout.is_dynamic() && !output_layout.is_dynamic() &&
           !static_cast<bool>(input_layout.data_padding) && !static_cast<bool>(output_layout.data_padding) && input_layout.get_linear_offset() == 0 &&
           output_layout.get_linear_offset() == 0 && input_layout.count() == output_layout.count() && input_layout.bytes_count() == output_layout.bytes_count();
}

std::unique_ptr<primitive_impl> ReorderImplementationManager::create_impl(const program_node& node, const kernel_impl_params&) const {
    OPENVINO_ASSERT(node.is_type<reorder>(), "[GPU][Vulkan] Invalid node type passed to Reorder manager");
    return std::make_unique<reorder_impl>();
}

}  // namespace vulkan
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::vulkan::reorder_impl)
