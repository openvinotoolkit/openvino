// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape.hpp"

#include <memory>

#include "intel_gpu/runtime/stream.hpp"
#include "registry/implementation_map.hpp"

namespace cldnn {
namespace vulkan {
namespace {

bool is_copy_compatible_format(format fmt) {
    return fmt == format::any || format::is_default_format(fmt);
}

struct reshape_impl : typed_primitive_impl<reshape> {
    using parent = typed_primitive_impl<reshape>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::vulkan::reshape_impl)

    reshape_impl() : parent("vulkan_reshape") {}

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<reshape_impl>(*this);
    }

    bool is_cpu() const override {
        return false;
    }

    bool requires_lockable_input() const override {
        return false;
    }

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}

    event::ptr execute_impl(const std::vector<event::ptr>& events, reshape_inst& instance) override {
        auto& stream = instance.get_network().get_stream();
        stream.wait_for_events(events);
        return instance.output_memory_ptr()->copy_from(stream, *instance.input_memory_ptr(), true);
    }
};

}  // namespace

bool ReshapeImplementationManager::validate_impl(const program_node& node) const {
    OPENVINO_ASSERT(node.is_type<reshape>(), "[GPU][Vulkan] Invalid node type passed to Reshape manager");
    if (node.get_program().get_engine().runtime_type() != runtime_types::vulkan) {
        return false;
    }

    const auto& input_layout = node.get_input_layout(0);
    const auto& output_layout = node.get_output_layout(0);
    return input_layout.data_type == output_layout.data_type && data_type_traits::size_of(input_layout.data_type) >= 1 &&
           is_copy_compatible_format(input_layout.format) && is_copy_compatible_format(output_layout.format) && !input_layout.is_dynamic() &&
           !output_layout.is_dynamic() && !static_cast<bool>(input_layout.data_padding) && !static_cast<bool>(output_layout.data_padding) &&
           input_layout.get_linear_offset() == 0 && output_layout.get_linear_offset() == 0 && input_layout.count() == output_layout.count() &&
           input_layout.bytes_count() == output_layout.bytes_count();
}

std::unique_ptr<primitive_impl> ReshapeImplementationManager::create_impl(const program_node& node, const kernel_impl_params&) const {
    OPENVINO_ASSERT(node.is_type<reshape>(), "[GPU][Vulkan] Invalid node type passed to Reshape manager");
    return std::make_unique<reshape_impl>();
}

}  // namespace vulkan
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::vulkan::reshape_impl)
