// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder.hpp"

#include <memory>
#include <utility>
#include <vector>

#include "common_utils/kernel_selector_data_adapter.hpp"
#include "common_utils/kernel_selector_primitive_impl.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "kernel_selector/kernels/reorder/reorder_kernel.h"
#include "openvino/core/except.hpp"

namespace cldnn::vulkan {
namespace {

const kernel_selector::ReorderKernelRef& get_reference_kernel() {
    static const kernel_selector::ReorderKernelRef kernel;
    return kernel;
}

bool is_supported_type(data_types type) {
    switch (type) {
    case data_types::f32:
    case data_types::f16:
    case data_types::bf16:
    case data_types::i8:
    case data_types::u8:
    case data_types::i16:
    case data_types::u16:
    case data_types::i32:
    case data_types::u32:
    case data_types::i64:
        return true;
    default:
        return false;
    }
}

bool is_structural_copy(const layout& input, const layout& output) {
    if (input.data_type != output.data_type || input.data_padding || output.data_padding) {
        return false;
    }
    const bool compatible_formats = input.format == output.format || (format::is_default_format(input.format) && format::is_default_format(output.format));
    if (!compatible_formats) {
        return false;
    }
    if (input.is_dynamic() || output.is_dynamic()) {
        return input.is_dynamic() && output.is_dynamic();
    }
    return input.count() == output.count() && input.bytes_count() == output.bytes_count() && input.get_linear_offset() == 0 && output.get_linear_offset() == 0;
}

kernel_selector::reorder_params make_reference_params(const kernel_impl_params& impl_params) {
    const auto& primitive = impl_params.typed_desc<reorder>();
    kernel_selector::reorder_params params;
    params.uniqueID = std::to_string(impl_params.hash());
    params.engineInfo = make_kernel_selector_engine_info(impl_params.get_program().get_engine().get_device_info());
    params.inputs[0] = convert_data_tensor(impl_params.get_input_layout(0));
    params.outputs[0] = convert_data_tensor(impl_params.get_output_layout(0));
    params.layerID = primitive->id;
    params.has_padded_output = static_cast<bool>(impl_params.get_output_layout().data_padding);
    params.mode = kernel_selector::MeanSubtractMode::NONE;
    params.surface_input = false;
    params.truncate = primitive->truncate;
    return params;
}

}  // namespace

class reorder_copy_impl final : public typed_primitive_impl<reorder> {
public:
    using parent = typed_primitive_impl<reorder>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::vulkan::reorder_copy_impl)

    reorder_copy_impl() : parent("vulkan_reorder_copy") {}

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<reorder_copy_impl>(*this);
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
        OPENVINO_ASSERT(is_structural_copy(instance.get_input_layout(0), instance.get_output_layout(0)),
                        "[GPU][Vulkan] Structural Reorder requires byte-compatible runtime layouts");
        stream.wait_for_events(events);
        return instance.output_memory_ptr()->copy_from(stream, *instance.input_memory_ptr(), true);
    }
};

class reorder_impl final : public typed_primitive_impl_kernel_selector<reorder> {
public:
    using parent = typed_primitive_impl_kernel_selector<reorder>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::vulkan::reorder_impl)

    reorder_impl() : parent("vulkan_reorder_clspv") {}

    explicit reorder_impl(kernel_selector::KernelData kernel_data) : parent(std::move(kernel_data)) {
        OPENVINO_ASSERT(_kernel_data.kernels.size() == 1, "[GPU][Vulkan] Reference Reorder expects exactly one kernel-selector dispatch");
    }

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<reorder_impl>(*this);
    }
};

bool ReorderImplementationManager::validate_impl(const program_node& node) const {
    OPENVINO_ASSERT(node.is_type<reorder>(), "[GPU][Vulkan] Invalid node type passed to Reorder manager");
    if (node.get_program().get_engine().runtime_type() != runtime_types::vulkan) {
        return false;
    }
    const auto& input = node.get_input_layout(0);
    const auto& output = node.get_output_layout(0);
    const auto& primitive = node.as<reorder>().get_primitive();
    const bool common_contract = is_supported_type(input.data_type) && is_supported_type(output.data_type) && format::is_simple_data_format(input.format) &&
                                 format::is_simple_data_format(output.format) && !format::is_weights_format(input.format) &&
                                 !format::is_weights_format(output.format) && !primitive->mean.is_valid() && primitive->subtract_per_feature.empty() &&
                                 !primitive->has_surface_input();
    if (!common_contract) {
        return false;
    }
    if (input.is_dynamic() || output.is_dynamic()) {
        return is_structural_copy(input, output);
    }
    return input.count() == output.count();
}

std::unique_ptr<primitive_impl> ReorderImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<reorder>(), "[GPU][Vulkan] Invalid node type passed to Reorder manager");
    if (is_structural_copy(node.get_input_layout(0), node.get_output_layout(0))) {
        return std::make_unique<reorder_copy_impl>();
    }
    auto candidates = get_reference_kernel().GetKernelsData(make_reference_params(params));
    OPENVINO_ASSERT(candidates.size() == 1, "[GPU][Vulkan] Kernel selector did not produce the generic reference Reorder kernel");
    return std::make_unique<reorder_impl>(std::move(candidates.front()));
}

}  // namespace cldnn::vulkan

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::vulkan::reorder_copy_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::vulkan::reorder_impl)
