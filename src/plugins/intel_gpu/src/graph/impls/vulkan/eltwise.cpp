// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise.hpp"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "common_utils/eltwise_kernel_params.hpp"
#include "common_utils/kernel_selector_data_adapter.hpp"
#include "common_utils/kernel_selector_primitive_impl.hpp"
#include "kernel_selector/kernels/eltwise/eltwise_kernel_ref.h"
#include "kernel_selector/kernels/reorder/reorder_kernel.h"
#include "openvino/core/except.hpp"

namespace cldnn::vulkan {
namespace {

const kernel_selector::KernelBase& get_reference_kernel() {
    static const kernel_selector::EltwiseKernelRef kernel;
    return kernel;
}

const kernel_selector::KernelBase& get_input_conversion_kernel() {
    static const kernel_selector::ReorderKernelRef kernel;
    return kernel;
}

bool use_input_conversion_prototype(const kernel_impl_params& params) {
    // Opt-in prototype, not a device-name rule or a retry after a driver error.
    static const bool enabled = [] {
        const auto* value = std::getenv("OV_GPU_VULKAN_LOGICAL_XOR_I32_INPUT");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled && params.typed_desc<eltwise>()->mode == eltwise_mode::logic_xor && params.input_layouts.size() == 2 &&
           params.get_input_layout(0).data_type == data_types::u8 && params.get_input_layout(1).data_type == data_types::u8 &&
           params.get_output_layout().data_type == data_types::u8;
}

layout converted_input_layout(const kernel_impl_params& params) {
    auto result = params.get_input_layout(0);
    result.data_type = data_types::i32;
    return result;
}

kernel_selector::reorder_params make_input_conversion_params(const kernel_impl_params& params, bool is_dynamic) {
    kernel_selector::reorder_params conversion;
    conversion.uniqueID = std::to_string(params.hash()) + "_logical_input_i32";
    conversion.layerID = params.desc->id;
    conversion.engineInfo = make_kernel_selector_engine_info(params.get_program().get_engine().get_device_info());
    // A broadcast input can be static even when Eltwise's output is dynamic.
    conversion.is_shape_agnostic = is_dynamic && params.get_input_layout(0).is_dynamic();
    conversion.inputs[0] = convert_data_tensor(params.get_input_layout(0));
    conversion.outputs[0] = convert_data_tensor(converted_input_layout(params));
    conversion.has_padded_output = static_cast<bool>(params.get_input_layout(0).data_padding);
    conversion.mode = kernel_selector::MeanSubtractMode::NONE;
    // The temporary has the first input's shape/padding, not Eltwise's output
    // shape. Both sides of Reorder use that existing shape-info record.
    conversion.set_dynamic_shape_offsets({{0, 0}}, {{0, 0}});
    return conversion;
}

bool is_portable_type(data_types type) {
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

bool has_portable_contract(const program_node& node) {
    if (node.has_fused_primitives() || !is_portable_type(node.get_output_layout(0).data_type)) {
        return false;
    }
    for (const auto& dependency : node.get_dependencies()) {
        if (!is_portable_type(dependency.first->get_output_layout(dependency.second).data_type)) {
            return false;
        }
    }
    return true;
}

kernel_selector::KernelsData get_reference_kernels_data(const kernel_impl_params& params) {
    auto canonical_params = canonicalize_eltwise_shapes(params);
    const bool convert_input = use_input_conversion_prototype(params);
    kernel_selector::KernelsData conversion;
    if (convert_input) {
        conversion = get_input_conversion_kernel().GetKernelsData(make_input_conversion_params(canonical_params, params.is_dynamic()));
        OPENVINO_ASSERT(conversion.size() == 1 && conversion.front().kernels.size() == 1,
                        "[GPU][Vulkan] Logical input conversion requires the reference Reorder kernel");
        canonical_params.input_layouts[0] = converted_input_layout(canonical_params);
    }
    auto kernel_params = make_unfused_eltwise_kernel_params(canonical_params, params.is_dynamic());
    auto candidates = get_reference_kernel().GetKernelsData(kernel_params);
    if (convert_input && candidates.size() == 1) {
        auto& data = candidates.front();
        OPENVINO_ASSERT(data.kernels.size() == 1, "[GPU][Vulkan] Expected one reference Eltwise dispatch");
        auto& reorder_dispatch = conversion.front().kernels.front();
        for (auto& argument : reorder_dispatch.params.arguments) {
            if (argument.t == argument_desc::Types::OUTPUT) {
                argument = {argument_desc::Types::INTERNAL_BUFFER, 0};
            }
        }
        for (auto& argument : data.kernels.front().params.arguments) {
            if (argument.t == argument_desc::Types::INPUT && argument.index == 0) {
                argument = {argument_desc::Types::INTERNAL_BUFFER, 0};
            }
        }
        data.kernels.insert(data.kernels.begin(), std::move(reorder_dispatch));
        data.internalBufferDataType = kernel_selector::Datatype::INT32;
        // The instance requests the full current layout through BufferDescriptor,
        // including padding and dynamic shape updates, rather than a fixed size.
        data.internalBuffers.emplace_back();
        data.needs_sub_kernels_sync = true;
        data.kernelName = "vulkan_eltwise_logical_input_i32_prototype";
    }
    return candidates;
}

}  // namespace

class eltwise_impl final : public typed_primitive_impl_kernel_selector<eltwise> {
public:
    using parent = typed_primitive_impl_kernel_selector<eltwise>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::vulkan::eltwise_impl)

    eltwise_impl() : parent("vulkan_eltwise_clspv") {}

    eltwise_impl(kernel_selector::KernelData kernel_data, bool is_dynamic) : parent(std::move(kernel_data), is_dynamic) {
        const size_t expected_dispatches = has_input_conversion() ? converted_dispatch_count : 1;
        OPENVINO_ASSERT(_kernel_data.kernels.size() == expected_dispatches, "[GPU][Vulkan] Unexpected reference Eltwise dispatch count");
    }

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<eltwise_impl>(*this);
    }

    kernel_impl_params canonicalize_shapes(const kernel_impl_params& params) const override {
        return canonicalize_eltwise_shapes(params);
    }

    std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        if (!has_input_conversion()) {
            return {};
        }
        return {BufferDescriptor(converted_input_layout(canonicalize_eltwise_shapes(params)))};
    }

protected:
    void update_dispatch_data(const kernel_impl_params& params) override {
        if (has_input_conversion()) {
            update_stage(conversion_stage, get_input_conversion_kernel(), make_input_conversion_params(params, true));
            auto converted_params = params;
            converted_params.input_layouts[0] = converted_input_layout(params);
            update_stage(eltwise_stage, get_reference_kernel(), make_unfused_eltwise_kernel_params(converted_params, true));
            return;
        }
        auto kernel_params = make_unfused_eltwise_kernel_params(params, true);
        if (_kernel_data.update_dispatch_data_func == nullptr) {
            get_reference_kernel().GetUpdateDispatchDataFunc(_kernel_data);
        }
        _kernel_data.update_dispatch_data_func(kernel_params, _kernel_data);
    }

private:
    static constexpr size_t conversion_stage = 0;
    static constexpr size_t eltwise_stage = 1;
    static constexpr size_t converted_dispatch_count = 2;

    bool has_input_conversion() const {
        return !_kernel_data.internalBuffers.empty();
    }

    void update_stage(size_t index, const kernel_selector::KernelBase& kernel, const kernel_selector::Params& params) {
        kernel_selector::KernelData stage;
        stage.kernels.push_back(_kernel_data.kernels.at(index));
        kernel.GetUpdateDispatchDataFunc(stage);
        stage.update_dispatch_data_func(params, stage);
        _kernel_data.kernels[index] = std::move(stage.kernels.front());
    }
};

bool EltwiseImplementationManager::validate_impl(const program_node& node) const {
    OPENVINO_ASSERT(node.is_type<eltwise>(), "[GPU][Vulkan] Invalid node type passed to Eltwise manager");
    return node.get_program().get_engine().runtime_type() == runtime_types::vulkan && has_portable_contract(node);
}

std::unique_ptr<primitive_impl> EltwiseImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<eltwise>(), "[GPU][Vulkan] Invalid node type passed to Eltwise manager");
    auto candidates = get_reference_kernels_data(params);
    OPENVINO_ASSERT(candidates.size() == 1, "[GPU][Vulkan] Kernel selector did not produce the generic reference Eltwise kernel");
    return std::make_unique<eltwise_impl>(std::move(candidates.front()), params.is_dynamic());
}

}  // namespace cldnn::vulkan

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::vulkan::eltwise_impl)
