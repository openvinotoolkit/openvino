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
#include "storage_type_jit.hpp"
#include "vulkan/vulkan_device.hpp"

namespace cldnn::vulkan {
namespace {

// Preserve the I64 tensor's two-word storage at the boundary of the common I32
// precision path. Reuse the reference kernel's indexing and conversion hooks;
// this adapter does not implement I64 Eltwise arithmetic.
class I64BoundaryReorderKernel : public kernel_selector::ReorderKernelRef {
public:
    kernel_selector::JitConstants GetJitConstants(const kernel_selector::reorder_params& params) const override {
        auto jit = ReorderKernelRef::GetJitConstants(params);
        const auto replace = [&](const std::string& name, const std::string& value) {
            jit.RemoveConstant(name);
            jit.AddConstant(kernel_selector::MakeJitConstant(name, value));
        };
        for (const auto* prefix : {"CALC", "INPUT_REORDER", "OUTPUT_REORDER"}) {
            for (const auto& definition :
                 kernel_selector::MakeTypeJitConstants(kernel_selector::Datatype::INT32, prefix).GetDefinitions()) {
                replace(definition.first, definition.second);
            }
        }
        if (params.inputs[0].GetDType() == kernel_selector::Datatype::INT64) {
            replace("INPUT_REORDER_TYPE", "int2");
            replace("INPUT_REORDER_TYPE_SIZE", std::to_string(sizeof(int64_t)));
            replace("DECODE_INPUT_REORDER_COMPUTE_TYPE(v)", "((v).s0)");
        } else {
            replace("OUTPUT_REORDER_TYPE", "int2");
            replace("OUTPUT_REORDER_TYPE_SIZE", std::to_string(sizeof(int64_t)));
            // Vector select uses the mask's sign bit: the high word is zero or
            // minus one, exactly the sign extension of the existing I32 result.
            replace("TO_OUTPUT_REORDER_TYPE(v)", "select((int2)((v), 0), (int2)((v), -1), (int2)(v))");
        }
        return jit;
    }
};

const kernel_selector::KernelBase& get_reference_kernel() {
    static const kernel_selector::ReorderKernelRef kernel;
    return kernel;
}

bool needs_i64_boundary(const kernel_impl_params& params) {
    const auto input_type = params.get_input_layout(0).data_type;
    const auto output_type = params.get_output_layout(0).data_type;
    if ((input_type == data_types::i64 && output_type == data_types::i32) ||
        (input_type == data_types::i32 && output_type == data_types::i64)) {
        const auto& device = static_cast<const vulkan_device&>(*params.get_program().get_engine().get_device());
        return !device.supports_arithmetic_type(data_types::i64);
    }
    return false;
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

kernel_selector::reorder_params make_reference_params(const kernel_impl_params& impl_params, bool is_shape_agnostic) {
    const auto& primitive = impl_params.typed_desc<reorder>();
    kernel_selector::reorder_params params;
    params.uniqueID = std::to_string(impl_params.hash());
    params.engineInfo = make_kernel_selector_engine_info(impl_params.get_program().get_engine().get_device_info());
    params.is_shape_agnostic = is_shape_agnostic;
    params.inputs[0] = convert_data_tensor(impl_params.get_input_layout(0));
    params.outputs[0] = convert_data_tensor(impl_params.get_output_layout(0));
    params.layerID = primitive->id;
    params.has_padded_output = static_cast<bool>(impl_params.get_output_layout().data_padding);
    params.mode = kernel_selector::MeanSubtractMode::NONE;
    params.surface_input = false;
    params.truncate = primitive->truncate;
    params.set_dynamic_shape_offsets();
    return params;
}

}  // namespace

class reorder_copy_impl final : public typed_primitive_impl<reorder> {
public:
    using parent = typed_primitive_impl<reorder>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::vulkan::reorder_copy_impl)

    explicit reorder_copy_impl(bool is_dynamic = false) : parent("vulkan_reorder_copy", is_dynamic) {}

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
        // Shape prediction can allocate more input storage than the current tensor requires.
        return instance.output_memory_ptr()->copy_from(stream, *instance.input_memory_ptr(), 0, 0, instance.get_output_layout().bytes_count(), true);
    }
};

class reorder_impl final : public typed_primitive_impl_kernel_selector<reorder> {
public:
    using parent = typed_primitive_impl_kernel_selector<reorder>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::vulkan::reorder_impl)

    reorder_impl() : parent("vulkan_reorder_clspv") {}

    reorder_impl(kernel_selector::KernelData kernel_data, bool is_dynamic) : parent(std::move(kernel_data), is_dynamic) {
        OPENVINO_ASSERT(_kernel_data.kernels.size() == 1, "[GPU][Vulkan] Reference Reorder expects exactly one kernel-selector dispatch");
    }

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<reorder_impl>(*this);
    }

protected:
    void update_dispatch_data(const kernel_impl_params& params) override {
        const auto kernel_params = make_reference_params(params, true);
        if (_kernel_data.update_dispatch_data_func == nullptr) {
            get_reference_kernel().GetUpdateDispatchDataFunc(_kernel_data);
        }
        _kernel_data.update_dispatch_data_func(kernel_params, _kernel_data);
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
    const bool common_contract = is_supported_type(input.data_type) && is_supported_type(output.data_type) && !input.format.is_image() &&
                                 !output.format.is_image() && !format::is_weights_format(input.format) && !format::is_weights_format(output.format) &&
                                 !primitive->mean.is_valid() && primitive->subtract_per_feature.empty() && !primitive->has_surface_input();
    if (!common_contract) {
        return false;
    }
    if (input.is_dynamic() || output.is_dynamic()) {
        return true;
    }
    return input.count() == output.count();
}

std::unique_ptr<primitive_impl> ReorderImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<reorder>(), "[GPU][Vulkan] Invalid node type passed to Reorder manager");
    if (is_structural_copy(node.get_input_layout(0), node.get_output_layout(0))) {
        return std::make_unique<reorder_copy_impl>(params.is_dynamic());
    }
    const auto& device = static_cast<const vulkan_device&>(*params.get_program().get_engine().get_device());
    const auto kernel_params = make_reference_params(params, params.is_dynamic());
    auto candidates = needs_i64_boundary(params)
                          ? StorageTypeKernel<I64BoundaryReorderKernel, kernel_selector::reorder_params>(device).GetKernelsData(kernel_params)
                          : StorageTypeKernel<kernel_selector::ReorderKernelRef, kernel_selector::reorder_params>(device).GetKernelsData(kernel_params);
    OPENVINO_ASSERT(candidates.size() == 1, "[GPU][Vulkan] Kernel selector did not produce the generic reference Reorder kernel");
    candidates.front().kernelName = get_reference_kernel().GetName();
    get_reference_kernel().GetUpdateDispatchDataFunc(candidates.front());
    return std::make_unique<reorder_impl>(std::move(candidates.front()), params.is_dynamic());
}

}  // namespace cldnn::vulkan

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::vulkan::reorder_copy_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::vulkan::reorder_impl)
