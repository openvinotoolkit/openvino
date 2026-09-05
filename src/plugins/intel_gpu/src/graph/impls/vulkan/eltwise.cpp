// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise.hpp"

#include <memory>
#include <utility>
#include <vector>

#include "common_utils/eltwise_kernel_params.hpp"
#include "common_utils/kernel_selector_primitive_impl.hpp"
#include "kernel_selector/kernels/eltwise/eltwise_kernel_ref.h"
#include "openvino/core/except.hpp"

namespace cldnn::vulkan {
namespace {

const kernel_selector::EltwiseKernelRef& get_reference_kernel() {
    static const kernel_selector::EltwiseKernelRef kernel;
    return kernel;
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
    const auto canonical_params = canonicalize_eltwise_shapes(params);
    auto kernel_params = make_unfused_eltwise_kernel_params(canonical_params, params.is_dynamic());
    return get_reference_kernel().GetKernelsData(kernel_params);
}

}  // namespace

class eltwise_impl final : public typed_primitive_impl_kernel_selector<eltwise> {
public:
    using parent = typed_primitive_impl_kernel_selector<eltwise>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::vulkan::eltwise_impl)

    eltwise_impl() : parent("vulkan_eltwise_clspv") {}

    eltwise_impl(kernel_selector::KernelData kernel_data, bool is_dynamic) : parent(std::move(kernel_data), is_dynamic) {
        OPENVINO_ASSERT(_kernel_data.kernels.size() == 1, "[GPU][Vulkan] Reference Eltwise expects exactly one kernel-selector dispatch");
    }

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<eltwise_impl>(*this);
    }

    kernel_impl_params canonicalize_shapes(const kernel_impl_params& params) const override {
        return canonicalize_eltwise_shapes(params);
    }

protected:
    void update_dispatch_data(const kernel_impl_params& params) override {
        auto kernel_params = make_unfused_eltwise_kernel_params(params, true);
        if (_kernel_data.update_dispatch_data_func == nullptr) {
            auto candidates = get_reference_kernel().GetKernelsData(kernel_params);
            OPENVINO_ASSERT(candidates.size() == 1, "[GPU][Vulkan] Kernel selector did not restore dynamic Eltwise dispatch");
            _kernel_data.update_dispatch_data_func = std::move(candidates.front().update_dispatch_data_func);
        }
        _kernel_data.update_dispatch_data_func(kernel_params, _kernel_data);
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
