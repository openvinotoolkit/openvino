// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/fully_connected_gguf.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

struct FullyConnectedGGUFImpl : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::fully_connected_gguf")

    explicit FullyConnectedGGUFImpl(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        // The decode kernels use int8 dot products.
        if (!node.get_program().get_engine().get_device_info().supports_imad) {
            return false;
        }
        const auto& in = node.get_input_layout(0);
        const auto& weights = node.get_input_layout(1);
        const auto& out = node.get_output_layout(0);
        if (!one_of(in.data_type, {data_types::f16, data_types::f32}) || !one_of(out.data_type, {data_types::f16, data_types::f32}) ||
            weights.data_type != data_types::u8) {
            return false;
        }
        if (!format::is_default_format(in.format) || !format::is_default_format(out.format) || in.data_padding || out.data_padding) {
            return false;
        }
        const auto& w_shape = weights.get_partial_shape();
        if (w_shape.is_dynamic() || w_shape.size() != 2) {
            return false;
        }
        const auto format = node.get_kernel_impl_params()->typed_desc<fully_connected_gguf>()->format;
        return w_shape[1].get_length() % static_cast<int64_t>(ov::op::internal::GGUFDequantize::block_bytes(format)) == 0;
    }
};

}  // namespace ov::intel_gpu::ocl
