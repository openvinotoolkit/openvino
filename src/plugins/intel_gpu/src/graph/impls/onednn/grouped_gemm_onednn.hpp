// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#ifndef ENABLE_ONEDNN_FOR_GPU
    #define ENABLE_ONEDNN_FOR_GPU 1
#endif

#include "grouped_matmul_inst.h"
#include "intel_gpu/runtime/utils.hpp"
#include "registry/implementation_manager.hpp"

#include <memory>

namespace cldnn {
namespace onednn {

struct GroupedMatmulImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("onednn::grouped_gemm")
    GroupedMatmulImplementationManager(shape_types shape_type)
        : ImplementationManager(impl_types::onednn, shape_type) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        OPENVINO_ASSERT(node.is_type<grouped_matmul>());
        const auto& config = node.get_program().get_config();
        const auto& info = node.get_program().get_engine().get_device_info();
        if (!info.supports_immad || info.arch == gpu_arch::unknown || !config.get_use_onednn())
            return false;

        static const std::vector<format> supported_fmts = {
            format::bfyx,
        };

        static const std::vector<ov::element::Type_t> supported_activation_types = {
            ov::element::f16,
        };

        static const std::vector<ov::element::Type_t> supported_compressed_weight_types = {
            ov::element::u8,
            ov::element::i8,
            ov::element::u4,
            ov::element::i4,
        };

        static const std::vector<ov::element::Type_t> supported_offset_types = {
            ov::element::i32,
        };

        const auto& in0 = node.get_input_layout(grouped_matmul::GroupedMatmulInputIdx::INPUT);
        const auto& in1 = node.get_input_layout(grouped_matmul::GroupedMatmulInputIdx::WEIGHT);
        const auto& off = node.get_input_layout(grouped_matmul::GroupedMatmulInputIdx::OFFSETS);
        const auto& out = node.get_output_layout(0);

        const auto* desc = node.as<grouped_matmul>().get_primitive().get();
        const bool compressed = desc && desc->compressed_weights;

        if (!one_of(in0.format, supported_fmts) || !one_of(in0.data_type, supported_activation_types))
            return false;
        if (!one_of(in1.format, supported_fmts))
            return false;
        if (compressed) {
            if (!one_of(in1.data_type, supported_compressed_weight_types))
                return false;
        } else if (!one_of(in1.data_type, supported_activation_types)) {
            return false;
        }
        if (!one_of(off.format, supported_fmts) || !one_of(off.data_type, supported_offset_types))
            return false;
        if (!one_of(out.format, supported_fmts) || !one_of(out.data_type, supported_activation_types))
            return false;

        // oneDNN does not support mixed fp16 x bf16 configurations
        if (!compressed &&
            ((in0.data_type == data_types::f16 && in1.data_type == data_types::bf16) ||
             (in0.data_type == data_types::bf16 && in1.data_type == data_types::f16)))
            return false;

        return true;
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        OPENVINO_ASSERT(node.is_type<grouped_matmul>());
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::bfyx);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::bfyx);
        return {in_fmts, out_fmts};
    }
};

}  // namespace onednn
}  // namespace cldnn
