// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

struct SegmentMax : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::segment_max")
    explicit SegmentMax(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_data_types = {
            ov::element::f32,
            ov::element::f16,
            ov::element::i8,
            ov::element::u8,
            ov::element::i32,
            ov::element::i64,
        };

        static constexpr std::array supported_segment_ids_types = {
            ov::element::i32,
            ov::element::i64,
        };

        const auto& in0_layout = node.get_input_layout(0);  // data
        const auto& in1_layout = node.get_input_layout(1);  // segment_ids
        const auto& out_layout = node.get_output_layout(0);

        if (!one_of(in0_layout.data_type, supported_data_types) || !one_of(out_layout.data_type, supported_data_types)) {
            return false;
        }

        if (!one_of(in1_layout.data_type, supported_segment_ids_types)) {
            return false;
        }

        if (node.has_fused_primitives()) {
            return false;
        }

        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
