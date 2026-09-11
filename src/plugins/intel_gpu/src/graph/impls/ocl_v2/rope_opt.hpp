// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "program_node.h"
#include "rope_inst.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

struct RopeOpt : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::rope::opt")
    explicit RopeOpt(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_fmts = {
            format::bfyx,
        };

        static constexpr std::array supported_types = {
            ov::element::f32,
            ov::element::f16,
            ov::element::bf16,
        };

        const auto& in0_layout = node.get_input_layout(0);
        const auto& out_layout = node.get_output_layout(0);
        if (!one_of(in0_layout.format, supported_fmts) || !one_of(out_layout.format, supported_fmts)) {
            return false;
        }

        // An i8 output is allowed so that a narrowing conversion after the rotation can ride
        // along in its store instead of costing a separate full-tensor pass. Any other differing
        // output type is rejected: VEC_SIZE is derived from the input type alone, so a
        // mixed-precision configuration would emit vector stores whose element type does not
        // match the output.
        static constexpr std::array supported_out_types = {
            ov::element::f32,
            ov::element::f16,
            ov::element::i8,
        };

        if (out_layout.data_type != in0_layout.data_type && !(out_layout.data_type == ov::element::i8 && i8_output_supported(node))) {
            return false;
        }

        return one_of(in0_layout.data_type, supported_types) && one_of(out_layout.data_type, supported_out_types);
    }

    // The narrowing store is written only in the interleaved body at VEC_SIZE 16, so accept an
    // i8 output only where that is the body this node compiles to. Accepting it anywhere else
    // makes has_impl_for say yes, lets remove_redundant_reorders fuse the conversion away, and
    // then fails as an OpenCL build error at network load instead of simply keeping the reorder.
    [[nodiscard]] static bool i8_output_supported(const program_node& node) {
        const auto& config = node.as<rope>().get_primitive()->config;
        if (config.is_qwen || config.is_chatglm || config.is_ltx_video || !config.is_interleaved) {
            return false;
        }
        // get_vec_size() reaches 16 only for an f16 input whose rotary width is a multiple of
        // 2 * 16, and drops to 1 when the cos/sin tables are f32 while the data is not.
        return node.get_input_layout(0).data_type == ov::element::f16 && node.get_input_layout(1).data_type == ov::element::f16 &&
               config.rotary_ndims % 32 == 0;
    }
};

}  // namespace ov::intel_gpu::ocl
